"""
SIMILARITY AGENT AI - Agent Intelligent pour Décision de Crédit
Refactorisé avec LangChain & LangGraph
"""

import os
import json
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, asdict

# LangChain / LangGraph Imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Original Imports maintained for logic
from qdrant_client import QdrantClient

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# SECURITE : On ne met plus de clés en dur ici. Elles doivent être dans le fichier .env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "credit_dataset"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_SIMILAR = 20

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = """Tu es un expert analyste credit senior avec 20 ans d'experience.
Tu analyses les demandes de credit en comparant avec des cas historiques.
Tu dois fournir une analyse objective et explicable. 
Reponds TOUJOURS en JSON valide."""


# ==============================================================================
# CREDIT PROFILE (Inchangé)
# ==============================================================================

@dataclass
class CreditProfile:
    loan_amount: float
    loan_duration: int
    monthly_income:  float
    other_income: float
    monthly_charges: float
    employment_type:  str
    contract_type: str
    seniority_years:  int
    marital_status: str
    number_of_children: int
    spouse_employed: Optional[bool]
    housing_status: str
    is_primary_holder: bool
    
    def to_text(self) -> str:
        total_income = self.monthly_income + (self.other_income or 0)
        return f"""
        Demande de pret:  {self.loan_amount} euros sur {self.loan_duration} mois
        Revenus:  {total_income} euros, Charges: {self.monthly_charges} euros
        Emploi: {self.employment_type} ({self.contract_type}), {self.seniority_years} ans
        Situation:  {self.marital_status}, {self.number_of_children} enfants
        Logement: {self.housing_status}
        """.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        total_income = self.monthly_income + (self.other_income or 0)
        monthly_payment = self.loan_amount / self.loan_duration if self.loan_duration > 0 else 0
        current_debt_ratio = (self.monthly_charges / total_income * 100) if total_income > 0 else 0
        projected_debt_ratio = ((self.monthly_charges + monthly_payment) / total_income * 100) if total_income > 0 else 0
        
        return {
            "loan_amount": self.loan_amount,
            "loan_duration":  self.loan_duration,
            "monthly_payment": round(monthly_payment, 2),
            "monthly_income": self.monthly_income,
            "other_income": self.other_income,
            "total_income": total_income,
            "monthly_charges": self. monthly_charges,
            "current_debt_ratio": round(current_debt_ratio, 2),
            "projected_debt_ratio": round(projected_debt_ratio, 2),
            "employment_type": self.employment_type,
            "contract_type": self.contract_type,
            "seniority_years": self.seniority_years,
            "marital_status": self.marital_status,
            "number_of_children": self.number_of_children,
            "spouse_employed":  self.spouse_employed,
            "housing_status": self.housing_status,
            "is_primary_holder": self.is_primary_holder
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreditProfile":
        return cls(
            loan_amount=data. get("loan_amount", 0),
            loan_duration=data.get("loan_duration", 0),
            monthly_income=data.get("monthly_income", 0),
            other_income=data.get("other_income", 0),
            monthly_charges=data.get("monthly_charges", 0),
            employment_type=data.get("employment_type", "unknown"),
            contract_type=data.get("contract_type", "unknown"),
            seniority_years=data.get("seniority_years", 0),
            marital_status=data. get("marital_status", "unknown"),
            number_of_children=data.get("number_of_children", 0),
            spouse_employed=data.get("spouse_employed"),
            housing_status=data.get("housing_status", "unknown"),
            is_primary_holder=data. get("is_primary_holder", True)
        )


# ==============================================================================
# LANGGRAPH STATE DEFINITION
# ==============================================================================

class AgentState(TypedDict):
    request_data: Dict[str, Any]      # Données brutes d'entrée
    profile: Optional[CreditProfile]  # Objet profil instancié
    profile_dict: Dict[str, Any]      # Profil formaté pour calculs
    query_vector: List[float]         # Embedding
    similar_cases: List[Dict]         # Résultats Qdrant
    stats: Dict[str, Any]             # Statistiques calculées
    ai_analysis: Dict[str, Any]       # Réponse du LLM
    final_output: Dict[str, Any]      # Résultat final formaté


# ==============================================================================
# SIMILARITY AGENT AI (LangChain Version)
# ==============================================================================

class SimilarityAgentAI:
    
    def __init__(self):
        print("Initialisation du Similarity Agent AI (LangChain/LangGraph)...")
        print("=" * 60)
        
        # 1. Init Vector DB Client (Qdrant)
        if not QDRANT_URL or not QDRANT_API_KEY:
             print("ATTENTION: QDRANT_URL ou QDRANT_API_KEY manquant dans les variables d'environnement")
        
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"Qdrant connecte: {str(QDRANT_URL)[:30]}...")
        
        # 2. Init Embeddings (LangChain HuggingFace Wrapper)
        # Remplace SentenceTransformer direct par LangChain wrapper
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("Modele d'embedding: " + EMBEDDING_MODEL)
        
        # 3. Init LLM (LangChain ChatOpenAI)
        if OPENAI_API_KEY: 
            self.llm = ChatOpenAI(
                api_key=OPENAI_API_KEY, 
                base_url=OPENAI_BASE_URL,
                model=LLM_MODEL,
                temperature=0.2,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            self.llm_enabled = True
            print("LLM connecte: " + LLM_MODEL)
        else:
            self.llm = None
            self.llm_enabled = False
            print("LLM non configure (OPENAI_API_KEY manquant)")
        
        self.collection_name = COLLECTION_NAME
        self.top_k = TOP_K_SIMILAR
        
        # 4. Construire le Graph LangGraph
        self.graph = self._build_graph()

        print("=" * 60)
        print("Similarity Agent AI initialise avec succes!")

    # --------------------------------------------------------------------------
    # LOGIQUE METIER (Helpers)
    # --------------------------------------------------------------------------
    
    def _format_cases_for_llm(self, cases: List[Dict]) -> str:
        if not cases:
            return "Aucun cas similaire."
        
        lines = []
        for i, c in enumerate(cases[: 10], 1):
            p = c["payload"]
            status = "DEFAUT" if c["defaulted"] else "OK"
            fraud = " FRAUDE" if c["fraud_flag"] else ""
            similarity_pct = int(c["similarity_score"] * 100)
            loan = p.get("loan_amount", 0)
            duration = p.get("loan_duration", 0)
            emp = p.get("employment_type", "? ")
            contract = p.get("contract_type", "?")
            lines.append(str(i) + ". [" + str(similarity_pct) + "%] " + status + fraud + " - " + str(loan) + "E/" + str(duration) + "m - " + emp + " (" + contract + ")")
        return "\n".join(lines)
    
    def _build_prompt_content(self, profile: Dict, cases: List[Dict], stats: Dict) -> str:
        cases_text = self._format_cases_for_llm(cases)
        success_pct = int(stats["success_rate"] * 100)
        default_pct = int(stats["default_rate"] * 100)
        fraud_pct = int(stats["fraud_rate"] * 100)
        similarity_pct = int(stats["avg_similarity"] * 100)
        
        prompt = """
## NOUVEAU DOSSIER: 

- Pret:  """ + str(profile["loan_amount"]) + """E sur """ + str(profile["loan_duration"]) + """ mois (mensualite: """ + str(profile["monthly_payment"]) + """E)
- Revenus: """ + str(profile["monthly_income"]) + """E/mois + """ + str(profile["other_income"]) + """E autres = """ + str(profile["total_income"]) + """E total
- Charges: """ + str(profile["monthly_charges"]) + """E/mois
- Ratio endettement: actuel """ + str(profile["current_debt_ratio"]) + """% - projete """ + str(profile["projected_debt_ratio"]) + """%
- Emploi: """ + str(profile["employment_type"]) + """ (""" + str(profile["contract_type"]) + """), """ + str(profile["seniority_years"]) + """ ans anciennete
- Situation: """ + str(profile["marital_status"]) + """, """ + str(profile["number_of_children"]) + """ enfant(s), conjoint employe:  """ + str(profile["spouse_employed"]) + """
- Logement: """ + str(profile["housing_status"]) + """

## CAS SIMILAIRES (""" + str(stats["total_similar"]) + """ trouves):
""" + cases_text + """

## STATISTIQUES:
- Succes: """ + str(stats["good_profiles"]) + """/""" + str(stats["total_similar"]) + """ (""" + str(success_pct) + """%)
- Defauts: """ + str(stats["bad_profiles"]) + """/""" + str(stats["total_similar"]) + """ (""" + str(default_pct) + """%)
- Fraudes: """ + str(stats["fraud_cases"]) + """/""" + str(stats["total_similar"]) + """ (""" + str(fraud_pct) + """%)
- Similarite moyenne: """ + str(similarity_pct) + """%

## REPONDS EN JSON VALIDE: 
{
    "recommendation": "APPROUVER ou APPROUVER_AVEC_CONDITIONS ou REVISER ou REFUSER",
    "confidence_level": "high ou medium ou low",
    "risk_score": "nombre entre 0.0 et 1.0",
    "risk_level": "faible ou modere ou eleve",
    "points_forts": ["point 1", "point 2"],
    "points_faibles":  ["point 1", "point 2"],
    "reasoning": "Explication en 2-3 phrases",
    "conditions": ["condition 1 si necessaire"],
    "red_flags": ["alerte si presente"],
    "summary": "Resume en 1 phrase"
}
"""
        return prompt

    def _fallback_analysis(self) -> Dict[str, Any]:
        return {
            "recommendation": "REVISER",
            "confidence_level": "low",
            "risk_score":  0.5,
            "risk_level": "modere",
            "points_forts": ["Analyse automatique non disponible"],
            "points_faibles": ["Analyse automatique non disponible"],
            "reasoning": "Le systeme AI n'est pas disponible.  Revision manuelle necessaire.",
            "conditions": ["Revision manuelle obligatoire"],
            "red_flags": ["Analyse automatique indisponible"],
            "summary": "Revision manuelle requise - systeme AI non disponible."
        }

    # --------------------------------------------------------------------------
    # LANGGRAPH NODES
    # --------------------------------------------------------------------------

    def node_extract_profile(self, state: AgentState) -> Dict:
        """Etape 1: Extraction du profil"""
        print("")
        print("Etape 1/5: Extraction du profil...")
        profile = CreditProfile.from_dict(state["request_data"])
        profile_dict = profile.to_dict()
        
        print("   Profil extrait:  " + str(profile.loan_amount) + "E sur " + str(profile.loan_duration) + " mois")
        print("   Ratio d'endettement projete: " + str(profile_dict["projected_debt_ratio"]) + "%")
        
        return {"profile": profile, "profile_dict": profile_dict}

    def node_generate_embedding(self, state: AgentState) -> Dict:
        """Etape 2: Generation de l'embedding"""
        print("")
        print("Etape 2/5: Generation de l'embedding...")
        profile = state.get("profile")
        if profile is None:
            raise ValueError("Profil manquant pour la generation d'embedding")

        text_to_embed = profile.to_text()
        
        # Utilisation de LangChain Embeddings
        query_vector = self.embedding_model.embed_query(text_to_embed)
        
        print("   Embedding genere: " + str(len(query_vector)) + " dimensions")
        return {"query_vector": query_vector}

    def node_search_similar(self, state: AgentState) -> Dict:
        """Etape 3: Recherche Qdrant"""
        print("")
        print("Etape 3/5: Recherche des " + str(self.top_k) + " cas similaires...")
        
        query_vector = state.get("query_vector", [])
        similar_cases: List[Dict[str, Any]] = []
        
        try:
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.top_k,
                with_payload=True
            )
            points = results.points if hasattr(results, "points") else results
        except Exception as e:
            print("Erreur Qdrant: " + str(e))
            points = []
        
        for result in points:
            if hasattr(result, "payload"):
                payload = getattr(result, "payload", {}) or {}
                score = float(getattr(result, "score", 0) or 0)
            elif isinstance(result, dict):
                payload = result.get("payload", {}) or {}
                score = float(result.get("score", 0) or 0)
            else:
                payload = {}
                score = 0.0

            similar_cases.append({
                "case_id": payload.get("case_id"),
                "similarity_score": score,
                "defaulted": payload.get("defaulted", False),
                "fraud_flag": payload.get("fraud_flag", False),
                "payload": payload
            })
            
        print("   " + str(len(similar_cases)) + " cas similaires trouves")
        
        if similar_cases:
            print("")
            print("   Top 5 des cas similaires:")
            for i, c in enumerate(similar_cases[:5], 1):
                status = "Defaut" if c["defaulted"] else "OK"
                fraud = " FRAUDE" if c["fraud_flag"] else ""
                score_pct = int(c["similarity_score"] * 100)
                print("      " + str(i) + ". Case #" + str(c["case_id"]) + ": " + str(score_pct) + "% | " + status + fraud)
                
        return {"similar_cases": similar_cases}

    def node_compute_stats(self, state: AgentState) -> Dict:
        """Etape 4: Calcul statistiques"""
        print("")
        print("Etape 4/5: Analyse statistique...")
        similar_cases = state["similar_cases"]
        
        if not similar_cases:
            stats = {
                "total_similar":  0, "good_profiles": 0, "bad_profiles": 0, "fraud_cases": 0,
                "success_rate": 0, "default_rate": 0, "fraud_rate": 0, "avg_similarity": 0
            }
        else:
            total = len(similar_cases)
            good = sum(1 for c in similar_cases if not c["defaulted"])
            bad = sum(1 for c in similar_cases if c["defaulted"])
            fraud = sum(1 for c in similar_cases if c["fraud_flag"])
            avg_sim = sum(c["similarity_score"] for c in similar_cases) / total
            
            stats = {
                "total_similar": total,
                "good_profiles": good,
                "bad_profiles": bad,
                "fraud_cases":  fraud,
                "success_rate": good / total,
                "default_rate": bad / total,
                "fraud_rate": fraud / total,
                "avg_similarity":  avg_sim
            }
            
        print("   Taux de succes historique: " + str(int(stats["success_rate"] * 100)) + "%")
        print("   Taux de defaut historique: " + str(int(stats["default_rate"] * 100)) + "%")
        print("   Taux de fraude historique: " + str(int(stats["fraud_rate"] * 100)) + "%")
        
        return {"stats": stats}

    def node_ai_analysis(self, state: AgentState) -> Dict:
        """Etape 5: Appel LLM via LangChain"""
        print("")
        print("Etape 5/5: Analyse AI en cours...")
        
        if not self.llm_enabled:
            print("   LLM non disponible, utilisation de l'analyse de secours")
            return {"ai_analysis": self._fallback_analysis()}
            
        profile_dict = state["profile_dict"]
        similar_cases = state["similar_cases"]
        stats = state["stats"]
        
        prompt_content = self._build_prompt_content(profile_dict, similar_cases, stats)
        
        try:
            # Appel LangChain ChatOpenAI
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_content)
            ]
            response = self.llm.invoke(messages)
            ai_analysis = json.loads(response.content)
            print("   Analyse LLM terminee")
            return {"ai_analysis": ai_analysis}
            
        except Exception as e:
            print("Erreur LLM: " + str(e))
            return {"ai_analysis": self._fallback_analysis()}

    def node_format_output(self, state: AgentState) -> Dict:
        """Etape Finale: Construction de la reponse"""
        stats = state["stats"]
        ai_analysis = state["ai_analysis"]
        
        result = {
            "profile": state["profile_dict"],
            "rag_statistics": {
                "total_similar_cases": stats["total_similar"],
                "similar_good_profiles": stats["good_profiles"],
                "similar_bad_profiles": stats["bad_profiles"],
                "fraud_cases": stats["fraud_cases"],
                "repayment_success_rate": round(stats["success_rate"], 2),
                "default_rate": round(stats["default_rate"], 2),
                "fraud_ratio": round(stats["fraud_rate"], 2),
                "average_similarity": round(stats["avg_similarity"], 4)
            },
            "ai_analysis": ai_analysis,
            "metadata": {
                "agent_version": "2.0-AI-LangChain",
                "llm_model": LLM_MODEL if self.llm_enabled else "fallback"
            }
        }
        return {"final_output": result}

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)
        
        # Ajout des noeuds
        workflow.add_node("extract_profile", self.node_extract_profile)
        workflow.add_node("generate_embedding", self.node_generate_embedding)
        workflow.add_node("search_similar", self.node_search_similar)
        workflow.add_node("compute_stats", self.node_compute_stats)
        workflow.add_node("ai_analysis", self.node_ai_analysis)
        workflow.add_node("format_output", self.node_format_output)
        
        # Definition des aretes (flux lineaire)
        workflow.set_entry_point("extract_profile")
        workflow.add_edge("extract_profile", "generate_embedding")
        workflow.add_edge("generate_embedding", "search_similar")
        workflow.add_edge("search_similar", "compute_stats")
        workflow.add_edge("compute_stats", "ai_analysis")
        workflow.add_edge("ai_analysis", "format_output")
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
    
    def analyze_similarity(self, request: Dict[str, Any]) -> Dict[str, Any]:
        print("")
        print("=" * 70)
        print("SIMILARITY AGENT AI - Analyse en cours...")
        print("=" * 70)
        
        initial_state = {"request_data": request}
        final_state = self.graph.invoke(initial_state)
        
        result = final_state["final_output"]
        ai_analysis = final_state["ai_analysis"]
        
        # Affichage du resultat (Logique d'affichage preservee)
        print("")
        print("=" * 70)
        print("ANALYSE TERMINEE")
        print("=" * 70)
        
        rec = ai_analysis. get("recommendation", "REVISER")
        print("")
        print("   RECOMMANDATION: " + rec)
        print("   Score de risque: " + str(ai_analysis.get("risk_score", 0.5)))
        print("   Niveau de confiance: " + str(ai_analysis.get("confidence_level", "low")))
        print("   Resume: " + str(ai_analysis.get("summary", "N/A")))
        
        if ai_analysis.get("red_flags"):
            print("")
            print("   ALERTES:")
            for flag in ai_analysis["red_flags"]:
                print("      - " + str(flag))
        
        if ai_analysis.get("conditions"):
            print("")
            print("   CONDITIONS:")
            for cond in ai_analysis["conditions"]:
                print("      - " + str(cond))
        
        if ai_analysis.get("points_forts"):
            print("")
            print("   POINTS FORTS:")
            for point in ai_analysis["points_forts"][:3]: 
                print("      - " + str(point))
        
        if ai_analysis.get("points_faibles"):
            print("")
            print("   POINTS FAIBLES:")
            for point in ai_analysis["points_faibles"][:3]:
                print("      - " + str(point))
        
        return result


# ==============================================================================
# WRAPPER FUNCTIONS
# ==============================================================================

_agent_instance = None

def get_agent() -> SimilarityAgentAI:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SimilarityAgentAI()
    return _agent_instance

def analyze_similarity(request) -> Dict[str, Any]: 
    return get_agent().analyze_similarity(request)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("")
    print("=" * 70)
    print("TEST DU SIMILARITY AGENT AI (LANGCHAIN)")
    print("=" * 70)
    
    test_case = {
        "loan_amount": 150000.0,
        "loan_duration": 240,
        "monthly_income": 4500.0,
        "other_income": 500.0,
        "monthly_charges": 1200.0,
        "employment_type": "employee",
        "contract_type":  "permanent",
        "seniority_years": 5,
        "marital_status": "married",
        "number_of_children": 2,
        "spouse_employed": True,
        "housing_status":  "owner",
        "is_primary_holder": True
    }
    
    print("")
    print("CAS DE TEST:")
    print("   Montant:  " + str(test_case["loan_amount"]) + "E sur " + str(test_case["loan_duration"]) + " mois")
    print("   Emploi: " + test_case["employment_type"] + " (" + test_case["contract_type"] + ")")
    print("   Revenus: " + str(test_case["monthly_income"]) + "E/mois")
    print("   Logement: " + test_case["housing_status"])
    print("   Situation: " + test_case["marital_status"] + ", " + str(test_case["number_of_children"]) + " enfant(s)")
    
    # Pour le test local, on peut essayer de lire les env vars
    # Si elles ne sont pas chargées, assurez-vous de faire: source backend/.env
    
    try:
        result = analyze_similarity(test_case)
        print("")
        print("=" * 70)
        print("RESULTAT COMPLET (JSON)")
        print("=" * 70)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\nERREUR: {e}")
        print("Assurez-vous que les variables d'environnement (QDRANT_API_KEY, etc.) sont bien définies.")