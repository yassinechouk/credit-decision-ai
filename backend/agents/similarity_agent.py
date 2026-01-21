"""
SIMILARITY AGENT AI - Agent Intelligent pour Décision de Crédit
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ==============================================================================
# CONFIGURATION
# ==============================================================================

QDRANT_URL = os. getenv(
    "QDRANT_URL", 
    "https://44775a69-b58f-449f-b5ca-b0f6ec6b5862.europe-west3-0.gcp.cloud.qdrant.io:6333"
)
QDRANT_API_KEY = os. getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.51Eobf7Ye3tWtM_4YRPqCtAAvPXIssDAJbgm3KHx9ic"
)
COLLECTION_NAME = "credit_dataset"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_SIMILAR = 20

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")


# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = """Tu es un expert analyste credit senior avec 20 ans d'experience.
Tu analyses les demandes de credit en comparant avec des cas historiques.
Tu dois fournir une analyse objective et explicable. 
Reponds TOUJOURS en JSON valide."""


# ==============================================================================
# CREDIT PROFILE
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
# SIMILARITY AGENT AI
# ==============================================================================

class SimilarityAgentAI:
    
    def __init__(self):
        print("Initialisation du Similarity Agent AI...")
        print("=" * 60)
        
        self. qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("Qdrant connecte:  " + QDRANT_URL[: 50] + "...")
        
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Modele d'embedding: " + EMBEDDING_MODEL)
        
        if OPENAI_API_KEY: 
            self.llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            self.llm_enabled = True
            print("LLM connecte: " + LLM_MODEL)
        else:
            self.llm_client = None
            self.llm_enabled = False
            print("LLM non configure (OPENAI_API_KEY manquant)")
        
        self.collection_name = COLLECTION_NAME
        self.top_k = TOP_K_SIMILAR
        
        print("=" * 60)
        print("Similarity Agent AI initialise avec succes!")
    
    def _create_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()
    
    def _search_similar_cases(self, query_vector: List[float]) -> List[Dict[str, Any]]: 
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
        
        similar_cases = []
        for result in points:
            payload = result.payload if hasattr(result, "payload") else result. get("payload", {})
            score = result.score if hasattr(result, "score") else result.get("score", 0)
            similar_cases.append({
                "case_id": payload.get("case_id"),
                "similarity_score": score,
                "defaulted": payload.get("defaulted", False),
                "fraud_flag": payload.get("fraud_flag", False),
                "payload": payload
            })
        return similar_cases
    
    def _compute_statistics(self, similar_cases: List[Dict]) -> Dict[str, Any]:
        if not similar_cases:
            return {
                "total_similar":  0,
                "good_profiles": 0,
                "bad_profiles": 0,
                "fraud_cases": 0,
                "success_rate": 0,
                "default_rate": 0,
                "fraud_rate": 0,
                "avg_similarity": 0
            }
        
        total = len(similar_cases)
        good = sum(1 for c in similar_cases if not c["defaulted"])
        bad = sum(1 for c in similar_cases if c["defaulted"])
        fraud = sum(1 for c in similar_cases if c["fraud_flag"])
        avg_sim = sum(c["similarity_score"] for c in similar_cases) / total
        
        return {
            "total_similar": total,
            "good_profiles": good,
            "bad_profiles": bad,
            "fraud_cases":  fraud,
            "success_rate": good / total,
            "default_rate": bad / total,
            "fraud_rate": fraud / total,
            "avg_similarity":  avg_sim
        }
    
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
    
    def _build_prompt(self, profile: Dict, cases: List[Dict], stats: Dict) -> str:
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
    
    def _call_llm(self, prompt:  str) -> Dict[str, Any]:
        if not self.llm_enabled:
            return self._fallback()
        
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1500
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print("Erreur LLM: " + str(e))
            return self._fallback()
    
    def _fallback(self) -> Dict[str, Any]:
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
    
    def analyze_similarity(self, request: Dict[str, Any]) -> Dict[str, Any]:
        print("")
        print("=" * 70)
        print("SIMILARITY AGENT AI - Analyse en cours...")
        print("=" * 70)
        
        # Etape 1: Extraction du profil
        print("")
        print("Etape 1/5: Extraction du profil...")
        profile = CreditProfile.from_dict(request)
        profile_dict = profile.to_dict()
        print("   Profil extrait:  " + str(profile. loan_amount) + "E sur " + str(profile.loan_duration) + " mois")
        print("   Ratio d'endettement projete: " + str(profile_dict["projected_debt_ratio"]) + "%")
        
        # Etape 2: Generation de l'embedding
        print("")
        print("Etape 2/5: Generation de l'embedding...")
        query_vector = self._create_embedding(profile.to_text())
        print("   Embedding genere: " + str(len(query_vector)) + " dimensions")
        
        # Etape 3: Recherche des cas similaires
        print("")
        print("Etape 3/5: Recherche des " + str(self.top_k) + " cas similaires...")
        similar_cases = self._search_similar_cases(query_vector)
        print("   " + str(len(similar_cases)) + " cas similaires trouves")
        
        if similar_cases:
            print("")
            print("   Top 5 des cas similaires:")
            for i, c in enumerate(similar_cases[:5], 1):
                status = "Defaut" if c["defaulted"] else "OK"
                fraud = " FRAUDE" if c["fraud_flag"] else ""
                score_pct = int(c["similarity_score"] * 100)
                print("      " + str(i) + ". Case #" + str(c["case_id"]) + ": " + str(score_pct) + "% | " + status + fraud)
        
        # Etape 4: Analyse statistique
        print("")
        print("Etape 4/5: Analyse statistique...")
        stats = self._compute_statistics(similar_cases)
        print("   Taux de succes historique: " + str(int(stats["success_rate"] * 100)) + "%")
        print("   Taux de defaut historique: " + str(int(stats["default_rate"] * 100)) + "%")
        print("   Taux de fraude historique: " + str(int(stats["fraud_rate"] * 100)) + "%")
        
        # Etape 5: Analyse AI
        print("")
        print("Etape 5/5: Analyse AI en cours...")
        if self.llm_enabled:
            prompt = self._build_prompt(profile_dict, similar_cases, stats)
            ai_analysis = self._call_llm(prompt)
            print("   Analyse LLM terminee")
        else:
            print("   LLM non disponible, utilisation de l'analyse de secours")
            ai_analysis = self._fallback()
        
        # Resultat final
        result = {
            "profile": profile_dict,
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
                "agent_version": "2.0-AI",
                "llm_model": LLM_MODEL if self.llm_enabled else "fallback"
            }
        }
        
        # Affichage du resultat
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
    print("TEST DU SIMILARITY AGENT AI")
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
    
    result = analyze_similarity(test_case)
    
    print("")
    print("=" * 70)
    print("RESULTAT COMPLET (JSON)")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))