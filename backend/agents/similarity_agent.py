"""
==============================================================================
SIMILARITY AGENT - Agent de Recherche de Similarité pour Décision de Crédit
==============================================================================

Cet agent compare un nouveau dossier de crédit avec les cas historiques 
stockés dans Qdrant pour évaluer le risque basé sur des profils similaires. 

FONCTIONNEMENT: 
1. Reçoit un profil de crédit (nouveau dossier)
2. Convertit le profil en texte descriptif
3. Génère un embedding (vecteur numérique) du texte
4. Recherche les K cas les plus similaires dans Qdrant
5. Analyse les résultats (défauts, fraudes, patterns)
6. Retourne une évaluation du risque avec insights

AUTEUR:  Équipe Credit Decision AI
DATE:  Janvier 2026
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


# ==============================================================================
# CONFIGURATION
# ==============================================================================

QDRANT_URL = os. getenv(
    "QDRANT_URL", 
    "https://44775a69-b58f-449f-b5ca-b0f6ec6b5862.europe-west3-0.gcp.cloud.qdrant.io:6333"
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.51Eobf7Ye3tWtM_4YRPqCtAAvPXIssDAJbgm3KHx9ic"
)

COLLECTION_NAME = "credit_dataset"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_SIMILAR = int(os.getenv("TOP_K_SIMILAR", "10"))


# ==============================================================================
# CLASSE DE DONNÉES :  Représentation d'un Dossier de Crédit
# ==============================================================================

@dataclass
class CreditProfile:
    """
    Représente un profil de demande de crédit. 
    """
    loan_amount: float
    loan_duration: int
    monthly_income: float
    other_income: float
    monthly_charges: float
    employment_type: str
    contract_type: str
    seniority_years: int
    marital_status: str
    number_of_children: int
    spouse_employed: Optional[bool]
    housing_status: str
    is_primary_holder: bool
    
    def to_text(self) -> str:
        """
        Convertit le profil en texte descriptif pour l'embedding.
        """
        if self.spouse_employed is True:
            spouse_status = "conjoint employé"
        elif self.spouse_employed is False:
            spouse_status = "conjoint non employé"
        else:
            spouse_status = "célibataire ou information non disponible"
        
        total_income = self.monthly_income + (self.other_income or 0)
        debt_ratio = (self.monthly_charges / total_income * 100) if total_income > 0 else 0
        
        text = f"""
        Demande de prêt: 
        - Montant demandé: {self.loan_amount}€ sur {self.loan_duration} mois
        - Revenu mensuel: {self.monthly_income}€
        - Autres revenus: {self.other_income}€
        - Revenu total: {total_income}€
        - Charges mensuelles: {self.monthly_charges}€
        - Ratio d'endettement: {debt_ratio:.1f}%
        - Type d'emploi: {self. employment_type}
        - Type de contrat: {self.contract_type}
        - Ancienneté: {self.seniority_years} ans
        - Statut marital: {self.marital_status}
        - Nombre d'enfants: {self.number_of_children}
        - Situation conjoint: {spouse_status}
        - Statut logement: {self.housing_status}
        - Titulaire principal: {'oui' if self.is_primary_holder else 'non'}
        """
        return text. strip()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CreditProfile':
        """Crée un CreditProfile à partir d'un dictionnaire."""
        return cls(
            loan_amount=data.get('loan_amount', 0),
            loan_duration=data.get('loan_duration', 0),
            monthly_income=data. get('monthly_income', 0),
            other_income=data.get('other_income', 0),
            monthly_charges=data.get('monthly_charges', 0),
            employment_type=data.get('employment_type', 'unknown'),
            contract_type=data.get('contract_type', 'unknown'),
            seniority_years=data.get('seniority_years', 0),
            marital_status=data.get('marital_status', 'unknown'),
            number_of_children=data.get('number_of_children', 0),
            spouse_employed=data.get('spouse_employed'),
            housing_status=data. get('housing_status', 'unknown'),
            is_primary_holder=data.get('is_primary_holder', True)
        )


# ==============================================================================
# CLASSE PRINCIPALE :  Similarity Agent
# ==============================================================================

class SimilarityAgent:
    """
    Agent de recherche de similarité pour l'évaluation de risque de crédit.
    """
    
    def __init__(self):
        """Initialise l'agent avec les connexions nécessaires."""
        print("🔄 Initialisation du Similarity Agent...")
        
        # Connexion à Qdrant
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        print(f"✅ Connecté à Qdrant: {QDRANT_URL}")
        
        # Chargement du modèle d'embedding
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Modèle d'embedding chargé:  {EMBEDDING_MODEL}")
        
        self.collection_name = COLLECTION_NAME
        self.top_k = TOP_K_SIMILAR
        
        print("✅ Similarity Agent initialisé avec succès!")
    
    def _create_embedding(self, text: str) -> List[float]:
        """Génère un embedding à partir d'un texte."""
        embedding = self.embedding_model.encode(text)
        return embedding. tolist()
    
    def _search_similar_cases(
        self, 
        query_vector: List[float],
        top_k: int = None,
        filter_conditions: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche les cas similaires dans Qdrant. 
        
        CORRECTION:  Utilise query_points() au lieu de search() pour les versions récentes de qdrant-client
        """
        if top_k is None:
            top_k = self.top_k
        
        # Construire le filtre si spécifié
        query_filter = None
        if filter_conditions: 
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        # ============================================================
        # CORRECTION : Utiliser query_points au lieu de search
        # ============================================================
        try:
            # Nouvelle API (qdrant-client >= 1.7.0)
            from qdrant_client.http.models import QueryRequest
            
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            # Extraire les points du résultat
            points = results.points if hasattr(results, 'points') else results
            
        except (ImportError, AttributeError, TypeError):
            # Ancienne API (fallback) - essayer search
            try:
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                )
                points = results
            except AttributeError:
                # Dernier recours:  utiliser la méthode REST directe
                points = self._search_via_rest(query_vector, top_k, query_filter)
        
        # Formater les résultats
        similar_cases = []
        for result in points:
            similar_cases.append({
                "case_id": result.payload.get("case_id") if hasattr(result, 'payload') else result.get("payload", {}).get("case_id"),
                "similarity_score": result.score if hasattr(result, 'score') else result.get("score", 0),
                "defaulted": result.payload.get("defaulted", False) if hasattr(result, 'payload') else result.get("payload", {}).get("defaulted", False),
                "fraud_flag": result.payload.get("fraud_flag", False) if hasattr(result, 'payload') else result.get("payload", {}).get("fraud_flag", False),
                "payload": result.payload if hasattr(result, 'payload') else result.get("payload", {})
            })
        
        return similar_cases
    
    def _search_via_rest(self, query_vector: List[float], top_k: int, query_filter) -> List[Dict]: 
        """
        Méthode de secours utilisant l'API REST directement.
        """
        import requests
        
        url = f"{QDRANT_URL}/collections/{self.collection_name}/points/search"
        headers = {
            "Content-Type": "application/json",
            "api-key": QDRANT_API_KEY
        }
        
        payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True
        }
        
        if query_filter:
            payload["filter"] = query_filter. dict() if hasattr(query_filter, 'dict') else query_filter
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json().get("result", [])
        else:
            print(f"❌ Erreur API REST: {response.status_code} - {response.text}")
            return []
    
    def _analyze_similar_cases(self, similar_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les cas similaires pour extraire des statistiques et insights."""
        if not similar_cases:
            return {
                "similar_good_profiles": 0,
                "similar_bad_profiles": 0,
                "repayment_success_rate": 0.0,
                "fraud_ratio": 0.0,
                "confidence":  "low",
                "dominant_patterns": [],
                "insight": "Aucun cas similaire trouvé dans l'historique"
            }
        
        good_profiles = sum(1 for c in similar_cases if not c["defaulted"])
        bad_profiles = sum(1 for c in similar_cases if c["defaulted"])
        fraud_cases = sum(1 for c in similar_cases if c["fraud_flag"])
        
        total = len(similar_cases)
        
        repayment_success_rate = good_profiles / total if total > 0 else 0
        fraud_ratio = fraud_cases / total if total > 0 else 0
        
        avg_similarity = sum(c["similarity_score"] for c in similar_cases) / total
        
        if total >= 10 and avg_similarity >= 0.8:
            confidence = "high"
        elif total >= 5 and avg_similarity >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        dominant_patterns = self._identify_patterns(similar_cases)
        
        insight = self._generate_insight(
            good_profiles, bad_profiles, fraud_cases,
            repayment_success_rate, dominant_patterns
        )
        
        return {
            "similar_good_profiles": good_profiles,
            "similar_bad_profiles": bad_profiles,
            "repayment_success_rate":  round(repayment_success_rate, 2),
            "fraud_ratio":  round(fraud_ratio, 2),
            "total_similar_cases": total,
            "average_similarity": round(avg_similarity, 4),
            "confidence": confidence,
            "dominant_patterns": dominant_patterns,
            "insight":  insight
        }
    
    def _identify_patterns(self, similar_cases: List[Dict[str, Any]]) -> List[str]:
        """Identifie les patterns dominants parmi les cas similaires."""
        patterns = []
        
        if not similar_cases:
            return patterns
        
        # Analyser les types d'emploi
        employment_types = [c["payload"].get("employment_type") for c in similar_cases if c["payload"].get("employment_type")]
        if employment_types:
            most_common_employment = max(set(employment_types), key=employment_types.count)
            employment_ratio = employment_types.count(most_common_employment) / len(employment_types)
            if employment_ratio >= 0.5:
                patterns.append(f"Majorité {most_common_employment}s ({employment_ratio*100:.0f}%)")
        
        # Analyser les types de contrat
        contract_types = [c["payload"].get("contract_type") for c in similar_cases if c["payload"].get("contract_type")]
        if contract_types:
            most_common_contract = max(set(contract_types), key=contract_types. count)
            contract_ratio = contract_types.count(most_common_contract) / len(contract_types)
            if contract_ratio >= 0.5:
                patterns.append(f"Contrat {most_common_contract} dominant ({contract_ratio*100:.0f}%)")
        
        # Analyser le statut logement
        housing_statuses = [c["payload"].get("housing_status") for c in similar_cases if c["payload"].get("housing_status")]
        if housing_statuses:
            most_common_housing = max(set(housing_statuses), key=housing_statuses.count)
            housing_ratio = housing_statuses.count(most_common_housing) / len(housing_statuses)
            if housing_ratio >= 0.5:
                patterns.append(f"Logement:  {most_common_housing} ({housing_ratio*100:.0f}%)")
        
        return patterns
    
    def _generate_insight(
        self,
        good_profiles: int,
        bad_profiles: int,
        fraud_cases: int,
        success_rate: float,
        patterns: List[str]
    ) -> str:
        """Génère un insight textuel basé sur l'analyse."""
        total = good_profiles + bad_profiles
        
        if total == 0:
            return "Aucun cas similaire trouvé pour établir une comparaison."
        
        if success_rate >= 0.8:
            risk_level = "faible"
            emoji = "✅"
        elif success_rate >= 0.6:
            risk_level = "modéré"
            emoji = "⚠️"
        else: 
            risk_level = "élevé"
            emoji = "❌"
        
        insight = f"{emoji} Profil à risque {risk_level}.  "
        insight += f"Sur {total} cas similaires trouvés, {good_profiles} ont remboursé avec succès "
        insight += f"({success_rate*100:.0f}% de taux de succès). "
        
        if fraud_cases > 0:
            insight += f"⚠️ {fraud_cases} cas de fraude détectés parmi les profils similaires. "
        
        if patterns:
            insight += f"Patterns:  {', '.join(patterns[: 2])}."
        
        return insight
    
    def analyze_similarity(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fonction principale - Analyse la similarité d'un nouveau dossier.
        """
        print(f"\n{'='*60}")
        print("🔍 SIMILARITY AGENT - Analyse en cours...")
        print(f"{'='*60}")
        
        # Étape 1: Extraire le profil
        print("\n📋 Étape 1: Extraction du profil...")
        
        if hasattr(request, '__dict__'):
            profile_data = {
                'loan_amount': getattr(request, 'requested_amount', 0),
                'loan_duration': getattr(request, 'loan_duration', 120),
                'monthly_income':  getattr(request, 'monthly_income', 0),
                'other_income': getattr(request, 'other_income', 0),
                'monthly_charges': getattr(request, 'monthly_charges', 0),
                'employment_type': getattr(request, 'employment_type', 'employee'),
                'contract_type':  getattr(request, 'contract_type', 'permanent'),
                'seniority_years': getattr(request, 'seniority_years', 0),
                'marital_status': getattr(request, 'marital_status', 'single'),
                'number_of_children': getattr(request, 'number_of_children', 0),
                'spouse_employed': getattr(request, 'spouse_employed', None),
                'housing_status':  getattr(request, 'housing_status', 'rent'),
                'is_primary_holder': getattr(request, 'is_primary_holder', True)
            }
        else:
            profile_data = request
        
        profile = CreditProfile.from_dict(profile_data)
        print(f"   ✓ Profil extrait:  Prêt de {profile.loan_amount}€ sur {profile.loan_duration} mois")
        
        # Étape 2: Générer l'embedding
        print("\n🧠 Étape 2: Génération de l'embedding...")
        profile_text = profile.to_text()
        query_vector = self._create_embedding(profile_text)
        print(f"   ✓ Embedding généré: vecteur de {len(query_vector)} dimensions")
        
        # Étape 3: Rechercher les cas similaires
        print(f"\n🔎 Étape 3: Recherche des {self.top_k} cas les plus similaires...")
        similar_cases = self._search_similar_cases(query_vector)
        print(f"   ✓ {len(similar_cases)} cas similaires trouvés")
        
        if similar_cases:
            print("\n   📊 Cas similaires trouvés:")
            for i, case in enumerate(similar_cases[: 5], 1):
                status = "❌ Défaut" if case["defaulted"] else "✅ OK"
                fraud = " 🚨 FRAUDE" if case["fraud_flag"] else ""
                print(f"      {i}. Case #{case['case_id']}: Score={case['similarity_score']:.4f} | {status}{fraud}")
        
        # Étape 4: Analyser les résultats
        print("\n📈 Étape 4: Analyse des résultats...")
        analysis = self._analyze_similar_cases(similar_cases)
        
        print(f"\n{'='*60}")
        print("✅ ANALYSE TERMINÉE")
        print(f"{'='*60}")
        print(f"   • Profils similaires OK: {analysis['similar_good_profiles']}")
        print(f"   • Profils similaires en défaut: {analysis['similar_bad_profiles']}")
        print(f"   • Taux de succès: {analysis['repayment_success_rate']*100:.1f}%")
        print(f"   • Ratio de fraude: {analysis['fraud_ratio']*100:.1f}%")
        print(f"   • Confiance: {analysis['confidence']}")
        print(f"   • Insight: {analysis['insight']}")
        
        return analysis


# ==============================================================================
# FONCTION WRAPPER (pour compatibilité avec l'orchestrateur existant)
# ==============================================================================

_agent_instance = None

def get_agent() -> SimilarityAgent: 
    """Retourne l'instance singleton de l'agent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SimilarityAgent()
    return _agent_instance

def analyze_similarity(request) -> Dict[str, Any]: 
    """
    Fonction wrapper pour compatibilité avec l'orchestrateur.
    """
    agent = get_agent()
    return agent.analyze_similarity(request)


# ==============================================================================
# SCRIPT DE TEST
# ==============================================================================

if __name__ == "__main__": 
    print("\n" + "="*70)
    print("🧪 TEST DU SIMILARITY AGENT")
    print("="*70)
    
    test_case = {
        "loan_amount": 1500000.0,
        "loan_duration": 6,
        "monthly_income": 10000.0,
        "other_income": 0.0,
        "monthly_charges": 5000.0,
        "employment_type": "employee",
        "contract_type": "permanent",
        "seniority_years": 1,
        "marital_status": "married",
        "number_of_children": 5,
        "spouse_employed": False,
        "housing_status":  "owner",
        "is_primary_holder": True
    }
    
    print("\n📝 Cas de test:")
    print(f"   Montant: {test_case['loan_amount']}€")
    print(f"   Durée: {test_case['loan_duration']} mois")
    print(f"   Revenu:  {test_case['monthly_income']}€/mois")
    print(f"   Emploi: {test_case['employment_type']} ({test_case['contract_type']})")
    
    result = analyze_similarity(test_case)
    
    print("\n" + "="*70)
    print("📊 RÉSULTAT FINAL")
    print("="*70)
    print(f"\n{result}")