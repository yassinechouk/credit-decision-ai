"""
SIMILARITY AGENT AI - Agent Intelligent pour Décision de Crédit
Refactorisé avec LangChain & LangGraph
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict, Tuple
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
try:
    TOP_K_SIMILAR = int(os.getenv("TOP_K_SIMILAR", "20"))
except ValueError:
    TOP_K_SIMILAR = 20
SIMILARITY_DATASET_PATH = os.getenv("SIMILARITY_DATASET_PATH", "")
QDRANT_AUTO_LOAD = os.getenv("QDRANT_AUTO_LOAD", "0") == "1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

try:
    QDRANT_TIMEOUT_SEC = float(os.getenv("QDRANT_TIMEOUT_SEC", "2.5"))
except ValueError:
    QDRANT_TIMEOUT_SEC = 2.5
try:
    QDRANT_RETRY_COUNT = int(os.getenv("QDRANT_RETRY_COUNT", "2"))
except ValueError:
    QDRANT_RETRY_COUNT = 2
try:
    EMBEDDING_TIMEOUT_SEC = float(os.getenv("EMBEDDING_TIMEOUT_SEC", "3.0"))
except ValueError:
    EMBEDDING_TIMEOUT_SEC = 3.0
try:
    EMBEDDING_RETRY_COUNT = int(os.getenv("EMBEDDING_RETRY_COUNT", "1"))
except ValueError:
    EMBEDDING_RETRY_COUNT = 1


def _embed_with_timeout(embedder, text: str, timeout_sec: float) -> Optional[List[float]]:
    result: Dict[str, Any] = {"value": None, "error": None}

    def _run() -> None:
        try:
            result["value"] = embedder.embed_query(text)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        return None
    if result.get("error") is not None:
        return None
    return result.get("value")


def _embed_with_retry(embedder, text: str, timeout_sec: float, retries: int) -> Optional[List[float]]:
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        vector = _embed_with_timeout(embedder, text, timeout_sec)
        if vector:
            return vector
        if attempt < attempts - 1:
            time.sleep(0.1 * (attempt + 1))
    return None

"""
QDRANT PAYLOAD SCHEMA (compatibilite "dataset actuel", sans changer les formulaires)

Contexte:
- Le projet charge `data/synthetic/credit_dataset.json` (records synthetiques).
- SimilarityAgent stocke ces records dans Qdrant comme `payload`.
- La collection supporte des vecteurs nommes (mini multi-vector): profile + payment.

Objectif:
- Definir un "schema" standard (champs attendus + types) pour rendre l'integration Qdrant
  robuste (filtrage, maintenance, docs), sans casser l'existant.
- Permettre des recherches par type de vecteur (vector_type).

Champs attendus (issus du dataset actuel):
- case_id (int)
- loan_amount (float)
- loan_duration (int)
- monthly_income (float)
- other_income (float)
- monthly_charges (float)
- employment_type (str)
- contract_type (str)
- seniority_years (int)
- marital_status (str)
- number_of_children (int)
- spouse_employed (bool)
- housing_status (str)
- is_primary_holder (bool)
- defaulted (bool)
- fraud_flag (bool)

Notes:
- Pas de `product_type`, `late_count`, `client_id`, `status` ici car ils n'existent pas dans le dataset
  et l'objectif est de ne rien changer cote formulaires.
- Les champs payment (late_installments, etc.) sont optionnels si la source est Postgres.
"""

# Payload schema types are used to build Qdrant payload indexes (optional but helpful for future filters).
QDRANT_CREDIT_CASE_PAYLOAD_SCHEMA: Dict[str, str] = {
    "case_id": "integer",
    # Present in Postgres-backed sync (optional for synthetic dataset).
    "user_id": "integer",
    "case_status": "keyword",
    "loan_amount": "float",
    "loan_duration": "integer",
    "monthly_income": "float",
    "other_income": "float",
    "monthly_charges": "float",
    "employment_type": "keyword",
    "contract_type": "keyword",
    "seniority_years": "integer",
    "marital_status": "keyword",
    "number_of_children": "integer",
    "spouse_employed": "bool",
    "housing_status": "keyword",
    "is_primary_holder": "bool",
    "defaulted": "bool",
    "fraud_flag": "bool",
    # Payment summary fields (optional).
    "late_installments": "integer",
    "missed_installments": "integer",
    "on_time_rate": "float",
    "avg_days_late": "float",
    "max_days_late": "integer",
    "last_payment_date": "datetime",
    # Sync metadata (optional).
    "updated_at": "datetime",
    "synced_at": "datetime",
    "loan_status": "keyword",
}


def _extract_payment_summary(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = request_data.get("payment_behavior_summary")
    if isinstance(summary, dict):
        return summary
    payment_history = request_data.get("payment_history")
    if isinstance(payment_history, dict):
        nested = payment_history.get("payment_behavior_summary")
        if isinstance(nested, dict):
            return nested
    return None


def _classify_payment_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    try:
        on_time_rate = float(summary.get("on_time_rate") or 0)
    except (TypeError, ValueError):
        on_time_rate = 0.0
    try:
        missed = int(summary.get("missed_installments") or 0)
    except (TypeError, ValueError):
        missed = 0
    try:
        max_late = int(summary.get("max_days_late") or 0)
    except (TypeError, ValueError):
        max_late = 0
    try:
        total_installments = int(summary.get("total_installments") or 0)
    except (TypeError, ValueError):
        total_installments = 0

    if total_installments == 0:
        return {
            "label": "unknown",
            "note": "Historique de paiement insuffisant",
            "risk_delta": 0.0,
            "strengths": [],
            "weaknesses": ["Historique de paiement insuffisant"],
            "red_flags": [],
        }

    if on_time_rate >= 0.95 and missed == 0 and max_late <= 3:
        return {
            "label": "good",
            "note": "Profil proche des clients qui paient toujours a temps",
            "risk_delta": -0.08,
            "strengths": ["Historique de paiement excellent"],
            "weaknesses": [],
            "red_flags": [],
        }
    if on_time_rate >= 0.85 and missed <= 1 and max_late <= 15:
        return {
            "label": "moderate",
            "note": "Profil proche des clients globalement fiables avec quelques retards",
            "risk_delta": -0.02,
            "strengths": ["Historique de paiement globalement fiable"],
            "weaknesses": ["Quelques retards observes"],
            "red_flags": [],
        }
    return {
        "label": "bad",
        "note": "Profil proche des clients avec retards frequents ou impayes",
        "risk_delta": 0.12,
        "strengths": [],
        "weaknesses": ["Retards ou impayes recurrents"],
        "red_flags": ["PAYMENT_HISTORY_WEAK"],
    }


def _format_payment_summary_for_prompt(summary: Optional[Dict[str, Any]]) -> str:
    if not isinstance(summary, dict):
        return ""
    try:
        on_time_rate = float(summary.get("on_time_rate") or 0)
    except (TypeError, ValueError):
        on_time_rate = 0.0
    try:
        avg_days_late = float(summary.get("avg_days_late") or 0)
    except (TypeError, ValueError):
        avg_days_late = 0.0
    try:
        max_days_late = int(summary.get("max_days_late") or 0)
    except (TypeError, ValueError):
        max_days_late = 0
    try:
        missed = int(summary.get("missed_installments") or 0)
    except (TypeError, ValueError):
        missed = 0
    return (
        "\n## HISTORIQUE DE PAIEMENT:\n"
        f"- Taux a l'heure: {round(on_time_rate * 100)}%\n"
        f"- Retard moyen: {round(avg_days_late, 1)} jours\n"
        f"- Retard max: {max_days_late} jours\n"
        f"- Tranches manquees: {missed}\n"
    )


def _build_payment_embedding_text(summary: Optional[Dict[str, Any]]) -> str:
    if not isinstance(summary, dict):
        return ""
    try:
        on_time_rate = float(summary.get("on_time_rate") or 0)
    except (TypeError, ValueError):
        on_time_rate = 0.0
    try:
        late_installments = int(summary.get("late_installments") or 0)
    except (TypeError, ValueError):
        late_installments = 0
    try:
        missed_installments = int(summary.get("missed_installments") or 0)
    except (TypeError, ValueError):
        missed_installments = 0
    try:
        avg_days_late = float(summary.get("avg_days_late") or 0)
    except (TypeError, ValueError):
        avg_days_late = 0.0
    try:
        max_days_late = int(summary.get("max_days_late") or 0)
    except (TypeError, ValueError):
        max_days_late = 0
    last_payment_date = summary.get("last_payment_date") or ""
    return (
        "Payment behavior summary: "
        f"on_time_rate={round(on_time_rate, 4)}, "
        f"late_installments={late_installments}, "
        f"missed_installments={missed_installments}, "
        f"avg_days_late={round(avg_days_late, 2)}, "
        f"max_days_late={max_days_late}, "
        f"last_payment_date={last_payment_date}"
    )


def _normalize_payload_credit_case(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise "doucement" le payload avant upsert Qdrant:
    - conserve les champs existants (compatibilite)
    - force les champs "core" a etre presents si possible
    - cast de types simples (int/float/bool/str) quand c'est safe
    """
    payload = dict(record or {})

    # Ensure case_id is present in payload when possible (useful for display).
    case_id = payload.get("case_id")
    if case_id is not None:
        try:
            payload["case_id"] = int(case_id)
        except Exception:
            # Keep original if conversion fails.
            pass

    for k in ("loan_duration", "seniority_years", "number_of_children"):
        if k in payload and payload[k] is not None:
            try:
                payload[k] = int(payload[k])
            except Exception:
                pass

    for k in ("user_id", "late_installments", "missed_installments", "max_days_late"):
        if k in payload and payload[k] is not None:
            try:
                payload[k] = int(payload[k])
            except Exception:
                pass

    for k in ("loan_amount", "monthly_income", "other_income", "monthly_charges"):
        if k in payload and payload[k] is not None:
            try:
                payload[k] = float(payload[k])
            except Exception:
                pass

    for k in ("on_time_rate", "avg_days_late"):
        if k in payload and payload[k] is not None:
            try:
                payload[k] = float(payload[k])
            except Exception:
                pass

    for k in ("defaulted", "fraud_flag", "spouse_employed", "is_primary_holder"):
        if k in payload and payload[k] is not None:
            # Accept booleans and 0/1.
            if isinstance(payload[k], bool):
                continue
            if str(payload[k]).strip() in ("0", "1"):
                payload[k] = str(payload[k]).strip() == "1"

    for k in ("employment_type", "contract_type", "marital_status", "housing_status"):
        if k in payload and payload[k] is not None:
            payload[k] = str(payload[k])

    for k in ("case_status", "loan_status"):
        if k in payload and payload[k] is not None:
            payload[k] = str(payload[k])

    return payload


def _case_status(case: Dict[str, Any]) -> str:
    if case.get("fraud_flag"):
        return "FRAUD"
    if case.get("defaulted"):
        return "DEFAULT"
    return "OK"


def _compact_similar_cases(
    cases: List[Dict[str, Any]],
    limit: int = 8,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    breakdown = {"ok": 0, "default": 0, "fraud": 0}
    compact: List[Dict[str, Any]] = []

    for case in cases:
        status = _case_status(case)
        if status == "FRAUD":
            breakdown["fraud"] += 1
        elif status == "DEFAULT":
            breakdown["default"] += 1
        else:
            breakdown["ok"] += 1

        if len(compact) < limit:
            payload = case.get("payload") or {}
            score = float(case.get("similarity_score") or 0)
            compact.append(
                {
                    "case_id": case.get("case_id"),
                    "similarity_score": round(score, 4),
                    "similarity_pct": int(score * 100),
                    "status": status,
                    "loan_amount": payload.get("loan_amount"),
                    "loan_duration": payload.get("loan_duration"),
                    "employment_type": payload.get("employment_type"),
                    "contract_type": payload.get("contract_type"),
                }
            )

    return compact, breakdown


def _build_similarity_report(
    stats: Dict[str, Any],
    breakdown: Dict[str, int],
    ai_analysis: Dict[str, Any],
) -> str:
    total = int(stats.get("total_similar", 0) or 0)
    avg_similarity = float(stats.get("avg_similarity", 0.0) or 0.0)
    min_similarity = float(stats.get("min_similarity", 0.0) or 0.0)
    max_similarity = float(stats.get("max_similarity", 0.0) or 0.0)
    median_similarity = float(stats.get("median_similarity", 0.0) or 0.0)
    default_rate = float(stats.get("default_rate", 0.0) or 0.0)
    fraud_rate = float(stats.get("fraud_rate", 0.0) or 0.0)

    recommendation = str(ai_analysis.get("recommendation", "REVISER"))
    risk_level = str(ai_analysis.get("risk_level", "modere"))
    try:
        risk_score = float(ai_analysis.get("risk_score", 0.5))
    except (TypeError, ValueError):
        risk_score = 0.5

    payment_assessment = ai_analysis.get("payment_history_assessment")
    payment_note = None
    if isinstance(payment_assessment, dict):
        note = payment_assessment.get("note")
        if isinstance(note, str) and note.strip():
            payment_note = note.strip()

    if total <= 0:
        report = (
            "Aucun dossier similaire trouve; comparaison limitee. "
            f"Recommandation: {recommendation} (risque {risk_level}, score {round(risk_score, 2)})."
        )
        if payment_note:
            report += f" Historique de paiement: {payment_note}."
        return report

    avg_pct = int(avg_similarity * 100)
    min_pct = int(min_similarity * 100)
    median_pct = int(median_similarity * 100)
    max_pct = int(max_similarity * 100)
    default_pct = int(default_rate * 100)
    fraud_pct = int(fraud_rate * 100)

    sentences = [
        (
            f"{total} dossiers similaires trouves; similarite moyenne {avg_pct}% "
            f"(min {min_pct}%, mediane {median_pct}%, max {max_pct}%)."
        ),
        (
            f"Historique pair: defaut {default_pct}%, fraude {fraud_pct}% "
            f"(OK {breakdown.get('ok', 0)}, defaut {breakdown.get('default', 0)}, fraude {breakdown.get('fraud', 0)})."
        ),
    ]
    if payment_note:
        sentences.append(f"Historique de paiement: {payment_note}.")
    sentences.append(
        f"Recommandation: {recommendation} (risque {risk_level}, score {round(risk_score, 2)})."
    )
    return " ".join(sentences)


def _build_similarity_buckets(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = [
        {"label": "Tres proche (>=0.8)", "min": 0.8, "max": 1.0},
        {"label": "Proche (0.6-0.8)", "min": 0.6, "max": 0.8},
        {"label": "Moyen (0.4-0.6)", "min": 0.4, "max": 0.6},
        {"label": "Faible (<0.4)", "min": 0.0, "max": 0.4},
    ]

    for bucket in buckets:
        bucket["count"] = 0
        bucket["default_count"] = 0
        bucket["fraud_count"] = 0
        bucket["avg_similarity"] = 0.0

    sums = {bucket["label"]: 0.0 for bucket in buckets}

    for case in cases:
        score = float(case.get("similarity_score") or 0.0)
        target = None
        if score >= 0.8:
            target = buckets[0]
        elif score >= 0.6:
            target = buckets[1]
        elif score >= 0.4:
            target = buckets[2]
        else:
            target = buckets[3]

        target["count"] += 1
        if case.get("defaulted"):
            target["default_count"] += 1
        if case.get("fraud_flag"):
            target["fraud_count"] += 1
        sums[target["label"]] += score

    for bucket in buckets:
        count = bucket["count"]
        bucket["default_rate"] = round(bucket["default_count"] / count, 3) if count else 0.0
        bucket["fraud_rate"] = round(bucket["fraud_count"] / count, 3) if count else 0.0
        bucket["avg_similarity"] = round(sums[bucket["label"]] / count, 4) if count else 0.0

    return buckets


def _augment_similarity_flags(ai_analysis: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    red_flags = ai_analysis.get("red_flags")
    if not isinstance(red_flags, list):
        red_flags = [] if red_flags is None else [str(red_flags)]

    def add_flag(flag: str) -> None:
        if flag not in red_flags:
            red_flags.append(flag)

    total = int(stats.get("total_similar", 0) or 0)
    avg_similarity = float(stats.get("avg_similarity", 0.0) or 0.0)
    default_rate = float(stats.get("default_rate", 0.0) or 0.0)
    fraud_rate = float(stats.get("fraud_rate", 0.0) or 0.0)

    if total == 0:
        add_flag("NO_SIMILAR_CASES")
    elif total < 3:
        add_flag("LOW_SIMILARITY_SAMPLE")

    if avg_similarity < 0.4:
        add_flag("LOW_AVG_SIMILARITY")
    if default_rate > 0.4:
        add_flag("PEER_DEFAULT_RATE_HIGH")
    if fraud_rate > 0.15:
        add_flag("PEER_FRAUD_RATE_HIGH")

    ai_analysis["red_flags"] = red_flags
    return ai_analysis


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
    query_vectors: Dict[str, List[float]]  # Multi-vector embeddings
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
        qdrant_url = QDRANT_URL or "http://localhost:6333"
        qdrant_key = QDRANT_API_KEY or None
        if not QDRANT_URL:
             print("ATTENTION: QDRANT_URL manquant, utilisation du local http://localhost:6333")
        try:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
            print(f"Qdrant connecte: {str(qdrant_url)[:30]}...")
        except Exception as exc:
            print("Erreur initialisation Qdrant: " + str(exc))
            self.qdrant_client = None
        
        # 2. Init Embeddings (LangChain HuggingFace Wrapper)
        # Remplace SentenceTransformer direct par LangChain wrapper
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            print("Modele d'embedding: " + EMBEDDING_MODEL)
        except Exception as exc:
            print("Erreur chargement embedding: " + str(exc))
            self.embedding_model = None
        
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
        self.dataset_path = self._find_dataset_path()
        self._dataset_cache: Optional[List[Dict[str, Any]]] = None
        self._dataset_stats: Optional[Dict[str, Tuple[float, float]]] = None

        if self.qdrant_client:
            self._ensure_collection()
            if QDRANT_AUTO_LOAD:
                self._load_dataset_into_qdrant_if_empty()
        
        # 4. Construire le Graph LangGraph
        self.graph = self._build_graph()

        print("=" * 60)
        print("Similarity Agent AI initialise avec succes!")

    # --------------------------------------------------------------------------
    # LOGIQUE METIER (Helpers)
    # --------------------------------------------------------------------------

    def _find_dataset_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        if SIMILARITY_DATASET_PATH:
            candidates.append(Path(SIMILARITY_DATASET_PATH))
        try:
            candidates.append(Path(__file__).resolve().parents[2] / "data" / "synthetic" / "credit_dataset.json")
        except Exception:
            pass
        candidates.append(Path("/app/data/synthetic/credit_dataset.json"))
        candidates.append(Path("/data/synthetic/credit_dataset.json"))
        for path in candidates:
            if path and path.exists():
                return path
        return None

    def _get_dataset(self) -> List[Dict[str, Any]]:
        if self._dataset_cache is not None:
            return self._dataset_cache
        if not self.dataset_path or not self.dataset_path.exists():
            self._dataset_cache = []
            return self._dataset_cache
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._dataset_cache = data
                else:
                    self._dataset_cache = []
        except Exception:
            self._dataset_cache = []
        return self._dataset_cache

    def _compute_dataset_stats(self, dataset: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        if self._dataset_stats is not None:
            return self._dataset_stats
        numeric_fields = [
            "loan_amount",
            "loan_duration",
            "monthly_income",
            "other_income",
            "monthly_charges",
            "seniority_years",
            "number_of_children",
        ]
        stats: Dict[str, Tuple[float, float]] = {}
        for field in numeric_fields:
            values = [float(rec.get(field, 0) or 0) for rec in dataset]
            if not values:
                stats[field] = (0.0, 1.0)
                continue
            stats[field] = (min(values), max(values))
        self._dataset_stats = stats
        return stats

    def _numeric_distance(self, profile: Dict[str, Any], record: Dict[str, Any], stats: Dict[str, Tuple[float, float]]) -> float:
        numeric_fields = [
            "loan_amount",
            "loan_duration",
            "monthly_income",
            "other_income",
            "monthly_charges",
            "seniority_years",
            "number_of_children",
        ]
        distances: List[float] = []
        for field in numeric_fields:
            p_val = float(profile.get(field, 0) or 0)
            r_val = float(record.get(field, 0) or 0)
            min_v, max_v = stats.get(field, (0.0, 1.0))
            denom = max(1e-6, max_v - min_v)
            distances.append(abs(p_val - r_val) / denom)
        return sum(distances) / max(1, len(distances))

    def _categorical_bonus(self, profile: Dict[str, Any], record: Dict[str, Any]) -> float:
        bonus = 0.0
        for field, weight in [
            ("employment_type", 0.05),
            ("contract_type", 0.05),
            ("marital_status", 0.03),
            ("housing_status", 0.03),
        ]:
            if profile.get(field) and record.get(field) and str(profile.get(field)).lower() == str(record.get(field)).lower():
                bonus += weight
        return bonus

    def _fallback_similar_cases(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        dataset = self._get_dataset()
        if not dataset:
            return []
        stats = self._compute_dataset_stats(dataset)
        scored: List[Dict[str, Any]] = []
        for record in dataset:
            dist = self._numeric_distance(profile, record, stats)
            score = max(0.0, 1.0 - dist)
            score += self._categorical_bonus(profile, record)
            score = max(0.0, min(1.0, score))
            scored.append({
                "case_id": record.get("case_id"),
                "similarity_score": score,
                "defaulted": record.get("defaulted", False),
                "fraud_flag": record.get("fraud_flag", False),
                "payload": record,
            })
        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[: self.top_k]

    def _ensure_collection(self) -> None:
        if not self.qdrant_client:
            return
        try:
            exists = self.qdrant_client.collection_exists(self.collection_name)
        except Exception:
            return
        if exists:
            # Collection exists: do not break existing deployments.
            # Best-effort: warn if config looks inconsistent and ensure payload indexes.
            try:
                info = self.qdrant_client.get_collection(self.collection_name)
                cfg = getattr(info, "config", None)
                params = getattr(cfg, "params", None) if cfg else None
                vectors = getattr(params, "vectors", None) if params else None
                vector_params = None
                if vectors is not None:
                    # Single-vector collections may expose "default"; multi-vector uses named configs.
                    vector_params = getattr(vectors, "default", None) if hasattr(vectors, "default") else None
                    if vector_params is None:
                        vector_params = getattr(vectors, "profile", None)
                if vector_params is not None:
                    expected_size = None
                    if self.embedding_model:
                        try:
                            expected_size = len(self.embedding_model.embed_query("seed"))
                        except Exception:
                            expected_size = None
                    actual_size = getattr(vector_params, "size", None)
                    actual_distance = getattr(vector_params, "distance", None)
                    if expected_size and actual_size and expected_size != actual_size:
                        print(
                            f"[WARN] Qdrant collection '{self.collection_name}' vector size mismatch: "
                            f"expected={expected_size}, actual={actual_size}"
                        )
                    if actual_distance and str(actual_distance).lower() not in ("cosine", "distance.cosine"):
                        print(
                            f"[WARN] Qdrant collection '{self.collection_name}' distance is {actual_distance}, "
                            "expected COSINE"
                        )
            except Exception:
                pass

            self._ensure_payload_indexes()
            return
        vector_size = 384
        if self.embedding_model:
            try:
                vector_size = len(self.embedding_model.embed_query("seed"))
            except Exception:
                vector_size = 384
        try:
            from qdrant_client.http.models import VectorParams, Distance, HnswConfigDiff

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "profile": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "payment": VectorParams(size=vector_size, distance=Distance.COSINE),
                },
                on_disk_payload=True,
                # Basic defaults (not aggressive); good enough for production and compatible with current queries.
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            )
            print(f"Collection Qdrant creee: {self.collection_name}")
            self._ensure_payload_indexes()
        except Exception as exc:
            print("Impossible de creer la collection Qdrant: " + str(exc))

    def _ensure_payload_indexes(self) -> None:
        """
        Best-effort creation of payload indexes (types) for the documented credit_case schema.
        Non-fatal: if index already exists or server rejects, we keep running.
        """
        if not self.qdrant_client:
            return
        try:
            from qdrant_client.http.models import PayloadSchemaType
        except Exception:
            return

        schema_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "bool": PayloadSchemaType.BOOL,
            "datetime": PayloadSchemaType.DATETIME,
            "text": PayloadSchemaType.TEXT,
            "uuid": PayloadSchemaType.UUID,
        }

        for field_name, schema_type in QDRANT_CREDIT_CASE_PAYLOAD_SCHEMA.items():
            qtype = schema_map.get(schema_type)
            if not qtype:
                continue
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=qtype,
                )
            except Exception:
                continue

    def _load_dataset_into_qdrant_if_empty(self) -> None:
        if not self.qdrant_client or not self.embedding_model:
            return
        dataset = self._get_dataset()
        if not dataset:
            return
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            points_count = getattr(info, "points_count", None)
            if points_count and points_count > 0:
                return
        except Exception:
            pass
        try:
            from qdrant_client.http.models import PointStruct
        except Exception:
            return

        batch = []
        for record in dataset:
            text = CreditProfile.from_dict(record).to_text()
            try:
                vector = self.embedding_model.embed_query(text)
            except Exception:
                continue
            case_id = record.get("case_id")
            if case_id is None:
                continue
            try:
                point_id = int(case_id)
            except Exception:
                # Qdrant point IDs must be int/uuid/string; keep safe.
                point_id = str(case_id)
            payload = _normalize_payload_credit_case(record)
            batch.append(PointStruct(id=point_id, vector={"profile": vector}, payload=payload))
            if len(batch) >= 100:
                self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)
                batch = []
        if batch:
            self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)
        print("Dataset charge dans Qdrant (auto-load).")
    
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

    def _apply_payment_assessment(self, ai_analysis: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(ai_analysis, dict):
            return ai_analysis
        risk_score = ai_analysis.get("risk_score", 0.5)
        try:
            risk_score = float(risk_score)
        except (TypeError, ValueError):
            risk_score = 0.5
        risk_score = max(0.0, min(1.0, risk_score + float(assessment.get("risk_delta", 0.0))))
        ai_analysis["risk_score"] = round(risk_score, 4)

        if risk_score >= 0.7:
            ai_analysis["risk_level"] = "eleve"
        elif risk_score >= 0.4:
            ai_analysis["risk_level"] = "modere"
        else:
            ai_analysis["risk_level"] = "faible"

        points_forts = ai_analysis.get("points_forts") or []
        if not isinstance(points_forts, list):
            points_forts = [str(points_forts)]
        points_faibles = ai_analysis.get("points_faibles") or []
        if not isinstance(points_faibles, list):
            points_faibles = [str(points_faibles)]
        red_flags = ai_analysis.get("red_flags") or []
        if not isinstance(red_flags, list):
            red_flags = [str(red_flags)]

        if assessment.get("strengths"):
            points_forts = list(points_forts) + assessment.get("strengths", [])
        if assessment.get("weaknesses"):
            points_faibles = list(points_faibles) + assessment.get("weaknesses", [])
        if assessment.get("red_flags"):
            red_flags = list(red_flags) + assessment.get("red_flags", [])

        ai_analysis["points_forts"] = points_forts
        ai_analysis["points_faibles"] = points_faibles
        ai_analysis["red_flags"] = red_flags
        ai_analysis["payment_history_assessment"] = {
            "label": assessment.get("label"),
            "note": assessment.get("note"),
        }
        return ai_analysis

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

        profile_text = profile.to_text()
        payment_summary = _extract_payment_summary(state.get("request_data") or {})
        payment_text = _build_payment_embedding_text(payment_summary)

        if not self.embedding_model:
            print("   Embedding indisponible, fallback sans vecteur")
            return {"query_vector": [], "query_vectors": {}}

        # Utilisation de LangChain Embeddings
        query_vectors: Dict[str, List[float]] = {}
        profile_vector = _embed_with_retry(self.embedding_model, profile_text, EMBEDDING_TIMEOUT_SEC, EMBEDDING_RETRY_COUNT)
        if not profile_vector:
            print("   Embedding profile indisponible, fallback sans vecteur")
            return {"query_vector": [], "query_vectors": {}}
        query_vectors["profile"] = profile_vector
        if payment_text:
            payment_vector = _embed_with_retry(self.embedding_model, payment_text, EMBEDDING_TIMEOUT_SEC, EMBEDDING_RETRY_COUNT)
            if payment_vector:
                query_vectors["payment"] = payment_vector

        print("   Embedding profile genere: " + str(len(profile_vector)) + " dimensions")
        return {"query_vector": profile_vector, "query_vectors": query_vectors}

    def node_search_similar(self, state: AgentState) -> Dict:
        """Etape 3: Recherche Qdrant"""
        print("")
        print("Etape 3/5: Recherche des " + str(self.top_k) + " cas similaires...")
        
        request_data = state.get("request_data") or {}
        vector_type = str(request_data.get("vector_type") or "profile").lower()
        query_vectors = state.get("query_vectors") or {}
        query_vector = query_vectors.get(vector_type) or query_vectors.get("profile") or state.get("query_vector", [])
        using_vector = vector_type if vector_type in query_vectors else "profile"
        profile_dict = state.get("profile_dict") or {}
        similar_cases: List[Dict[str, Any]] = []
        if not self.qdrant_client or not query_vector:
            print("   Qdrant ou embedding indisponible, aucun cas similaire recherche")
            points = []
        else:
            def _query_vector(name: str, vector: List[float]) -> List[Any]:
                if not vector:
                    return []
                attempts = max(1, QDRANT_RETRY_COUNT + 1)
                for attempt in range(attempts):
                    try:
                        results = self.qdrant_client.query_points(
                            collection_name=self.collection_name,
                            query=vector,
                            using=name,
                            limit=self.top_k,
                            with_payload=True,
                            timeout=QDRANT_TIMEOUT_SEC,
                        )
                        return results.points if hasattr(results, "points") else results
                    except Exception:
                        if attempt < attempts - 1:
                            time.sleep(0.1 * (attempt + 1))
                        continue
                return []

            if vector_type in {"hybrid", "profile+payment", "profile_payment"}:
                profile_vector = query_vectors.get("profile") or state.get("query_vector", [])
                payment_vector = query_vectors.get("payment")

                profile_points = _query_vector("profile", profile_vector)
                payment_points = _query_vector("payment", payment_vector or [])

                # If payment vector missing, fallback to profile only.
                if not payment_points:
                    points = profile_points
                else:
                    profile_weight = float(request_data.get("profile_weight", 0.6))
                    payment_weight = float(request_data.get("payment_weight", 0.4))
                    combined: Dict[Any, Dict[str, Any]] = {}

                    def _ingest(points_list: List[Any], weight: float):
                        for result in points_list:
                            if hasattr(result, "payload"):
                                payload = getattr(result, "payload", {}) or {}
                                score = float(getattr(result, "score", 0) or 0)
                            elif isinstance(result, dict):
                                payload = result.get("payload", {}) or {}
                                score = float(result.get("score", 0) or 0)
                            else:
                                payload = {}
                                score = 0.0
                            case_id = payload.get("case_id")
                            if case_id is None:
                                continue
                            entry = combined.setdefault(
                                case_id,
                                {"payload": payload, "score": 0.0},
                            )
                            entry["score"] += weight * score

                    _ingest(profile_points, profile_weight)
                    _ingest(payment_points, payment_weight)
                    points = []
                    for case_id, entry in combined.items():
                        points.append({"payload": entry["payload"], "score": entry["score"]})
                    points.sort(key=lambda x: x.get("score", 0), reverse=True)
                    points = points[: self.top_k]
            else:
                try:
                    results = self.qdrant_client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        using=using_vector,
                        limit=self.top_k,
                        with_payload=True,
                        timeout=QDRANT_TIMEOUT_SEC,
                    )
                    points = results.points if hasattr(results, "points") else results
                except Exception as e:
                    # Retry with profile vector if a specific vector name fails.
                    if using_vector != "profile":
                        try:
                            results = self.qdrant_client.query_points(
                                collection_name=self.collection_name,
                                query=query_vectors.get("profile") or query_vector,
                                using="profile",
                                limit=self.top_k,
                                with_payload=True,
                                timeout=QDRANT_TIMEOUT_SEC,
                            )
                            points = results.points if hasattr(results, "points") else results
                            print("   Fallback Qdrant: using=profile")
                        except Exception as e2:
                            print("Erreur Qdrant: " + str(e2))
                            points = []
                    else:
                        print("Erreur Qdrant: " + str(e))
                    if hasattr(self.qdrant_client, "search"):
                        try:
                            results = self.qdrant_client.search(
                                collection_name=self.collection_name,
                                query_vector=query_vector,
                                limit=self.top_k,
                                with_payload=True,
                                using=using_vector,
                                timeout=QDRANT_TIMEOUT_SEC,
                            )
                            points = results if isinstance(results, list) else getattr(results, "points", [])
                            print("   Fallback Qdrant: search() utilise")
                        except Exception as search_exc:
                            print("Erreur Qdrant search: " + str(search_exc))
                            points = []
                    else:
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
            
        if not similar_cases:
            fallback_cases = self._fallback_similar_cases(profile_dict)
            if fallback_cases:
                similar_cases = fallback_cases
                print("   Fallback local: " + str(len(similar_cases)) + " cas similaires trouves")
            else:
                print("   " + str(len(similar_cases)) + " cas similaires trouves")
        else:
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
                "success_rate": 0, "default_rate": 0, "fraud_rate": 0, "avg_similarity": 0,
                "min_similarity": 0.0, "median_similarity": 0.0, "max_similarity": 0.0,
            }
        else:
            total = len(similar_cases)
            good = sum(1 for c in similar_cases if not c["defaulted"])
            bad = sum(1 for c in similar_cases if c["defaulted"])
            fraud = sum(1 for c in similar_cases if c["fraud_flag"])
            avg_sim = sum(c["similarity_score"] for c in similar_cases) / total
            scores = sorted(float(c.get("similarity_score") or 0.0) for c in similar_cases)
            mid = total // 2
            if total % 2 == 0:
                median_sim = (scores[mid - 1] + scores[mid]) / 2
            else:
                median_sim = scores[mid]
            
            stats = {
                "total_similar": total,
                "good_profiles": good,
                "bad_profiles": bad,
                "fraud_cases":  fraud,
                "success_rate": good / total,
                "default_rate": bad / total,
                "fraud_rate": fraud / total,
                "avg_similarity":  avg_sim,
                "min_similarity": scores[0] if scores else 0.0,
                "median_similarity": median_sim if scores else 0.0,
                "max_similarity": scores[-1] if scores else 0.0,
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
            ai_analysis = self._fallback_analysis()
            payment_summary = _extract_payment_summary(state.get("request_data", {}))
            if payment_summary:
                assessment = _classify_payment_summary(payment_summary)
                ai_analysis = self._apply_payment_assessment(ai_analysis, assessment)
            ai_analysis = _augment_similarity_flags(ai_analysis, state.get("stats", {}))
            return {"ai_analysis": ai_analysis}
            
        profile_dict = state["profile_dict"]
        similar_cases = state["similar_cases"]
        stats = state["stats"]
        
        prompt_content = self._build_prompt_content(profile_dict, similar_cases, stats)
        payment_summary = _extract_payment_summary(state.get("request_data", {}))
        if payment_summary:
            prompt_content += _format_payment_summary_for_prompt(payment_summary)
        
        try:
            # Appel LangChain ChatOpenAI
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_content)
            ]
            response = self.llm.invoke(messages)
            ai_analysis = json.loads(response.content)
            print("   Analyse LLM terminee")
            payment_summary = _extract_payment_summary(state.get("request_data", {}))
            if payment_summary:
                assessment = _classify_payment_summary(payment_summary)
                ai_analysis = self._apply_payment_assessment(ai_analysis, assessment)
            ai_analysis = _augment_similarity_flags(ai_analysis, state.get("stats", {}))
            return {"ai_analysis": ai_analysis}
            
        except Exception as e:
            print("Erreur LLM: " + str(e))
            ai_analysis = self._fallback_analysis()
            payment_summary = _extract_payment_summary(state.get("request_data", {}))
            if payment_summary:
                assessment = _classify_payment_summary(payment_summary)
                ai_analysis = self._apply_payment_assessment(ai_analysis, assessment)
            ai_analysis = _augment_similarity_flags(ai_analysis, state.get("stats", {}))
            return {"ai_analysis": ai_analysis}

    def node_format_output(self, state: AgentState) -> Dict:
        """Etape Finale: Construction de la reponse"""
        stats = state["stats"]
        ai_analysis = state["ai_analysis"]
        compact_cases, breakdown = _compact_similar_cases(state.get("similar_cases", []), limit=8)
        similarity_buckets = _build_similarity_buckets(state.get("similar_cases", []))
        similarity_report = _build_similarity_report(stats, breakdown, ai_analysis)

        if not ai_analysis.get("summary"):
            ai_analysis["summary"] = similarity_report

        confidence_level = str(ai_analysis.get("confidence_level", "low")).lower()
        confidence_map = {"high": 0.85, "medium": 0.65, "low": 0.45}
        confidence = confidence_map.get(confidence_level, 0.55)
        if stats.get("total_similar", 0) >= 10:
            confidence += 0.05
        avg_similarity = stats.get("avg_similarity", 0.0)
        if avg_similarity >= 0.7:
            confidence += 0.05
        if stats.get("total_similar", 0) == 0:
            confidence -= 0.15
        if avg_similarity < 0.4:
            confidence -= 0.1
        confidence = max(0.0, min(1.0, confidence))
        
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
                "average_similarity": round(stats["avg_similarity"], 4),
                "min_similarity": round(stats.get("min_similarity", 0.0), 4),
                "median_similarity": round(stats.get("median_similarity", 0.0), 4),
                "max_similarity": round(stats.get("max_similarity", 0.0), 4),
            },
            "ai_analysis": ai_analysis,
            "similarity_report": similarity_report,
            "similarity_breakdown": breakdown,
            "similarity_cases": compact_cases,
            "similarity_buckets": similarity_buckets,
            "confidence": round(confidence, 4),
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
