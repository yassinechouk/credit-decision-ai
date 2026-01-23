"""Fraud Detection Agent using LangChain tools and LangGraph orchestration.

Features:
- Aggregates fraud signals from documents, behavior, transactions, and images.
- Detects known fraud patterns and flags them.
- Computes fraud risk score (0.0-1.0) and risk level.
- Generates audit-friendly explanations via LLM with graceful fallback.
- Provides LangChain tools and a LangGraph pipeline; falls back to pure Python if missing.
"""
from __future__ import annotations

import importlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Environment-driven LLM config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_client():
    if not OPENAI_API_KEY:
        return None
    try:
        openai_mod = importlib.import_module("openai")
        OpenAI = getattr(openai_mod, "OpenAI")
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except Exception:
        return None


def _safe_list(val: Any) -> List[Any]:
    return val if isinstance(val, list) else []


def _regex_flags_from_text(texts: List[str]) -> List[str]:
    flags: List[str] = []
    patterns = {
        "DOC_TAMPER": r"(tamper|forgery|alt[ée]r[ée]|falsifi[ée])",
        "SUSPICIOUS_AMOUNT": r"(montant\s+(eleve|suspect|anormal)|large\s+transfer)",
        "INCOME_MISMATCH": r"(income\s+mismatch|revenu\s+decl|salary\s+diff)",
    }
    for text in texts:
        lower = text.lower()
        for flag, pat in patterns.items():
            if re.search(pat, lower):
                flags.append(flag)
    # Deduplicate preserving order
    seen = set()
    deduped: List[str] = []
    for f in flags:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    return deduped


def _aggregate_signals(payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    supporting: List[str] = []
    detected: List[str] = []

    doc_flags = _safe_list(payload.get("document_flags"))
    beh_flags = _safe_list(payload.get("behavior_flags"))
    txn_flags = _safe_list(payload.get("transaction_flags"))
    img_flags = _safe_list(payload.get("image_flags"))

    supporting.extend(doc_flags + beh_flags + txn_flags + img_flags)
    detected.extend(doc_flags + txn_flags + img_flags)

    # Dedup
    def dedup(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for it in items:
            if it not in seen:
                out.append(it)
                seen.add(it)
        return out

    supporting = dedup(supporting)
    detected = dedup(detected)

    # Add regex-derived flags from any text blobs if provided
    text_blobs = _safe_list(payload.get("free_text"))
    detected_from_text = _regex_flags_from_text(text_blobs)
    for f in detected_from_text:
        if f not in detected:
            detected.append(f)
        if f not in supporting:
            supporting.append(f)

    return supporting, detected


def _score_risk(detected_flags: List[str], supporting_signals: List[str]) -> Tuple[float, str]:
    weights = {
        "DOC_TAMPER": 0.4,
        "SUSPICIOUS_AMOUNT": 0.35,
        "INCOME_MISMATCH": 0.2,
        "HIGH_RISK_PATTERN": 0.35,
        "MULTIPLE_LOGIN_LOCATIONS": 0.2,
    }
    score = 0.1  # base
    for flag in detected_flags:
        score += weights.get(flag, 0.1)
    # add small bump for volume of signals
    score += min(len(supporting_signals) * 0.02, 0.1)
    score = max(0.0, min(1.0, score))

    if score >= 0.75:
        level = "HIGH"
    elif score >= 0.45:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level


def _explain_flags_llm(flags: List[str], score: float, level: str, signals: List[str]) -> Dict[str, Any]:
    client = _llm_client()
    prompt = f"""
Tu es un analyste fraude. Explique de facon concise les flags et le niveau de risque.
Flags: {flags}
Signals: {signals}
Score: {score:.2f}, Level: {level}
Reponds en JSON: {{"flag_explanations": {{"FLAG": "..."}}, "global_summary": "..."}}
"""
    if client:
        content: Optional[str] = None
        try:
            resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=300)
            content = resp.output_text
        except Exception:
            try:
                chat = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                content = chat.choices[0].message.content  # type: ignore[index]
            except Exception:
                content = None
        if content:
            import json
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "flag_explanations" in parsed:
                    return parsed
            except Exception:
                pass
    # Fallback deterministic explanations
    default_map = {
        "SUSPICIOUS_AMOUNT": "Transaction inhabituelle depassant le seuil defini.",
        "DOC_TAMPER": "Le document semble modifie ou falsifie.",
        "INCOME_MISMATCH": "Les revenus declares different des documents fournis.",
        "HIGH_RISK_PATTERN": "Pattern historique associe a des cas de fraude.",
        "MULTIPLE_LOGIN_LOCATIONS": "Connexions multiples depuis des localisations distantes.",
    }
    flag_expl = {f: default_map.get(f, "Signal de fraude a examiner.") for f in flags}
    summary = "Plusieurs signaux de fraude detectes, revue manuelle requise." if flags else "Peu de signaux de fraude identifies."
    return {"flag_explanations": flag_expl, "global_summary": summary}


# ---------------------------------------------------------------------------
# LangChain tools (lazy)
# ---------------------------------------------------------------------------
try:
    _lc_module = importlib.import_module("langchain_core.tools")
    _lc_tool = getattr(_lc_module, "tool")
except Exception:
    _lc_tool = None

if _lc_tool:

    @_lc_tool
    def extract_doc_flags_tool(document_flags: List[str]) -> List[str]:
        """Return document-level fraud flags."""
        return _safe_list(document_flags)

    @_lc_tool
    def extract_behavior_flags_tool(behavior_flags: List[str]) -> List[str]:
        """Return behavior-level fraud flags."""
        return _safe_list(behavior_flags)

    @_lc_tool
    def extract_transaction_flags_tool(transaction_flags: List[str]) -> List[str]:
        """Return transaction-level fraud flags."""
        return _safe_list(transaction_flags)

    @_lc_tool
    def extract_image_flags_tool(image_flags: List[str]) -> List[str]:
        """Return image/document-forensics flags."""
        return _safe_list(image_flags)

    @_lc_tool
    def generate_explanation_tool(flags: List[str], score: float, level: str, signals: List[str]) -> Dict[str, Any]:
        """Generate audit-friendly fraud explanations using LLM fallback."""
        return _explain_flags_llm(flags, score, level, signals)

    FRAUD_TOOLS = [
        extract_doc_flags_tool,
        extract_behavior_flags_tool,
        extract_transaction_flags_tool,
        extract_image_flags_tool,
        generate_explanation_tool,
    ]
else:
    FRAUD_TOOLS: List[Any] = []


# ---------------------------------------------------------------------------
# LangGraph orchestration (lazy)
# ---------------------------------------------------------------------------
StateGraph = None  # type: ignore[assignment]
END = "END"  # type: ignore[assignment]
_LANGGRAPH_AVAILABLE = False
try:
    _lg_module = importlib.import_module("langgraph.graph")
    StateGraph = getattr(_lg_module, "StateGraph")
    END = getattr(_lg_module, "END")
    _LANGGRAPH_AVAILABLE = True
except Exception:
    _LANGGRAPH_AVAILABLE = False

FraudState = Dict[str, Any]


def node_collect_signals(state: FraudState) -> FraudState:
    payload = state.get("payload") or {}
    supporting, detected = _aggregate_signals(payload)
    return {**state, "supporting_signals": supporting, "detected_flags": detected}


def node_score(state: FraudState) -> FraudState:
    detected = state.get("detected_flags", [])
    supporting = state.get("supporting_signals", [])
    score, level = _score_risk(_safe_list(detected), _safe_list(supporting))
    return {**state, "fraud_score": score, "risk_level": level}


def node_explain(state: FraudState) -> FraudState:
    flags = _safe_list(state.get("detected_flags"))
    score = float(state.get("fraud_score", 0.0))
    level = str(state.get("risk_level", "LOW"))
    signals = _safe_list(state.get("supporting_signals"))
    explanations = _explain_flags_llm(flags, score, level, signals)
    return {**state, "explanations": explanations}


def node_finalize(state: FraudState) -> FraudState:
    flags = _safe_list(state.get("detected_flags"))
    supporting = _safe_list(state.get("supporting_signals"))
    score = float(state.get("fraud_score", 0.0))
    level = str(state.get("risk_level", "LOW"))
    explanations = state.get("explanations") or {}
    case_id = state.get("case_id")

    # Confidence heuristic
    confidence = 0.5
    if flags:
        confidence += 0.2
    if score >= 0.75:
        confidence += 0.1
    if OPENAI_API_KEY:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))

    output = {
        "case_id": case_id,
        "fraud_analysis": {
            "fraud_score": round(score, 4),
            "risk_level": level,
            "detected_flags": flags,
            "supporting_signals": supporting,
            "explanations": explanations,
        },
        "confidence": round(confidence, 4),
    }
    return {**state, "output": output}


def build_fraud_graph():
    if not _LANGGRAPH_AVAILABLE or StateGraph is None:
        raise ImportError("langgraph is not installed")
    graph = StateGraph(FraudState)
    graph.add_node("collect_signals", node_collect_signals)
    graph.add_node("score", node_score)
    graph.add_node("explain", node_explain)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("collect_signals")
    graph.add_edge("collect_signals", "score")
    graph.add_edge("score", "explain")
    graph.add_edge("explain", "finalize")
    graph.add_edge("finalize", END)
    return graph


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def analyze_fraud(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run fraud detection pipeline on provided signals payload.

    Expected payload keys: case_id, document_flags, behavior_flags, transaction_flags,
    image_flags, and optional free_text list for regex extraction.
    """

    case_id = payload.get("case_id")
    state: FraudState = {"case_id": case_id, "payload": payload}

    if _LANGGRAPH_AVAILABLE:
        graph = build_fraud_graph().compile()
        result = graph.invoke(state)
        return result.get("output", {})

    # Pure Python fallback
    state = node_collect_signals(state)
    state = node_score(state)
    state = node_explain(state)
    state = node_finalize(state)
    return state.get("output", {})


if __name__ == "__main__":
    sample = {
        "case_id": "demo-case",
        "document_flags": ["INCOME_MISMATCH"],
        "behavior_flags": ["MULTIPLE_LOGIN_LOCATIONS"],
        "transaction_flags": ["SUSPICIOUS_AMOUNT"],
        "image_flags": ["DOC_TAMPER"],
        "free_text": ["Document tamper suspicion"],
    }
    from pprint import pprint

    pprint(analyze_fraud(sample))
