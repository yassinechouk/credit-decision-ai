"""Explanation Agent (LangChain + LangGraph).

Transforms multi-agent results into dual-audience explanations (internal vs client)
using LangChain tools and a LangGraph workflow. Includes LLM-backed summaries with
graceful fallback when LLM is unavailable.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

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


def _rag_lookup_reason(reason_code: str) -> Dict[str, str]:
    catalog = {
        "HIGH_DTI": {
            "description": "Debt-to-income ratio exceeds internal policy thresholds.",
            "source": "policy_credit_v3",
        },
        "INCOME_INSTABILITY": {
            "description": "Declared income shows variability or inconsistencies across documents.",
            "source": "document_analysis",
        },
        "PEER_RISK_HIGH": {
            "description": "Similar past cases exhibited higher observed default probability.",
            "source": "peer_analysis_qdrant",
        },
        "DOC_INCONSISTENCY": {
            "description": "Document checks surfaced inconsistencies requiring manual review.",
            "source": "document_analysis",
        },
    }
    return catalog.get(reason_code, {
        "description": "Reason code definition not available in local cache.",
        "source": "rag_lookup_missing",
    })


def _generate_customer_explanation(flags: List[str], key_factors: List[Dict[str, Any]], next_steps: List[str]) -> Dict[str, Any]:
    client = _llm_client()
    prompt = f"""
Tu es un conseiller client. Résume en termes simples pourquoi le dossier est en revue et quelles sont les prochaines étapes.
Flags: {flags}
Key factors: {key_factors}
Next steps: {next_steps}
Réponds en JSON: {{"summary": "...", "main_reasons": [..], "next_steps": [...]}}.
"""
    if client:
        content: Optional[str] = None
        try:
            resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=400)
            content = resp.output_text
        except Exception:
            try:
                chat = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                )
                content = chat.choices[0].message.content  # type: ignore[index]
            except Exception:
                content = None
        if content:
            import json
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "summary" in parsed:
                    return parsed
            except Exception:
                pass

    fallback_summary = "Votre dossier est en cours de revue. Nous clarifions certains éléments avant décision."
    fallback_reasons = [kf.get("description", "Raison à préciser") for kf in key_factors] or flags or ["Revue complémentaire requise."]
    return {
        "summary": fallback_summary,
        "main_reasons": fallback_reasons,
        "next_steps": next_steps,
    }


# ---------------------------------------------------------------------------
# LangChain tools (lazy)
# ---------------------------------------------------------------------------
_lc_tool = None
try:
    _lc_module = importlib.import_module("langchain_core.tools")
    _lc_tool = getattr(_lc_module, "tool")
except Exception:
    _lc_tool = None

if _lc_tool:

    @_lc_tool
    def extract_doc_flags_tool(document_result: Dict[str, Any]) -> List[str]:
        """Extract flags from document analysis result."""
        doc_analysis = document_result.get("document_analysis", {}) if isinstance(document_result, dict) else {}
        flags = doc_analysis.get("flags") or document_result.get("flags") or []
        return _safe_list(flags)

    @_lc_tool
    def extract_peer_patterns_tool(sim_result: Dict[str, Any]) -> List[str]:
        """Extract dominant peer patterns from similarity agent result."""
        patterns = sim_result.get("dominant_patterns") or sim_result.get("patterns") or []
        return _safe_list(patterns)

    @_lc_tool
    def generate_customer_explanation_tool(flags: List[str], key_factors: List[Dict[str, Any]], next_steps: List[str]) -> Dict[str, Any]:
        """Generate customer-facing explanation using LLM with fallback."""
        return _generate_customer_explanation(flags, key_factors, next_steps)

    EXPLANATION_TOOLS = [
        extract_doc_flags_tool,
        extract_peer_patterns_tool,
        generate_customer_explanation_tool,
    ]
else:
    extract_doc_flags_tool = None  # type: ignore[assignment]
    extract_peer_patterns_tool = None  # type: ignore[assignment]
    generate_customer_explanation_tool = None  # type: ignore[assignment]
    EXPLANATION_TOOLS: List[Any] = []


# ---------------------------------------------------------------------------
# LangGraph orchestration (lazy)
# ---------------------------------------------------------------------------
_LANGGRAPH_AVAILABLE = False
StateGraph = None  # type: ignore[assignment]
END = "END"  # type: ignore[assignment]
try:
    _lg_module = importlib.import_module("langgraph.graph")
    StateGraph = getattr(_lg_module, "StateGraph")
    END = getattr(_lg_module, "END")
    _LANGGRAPH_AVAILABLE = True
except Exception:
    _LANGGRAPH_AVAILABLE = False


ExplanationState = Dict[str, Any]


def node_collect_signals(state: ExplanationState) -> ExplanationState:
    doc_result = state.get("doc_result") or {}
    sim_result = state.get("sim_result") or {}
    behavior_result = state.get("behavior_result") or {}
    fraud_result = state.get("fraud_result") or {}
    image_result = state.get("image_result") or {}

    signals: List[str] = []

    doc_flags = doc_result.get("document_analysis", {}).get("flags") or doc_result.get("flags")
    if doc_flags:
        signals.extend(_safe_list(doc_flags))

    peer_patterns = sim_result.get("dominant_patterns") or sim_result.get("patterns")
    if peer_patterns:
        signals.extend(_safe_list(peer_patterns))

    b_flags = behavior_result.get("behavior_analysis", {}).get("behavior_flags") or behavior_result.get("behavior_flags")
    if b_flags:
        signals.extend(_safe_list(b_flags))

    f_flags = fraud_result.get("fraud_flags")
    if f_flags:
        signals.extend(_safe_list(f_flags))

    i_flags = image_result.get("tamper_flags") or image_result.get("flags")
    if i_flags:
        signals.extend(_safe_list(i_flags))

    deduped = []
    seen = set()
    for s in signals:
        if s not in seen:
            deduped.append(s)
            seen.add(s)

    return {**state, "supporting_signals": deduped}


def node_map_reason_codes(state: ExplanationState) -> ExplanationState:
    decision = state.get("decision") or {}
    reason_codes = decision.get("reason_codes") or []
    key_factors: List[Dict[str, Any]] = []
    for rc in _safe_list(reason_codes)[:3]:
        rag = _rag_lookup_reason(rc)
        key_factors.append({
            "reason_code": rc,
            "description": rag.get("description"),
            "source": rag.get("source"),
        })
    return {**state, "key_factors": key_factors, "reason_codes": _safe_list(reason_codes)}


def node_generate_internal_summary(state: ExplanationState) -> ExplanationState:
    decision = state.get("decision") or {}
    key_factors = state.get("key_factors", [])
    signals = state.get("supporting_signals", [])

    parts = []
    if decision.get("decision"):
        parts.append(f"Decision: {decision.get('decision')} (pending review)")
    if key_factors:
        parts.append("Key factors: " + ", ".join([kf.get("reason_code", "") for kf in key_factors]))
    if signals:
        parts.append("Supporting signals: " + ", ".join(signals[:5]))

    summary = " | ".join(parts) or "Application under review; no dominant factors listed."
    return {**state, "internal_summary": summary}


def node_generate_customer_summary(state: ExplanationState) -> ExplanationState:
    flags = state.get("supporting_signals", [])
    key_factors = state.get("key_factors", [])
    next_steps = state.get("customer_next_steps") or [
        "Fournir des justificatifs actualisés si demandé",
        "Vérifier les informations d'emploi et de revenus",
        "Attendre le retour du conseiller suite à la revue",
    ]

    customer_expl = _generate_customer_explanation(flags, key_factors, next_steps)
    return {**state, "customer_explanation": customer_expl}


def node_finalize(state: ExplanationState) -> ExplanationState:
    case_id = state.get("case_id") or (state.get("decision") or {}).get("case_id")
    key_factors = state.get("key_factors", [])
    supporting_signals = state.get("supporting_signals", [])
    internal_summary = state.get("internal_summary", "")
    customer_expl = state.get("customer_explanation", {})

    confidence = 0.5
    if key_factors:
        confidence += 0.2
    if supporting_signals:
        confidence += 0.2
    confidence = max(0.0, min(1.0, confidence))

    output = {
        "case_id": case_id,
        "explanation": {
            "internal_explanation": {
                "summary": internal_summary,
                "key_factors": key_factors,
                "supporting_signals": supporting_signals,
            },
            "customer_explanation": customer_expl,
        },
        "explanation_confidence": round(confidence, 4),
    }
    return {**state, "output": output}


def build_explanation_graph():
    if not _LANGGRAPH_AVAILABLE or StateGraph is None:
        raise ImportError("langgraph is not installed")

    graph = StateGraph(ExplanationState)
    graph.add_node("collect_signals", node_collect_signals)
    graph.add_node("map_reason_codes", node_map_reason_codes)
    graph.add_node("generate_internal_summary", node_generate_internal_summary)
    graph.add_node("generate_customer_summary", node_generate_customer_summary)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("collect_signals")
    graph.add_edge("collect_signals", "map_reason_codes")
    graph.add_edge("map_reason_codes", "generate_internal_summary")
    graph.add_edge("generate_internal_summary", "generate_customer_summary")
    graph.add_edge("generate_customer_summary", "finalize")
    graph.add_edge("finalize", END)
    return graph


def visualize_explanation_graph() -> str:
    # Some langgraph versions raise on draw; return a static Mermaid diagram instead.
    return """
flowchart TD
    collect_signals([collect_signals]) --> map_reason_codes([map_reason_codes])
    map_reason_codes --> generate_internal_summary([generate_internal_summary])
    generate_internal_summary --> generate_customer_summary([generate_customer_summary])
    generate_customer_summary --> finalize([finalize])
    finalize --> END
    subgraph Outputs
        internal_explanation
        customer_explanation
    end
    finalize --> internal_explanation
    finalize --> customer_explanation
""".strip()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def explain_decision(
    decision: Any,
    doc_result: Dict[str, Any],
    sim_result: Dict[str, Any],
    behavior_result: Optional[Dict[str, Any]] = None,
    fraud_result: Optional[Dict[str, Any]] = None,
    image_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(decision, dict):
        decision = {"decision": decision}

    state: ExplanationState = {
        "case_id": decision.get("case_id") or doc_result.get("case_id"),
        "decision": decision,
        "doc_result": doc_result,
        "sim_result": sim_result,
        "behavior_result": behavior_result or {},
        "fraud_result": fraud_result or {},
        "image_result": image_result or {},
    }

    if _LANGGRAPH_AVAILABLE:
        graph = build_explanation_graph().compile()
        result_state = graph.invoke(state)
        return result_state.get("output", {})

    # Fallback pure-Python (sequential nodes)
    state = node_collect_signals(state)
    state = node_map_reason_codes(state)
    state = node_generate_internal_summary(state)
    state = node_generate_customer_summary(state)
    state = node_finalize(state)
    return state.get("output", {})


