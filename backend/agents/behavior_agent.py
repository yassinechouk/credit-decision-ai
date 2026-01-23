"""Behavioral Analysis Agent (LangChain + LangGraph).

This module exposes behavioral heuristics as LangChain tools and orchestrates
them with a LangGraph state machine. It produces a single JSON output matching the
contract required by the credit decision system. No profiling, biometrics, or final
decisions are performed here.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Pure Python helpers (no external deps, always testable)
# ---------------------------------------------------------------------------

def _get_metric(telemetry: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely extract a float metric from telemetry. Returns default on errors/missing."""
    try:
        val = telemetry.get(key, default)
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _detect_flags(telemetry: Dict[str, Any]) -> List[str]:
    """Detect behavioral flags from telemetry (non-intrusive signals)."""
    flags: List[str] = []

    submission_duration = _get_metric(telemetry, "submission_duration_seconds")
    number_of_edits = _get_metric(telemetry, "number_of_edits")
    income_edits = _get_metric(telemetry, "income_field_edits")
    doc_reuploads = _get_metric(telemetry, "document_reuploads")
    back_nav = _get_metric(telemetry, "back_navigation_count")
    abandon_attempts = _get_metric(telemetry, "form_abandon_attempts")

    if submission_duration > 0 and submission_duration < 120:
        flags.append("RAPID_SUBMISSION")
    if submission_duration > 900:
        flags.append("LONG_HESITATION")
    if number_of_edits >= 10:
        flags.append("MULTIPLE_EDITS")
    if income_edits >= 3:
        flags.append("INCOME_REWRITES")
    if doc_reuploads >= 2:
        flags.append("DOCUMENT_REUPLOADS")
    if back_nav >= 4:
        flags.append("BACK_AND_FORTH")
    if abandon_attempts >= 1 and "LONG_HESITATION" not in flags:
        flags.append("LONG_HESITATION")

    seen: set[str] = set()
    unique_flags: List[str] = []
    for f in flags:
        if f not in seen:
            unique_flags.append(f)
            seen.add(f)
    return unique_flags


def _score_behavior(flags: List[str], telemetry: Dict[str, Any]) -> float:
    """Compute behavioral risk score (0-1) based on flags and telemetry intensity."""
    base = 0.15
    penalty_per_flag = 0.15
    score = base + penalty_per_flag * len(flags)

    score += min(_get_metric(telemetry, "document_reuploads") * 0.02, 0.1)
    score += min(_get_metric(telemetry, "income_field_edits") * 0.02, 0.08)

    jitter = (hash(str(telemetry)) % 5) / 1000.0  # up to 0.005, deterministic
    score += jitter

    return max(0.0, min(1.0, score))


def _level_from_score(score: float) -> str:
    """Map brs_score to qualitative level: LOW / MEDIUM / HIGH."""
    if score < 0.33:
        return "LOW"
    if score < 0.66:
        return "MEDIUM"
    return "HIGH"


def _build_supporting_metrics(telemetry: Dict[str, Any]) -> Dict[str, float]:
    return {
        "submission_duration_seconds": _get_metric(telemetry, "submission_duration_seconds", 0.0),
        "number_of_edits": _get_metric(telemetry, "number_of_edits", 0.0),
        "income_field_edits": _get_metric(telemetry, "income_field_edits", 0.0),
        "document_reuploads": _get_metric(telemetry, "document_reuploads", 0.0),
        "back_navigation_count": _get_metric(telemetry, "back_navigation_count", 0.0),
        "form_abandon_attempts": _get_metric(telemetry, "form_abandon_attempts", 0.0),
    }


def _compute_confidence(telemetry: Dict[str, Any], flags: List[str]) -> float:
    confidence = 0.6
    if telemetry:
        confidence += 0.1
    confidence += min(0.05 * len(flags), 0.2)
    return max(0.0, min(1.0, confidence))


def _llm_client():
    if not OPENAI_API_KEY:
        return None
    try:
        openai_mod = importlib.import_module("openai")
        OpenAI = getattr(openai_mod, "OpenAI")
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except Exception:
        return None


def _generate_behavior_explanations(state: Dict[str, Any]) -> Dict[str, Any]:
    flags = state.get("flags", [])
    brs_score = state.get("brs_score", 0.0)
    behavior_level = state.get("behavior_level", "LOW")
    supporting_metrics = state.get("supporting_metrics", {}) or {}

    client = _llm_client()
    if not client:
        return {
            "flags": {flag: "LLM non disponible" for flag in flags},
            "summary": f"Score global {round(brs_score,4)} -> niveau {behavior_level} (LLM indisponible)",
        }

    prompt = f"""
Tu es un analyste comportemental senior. Explique en français chaque flag et résume le score.
Flags: {flags}
Score: {round(brs_score,4)}
Niveau: {behavior_level}
Metrics: {supporting_metrics}
Réponds en JSON du type {{"flags": {{flag: explication}}, "summary": "..."}}.
"""
    content: Optional[str] = None

    # Try new Responses API first (OpenAI 2.x). Groq may not support it; fallback to chat.completions.
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
            return {
                "flags": {flag: "LLM indisponible" for flag in flags},
                "summary": "LLM indisponible, résumé non généré.",
            }

    import json
    try:
        if content is not None:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "flags" in parsed and "summary" in parsed:
                return parsed
    except Exception:
        pass
    fallback_text = content if content is not None else "LLM indisponible"
    return {
        "flags": {flag: fallback_text for flag in flags},
        "summary": fallback_text,
    }


# ---------------------------------------------------------------------------
# LangChain tool wrappers (lazy import to avoid hard failure if not installed)
# ---------------------------------------------------------------------------

_lc_tool = None
try:
    _lc_module = importlib.import_module("langchain_core.tools")
    _lc_tool = getattr(_lc_module, "tool")
except Exception:
    _lc_tool = None

if _lc_tool:

    @_lc_tool
    def lc_get_metric(telemetry: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely extract a float metric from telemetry."""
        return _get_metric(telemetry, key, default)

    @_lc_tool
    def lc_detect_flags(telemetry: Dict[str, Any]) -> List[str]:
        """Detect behavioral flags from telemetry."""
        return _detect_flags(telemetry)

    @_lc_tool
    def lc_score_behavior(flags: List[str], telemetry: Dict[str, Any]) -> float:
        """Compute behavioral risk score (0-1)."""
        return _score_behavior(flags, telemetry)

    @_lc_tool
    def lc_level_from_score(score: float) -> str:
        """Map brs_score to qualitative level: LOW / MEDIUM / HIGH."""
        return _level_from_score(score)

    @_lc_tool
    def generate_behavior_explanation_tool(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanations for behavior flags and score."""
        return _generate_behavior_explanations(state)

    BEHAVIOR_TOOLS = [
        lc_get_metric,
        lc_detect_flags,
        lc_score_behavior,
        lc_level_from_score,
        generate_behavior_explanation_tool,
    ]
else:
    lc_get_metric = None  # type: ignore[assignment]
    lc_detect_flags = None  # type: ignore[assignment]
    lc_score_behavior = None  # type: ignore[assignment]
    lc_level_from_score = None  # type: ignore[assignment]
    generate_behavior_explanation_tool = None  # type: ignore[assignment]
    BEHAVIOR_TOOLS = []


# ---------------------------------------------------------------------------
# LangGraph state machine (lazy import)
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

BehaviorState = Dict[str, Any]


def _node_detect_flags(state: BehaviorState) -> BehaviorState:
    telemetry = state.get("telemetry") or {}
    flags = _detect_flags(telemetry)
    return {**state, "flags": flags}


def _node_score(state: BehaviorState) -> BehaviorState:
    flags = state.get("flags") or []
    telemetry = state.get("telemetry") or {}
    brs = _score_behavior(flags, telemetry)
    return {**state, "brs_score": brs}


def _node_level(state: BehaviorState) -> BehaviorState:
    score = state.get("brs_score") or 0.0
    level = _level_from_score(score)
    return {**state, "behavior_level": level}


def _node_explain(state: BehaviorState) -> BehaviorState:
    telemetry = state.get("telemetry") or {}
    supporting_metrics = _build_supporting_metrics(telemetry)
    enriched_state = {**state, "supporting_metrics": supporting_metrics}
    explanations = _generate_behavior_explanations(enriched_state)
    return {**enriched_state, "explanations": explanations}


def _node_finalize(state: BehaviorState) -> BehaviorState:
    telemetry = state.get("telemetry") or {}
    flags = state.get("flags") or []
    supporting_metrics = state.get("supporting_metrics") or _build_supporting_metrics(telemetry)
    brs_score = state.get("brs_score") or 0.0
    behavior_level = state.get("behavior_level") or "LOW"
    confidence = _compute_confidence(telemetry, flags)
    explanations = state.get("explanations", {})

    output = {
        "case_id": state.get("case_id"),
        "behavior_analysis": {
            "brs_score": round(brs_score, 4),
            "behavior_level": behavior_level,
            "behavior_flags": flags,
            "supporting_metrics": supporting_metrics,
            "explanations": explanations,
        },
        "confidence": round(confidence, 4),
    }
    return {**state, "supporting_metrics": supporting_metrics, "output": output}


def build_behavior_graph():
    """Build the LangGraph for behavior scoring."""
    if not _LANGGRAPH_AVAILABLE or StateGraph is None:
        raise ImportError("langgraph is not installed")

    graph = StateGraph(BehaviorState)
    graph.add_node("detect_flags", _node_detect_flags)
    graph.add_node("score", _node_score)
    graph.add_node("level", _node_level)
    graph.add_node("generate_explanation", _node_explain)
    graph.add_node("finalize", _node_finalize)

    graph.set_entry_point("detect_flags")
    graph.add_edge("detect_flags", "score")
    graph.add_edge("score", "level")
    graph.add_edge("level", "generate_explanation")
    graph.add_edge("generate_explanation", "finalize")
    graph.add_edge("finalize", END)
    return graph


def visualize_behavior_graph() -> str:
    """Return a mermaid diagram of the behavior graph."""
    graph = build_behavior_graph()
    return graph.get_graph().draw_mermaid()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_behavior(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run the behavior analysis and return the standardized JSON output.

    Uses LangGraph if available; otherwise falls back to pure Python execution.
    """
    case_id = request.get("case_id")
    telemetry = request.get("telemetry") or {}
    if not isinstance(telemetry, dict):
        telemetry = {}

    if _LANGGRAPH_AVAILABLE:
        graph = build_behavior_graph().compile()
        result_state = graph.invoke({"case_id": case_id, "telemetry": telemetry})
        return result_state.get("output", {})

    flags = _detect_flags(telemetry) if telemetry else ["MISSING_TELEMETRY"]
    brs_score = _score_behavior(flags, telemetry)
    behavior_level = _level_from_score(brs_score)
    supporting_metrics = _build_supporting_metrics(telemetry)
    confidence = _compute_confidence(telemetry, flags)

    return {
        "case_id": case_id,
        "behavior_analysis": {
            "brs_score": round(brs_score, 4),
            "behavior_level": behavior_level,
            "behavior_flags": flags,
            "supporting_metrics": supporting_metrics,
        },
        "confidence": round(confidence, 4),
    }


