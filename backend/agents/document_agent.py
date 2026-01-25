"""Document Analysis Agent (LangChain + LangGraph).

Analyzes OCR'ed textual documents against the declared customer profile. Produces
structured, audit-friendly JSON without making credit decisions. Uses LangChain tools
for extraction and explanations, orchestrated via LangGraph with a fallback pure-Python
path if those libraries are unavailable.
"""

from __future__ import annotations

import importlib
import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Chunking & extraction helpers (pure Python, testable)
# ---------------------------------------------------------------------------
def _chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    if not words:
        return chunks
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    for start in range(0, len(words), step):
        slice_words = words[start : start + max_tokens]
        chunks.append(" ".join(slice_words))
    return chunks


INCOME_PATTERN = re.compile(r"(\d+[\d\s.,]{0,6})\s*(?:eur|€)", re.IGNORECASE)
SENIORITY_PATTERN = re.compile(r"(\d{1,2})\s*(?:ans?|years?)", re.IGNORECASE)
CONTRACT_KEYWORDS = {
    "cdi": "permanent",
    "permanent": "permanent",
    "cdd": "temporary",
    "interim": "temporary",
    "freelance": "freelance",
    "independent": "freelance",
    "contractor": "freelance",
}


def _extract_income(text: str) -> Optional[float]:
    matches = INCOME_PATTERN.findall(text)
    incomes: List[float] = []
    for raw in matches:
        cleaned = raw.replace(" ", "").replace(",", ".")
        try:
            incomes.append(float(cleaned))
        except ValueError:
            continue
    if not incomes:
        return None
    return mean(incomes)


def _extract_contract(text: str) -> Optional[str]:
    lower = text.lower()
    for key, normalized in CONTRACT_KEYWORDS.items():
        if key in lower:
            return normalized
    return None


def _extract_seniority(text: str) -> Optional[int]:
    match = SENIORITY_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _detect_generic_pattern(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    unique_ratio = len(set(lines)) / max(1, len(lines))
    return unique_ratio < 0.6


def _llm_client():
    if not OPENAI_API_KEY:
        return None
    try:
        openai_mod = importlib.import_module("openai")
        OpenAI = getattr(openai_mod, "OpenAI")
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except Exception:
        return None


def _llm_extract_fields(doc_summary: str) -> Dict[str, Any]:
    client = _llm_client()
    if not client or not doc_summary.strip():
        return {}

    prompt = f"""
Tu es un extracteur documentaire. Extrait si possible:
- income_documented (nombre)
- contract_type_detected (permanent/temporary/freelance)
- seniority_detected_years (entier)

Texte:
{doc_summary[:2000]}

Reponds en JSON: {{"income_documented": 0, "contract_type_detected": "", "seniority_detected_years": 0}}
Si absent, mets null.
"""
    try:
        resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=200)
        content = resp.output_text
    except Exception:
        try:
            chat = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            content = chat.choices[0].message.content  # type: ignore[index]
        except Exception:
            return {}

    import json
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {
                "income_documented": parsed.get("income_documented"),
                "contract_type_detected": parsed.get("contract_type_detected"),
                "seniority_detected_years": parsed.get("seniority_detected_years"),
            }
    except Exception:
        return {}
    return {}


def _generate_llm_explanations(flags: List[str], extracted: Dict[str, Any], declared: Dict[str, Any], doc_summary: str) -> Dict[str, Any]:
    client = _llm_client()
    if not client:
        return {
            "flag_explanations": {flag: "LLM non disponible" for flag in flags},
            "global_summary": "LLM non disponible, explication générique basée sur règles."
        }

    prompt = f"""
Tu es un analyste documentaire senior. Explique chaque incohérence détectée et résume les documents.
Flags: {flags}
Champs extraits: {extracted}
Profil déclaré: {declared}
Résumé texte: {doc_summary[:2000]}
Réponds en JSON: {{"flag_explanations": {{flag: texte}}, "global_summary": "..."}}
"""
    try:
        resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=600)
        content = resp.output_text
    except Exception:
        return {
            "flag_explanations": {flag: "LLM indisponible (fallback)." for flag in flags},
            "global_summary": "LLM indisponible, résumé non généré."
        }

    import json
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "flag_explanations" in parsed and "global_summary" in parsed:
            return parsed
    except Exception:
        pass
    return {
        "flag_explanations": {flag: content for flag in flags},
        "global_summary": content,
    }


# ---------------------------------------------------------------------------
# LangChain tools (lazily imported)
# ---------------------------------------------------------------------------
_lc_tool = None
try:
    _lc_module = importlib.import_module("langchain_core.tools")
    _lc_tool = getattr(_lc_module, "tool")
except Exception:
    _lc_tool = None

if _lc_tool:

    @_lc_tool
    def extract_income_tool(document_text: str) -> Optional[float]:
        """Extract documented income from text."""
        return _extract_income(document_text)

    @_lc_tool
    def extract_contract_tool(document_text: str) -> Optional[str]:
        """Detect contract type from text (permanent/temporary/freelance)."""
        return _extract_contract(document_text)

    @_lc_tool
    def extract_seniority_tool(document_text: str) -> Optional[int]:
        """Detect seniority in years from text."""
        return _extract_seniority(document_text)

    @_lc_tool
    def generate_explanation_tool(flags: List[str], extracted: Dict[str, Any], declared: Dict[str, Any], doc_summary: str) -> Dict[str, Any]:
        """Generate human-readable explanations for flags and a global summary."""
        return _generate_llm_explanations(flags, extracted, declared, doc_summary)

    DOCUMENT_TOOLS = [
        extract_income_tool,
        extract_contract_tool,
        extract_seniority_tool,
        generate_explanation_tool,
    ]
else:
    extract_income_tool = None  # type: ignore[assignment]
    extract_contract_tool = None  # type: ignore[assignment]
    extract_seniority_tool = None  # type: ignore[assignment]
    generate_explanation_tool = None  # type: ignore[assignment]
    DOCUMENT_TOOLS: List[Any] = []


# ---------------------------------------------------------------------------
# LangGraph orchestration (lazy import) + nodes
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


DocumentState = Dict[str, Any]


def _node_extract(state: DocumentState) -> DocumentState:
    declared = state.get("declared_profile") or {}
    documents = state.get("documents") or []

    incomes: List[float] = []
    contract_votes: List[str] = []
    seniority_vals: List[int] = []
    missing_documents: List[str] = []
    suspicious_patterns: List[str] = []

    doc_types_present = {doc.get("doc_type", ""): True for doc in documents}
    if not doc_types_present.get("salary_slip"):
        missing_documents.append("salary_slip")
    if not doc_types_present.get("employment_contract"):
        missing_documents.append("employment_contract")

    summaries: List[str] = []

    for doc in documents:
        raw_text = doc.get("raw_text", "") or ""
        summaries.append(raw_text[:500])
        doc_income = _extract_income(raw_text)
        if doc_income is not None:
            incomes.append(doc_income)
        contract = _extract_contract(raw_text)
        if contract:
            contract_votes.append(contract)
        seniority = _extract_seniority(raw_text)
        if seniority is not None:
            seniority_vals.append(seniority)
        if _detect_generic_pattern(raw_text):
            suspicious_patterns.append("GENERIC_TEXT_TEMPLATE")

    income_documented = mean(incomes) if incomes else None
    contract_detected = max(contract_votes, key=contract_votes.count) if contract_votes else None
    seniority_detected = round(mean(seniority_vals)) if seniority_vals else None

    extracted_fields = {
        "income_documented": income_documented,
        "contract_type_detected": contract_detected,
        "seniority_detected_years": seniority_detected,
    }
    doc_summary = "\n".join(summaries[:5])

    if not any(extracted_fields.values()) and doc_summary.strip():
        llm_extracted = _llm_extract_fields(doc_summary)
        if isinstance(llm_extracted, dict):
            for key, value in llm_extracted.items():
                if extracted_fields.get(key) in (None, "", 0) and value not in (None, "", 0):
                    extracted_fields[key] = value

    return {
        **state,
        "extracted_fields": extracted_fields,
        "missing_documents": missing_documents,
        "suspicious_patterns": list(dict.fromkeys(suspicious_patterns)),
        "doc_summary": doc_summary,
    }


def _node_flags(state: DocumentState) -> DocumentState:
    declared = state.get("declared_profile") or {}
    extracted = state.get("extracted_fields") or {}
    flags: List[str] = []

    income_documented = extracted.get("income_documented")
    declared_income = declared.get("monthly_income")
    if income_documented is not None and declared_income:
        diff_ratio = abs(income_documented - declared_income) / max(1e-6, declared_income)
        # Stricter threshold to raise a flag even on moderate discrepancies (e.g., 6-10%)
        if diff_ratio > 0.05:
            flags.append("INCOME_MISMATCH")

    contract_detected = extracted.get("contract_type_detected")
    declared_contract = declared.get("contract_type")
    if contract_detected and declared_contract and contract_detected != declared_contract:
        flags.append("CONTRACT_MISMATCH")

    seniority_detected = extracted.get("seniority_detected_years")
    declared_seniority = declared.get("seniority_years")
    if seniority_detected is not None and declared_seniority is not None:
        # Stricter threshold: flag when gap exceeds 1 year
        if abs(seniority_detected - declared_seniority) > 1:
            flags.append("SENIORITY_MISMATCH")

    if not any(extracted.values()) and not state.get("doc_summary"):
        flags.append("MISSING_KEY_FIELDS")

    if state.get("missing_documents"):
        flags.append("MISSING_DOCUMENTS")

    if "GENERIC_TEXT_TEMPLATE" in state.get("suspicious_patterns", []):
        flags.append("GENERIC_TEXT_TEMPLATE")

    return {**state, "flags": list(dict.fromkeys(flags))}


def _node_score(state: DocumentState) -> DocumentState:
    flags = state.get("flags", [])
    suspicious_patterns = state.get("suspicious_patterns", [])
    penalties = 0.0
    penalties += 0.2 * len(flags)
    penalties += 0.1 * len(suspicious_patterns)
    dds_score = max(0.0, min(1.0, 1.0 - penalties))
    if dds_score >= 0.8:
        level = "HIGH"
    elif dds_score >= 0.6:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {**state, "dds_score": round(dds_score, 4), "consistency_level": level}


def _node_llm_explain(state: DocumentState) -> DocumentState:
    explanations = _generate_llm_explanations(
        flags=state.get("flags", []),
        extracted=state.get("extracted_fields", {}),
        declared=state.get("declared_profile", {}),
        doc_summary=state.get("doc_summary", ""),
    )
    return {**state, "explanations": explanations}


def _node_finalize(state: DocumentState) -> DocumentState:
    extracted = state.get("extracted_fields", {})
    flags = state.get("flags", [])
    missing_documents = state.get("missing_documents", [])
    suspicious_patterns = state.get("suspicious_patterns", [])
    dds_score = state.get("dds_score", 0.0)
    consistency_level = state.get("consistency_level", "LOW")
    explanations = state.get("explanations", {})

    confidence = 0.5
    if extracted.get("income_documented"):
        confidence += 0.15
    if extracted.get("contract_type_detected"):
        confidence += 0.1
    if extracted.get("seniority_detected_years") is not None:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))

    output = {
        "case_id": state.get("case_id"),
        "document_analysis": {
            "dds_score": dds_score,
            "consistency_level": consistency_level,
            "extracted_fields": extracted,
            "flags": flags,
            "missing_documents": missing_documents,
            "suspicious_patterns": suspicious_patterns,
            "explanations": explanations,
        },
        "confidence": round(confidence, 4),
    }
    return {**state, "output": output}


def build_document_graph():
    if not _LANGGRAPH_AVAILABLE or StateGraph is None:
        raise ImportError("langgraph is not installed")

    graph = StateGraph(DocumentState)
    graph.add_node("extract", _node_extract)
    graph.add_node("flags", _node_flags)
    graph.add_node("score", _node_score)
    graph.add_node("llm_explain", _node_llm_explain)
    graph.add_node("finalize", _node_finalize)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "flags")
    graph.add_edge("flags", "score")
    graph.add_edge("score", "llm_explain")
    graph.add_edge("llm_explain", "finalize")
    graph.add_edge("finalize", END)
    return graph


def visualize_document_graph() -> str:
    graph = build_document_graph()
    return graph.get_graph().draw_mermaid()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def analyze_documents(request: Dict[str, Any]) -> Dict[str, Any]:
    case_id = request.get("case_id")
    declared = request.get("declared_profile", {}) or {}
    documents = request.get("documents", []) or []

    if _LANGGRAPH_AVAILABLE:
        graph = build_document_graph().compile()
        result_state = graph.invoke({
            "case_id": case_id,
            "declared_profile": declared,
            "documents": documents,
        })
        return result_state.get("output", {})

    # Fallback pure-Python path
    state: DocumentState = {
        "case_id": case_id,
        "declared_profile": declared,
        "documents": documents,
    }
    state = _node_extract(state)
    state = _node_flags(state)
    state = _node_score(state)
    state = _node_llm_explain(state)
    state = _node_finalize(state)
    return state.get("output", {})

