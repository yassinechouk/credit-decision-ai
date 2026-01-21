"""Document Analysis Agent for Credit Decision Memory.

This agent inspects textual documents (already OCR'ed) against the customer's declared
profile to produce a structured, audit-friendly signal. It does NOT make approval
decisions; it only surfaces documentary consistency insights.
"""

import re
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Simple chunker: approximates token sizes by words to keep dependencies light.
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


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
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
    # Simple heuristic: many identical lines or very low unique ratio.
    unique_ratio = len(set(lines)) / max(1, len(lines))
    return unique_ratio < 0.6


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def analyze_documents(request: Dict[str, Any]) -> Dict[str, Any]:
    case_id = request.get("case_id")
    declared = request.get("declared_profile", {}) or {}
    documents = request.get("documents", []) or []

    # Aggregate extraction
    incomes: List[float] = []
    contract_votes: List[str] = []
    seniority_vals: List[int] = []
    flags: List[str] = []
    suspicious_patterns: List[str] = []

    missing_documents: List[str] = []
    doc_types_present = {doc.get("doc_type", ""): True for doc in documents}
    if not doc_types_present.get("salary_slip"):
        missing_documents.append("salary_slip")
    if not doc_types_present.get("employment_contract"):
        missing_documents.append("employment_contract")

    chunked_docs: List[Tuple[str, str, int, str]] = []

    for doc in documents:
        doc_id = doc.get("doc_id", "")
        doc_type = doc.get("doc_type", "")
        raw_text = doc.get("raw_text", "") or ""
        chunks = _chunk_text(raw_text, max_tokens=400, overlap=50)
        for idx, chunk in enumerate(chunks):
            chunked_docs.append((doc_id, doc_type, idx, chunk))

        # Extraction per document (aggregate across chunks)
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

    # Consistency checks
    declared_income = declared.get("monthly_income")
    if income_documented is not None and declared_income:
        diff_ratio = abs(income_documented - declared_income) / declared_income
        if diff_ratio > 0.15:
            flags.append("INCOME_MISMATCH")

    declared_contract = declared.get("contract_type")
    if contract_detected and declared_contract and contract_detected != declared_contract:
        flags.append("CONTRACT_MISMATCH")

    declared_seniority = declared.get("seniority_years")
    if seniority_detected is not None and declared_seniority is not None:
        if abs(seniority_detected - declared_seniority) > 2:
            flags.append("SENIORITY_MISMATCH")

    if not incomes and not contract_votes and not seniority_vals:
        flags.append("MISSING_KEY_FIELDS")

    if missing_documents:
        flags.append("MISSING_DOCUMENTS")

    # Score and consistency level
    penalties = 0
    penalties += 0.25 * len(flags)
    penalties += 0.1 * len(suspicious_patterns)
    base = 1.0
    dds_score = max(0.0, min(1.0, base - penalties))

    if dds_score >= 0.8:
        consistency_level = "HIGH"
    elif dds_score >= 0.6:
        consistency_level = "MEDIUM"
    else:
        consistency_level = "LOW"

    confidence = 0.5
    if incomes:
        confidence += 0.2
    if contract_detected:
        confidence += 0.15
    if seniority_detected is not None:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))

    return {
        "case_id": case_id,
        "document_analysis": {
            "dds_score": round(dds_score, 4),
            "consistency_level": consistency_level,
            "extracted_fields": {
                "income_documented": income_documented,
                "contract_type_detected": contract_detected,
                "seniority_detected_years": seniority_detected,
            },
            "flags": list(dict.fromkeys(flags)),
            "missing_documents": missing_documents,
            "suspicious_patterns": list(dict.fromkeys(suspicious_patterns)),
        },
        "confidence": round(confidence, 4),
    }

