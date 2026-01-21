"""Behavioral Analysis Agent for Credit Decision Memory.

Analyzes non-intrusive telemetry from the form submission to surface
behavioral stability signals. Does not perform profiling, biometric use,
or final credit decisions.
"""

from typing import Any, Dict, List, Optional


def _get_metric(telemetry: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        val = telemetry.get(key, default)
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _detect_flags(telemetry: Dict[str, Any]) -> List[str]:
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

    # Deduplicate while preserving order
    seen = set()
    unique_flags: List[str] = []
    for f in flags:
        if f not in seen:
            unique_flags.append(f)
            seen.add(f)
    return unique_flags


def _score_behavior(flags: List[str], telemetry: Dict[str, Any]) -> float:
    # Base stability, then add penalties per flag and mild contributions from metrics.
    base = 0.15
    penalty_per_flag = 0.15
    score = base + penalty_per_flag * len(flags)

    # Mild contributions from intensity of some metrics (clamped)
    score += min(_get_metric(telemetry, "document_reuploads") * 0.02, 0.1)
    score += min(_get_metric(telemetry, "income_field_edits") * 0.02, 0.08)

    # Add a tiny deterministic jitter for non-determinism without randomness
    jitter = (hash(str(telemetry)) % 5) / 1000.0  # up to 0.005
    score += jitter

    return max(0.0, min(1.0, score))


def _level_from_score(score: float) -> str:
    if score < 0.33:
        return "LOW"
    if score < 0.66:
        return "MEDIUM"
    return "HIGH"


def analyze_behavior(request: Dict[str, Any]) -> Dict[str, Any]:
    case_id = request.get("case_id")
    telemetry = request.get("telemetry") or {}

    if not isinstance(telemetry, dict):
        telemetry = {}

    flags = _detect_flags(telemetry)

    if not telemetry:
        flags = ["MISSING_TELEMETRY"]

    brs_score = _score_behavior(flags, telemetry)
    behavior_level = _level_from_score(brs_score)

    confidence = 0.6
    if telemetry:
        confidence += 0.1
    confidence += min(0.05 * len(flags), 0.2)
    confidence = max(0.0, min(1.0, confidence))

    supporting_metrics = {
        "submission_duration_seconds": _get_metric(telemetry, "submission_duration_seconds", 0.0),
        "number_of_edits": _get_metric(telemetry, "number_of_edits", 0.0),
        "income_field_edits": _get_metric(telemetry, "income_field_edits", 0.0),
        "document_reuploads": _get_metric(telemetry, "document_reuploads", 0.0),
        "back_navigation_count": _get_metric(telemetry, "back_navigation_count", 0.0),
        "form_abandon_attempts": _get_metric(telemetry, "form_abandon_attempts", 0.0),
    }

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
