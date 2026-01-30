"""Decision Agent (LLM-first with rule-based fallback).

Uses scores and flags from other agents to produce a final credit decision.
Falls back to deterministic rules when LLM is unavailable or returns invalid data.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Any, Dict, List, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
DECISION_LLM_MODEL = os.getenv("DECISION_LLM_MODEL", os.getenv("LLM_MODEL", "llama-3.1-8b-instant"))
DECISION_LLM_FALLBACK_MODEL = os.getenv("DECISION_LLM_FALLBACK_MODEL", "llama-3.1-8b-instant")
try:
    DECISION_LLM_MAX_OUTPUT_TOKENS = int(os.getenv("DECISION_LLM_MAX_OUTPUT_TOKENS", "300"))
except ValueError:
    DECISION_LLM_MAX_OUTPUT_TOKENS = 300


def _llm_client():
    if not OPENAI_API_KEY:
        return None
    try:
        openai_mod = importlib.import_module("openai")
        OpenAI = getattr(openai_mod, "OpenAI")
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except Exception:
        return None


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_decision_label(label: Optional[str]) -> str:
    mapping = {
        "APPROVE": "approve",
        "APPROUVER": "approve",
        "APPROUVER_AVEC_CONDITIONS": "review",
        "MANUAL_REVIEW": "review",
        "REVISER": "review",
        "REVIEW": "review",
        "REJECT": "reject",
        "REFUSER": "reject",
    }
    if not label:
        return "review"
    normalized = str(label).strip().upper()
    return mapping.get(normalized, str(label).strip().lower())


def _extract_json_text(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.replace("```json", "```").replace("```", "").strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _parse_llm_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    json_text = _extract_json_text(raw)
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_payment_summary(behavior_result, payment_summary):
    if isinstance(payment_summary, dict):
        return payment_summary
    if isinstance(behavior_result, dict):
        summary = (behavior_result.get("behavior_analysis") or {}).get("payment_behavior_summary")
        if isinstance(summary, dict):
            return summary
    return None


def _payment_quality(summary):
    if not isinstance(summary, dict):
        return "unknown"
    on_time_rate = _safe_float(summary.get("on_time_rate"), 0.0)
    try:
        missed = int(summary.get("missed_installments") or 0)
    except (TypeError, ValueError):
        missed = 0
    try:
        max_late = int(summary.get("max_days_late") or 0)
    except (TypeError, ValueError):
        max_late = 0

    if on_time_rate >= 0.95 and missed == 0 and max_late <= 3:
        return "good"
    if on_time_rate <= 0.8 or missed >= 2 or max_late >= 30:
        return "bad"
    return "mixed"


def _build_decision_signals(
    doc_result: Dict[str, Any],
    sim_result: Dict[str, Any],
    behavior_result: Optional[Dict[str, Any]] = None,
    fraud_result: Optional[Dict[str, Any]] = None,
    image_result: Optional[Dict[str, Any]] = None,
    payment_summary: Optional[Dict[str, Any]] = None,
    orchestrator_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    doc_analysis = (doc_result or {}).get("document_analysis", {}) if isinstance(doc_result, dict) else {}
    sim_analysis = (sim_result or {}).get("ai_analysis", {}) if isinstance(sim_result, dict) else {}
    behavior_analysis = (behavior_result or {}).get("behavior_analysis", {}) if isinstance(behavior_result, dict) else {}
    fraud_analysis = (fraud_result or {}).get("fraud_analysis", {}) if isinstance(fraud_result, dict) else {}
    image_analysis = (image_result or {}).get("image_analysis", {}) if isinstance(image_result, dict) else {}

    payment_summary = _extract_payment_summary(behavior_result, payment_summary)
    signals: Dict[str, Any] = {
        "document": {
            "score": doc_analysis.get("dds_score"),
            "consistency_level": doc_analysis.get("consistency_level"),
            "flags": _safe_list(doc_analysis.get("flags", [])),
        },
        "similarity": {
            "risk_score": sim_analysis.get("risk_score"),
            "risk_level": sim_analysis.get("risk_level"),
            "recommendation": sim_analysis.get("recommendation"),
            "red_flags": _safe_list(sim_analysis.get("red_flags", [])),
        },
        "behavior": {
            "score": behavior_analysis.get("brs_score"),
            "behavior_level": behavior_analysis.get("behavior_level"),
            "flags": _safe_list(behavior_analysis.get("behavior_flags", [])),
        },
        "fraud": {
            "score": fraud_analysis.get("fraud_score"),
            "risk_level": fraud_analysis.get("risk_level"),
            "flags": _safe_list(fraud_analysis.get("detected_flags", [])) or _safe_list((fraud_result or {}).get("fraud_flags", [])),
        },
        "image": {
            "score": image_analysis.get("ifs_score"),
            "risk_level": image_analysis.get("risk_level"),
            "flags": _safe_list(image_analysis.get("flags", [])),
        },
    }

    if payment_summary:
        signals["behavior"]["payment_summary"] = payment_summary
        signals["behavior"]["payment_quality"] = _payment_quality(payment_summary)

    if isinstance(orchestrator_output, dict) and orchestrator_output:
        signals["orchestrator"] = {
            "proposed_decision": orchestrator_output.get("proposed_decision"),
            "decision_confidence": orchestrator_output.get("decision_confidence"),
            "human_review_required": orchestrator_output.get("human_review_required"),
            "detected_conflicts": orchestrator_output.get("detected_conflicts"),
            "review_triggers": orchestrator_output.get("review_triggers"),
            "risk_indicators": orchestrator_output.get("risk_indicators"),
            "aggregated_signals": orchestrator_output.get("aggregated_signals"),
        }

    return signals


def _build_prompt(signals: Dict[str, Any], candidate_reason_codes: Optional[List[str]]) -> str:
    payload = {
        "signals": signals,
        "candidate_reason_codes": candidate_reason_codes or [],
        "decision_options": ["approve", "reject", "review"],
    }
    context = json.dumps(payload, ensure_ascii=True)
    return (
        "Tu es l'agent decisionnel credit. Tu dois produire une decision finale.\n"
        "Regles:\n"
        "- Utilise uniquement les scores, flags et indicateurs fournis.\n"
        "- Si les signaux sont conflictuels ou insuffisants, choisis 'review'.\n"
        "- Si des flags de fraude critiques existent, choisis 'reject' ou 'review'.\n"
        "- decision_confidence doit etre entre 0 et 1.\n"
        "- human_review_required est true si decision='review' ou confidence faible.\n"
        "- reason_codes doit etre une sous-liste de candidate_reason_codes si fournis.\n"
        "Reponds uniquement en JSON strict avec les cles:\n"
        '{"decision": "...", "decision_confidence": 0.0, "human_review_required": true, "reason_codes": [], "summary": "..."}\n'
        f"Contexte JSON:\n{context}\n"
    )


def _call_llm(client, prompt: str) -> Optional[str]:
    if not client:
        return None
    for model in (DECISION_LLM_MODEL, DECISION_LLM_FALLBACK_MODEL):
        if not model:
            continue
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=DECISION_LLM_MAX_OUTPUT_TOKENS,
            )
            content = getattr(resp, "output_text", None)
            if content:
                return content
        except Exception:
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=DECISION_LLM_MAX_OUTPUT_TOKENS,
                )
                content = chat.choices[0].message.content  # type: ignore[index]
                if content:
                    return content
            except Exception:
                continue
    return None


def _coerce_confidence(value: Any, fallback: float) -> float:
    confidence = _safe_float(value, fallback)
    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0
    return round(confidence, 4)


def _extract_reason_codes(value: Any, candidate_reason_codes: Optional[List[str]]) -> List[str]:
    candidate_reason_codes = [code for code in _safe_list(candidate_reason_codes) if isinstance(code, str)]
    raw_codes = [code for code in _safe_list(value) if isinstance(code, str)]
    if candidate_reason_codes:
        filtered = [code for code in raw_codes if code in candidate_reason_codes]
        return filtered or candidate_reason_codes
    return raw_codes


def _requires_review_from_signals(signals: Dict[str, Any], decision: str, confidence: float) -> bool:
    if decision == "review":
        return True
    if confidence < 0.65:
        return True
    orchestrator = signals.get("orchestrator") or {}
    if _safe_list(orchestrator.get("detected_conflicts")):
        return True
    critical_flags = {"FRAUD", "DOC_TAMPER", "MISSING_DOCUMENTS", "INCOME_MISMATCH"}
    doc_flags = set(_safe_list((signals.get("document") or {}).get("flags")))
    image_flags = set(_safe_list((signals.get("image") or {}).get("flags")))
    fraud_flags = set(_safe_list((signals.get("fraud") or {}).get("flags")))
    behavior_flags = set(_safe_list((signals.get("behavior") or {}).get("flags")))
    if fraud_flags:
        return True
    if critical_flags.intersection(doc_flags | image_flags | behavior_flags):
        return True
    return False


def _make_decision_payload_rule_based(doc_result, sim_result, behavior_result=None, payment_summary=None):
    sim_analysis = (sim_result or {}).get("ai_analysis") or {}
    risk_score = sim_analysis.get("risk_score", 0.5)
    risk_score = _safe_float(risk_score, 0.5)

    confidence = 0.55
    if isinstance(sim_result, dict) and sim_result.get("confidence") is not None:
        confidence = _safe_float(sim_result.get("confidence"), 0.55)

    payment_summary = _extract_payment_summary(behavior_result, payment_summary)
    quality = _payment_quality(payment_summary)
    if quality == "good":
        confidence = min(1.0, confidence + 0.15)
        risk_score = max(0.0, risk_score - 0.1)
    elif quality == "bad":
        confidence = max(0.0, confidence - 0.2)
        risk_score = min(1.0, risk_score + 0.15)

    decision = "review"
    if risk_score < 0.4:
        decision = "approve"
    elif risk_score >= 0.75:
        decision = "reject"

    confidence = _coerce_confidence(confidence, 0.55)

    return {
        "decision": decision,
        "decision_confidence": confidence,
        "confidence": confidence,
        "risk_score": round(risk_score, 4),
        "payment_quality": quality,
    }


def make_decision_payload(
    doc_result: Dict[str, Any],
    sim_result: Dict[str, Any],
    behavior_result: Optional[Dict[str, Any]] = None,
    payment_summary: Optional[Dict[str, Any]] = None,
    fraud_result: Optional[Dict[str, Any]] = None,
    image_result: Optional[Dict[str, Any]] = None,
    orchestrator_output: Optional[Dict[str, Any]] = None,
    reason_codes: Optional[List[str]] = None,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    fallback_payload = _make_decision_payload_rule_based(doc_result, sim_result, behavior_result, payment_summary)

    signals = _build_decision_signals(
        doc_result,
        sim_result,
        behavior_result=behavior_result,
        fraud_result=fraud_result,
        image_result=image_result,
        payment_summary=payment_summary,
        orchestrator_output=orchestrator_output,
    )

    client = _llm_client()
    llm_payload: Dict[str, Any] = {}
    if client:
        prompt = _build_prompt(signals, reason_codes)
        content = _call_llm(client, prompt)
        parsed = _parse_llm_json(content)
        if parsed:
            decision = _normalize_decision_label(parsed.get("decision"))
            if decision not in {"approve", "reject", "review"}:
                decision = fallback_payload.get("decision", "review")

            confidence = _coerce_confidence(
                parsed.get("decision_confidence", parsed.get("confidence")),
                fallback_payload.get("decision_confidence", 0.55),
            )

            human_review_required = parsed.get("human_review_required")
            if not isinstance(human_review_required, bool):
                human_review_required = _requires_review_from_signals(signals, decision, confidence)

            llm_payload = {
                "decision": decision,
                "decision_raw": parsed.get("decision"),
                "decision_confidence": confidence,
                "confidence": confidence,
                "human_review_required": human_review_required,
                "reason_codes": _extract_reason_codes(parsed.get("reason_codes"), reason_codes),
            }
            if isinstance(parsed.get("summary"), str):
                llm_payload["summary"] = parsed.get("summary")

    merged = {**fallback_payload, **llm_payload}
    if isinstance(reason_codes, list):
        merged.setdefault("reason_codes", reason_codes)
    if case_id is not None:
        merged.setdefault("case_id", case_id)
    if "human_review_required" not in merged:
        merged["human_review_required"] = _requires_review_from_signals(
            signals,
            merged.get("decision", "review"),
            _safe_float(merged.get("decision_confidence"), 0.55),
        )
    if "decision_confidence" not in merged and "confidence" in merged:
        merged["decision_confidence"] = merged.get("confidence")
    return merged


def make_decision(doc_result, sim_result, behavior_result=None, payment_summary=None):
    payload = make_decision_payload(doc_result, sim_result, behavior_result, payment_summary)
    return payload.get("decision", "review")

