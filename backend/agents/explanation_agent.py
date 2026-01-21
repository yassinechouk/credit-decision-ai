"""Explanation Agent for Credit Decision Memory.

Transforms multi-agent outputs into dual-audience explanations (internal vs client)
without inventing rules or decisions. No decision-making here, only narration of
what upstream agents already concluded.
"""

from typing import Any, Dict, List, Optional


def _rag_lookup_reason(reason_code: str) -> Dict[str, str]:
    """Placeholder RAG lookup for reason code definitions and source.

    In production, this should query Qdrant (vector DB) for policy snippets,
    thresholds, or historical precedents. Here we provide a minimal, deterministic
    fallback to stay audit-friendly without hallucinating new rules.
    """

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


def _collect_supporting_signals(
    doc_result: Dict[str, Any],
    sim_result: Dict[str, Any],
    behavior_result: Optional[Dict[str, Any]] = None,
    fraud_result: Optional[Dict[str, Any]] = None,
    image_result: Optional[Dict[str, Any]] = None,
) -> List[str]:
    signals: List[str] = []

    # Document flags
    doc_flags = doc_result.get("document_analysis", {}).get("flags") or doc_result.get("flags")
    if doc_flags:
        signals.extend(doc_flags)

    # Similarity / peer patterns
    dom_patterns = sim_result.get("dominant_patterns")
    if dom_patterns:
        signals.extend(dom_patterns)

    # Behavior flags
    if behavior_result:
        b_flags = behavior_result.get("behavior_flags")
        if b_flags:
            signals.extend(b_flags)

    # Fraud flags
    if fraud_result:
        f_flags = fraud_result.get("fraud_flags")
        if f_flags:
            signals.extend(f_flags)

    # Image flags
    if image_result:
        i_flags = image_result.get("tamper_flags") or image_result.get("flags")
        if i_flags:
            signals.extend(i_flags)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for s in signals:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def explain_decision(
    decision: Any,
    doc_result: Dict[str, Any],
    sim_result: Dict[str, Any],
    behavior_result: Optional[Dict[str, Any]] = None,
    fraud_result: Optional[Dict[str, Any]] = None,
    image_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce dual-audience explanations with traceable references.

    - internal_explanation: structured, referenced, audit-ready
    - customer_explanation: simple, non accusatory, no scores/thresholds
    """

    if not isinstance(decision, dict):
        decision = {"decision": decision}

    case_id = decision.get("case_id") or doc_result.get("case_id")
    reason_codes: List[str] = decision.get("reason_codes") or []

    # Limit to top 3 reason codes as per requirements
    key_factors = []
    for rc in reason_codes[:3]:
        rag_data = _rag_lookup_reason(rc)
        key_factors.append({
            "reason_code": rc,
            "description": rag_data.get("description"),
            "source": rag_data.get("source"),
        })

    supporting_signals = _collect_supporting_signals(doc_result, sim_result, behavior_result, fraud_result, image_result)

    # Internal summary: focus on decision + top reasons
    internal_summary_parts = []
    if decision.get("decision"):
        internal_summary_parts.append(
            f"The application is marked '{decision['decision']}' pending review based on key risk signals."
        )
    if key_factors:
        internal_summary_parts.append("Key factors: " + ", ".join([kf["reason_code"] for kf in key_factors]))
    internal_summary = " ".join(internal_summary_parts) or "The application requires review; no dominant factors available."

    # Customer-facing explanation: simplify language, no scores, no thresholds
    customer_main_reasons: List[str] = []
    for kf in key_factors:
        rc = kf.get("reason_code")
        desc = kf.get("description")
        if rc == "HIGH_DTI":
            customer_main_reasons.append("Votre niveau d'engagement actuel est élevé par rapport à vos revenus.")
        elif rc == "INCOME_INSTABILITY":
            customer_main_reasons.append("Les justificatifs de revenus montrent des montants variables." )
        elif rc == "PEER_RISK_HIGH":
            customer_main_reasons.append("Des profils similaires ont parfois rencontré des difficultés de remboursement." )
        elif rc == "DOC_INCONSISTENCY":
            customer_main_reasons.append("Certaines informations documentaires doivent être clarifiées." )
        elif desc:
            customer_main_reasons.append(desc)

    if not customer_main_reasons:
        customer_main_reasons.append("Votre demande nécessite une revue complémentaire pour confirmer les informations.")

    customer_next_steps = [
        "Fournir des justificatifs de revenus mis à jour",
        "Préciser les informations d'emploi et d'ancienneté",
        "Adapter le montant ou la durée si nécessaire",
    ]

    # Confidence: heuristic based on presence of reason codes and signals
    confidence = 0.5
    if reason_codes:
        confidence += 0.25
    if supporting_signals:
        confidence += 0.15
    confidence = max(0.0, min(1.0, confidence))

    return {
        "case_id": case_id,
        "explanation": {
            "internal_explanation": {
                "summary": internal_summary,
                "key_factors": key_factors,
                "supporting_signals": supporting_signals,
            },
            "customer_explanation": {
                "summary": "Votre dossier doit être revu avant décision finale.",
                "main_reasons": customer_main_reasons,
                "next_steps": customer_next_steps,
            },
        },
        "explanation_confidence": round(confidence, 4),
    }

