from typing import Any, Dict

try:
    from agents.similarity_agent import analyze_similarity  # type: ignore
except Exception:  # pragma: no cover
    analyze_similarity = None  # type: ignore

try:
    from agents.behavior_agent import analyze_behavior  # type: ignore
except Exception:  # pragma: no cover
    analyze_behavior = None  # type: ignore

try:
    from agents.fraud_agent import analyze_fraud  # type: ignore
except Exception:  # pragma: no cover
    analyze_fraud = None  # type: ignore

try:
    from agents.explanation_agent import explain_decision  # type: ignore
except Exception:  # pragma: no cover
    explain_decision = None  # type: ignore

try:
    from agents.orchestrator import orchestrate_decision  # type: ignore
except Exception:  # pragma: no cover
    orchestrate_decision = None  # type: ignore


def _adapt_similarity_input(request_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": request_data.get("case_id"),
        "loan_amount": request_data.get("loan_amount", request_data.get("amount", 0)),
        "loan_duration": request_data.get("loan_duration", request_data.get("duration_months", 0)),
        "monthly_income": request_data.get("monthly_income", 0),
        "other_income": request_data.get("other_income", 0),
        "monthly_charges": request_data.get("monthly_charges", 0),
        "employment_type": request_data.get("employment_type", "unknown"),
        "contract_type": request_data.get("contract_type", "unknown"),
        "seniority_years": request_data.get("seniority_years", 0),
        "marital_status": request_data.get("marital_status", request_data.get("family_status", "unknown")),
        "number_of_children": request_data.get("number_of_children", 0),
        "spouse_employed": request_data.get("spouse_employed"),
        "housing_status": request_data.get("housing_status", "unknown"),
        "is_primary_holder": request_data.get("is_primary_holder", True),
    }


def run_orchestrator(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run similarity + behavior (+ fraud + explanation) and return unified payload."""

    case_id = request_data.get("case_id")

    sim_payload = _adapt_similarity_input(request_data)
    if analyze_similarity:
        try:
            sim_result = analyze_similarity(sim_payload)
        except Exception:
            sim_result = {"summary": "similarity error"}
    else:
        sim_result = {"summary": "similarity stub"}

    if isinstance(sim_result, dict) and "confidence" not in sim_result:
        ai_conf = (sim_result.get("ai_analysis") or {}).get("confidence", 0.6)
        sim_result["confidence"] = ai_conf
    if isinstance(sim_result, dict) and "explanations" not in sim_result:
        summary = (sim_result.get("ai_analysis") or {}).get("summary")
        if summary:
            sim_result["explanations"] = {"global_summary": summary}

    behavior_result = analyze_behavior(request_data) if analyze_behavior else {"flags": [], "brs_score": 0.0, "confidence": 0.0}
    if isinstance(behavior_result, dict) and "explanations" not in behavior_result:
        behavior_expl = (behavior_result.get("behavior_analysis") or {}).get("explanations")
        if behavior_expl:
            behavior_result["explanations"] = behavior_expl

    fraud_payload = {
        "case_id": case_id,
        "document_flags": [],
        "behavior_flags": (behavior_result.get("behavior_analysis") or {}).get("behavior_flags", []),
        "transaction_flags": request_data.get("transaction_flags", []),
        "image_flags": request_data.get("image_flags", []),
        "free_text": request_data.get("free_text", []),
    }
    if analyze_fraud:
        try:
            fraud_result = analyze_fraud(fraud_payload)
        except Exception:
            fraud_result = {"summary": "fraud error", "fraud_flags": []}
    else:
        fraud_result = {"summary": "fraud stub", "fraud_flags": []}

    if isinstance(fraud_result, dict) and "fraud_flags" not in fraud_result:
        fraud_result["fraud_flags"] = (fraud_result.get("fraud_analysis") or {}).get("detected_flags", [])
        fraud_result["fraud_score"] = (fraud_result.get("fraud_analysis") or {}).get("fraud_score", 0.0)
    if isinstance(fraud_result, dict) and "explanations" not in fraud_result:
        fraud_expl = (fraud_result.get("fraud_analysis") or {}).get("explanations")
        if fraud_expl:
            fraud_result["explanations"] = fraud_expl

    agent_results = {
        "similarity_agent": sim_result,
        "behavior_agent": behavior_result,
        "fraud_agent": fraud_result,
    }
    orchestration = orchestrate_decision(request_data, agent_results) if orchestrate_decision else {}

    proposed_decision = (orchestration or {}).get("proposed_decision", "MANUAL_REVIEW")
    decision = {
        "case_id": case_id,
        "decision": proposed_decision.lower() if isinstance(proposed_decision, str) else "review",
        "reason_codes": [],
    }

    explanation = {}
    if explain_decision:
        try:
            explanation = explain_decision(
                decision=decision,
                doc_result={},
                sim_result=sim_result,
                behavior_result=behavior_result,
                fraud_result=fraud_result,
            )
        except Exception:
            explanation = {"explanation": {"internal_explanation": {"summary": "LLM indisponible"}}}

    explanation_payload = explanation.get("explanation") if isinstance(explanation, dict) else explanation
    if isinstance(explanation_payload, dict):
        internal_summary = (explanation_payload.get("internal_explanation") or {}).get("summary")
        if internal_summary:
            explanation_payload["explanations"] = {"global_summary": internal_summary}
        if "confidence" not in explanation_payload:
            explanation_payload["confidence"] = explanation.get("explanation_confidence", 0.5) if isinstance(explanation, dict) else 0.5

    summary = (
        (explanation_payload.get("internal_explanation") or {}).get("summary")
        if isinstance(explanation_payload, dict)
        else None
    )
    if not summary:
        summary = (sim_result.get("ai_analysis") or {}).get("summary") if isinstance(sim_result, dict) else None
    if not summary:
        summary = "Analyse en cours"

    return {
        "decision": decision,
        "explanation": explanation.get("explanation") if isinstance(explanation, dict) else explanation,
        "behavior": behavior_result,
        "fraud": fraud_result,
        "similarity": sim_result,
        "agents": {
            "similarity": sim_result,
            "behavior": behavior_result,
            "fraud": fraud_result,
            "explanation": explanation_payload,
        },
        "summary": summary,
        "orchestration": orchestration,
    }
