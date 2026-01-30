from typing import Any, Dict, List, Optional

from core.db import fetch_payment_context

try:
    from agents.document_agent import analyze_documents  # type: ignore
except Exception:  # pragma: no cover - fallback
    analyze_documents = None  # type: ignore

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
    from agents.decision_agent import make_decision_payload as make_decision_payload  # type: ignore
except Exception:  # pragma: no cover
    make_decision_payload = None  # type: ignore

try:
    from agents.orchestrator import orchestrate_decision  # type: ignore
except Exception:  # pragma: no cover
    orchestrate_decision = None  # type: ignore

try:
    from agents.image_agent import analyze_images  # type: ignore
except Exception:  # pragma: no cover
    analyze_images = None  # type: ignore


def _normalize_contract_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    normalized = raw.strip().lower().replace("-", "").replace("_", "")
    mapping = {
        "cdi": "permanent",
        "permanent": "permanent",
        "permanentcontract": "permanent",
        "cdd": "temporary",
        "temporary": "temporary",
        "interim": "temporary",
        "interimaire": "temporary",
        "freelance": "freelance",
        "independent": "freelance",
        "contractor": "freelance",
        "consultant": "freelance",
    }
    return mapping.get(normalized, normalized)


def _infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    if "salary" in lower or "pay" in lower or "bulletin" in lower or "payslip" in lower:
        return "salary_slip"
    if "contract" in lower or "contrat" in lower or "employment" in lower:
        return "employment_contract"
    if "bank" in lower or "statement" in lower or "releve" in lower:
        return "bank_statement"
    if "invoice" in lower or "facture" in lower:
        return "invoice"
    return "other"


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _merge_flags(*collections: Any) -> List[str]:
    seen = set()
    merged: List[str] = []
    for items in collections:
        for item in _safe_list(items):
            if item not in seen:
                merged.append(item)
                seen.add(item)
    return merged


def _normalize_explanations(
    raw: Optional[Dict[str, Any]],
    flags: List[str],
    summary: Optional[str] = None,
    fallback_prefix: str = "Signal detecte",
) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}

    flag_explanations: Optional[Dict[str, str]] = None
    for key in ("flag_explanations", "flags"):
        candidate = raw.get(key)
        if isinstance(candidate, dict) and candidate:
            flag_explanations = {str(k): str(v) for k, v in candidate.items()}
            break

    if flag_explanations is None and flags:
        flag_explanations = {flag: f"{fallback_prefix}: {flag}." for flag in flags}

    if not summary:
        for key in ("global_summary", "summary", "reasoning"):
            candidate = raw.get(key)
            if isinstance(candidate, str) and candidate.strip():
                summary = candidate.strip()
                break

    normalized: Dict[str, Any] = {}
    if flag_explanations:
        normalized["flag_explanations"] = flag_explanations
    if summary:
        normalized["global_summary"] = summary
    return normalized


_DECISION_REASON_LABELS = {
    "HIGH_DTI": "Ratio d'endettement élevé",
    "DOC_INCONSISTENCY": "Incohérences documentaires",
    "INCOME_INSTABILITY": "Revenus instables ou incohérents",
    "PEER_RISK_HIGH": "Profil similaire à des cas risqués",
    "PAYMENT_HISTORY_GOOD": "Historique de paiement solide",
    "PAYMENT_HISTORY_POOR": "Historique de paiement dégradé",
}


def _decision_reason_label(code: str) -> str:
    return _DECISION_REASON_LABELS.get(code, code.replace("_", " ").title())


def _build_decision_agent_output(
    decision_payload: Dict[str, Any],
    orchestrator_output: Dict[str, Any],
    explanation_result: Dict[str, Any],
) -> Dict[str, Any]:
    reason_codes = _safe_list(decision_payload.get("reason_codes", []))
    decision = decision_payload.get("decision") or orchestrator_output.get("proposed_decision") or "review"
    confidence = decision_payload.get("decision_confidence", orchestrator_output.get("decision_confidence"))
    human_review_required = decision_payload.get("human_review_required", orchestrator_output.get("human_review_required"))

    internal_expl = (explanation_result.get("explanation") or {}).get("internal_explanation", {}) if isinstance(explanation_result, dict) else {}
    internal_summary = internal_expl.get("summary") if isinstance(internal_expl, dict) else None

    reasons = [
        {"code": code, "label": _decision_reason_label(code)}
        for code in reason_codes
        if isinstance(code, str)
    ]
    reason_text = ", ".join([r["label"] for r in reasons]) if reasons else "aucun signal majeur"
    summary = internal_summary or f"Recommandation: {decision}. Raisons principales: {reason_text}."
    if human_review_required:
        summary += " Revue humaine requise."

    flag_explanations = {code: _decision_reason_label(code) for code in reason_codes if isinstance(code, str)}
    explanations = _normalize_explanations(
        {"flag_explanations": flag_explanations},
        reason_codes,
        summary=summary,
        fallback_prefix="Raison",
    )

    decision_details = {
        "recommendation": decision,
        "confidence": confidence,
        "human_review_required": human_review_required,
        "reasons": reasons,
        "review_triggers": orchestrator_output.get("review_triggers"),
        "conflicts": orchestrator_output.get("detected_conflicts"),
        "risk_indicators": orchestrator_output.get("risk_indicators"),
        "summary": summary,
    }

    explanations["decision_details"] = decision_details

    return {
        "name": "decision",
        "score": confidence,
        "flags": reason_codes,
        "explanations": explanations,
        "confidence": confidence,
    }


def _build_documents_payload(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    document_texts = request_data.get("document_texts") or {}

    provided_payloads = request_data.get("documents_payloads") or request_data.get("documents_payload")
    if isinstance(provided_payloads, list):
        for item in provided_payloads:
            if isinstance(item, dict):
                filename = item.get("filename") or item.get("name") or ""
                payloads.append({
                    "doc_type": item.get("doc_type") or _infer_doc_type(filename),
                    "raw_text": item.get("raw_text") or item.get("text") or "",
                    "filename": filename or None,
                })
        if payloads:
            return payloads

    documents = request_data.get("documents") or []
    if isinstance(documents, list):
        for doc in documents:
            if isinstance(doc, dict):
                filename = str(doc.get("filename") or doc.get("name") or doc.get("file_path") or "")
                if "/" in filename:
                    filename = filename.split("/")[-1]
                payloads.append({
                    "doc_type": doc.get("doc_type") or doc.get("document_type") or _infer_doc_type(filename),
                    "raw_text": doc.get("raw_text") or doc.get("text") or "",
                    "filename": filename or None,
                })
            else:
                filename = str(doc)
                payloads.append({
                    "doc_type": _infer_doc_type(filename),
                    "raw_text": document_texts.get(filename, ""),
                    "filename": filename,
                })
    return payloads


def _build_declared_profile(request_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "monthly_income": request_data.get("monthly_income"),
        "contract_type": _normalize_contract_type(request_data.get("contract_type")),
        "seniority_years": request_data.get("seniority_years"),
        "employment_type": request_data.get("employment_type"),
    }


def _build_similarity_payload(request_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loan_amount": request_data.get("amount", request_data.get("loan_amount", 0.0)),
        "loan_duration": request_data.get("duration_months", request_data.get("loan_duration", 0)),
        "monthly_income": request_data.get("monthly_income", 0.0),
        "other_income": request_data.get("other_income", 0.0) or 0.0,
        "monthly_charges": request_data.get("monthly_charges", 0.0),
        "employment_type": request_data.get("employment_type", "unknown"),
        "contract_type": _normalize_contract_type(request_data.get("contract_type")) or "unknown",
        "seniority_years": request_data.get("seniority_years", 0),
        "marital_status": request_data.get("marital_status") or request_data.get("family_status", "unknown"),
        "number_of_children": request_data.get("number_of_children", 0),
        "spouse_employed": request_data.get("spouse_employed"),
        "housing_status": request_data.get("housing_status", "unknown"),
        "is_primary_holder": request_data.get("is_primary_holder", True),
    }


def _derive_reason_codes(
    request_data: Dict[str, Any],
    doc_result: Dict[str, Any],
    behavior_result: Dict[str, Any],
    sim_result: Dict[str, Any],
    fraud_result: Dict[str, Any],
    image_result: Optional[Dict[str, Any]] = None,
) -> List[str]:
    reason_codes: List[str] = []

    # Debt-to-income ratio
    monthly_income = float(request_data.get("monthly_income", 0.0) or 0.0)
    other_income = float(request_data.get("other_income", 0.0) or 0.0)
    total_income = monthly_income + other_income
    monthly_charges = float(request_data.get("monthly_charges", 0.0) or 0.0)
    if total_income > 0:
        dti = monthly_charges / total_income
        if dti > 0.45:
            reason_codes.append("HIGH_DTI")

    doc_flags = doc_result.get("document_analysis", {}).get("flags") or []
    if any(flag in doc_flags for flag in ["INCOME_MISMATCH", "MISSING_DOCUMENTS", "GENERIC_TEXT_TEMPLATE"]):
        reason_codes.append("DOC_INCONSISTENCY")
    if "INCOME_MISMATCH" in doc_flags:
        reason_codes.append("INCOME_INSTABILITY")

    behavior_flags = behavior_result.get("behavior_analysis", {}).get("behavior_flags") or []
    if "MULTIPLE_EDITS" in behavior_flags or "INCOME_REWRITES" in behavior_flags:
        reason_codes.append("DOC_INCONSISTENCY")

    sim_stats = sim_result.get("rag_statistics", {})
    if sim_stats and sim_stats.get("repayment_success_rate", 1.0) < 0.6:
        reason_codes.append("PEER_RISK_HIGH")
    sim_analysis = sim_result.get("ai_analysis", {})
    if sim_analysis and str(sim_analysis.get("risk_level", "")).lower() in {"eleve", "high"}:
        reason_codes.append("PEER_RISK_HIGH")

    fraud_level = fraud_result.get("fraud_analysis", {}).get("risk_level")
    if fraud_level == "HIGH":
        reason_codes.append("DOC_INCONSISTENCY")

    payment_summary = request_data.get("payment_behavior_summary")
    if not payment_summary and request_data.get("payment_history"):
        payment_summary = (request_data.get("payment_history") or {}).get("payment_behavior_summary")
    if isinstance(payment_summary, dict):
        try:
            on_time_rate = float(payment_summary.get("on_time_rate") or 0)
        except (TypeError, ValueError):
            on_time_rate = 0.0
        missed = int(payment_summary.get("missed_installments") or 0)
        max_late = int(payment_summary.get("max_days_late") or 0)
        if on_time_rate >= 0.95 and missed == 0 and max_late <= 3:
            reason_codes.append("PAYMENT_HISTORY_GOOD")
        elif on_time_rate <= 0.8 or missed >= 2 or max_late >= 30:
            reason_codes.append("PAYMENT_HISTORY_POOR")

    image_flags = []
    if image_result:
        image_flags = image_result.get("image_analysis", {}).get("flags", []) or image_result.get("image_flags", [])
    if any(flag in image_flags for flag in ["DOC_TAMPER", "INCONSISTENT_LAYOUT", "CROP_SUSPICIOUS"]):
        reason_codes.append("DOC_INCONSISTENCY")

    # Deduplicate preserving order
    return list(dict.fromkeys(reason_codes))


def _compute_projected_debt_ratio(request_data: Dict[str, Any]) -> float:
    monthly_income = float(request_data.get("monthly_income", 0.0) or 0.0)
    other_income = float(request_data.get("other_income", 0.0) or 0.0)
    total_income = monthly_income + other_income
    monthly_charges = float(request_data.get("monthly_charges", 0.0) or 0.0)
    loan_amount = float(request_data.get("amount", request_data.get("loan_amount", 0.0)) or 0.0)
    loan_duration = float(request_data.get("duration_months", request_data.get("loan_duration", 0.0)) or 0.0)
    monthly_payment = loan_amount / loan_duration if loan_duration > 0 else 0.0
    if total_income <= 0:
        return 0.0
    return ((monthly_charges + monthly_payment) / total_income) * 100


def _normalize_decision_label(label: str) -> str:
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
    return mapping.get(label, label.lower() if label else "review")


def _build_agent_bundle(
    doc_result: Dict[str, Any],
    sim_result: Dict[str, Any],
    behavior_result: Dict[str, Any],
    fraud_result: Dict[str, Any],
    image_result: Dict[str, Any],
    explanation_result: Dict[str, Any],
) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}

    doc_analysis = doc_result.get("document_analysis", {})
    doc_flags = _safe_list(doc_analysis.get("flags", []))
    doc_explanations = _normalize_explanations(
        doc_analysis.get("explanations"),
        doc_flags,
        fallback_prefix="Signal document",
    )
    document_details: Dict[str, Any] = {
        "consistency_level": doc_analysis.get("consistency_level"),
        "dds_score": doc_analysis.get("dds_score"),
        "extracted_fields": doc_analysis.get("extracted_fields"),
        "missing_documents": doc_analysis.get("missing_documents"),
        "suspicious_patterns": doc_analysis.get("suspicious_patterns"),
    }
    if any(value for value in document_details.values()):
        doc_explanations["document_details"] = document_details
    bundle["document"] = {
        "name": "document",
        "score": doc_analysis.get("dds_score"),
        "flags": doc_flags,
        "explanations": doc_explanations,
        "confidence": doc_result.get("confidence"),
    }

    sim_result = sim_result if isinstance(sim_result, dict) else {}
    sim_analysis = sim_result.get("ai_analysis", {})
    sim_report = sim_result.get("similarity_report")
    sim_summary = sim_analysis.get("summary") or sim_analysis.get("reasoning") or sim_report
    if sim_report and (not sim_summary or len(str(sim_summary)) < 120):
        sim_summary = sim_report
    sim_flags = _safe_list(sim_analysis.get("red_flags", []))
    sim_explanations = _normalize_explanations(
        {},
        sim_flags,
        summary=sim_summary,
        fallback_prefix="Signal similarite",
    )
    similarity_details: Dict[str, Any] = {}
    if isinstance(sim_report, str) and sim_report:
        similarity_details["report"] = sim_report
    breakdown = sim_result.get("similarity_breakdown")
    if isinstance(breakdown, dict) and breakdown:
        similarity_details["breakdown"] = breakdown
    cases = sim_result.get("similarity_cases")
    if isinstance(cases, list) and cases:
        similarity_details["cases"] = cases
    buckets = sim_result.get("similarity_buckets")
    if isinstance(buckets, list) and buckets:
        similarity_details["buckets"] = buckets
    stats = sim_result.get("rag_statistics")
    if isinstance(stats, dict) and stats:
        similarity_details["stats"] = stats
    analysis_details: Dict[str, Any] = {}
    for key in (
        "recommendation",
        "risk_level",
        "risk_score",
        "confidence_level",
        "points_forts",
        "points_faibles",
        "conditions",
        "summary",
        "reasoning",
        "payment_history_assessment",
    ):
        if key in sim_analysis:
            analysis_details[key] = sim_analysis.get(key)
    if analysis_details:
        similarity_details["analysis"] = analysis_details
    if similarity_details:
        sim_explanations["similarity_details"] = similarity_details
    bundle["similarity"] = {
        "name": "similarity",
        "score": sim_analysis.get("risk_score"),
        "flags": sim_flags,
        "explanations": sim_explanations,
        "confidence": sim_result.get("confidence"),
    }

    behavior_analysis = behavior_result.get("behavior_analysis", {})
    behavior_flags = _safe_list(behavior_analysis.get("behavior_flags", []))
    behavior_explanations = _normalize_explanations(
        behavior_analysis.get("explanations"),
        behavior_flags,
        fallback_prefix="Signal comportement",
    )
    bundle["behavior"] = {
        "name": "behavior",
        "score": behavior_analysis.get("brs_score"),
        "flags": behavior_flags,
        "explanations": behavior_explanations,
        "confidence": behavior_result.get("confidence"),
    }

    fraud_analysis = fraud_result.get("fraud_analysis", {})
    fraud_flags = _safe_list(fraud_analysis.get("detected_flags", []))
    fraud_explanations = _normalize_explanations(
        fraud_analysis.get("explanations"),
        fraud_flags,
        fallback_prefix="Signal fraude",
    )
    bundle["fraud"] = {
        "name": "fraud",
        "score": fraud_analysis.get("fraud_score"),
        "flags": fraud_flags,
        "explanations": fraud_explanations,
        "confidence": fraud_result.get("confidence"),
    }

    image_analysis = image_result.get("image_analysis", {})
    image_flags = _safe_list(image_analysis.get("flags", []))
    image_explanations = _normalize_explanations(
        image_analysis.get("explanations"),
        image_flags,
        fallback_prefix="Signal image",
    )
    bundle["image"] = {
        "name": "image",
        "score": image_analysis.get("ifs_score"),
        "flags": image_flags,
        "explanations": image_explanations,
        "confidence": image_result.get("confidence"),
    }

    customer_expl = (explanation_result.get("explanation") or {}).get("customer_explanation", {})
    internal_expl = (explanation_result.get("explanation") or {}).get("internal_explanation", {})
    bundle["explanation"] = {
        "name": "explanation",
        "score": None,
        "flags": None,
        "explanations": {
            "global_summary": customer_expl.get("summary") or internal_expl.get("summary"),
            "customer_explanation": customer_expl,
            "internal_explanation": internal_expl,
        },
        "confidence": explanation_result.get("explanation_confidence"),
    }

    return bundle


def run_orchestrator(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute agents, aggregate outputs, and produce an explanation-ready payload."""

    case_id = request_data.get("case_id") or request_data.get("request_id") or "unknown"
    user_id = request_data.get("user_id")

    payment_context = request_data.get("payment_history")
    if not payment_context and user_id:
        try:
            payment_context = fetch_payment_context(int(user_id), int(case_id) if str(case_id).isdigit() else None)
        except Exception:
            payment_context = None
    if payment_context:
        request_data = {**request_data, "payment_history": payment_context}

    documents_payload = _build_documents_payload(request_data)
    declared_profile = request_data.get("declared_profile") or _build_declared_profile(request_data)
    telemetry = request_data.get("telemetry") or {}
    payment_summary = request_data.get("payment_behavior_summary")
    if not payment_summary and payment_context:
        payment_summary = payment_context.get("payment_behavior_summary")
    if payment_summary:
        request_data = {**request_data, "payment_behavior_summary": payment_summary}

    doc_request = {
        "case_id": case_id,
        "declared_profile": declared_profile,
        "documents": documents_payload,
    }

    if analyze_documents:
        try:
            doc_result = analyze_documents(doc_request)
        except Exception:
            doc_result = {"case_id": case_id, "document_analysis": {}, "confidence": 0.4}
    else:
        doc_result = {"case_id": case_id, "document_analysis": {}, "confidence": 0.0}

    if analyze_behavior:
        try:
            behavior_result = analyze_behavior({
                "case_id": case_id,
                "telemetry": telemetry,
                "payment_behavior_summary": payment_summary,
                "payment_history": payment_context,
            })
        except Exception:
            behavior_result = {"case_id": case_id, "behavior_analysis": {}, "confidence": 0.4}
    else:
        behavior_result = {"case_id": case_id, "behavior_analysis": {}, "confidence": 0.0}

    if analyze_similarity:
        try:
            sim_result = analyze_similarity({**_build_similarity_payload(request_data), "payment_behavior_summary": payment_summary})
        except Exception:
            sim_result = {"case_id": case_id, "ai_analysis": {}, "rag_statistics": {}, "confidence": 0.3}
    else:
        sim_result = {"case_id": case_id, "ai_analysis": {}, "rag_statistics": {}, "confidence": 0.0}

    request_image_flags = _safe_list(request_data.get("image_flags", []))
    if analyze_images:
        try:
            image_result = analyze_images({
                "case_id": case_id,
                "image_flags": request_image_flags,
            })
        except Exception:
            image_result = {"case_id": case_id, "image_analysis": {}, "confidence": 0.4}
    else:
        image_result = {"case_id": case_id, "image_analysis": {}, "confidence": 0.0}

    doc_flags = _safe_list(doc_result.get("document_analysis", {}).get("flags", []))
    behavior_flags = _safe_list(behavior_result.get("behavior_analysis", {}).get("behavior_flags", []))
    image_flags = _merge_flags(request_image_flags, image_result.get("image_analysis", {}).get("flags", []))

    sim_analysis = sim_result.get("ai_analysis", {}) if isinstance(sim_result, dict) else {}
    sim_red_flags = _safe_list(sim_analysis.get("red_flags", []))
    sim_risk_level = str(sim_analysis.get("risk_level", "")).lower()
    similarity_flags: List[str] = []
    if sim_risk_level in {"eleve", "high"}:
        similarity_flags.append("PEER_RISK_HIGH")
    similarity_flags.extend(sim_red_flags)

    projected_ratio = _compute_projected_debt_ratio(request_data)
    txn_flags = _safe_list(request_data.get("transaction_flags", []))
    if projected_ratio >= 45:
        txn_flags.append("HIGH_DTI")

    fraud_payload = {
        "case_id": case_id,
        "document_flags": doc_flags,
        "behavior_flags": behavior_flags,
        "transaction_flags": txn_flags,
        "image_flags": image_flags,
        "similarity_flags": similarity_flags,
        "free_text": _safe_list(request_data.get("free_text", [])),
        "payment_history": payment_context,
    }

    if analyze_fraud:
        try:
            fraud_result = analyze_fraud(fraud_payload)
        except Exception:
            fraud_result = {"case_id": case_id, "fraud_analysis": {}, "confidence": 0.4}
    else:
        fraud_result = {"case_id": case_id, "fraud_analysis": {}, "confidence": 0.0}

    agent_results = {
        "document_agent": doc_result,
        "similarity_agent": sim_result,
        "behavior_agent": behavior_result,
        "fraud_agent": fraud_result,
        "image_agent": image_result,
    }

    if orchestrate_decision:
        orchestrator_output = orchestrate_decision({"case_id": case_id, **request_data}, agent_results)
        proposed = orchestrator_output.get("proposed_decision", "review")
        confidence = orchestrator_output.get("decision_confidence", 0.0)
    else:
        orchestrator_output = {
            "case_id": case_id,
            "proposed_decision": "review",
            "decision_confidence": 0.0,
            "human_review_required": True,
            "aggregated_signals": {},
        }
        proposed = "review"
        confidence = 0.0

    reason_codes = _derive_reason_codes(request_data, doc_result, behavior_result, sim_result, fraud_result, image_result)

    decision_agent_raw: Dict[str, Any] = {}
    if make_decision_payload:
        try:
            decision_agent_raw = make_decision_payload(
                doc_result,
                sim_result,
                behavior_result=behavior_result,
                payment_summary=payment_summary,
                fraud_result=fraud_result,
                image_result=image_result,
                orchestrator_output=orchestrator_output,
                reason_codes=reason_codes,
                case_id=str(case_id),
            )
        except Exception:
            decision_agent_raw = {}

    decision_choice = decision_agent_raw.get("decision") or proposed
    decision_label = _normalize_decision_label(str(decision_choice))
    decision_confidence = decision_agent_raw.get("decision_confidence", decision_agent_raw.get("confidence", confidence))
    try:
        decision_confidence = float(decision_confidence)
    except (TypeError, ValueError):
        decision_confidence = confidence
    if decision_confidence < 0:
        decision_confidence = 0.0
    if decision_confidence > 1:
        decision_confidence = 1.0

    human_review_required = decision_agent_raw.get("human_review_required")
    if not isinstance(human_review_required, bool):
        human_review_required = orchestrator_output.get("human_review_required", True)

    decision_reason_codes = decision_agent_raw.get("reason_codes")
    if not isinstance(decision_reason_codes, list):
        decision_reason_codes = reason_codes

    decision_payload = {
        "case_id": case_id,
        "decision": decision_label,
        "decision_raw": decision_agent_raw.get("decision_raw", decision_choice),
        "reason_codes": decision_reason_codes,
        "decision_confidence": round(decision_confidence, 4),
        "human_review_required": human_review_required,
    }
    if isinstance(decision_agent_raw.get("summary"), str):
        decision_payload["summary"] = decision_agent_raw.get("summary")

    agent_results["decision_agent"] = decision_agent_raw or {
        "decision": decision_payload.get("decision"),
        "decision_confidence": decision_payload.get("decision_confidence"),
        "reason_codes": decision_payload.get("reason_codes"),
        "human_review_required": decision_payload.get("human_review_required"),
    }

    if explain_decision:
        try:
            explanation_result = explain_decision(
                decision_payload,
                doc_result,
                sim_result,
                behavior_result=behavior_result,
                fraud_result=fraud_result,
                image_result=image_result,
                payment_behavior_summary=payment_summary,
            )
        except Exception:
            explanation_result = {"case_id": case_id, "explanation": {}, "explanation_confidence": 0.3}
    else:
        explanation_result = {"case_id": case_id, "explanation": {}, "explanation_confidence": 0.0}

    agent_bundle = _build_agent_bundle(doc_result, sim_result, behavior_result, fraud_result, image_result, explanation_result)
    agent_bundle["decision"] = _build_decision_agent_output(
        decision_payload,
        orchestrator_output,
        explanation_result,
    )
    customer_expl = (explanation_result.get("explanation") or {}).get("customer_explanation", {})
    customer_summary = customer_expl.get("summary")
    if customer_summary:
        summary = customer_summary
    else:
        final_reasons = decision_payload.get("reason_codes") or reason_codes
        reasons = ", ".join(final_reasons) if final_reasons else "aucun signal majeur"
        summary = f"Decision proposee: {decision_payload['decision']} | Raisons: {reasons}"

    return {
        "case_id": case_id,
        "decision": decision_payload,
        "orchestrator": orchestrator_output,
        "explanation": explanation_result,
        "agents": agent_bundle,
        "agents_raw": agent_results,
        "summary": summary,
        "customer_explanation": customer_expl.get("summary"),
    }
