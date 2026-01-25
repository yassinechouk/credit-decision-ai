from typing import Any, Dict, List, Optional

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
    bundle["document"] = {
        "name": "document",
        "score": doc_analysis.get("dds_score"),
        "flags": doc_flags,
        "explanations": doc_explanations,
        "confidence": doc_result.get("confidence"),
    }

    sim_analysis = sim_result.get("ai_analysis", {})
    sim_summary = sim_analysis.get("summary") or sim_analysis.get("reasoning")
    sim_flags = _safe_list(sim_analysis.get("red_flags", []))
    sim_explanations = _normalize_explanations(
        {},
        sim_flags,
        summary=sim_summary,
        fallback_prefix="Signal similarite",
    )
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

    documents_payload = _build_documents_payload(request_data)
    declared_profile = request_data.get("declared_profile") or _build_declared_profile(request_data)
    telemetry = request_data.get("telemetry") or {}

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
            behavior_result = analyze_behavior({"case_id": case_id, "telemetry": telemetry})
        except Exception:
            behavior_result = {"case_id": case_id, "behavior_analysis": {}, "confidence": 0.4}
    else:
        behavior_result = {"case_id": case_id, "behavior_analysis": {}, "confidence": 0.0}

    if analyze_similarity:
        try:
            sim_result = analyze_similarity(_build_similarity_payload(request_data))
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
    decision_payload = {
        "case_id": case_id,
        "decision": _normalize_decision_label(str(proposed)),
        "decision_raw": proposed,
        "reason_codes": reason_codes,
        "decision_confidence": confidence,
        "human_review_required": orchestrator_output.get("human_review_required", True),
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
            )
        except Exception:
            explanation_result = {"case_id": case_id, "explanation": {}, "explanation_confidence": 0.3}
    else:
        explanation_result = {"case_id": case_id, "explanation": {}, "explanation_confidence": 0.0}

    agent_bundle = _build_agent_bundle(doc_result, sim_result, behavior_result, fraud_result, image_result, explanation_result)
    customer_expl = (explanation_result.get("explanation") or {}).get("customer_explanation", {})
    customer_summary = customer_expl.get("summary")
    if customer_summary:
        summary = customer_summary
    else:
        reasons = ", ".join(reason_codes) if reason_codes else "aucun signal majeur"
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
