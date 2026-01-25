"""Agent chat helper for banker interactions.

Generates agent-specific replies using available agent outputs and optional LLM.
Falls back to a deterministic summary when no LLM is configured.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, List


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def _llm_client():
    if not OPENAI_API_KEY:
        return None
    try:
        openai_mod = importlib.import_module("openai")
        OpenAI = getattr(openai_mod, "OpenAI")
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except Exception:
        return None


def _safe_dict(val: Any) -> Dict[str, Any]:
    return val if isinstance(val, dict) else {}


def _summarize_document(agent_data: Dict[str, Any]) -> str:
    doc = _safe_dict(agent_data.get("document_analysis"))
    flags = doc.get("flags") or []
    extracted = doc.get("extracted_fields") or {}
    return (
        "Document: "
        f"score={doc.get('dds_score', 'n/a')}, "
        f"consistency={doc.get('consistency_level', 'n/a')}, "
        f"flags={flags}, extracted={extracted}"
    )


def _summarize_behavior(agent_data: Dict[str, Any]) -> str:
    beh = _safe_dict(agent_data.get("behavior_analysis"))
    flags = beh.get("behavior_flags") or []
    return (
        "Behavior: "
        f"score={beh.get('brs_score', 'n/a')}, "
        f"level={beh.get('behavior_level', 'n/a')}, "
        f"flags={flags}"
    )


def _summarize_similarity(agent_data: Dict[str, Any]) -> str:
    ai = _safe_dict(agent_data.get("ai_analysis"))
    stats = _safe_dict(agent_data.get("rag_statistics"))
    return (
        "Similarity: "
        f"rec={ai.get('recommendation', 'n/a')}, "
        f"risk={ai.get('risk_score', 'n/a')}, "
        f"level={ai.get('risk_level', 'n/a')}, "
        f"success_rate={stats.get('repayment_success_rate', 'n/a')}"
    )


def _summarize_fraud(agent_data: Dict[str, Any]) -> str:
    fraud = _safe_dict(agent_data.get("fraud_analysis"))
    flags = fraud.get("detected_flags") or agent_data.get("fraud_flags") or []
    return (
        "Fraud: "
        f"score={fraud.get('fraud_score', agent_data.get('fraud_score', 'n/a'))}, "
        f"level={fraud.get('risk_level', agent_data.get('risk_level', 'n/a'))}, "
        f"flags={flags}"
    )


def _summarize_explanation(agent_data: Dict[str, Any]) -> str:
    expl = _safe_dict(agent_data.get("explanation"))
    if not expl:
        expl = _safe_dict(agent_data.get("explanations"))
    if not expl:
        expl = _safe_dict(agent_data)
    customer = _safe_dict(expl.get("customer_explanation"))
    internal = _safe_dict(expl.get("internal_explanation"))
    summary = (
        expl.get("summary")
        or expl.get("global_summary")
        or customer.get("summary")
        or internal.get("summary")
    )
    return (
        "Explanation: "
        f"client_summary={summary or 'n/a'}, "
        f"internal_summary={internal.get('summary', 'n/a')}"
    )


def _summarize_image(agent_data: Dict[str, Any]) -> str:
    img = _safe_dict(agent_data.get("image_analysis"))
    flags = img.get("flags") or []
    return (
        "Image: "
        f"score={img.get('ifs_score', 'n/a')}, "
        f"level={img.get('risk_level', 'n/a')}, "
        f"flags={flags}"
    )


def _summarize_orchestrator(agent_data: Dict[str, Any]) -> str:
    return (
        "Orchestrator: "
        f"decision={agent_data.get('proposed_decision', 'n/a')}, "
        f"confidence={agent_data.get('decision_confidence', 'n/a')}, "
        f"human_review={agent_data.get('human_review_required', 'n/a')}"
    )


def _build_agent_context(agent_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
    agents_raw = _safe_dict(request.get("agents_raw"))
    agents_compact = _safe_dict(request.get("agents"))
    orchestrator = _safe_dict(request.get("orchestrator"))

    mapping = {
        "document": "document_agent",
        "behavior": "behavior_agent",
        "similarity": "similarity_agent",
        "image": "image_agent",
        "fraud": "fraud_agent",
        "explanation": "explanation_agent",
    }
    agent_key = mapping.get(agent_name, agent_name)
    agent_payload = _safe_dict(agents_raw.get(agent_key))
    if not agent_payload and agent_name in agents_compact:
        agent_payload = _safe_dict(agents_compact.get(agent_name))

    case_context = {
        "case_id": request.get("id") or request.get("case_id"),
        "amount": request.get("amount"),
        "duration_months": request.get("duration_months"),
        "monthly_income": request.get("monthly_income"),
        "other_income": request.get("other_income"),
        "monthly_charges": request.get("monthly_charges"),
        "employment_type": request.get("employment_type"),
        "contract_type": request.get("contract_type"),
        "seniority_years": request.get("seniority_years"),
        "family_status": request.get("family_status"),
        "marital_status": request.get("marital_status"),
        "number_of_children": request.get("number_of_children"),
        "spouse_employed": request.get("spouse_employed"),
        "housing_status": request.get("housing_status"),
        "is_primary_holder": request.get("is_primary_holder"),
        "documents": request.get("documents"),
    }

    return {
        "case": case_context,
        "agent_name": agent_name,
        "agent_output": agent_payload,
        "orchestrator": orchestrator,
    }


def _fallback_reply(agent_name: str, request: Dict[str, Any]) -> str:
    agents_raw = _safe_dict(request.get("agents_raw"))
    orchestrator = _safe_dict(request.get("orchestrator"))

    summaries: List[str] = []
    if agent_name == "document":
        summaries.append(_summarize_document(_safe_dict(agents_raw.get("document_agent"))))
    elif agent_name == "behavior":
        summaries.append(_summarize_behavior(_safe_dict(agents_raw.get("behavior_agent"))))
    elif agent_name == "similarity":
        summaries.append(_summarize_similarity(_safe_dict(agents_raw.get("similarity_agent"))))
    elif agent_name == "image":
        summaries.append(_summarize_image(_safe_dict(agents_raw.get("image_agent"))))
    elif agent_name == "fraud":
        summaries.append(_summarize_fraud(_safe_dict(agents_raw.get("fraud_agent"))))
    elif agent_name == "explanation":
        expl_payload = _safe_dict(request.get("explanation", {}))
        if not expl_payload:
            expl_payload = _safe_dict(_safe_dict(request.get("agents", {})).get("explanation", {}))
        summaries.append(_summarize_explanation(expl_payload))
    elif agent_name == "orchestrator":
        summaries.append(_summarize_orchestrator(orchestrator))
    else:
        summaries.append("Agent inconnu: aucune donnée disponible.")

    return "Résumé rapide (LLM indisponible): " + " | ".join(summaries)


def build_initial_agent_reply(agent_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
    agent_name = agent_name.lower()
    context = _build_agent_context(agent_name, request)
    raw_summary = _fallback_reply(agent_name, request)
    summary = raw_summary.replace("Résumé rapide (LLM indisponible): ", "Analyse initiale: ")
    if summary == raw_summary:
        summary = "Analyse initiale: " + raw_summary
    return {
        "summary": summary,
        "structured_output": context.get("agent_output", {}),
    }


def generate_agent_reply(agent_name: str, request: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    agent_name = agent_name.lower()
    client = _llm_client()
    context = _build_agent_context(agent_name, request)

    if not client:
        return {
            "summary": _fallback_reply(agent_name, request),
            "structured_output": context.get("agent_output", {}),
        }

    system_prompt = (
        "Tu es un agent interne qui aide un analyste bancaire. "
        "Réponds en français, de manière concise, factuelle et actionnable. "
        "Appuie tes réponses sur le contexte fourni. "
        "Si l'information manque, dis-le explicitement. "
        "N'annonce pas une décision finale; tu peux proposer des vérifications."
    )

    history_messages = []
    for msg in history[-8:]:
        role = "assistant"
        if msg.get("role") in {"banker", "user"}:
            role = "user"
        history_messages.append({"role": role, "content": str(msg.get("content", ""))})

    prompt_context = "CONTEXTE (JSON):\n" + json.dumps(context, ensure_ascii=False, indent=2)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_context}]
    messages.extend(history_messages)

    try:
        chat = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.2,
        )
        summary = chat.choices[0].message.content or _fallback_reply(agent_name, request)
        return {
            "summary": summary,
            "structured_output": context.get("agent_output", {}),
        }
    except Exception:
        return {
            "summary": _fallback_reply(agent_name, request),
            "structured_output": context.get("agent_output", {}),
        }
