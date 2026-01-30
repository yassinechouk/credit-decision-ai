from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Optional, List, Any

import json
import os
import hashlib
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse

from api.schemas import (
    LoginRequest,
    LoginResponse,
    CreditRequestCreate,
    CreditRequest,
    BankerRequest,
    DecisionCreate,
    CommentCreate,
    AgentBundle,
    AgentResult,
    DocumentInfo,
    DecisionInfo,
    Comment,
    AgentChatRequest,
    AgentChatResponse,
    AgentChatMessage,
    LoanInfo,
    InstallmentInfo,
    PaymentInfo,
    PaymentCreate,
    PaymentBehaviorSummary,
)
from api.deps import get_current_user
from core.orchestrator import run_orchestrator
from core.db import (
    fetch_user_by_email,
    create_credit_request as create_credit_request_db,
    list_cases_for_banker,
    fetch_case_detail,
    fetch_case_detail_for_client,
    upsert_decision,
    add_comment as add_comment_db,
    list_cases_for_client,
    save_orchestration,
    add_case_documents,
    get_agent_session,
    upsert_agent_session,
    ensure_agent_session_snapshot,
    clear_agent_sessions_for_banker,
    resubmit_credit_request_db,
    create_payment_for_case,
)
from agents.chat_agent import generate_agent_reply, build_initial_agent_reply

try:
    from services.vector_sync import sync_credit_case_to_qdrant  # type: ignore
except Exception:  # pragma: no cover
    sync_credit_case_to_qdrant = None  # type: ignore


router = APIRouter(prefix="/api")


UPLOAD_ROOT = os.getenv("UPLOAD_DIR", "/app/data/uploads")


def _require_role(user: Dict[str, Any], role: str) -> None:
    if user.get("role") != role:
        raise HTTPException(status_code=403, detail="Forbidden")


def _parse_json_field(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc


def _infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    if "salary" in lower or "pay" in lower or "bulletin" in lower or "payslip" in lower:
        return "salary_slip"
    if "contract" in lower or "contrat" in lower:
        return "employment_contract"
    if "bank" in lower or "statement" in lower or "releve" in lower:
        return "bank_statement"
    return "other"


def _extract_pdf_text(content: bytes) -> str:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ""
    try:
        import io
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        return ""


async def _store_files(case_id: int, files: Optional[List[UploadFile]]) -> List[Dict[str, Any]]:
    if not files:
        return []
    case_dir = Path(UPLOAD_ROOT) / str(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)

    stored: List[Dict[str, Any]] = []
    for upload in files:
        raw = await upload.read()
        filename = upload.filename or f"document-{uuid4()}.pdf"
        safe_name = filename.replace("..", ".").replace("/", "_")
        file_path = case_dir / safe_name
        file_path.write_bytes(raw)

        file_hash = hashlib.sha256(raw).hexdigest()
        text = ""
        if safe_name.lower().endswith(".pdf"):
            text = _extract_pdf_text(raw)
        else:
            try:
                text = raw.decode("utf-8")
            except Exception:
                text = ""

        stored.append({
            "document_type": _infer_doc_type(safe_name),
            "file_path": str(file_path),
            "file_hash": file_hash,
            "filename": safe_name,
            "raw_text": text,
        })
    return stored


def _map_status(db_status: str, decision_row: Optional[Dict[str, Any]]) -> str:
    if decision_row:
        decision_value = decision_row.get("decision")
        if decision_value == "APPROVE":
            return "approved"
        if decision_value == "REJECT":
            return "rejected"
        return "in_review"
    if db_status == "SUBMITTED":
        return "pending"
    if db_status == "UNDER_REVIEW":
        return "in_review"
    return "pending"


def _map_decision(decision_row: Optional[Dict[str, Any]]) -> Optional[DecisionInfo]:
    if not decision_row:
        return None
    decision = decision_row.get("decision")
    if decision not in {"APPROVE", "REJECT", "REVIEW"}:
        return None
    decision_lower = decision.lower()
    return DecisionInfo(
        decision=decision_lower,
        confidence=decision_row.get("confidence"),
        reason_codes=decision_row.get("reason_codes"),
        note=decision_row.get("note"),
        decided_by=str(decision_row.get("decided_by")) if decision_row.get("decided_by") is not None else None,
        decided_at=decision_row.get("decided_at"),
    )


def _map_agent_outputs(agent_rows: List[Dict[str, Any]]) -> Optional[AgentBundle]:
    if not agent_rows:
        return None
    bundle = AgentBundle()
    for row in agent_rows:
        name = row.get("agent_name")
        output = row.get("output_json") or {}
        if isinstance(output, str):
            try:
                import json

                output = json.loads(output)
            except Exception:
                output = {"summary": output}
        result = AgentResult(
            name=name or "agent",
            score=output.get("score"),
            flags=output.get("flags") or output.get("flag"),
            confidence=output.get("confidence"),
            explanations=output.get("explanations")
            or ({"global_summary": output.get("summary")} if output.get("summary") else None),
        )
        if name == "document":
            bundle.document = result
        elif name == "similarity":
            bundle.similarity = result
        elif name == "fraud":
            bundle.fraud = result
        elif name == "decision":
            bundle.decision = result
        elif name == "behavior":
            bundle.behavior = result
        elif name == "image":
            bundle.image = result
        elif name == "explanation":
            bundle.explanation = result
    return bundle


def _map_documents(doc_rows: List[Dict[str, Any]]) -> List[DocumentInfo]:
    return [
        DocumentInfo(
            document_id=row["document_id"],
            document_type=row["document_type"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            uploaded_at=row["uploaded_at"],
        )
        for row in doc_rows
    ]


def _map_comments(comment_rows: List[Dict[str, Any]]) -> List[Comment]:
    return [
        Comment(
            author_id=str(row.get("author_id")),
            message=row.get("message"),
            created_at=row.get("created_at"),
            is_public=row.get("is_public"),
        )
        for row in comment_rows
    ]


def _map_loan(loan_row: Optional[Dict[str, Any]]) -> Optional[LoanInfo]:
    if not loan_row:
        return None
    return LoanInfo(
        loan_id=int(loan_row["loan_id"]),
        user_id=int(loan_row["user_id"]),
        case_id=loan_row.get("case_id"),
        principal_amount=float(loan_row["principal_amount"]),
        interest_rate=float(loan_row["interest_rate"]),
        term_months=int(loan_row["term_months"]),
        status=str(loan_row["status"]),
        approved_at=loan_row.get("approved_at"),
        start_date=loan_row.get("start_date"),
        end_date=loan_row.get("end_date"),
        created_at=loan_row.get("created_at"),
    )


def _map_installments(rows: List[Dict[str, Any]]) -> List[InstallmentInfo]:
    return [
        InstallmentInfo(
            installment_id=int(row["installment_id"]),
            loan_id=int(row["loan_id"]),
            installment_number=int(row["installment_number"]),
            due_date=row["due_date"],
            amount_due=float(row["amount_due"]),
            status=str(row["status"]),
            amount_paid=float(row.get("amount_paid") or 0),
            paid_at=row.get("paid_at"),
            days_late=row.get("days_late"),
            created_at=row.get("created_at"),
        )
        for row in rows
    ]


def _map_payments(rows: List[Dict[str, Any]]) -> List[PaymentInfo]:
    return [
        PaymentInfo(
            payment_id=int(row["payment_id"]),
            loan_id=int(row["loan_id"]),
            installment_id=row.get("installment_id"),
            payment_date=row["payment_date"],
            amount=float(row["amount"]),
            channel=str(row["channel"]),
            status=str(row["status"]),
            is_reversal=bool(row.get("is_reversal")),
            reversal_of=row.get("reversal_of"),
            created_at=row.get("created_at"),
        )
        for row in rows
    ]


def _map_payment_summary(row: Optional[Dict[str, Any]]) -> Optional[PaymentBehaviorSummary]:
    if not row:
        return None
    return PaymentBehaviorSummary(
        summary_id=int(row["summary_id"]),
        user_id=int(row["user_id"]),
        total_loans=int(row.get("total_loans") or 0),
        total_installments=int(row.get("total_installments") or 0),
        on_time_installments=int(row.get("on_time_installments") or 0),
        late_installments=int(row.get("late_installments") or 0),
        missed_installments=int(row.get("missed_installments") or 0),
        on_time_rate=float(row.get("on_time_rate") or 0),
        avg_days_late=float(row.get("avg_days_late") or 0),
        max_days_late=int(row.get("max_days_late") or 0),
        avg_payment_amount=float(row.get("avg_payment_amount") or 0),
        last_payment_date=row.get("last_payment_date"),
        updated_at=row.get("updated_at"),
    )


def _build_agents_raw(detail: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "document": "document_agent",
        "similarity": "similarity_agent",
        "behavior": "behavior_agent",
        "fraud": "fraud_agent",
        "image": "image_agent",
        "explanation": "explanation_agent",
    }
    agents_raw: Dict[str, Any] = {}
    for row in detail.get("agent_outputs", []) or []:
        name = row.get("agent_name")
        key = mapping.get(name, name)
        output = row.get("output_json") or {}
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                output = {"summary": output}
        agents_raw[key] = output
    return agents_raw


def _build_chat_snapshot(detail: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = _build_banker_request(detail).model_dump(mode="json")
    snapshot["agents_raw"] = _build_agents_raw(detail)
    snapshot["orchestrator"] = {
        "proposed_decision": detail.get("auto_decision"),
        "decision_confidence": detail.get("auto_decision_confidence"),
        "human_review_required": detail.get("auto_review_required"),
    }
    return snapshot


def _prime_agent_sessions(detail: Dict[str, Any], banker_id: Optional[int] = None) -> None:
    if banker_id is None:
        return
    case_id = int(detail["case_id"])
    snapshot = _build_chat_snapshot(detail)
    for agent_name in ("document", "behavior", "similarity", "fraud", "image", "decision", "explanation"):
        ensure_agent_session_snapshot(case_id, agent_name, snapshot, banker_id)


def _parse_json_field(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _infer_doc_type(filename: str, raw_text: str = "") -> str:
    lower = filename.lower()
    text = (raw_text or "").lower()
    if "salary" in lower or "pay" in lower or "bulletin" in lower or "payslip" in lower:
        return "salary_slip"
    if "contract" in lower or "contrat" in lower or "employment" in lower:
        return "employment_contract"
    if "bank" in lower or "statement" in lower or "releve" in lower:
        return "bank_statement"
    if "invoice" in lower or "facture" in lower:
        return "invoice"

    # Fallback: infer from extracted text when filename is generic (ex: 1.pdf).
    if "fiche de paie" in text or "salaire" in text or "payslip" in text:
        return "salary_slip"
    if "contrat de travail" in text or "cdi" in text or "cdd" in text:
        return "employment_contract"
    if "relev" in text or "iban" in text or "banque" in text:
        return "bank_statement"
    return "uploaded"


def _safe_filename(name: str) -> str:
    return os.path.basename(name).replace("\\", "/").split("/")[-1]


def _hydrate_documents(detail: Dict[str, Any]) -> None:
    """
    Ensure documents in detail include raw_text and inferred document_type,
    even if filenames are generic (ex: 1.pdf).
    """
    documents = detail.get("documents") or []
    updated: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        file_path = str(doc.get("file_path") or "")
        filename = _safe_filename(file_path) or str(doc.get("filename") or "")
        raw_text = str(doc.get("raw_text") or "")
        if not raw_text and file_path and os.path.exists(file_path):
            try:
                data = Path(file_path).read_bytes()
                if filename.lower().endswith(".pdf"):
                    raw_text = _extract_pdf_text(data)
                else:
                    try:
                        raw_text = data.decode("utf-8")
                    except Exception:
                        raw_text = ""
            except Exception:
                raw_text = ""
        document_type = doc.get("document_type") or doc.get("doc_type") or "uploaded"
        # Improve doc type detection using raw text.
        document_type = _infer_doc_type(filename, raw_text)
        updated.append(
            {
                **doc,
                "filename": filename or doc.get("filename"),
                "raw_text": raw_text,
                "document_type": document_type,
                "doc_type": document_type,
            }
        )
    detail["documents"] = updated


async def _store_files(case_id: int, files: Optional[List[UploadFile]]) -> List[Dict[str, str]]:
    stored_documents: List[Dict[str, str]] = []
    files = files or []
    if not files:
        return stored_documents
    dest_dir = Path(UPLOAD_ROOT) / str(case_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        filename = _safe_filename(file.filename or "document")
        dest_path = dest_dir / filename
        data = await file.read()
        dest_path.write_bytes(data)
        raw_text = ""
        if filename.lower().endswith(".pdf"):
            raw_text = _extract_pdf_text(data)
        else:
            try:
                raw_text = data.decode("utf-8")
            except Exception:
                raw_text = ""
        doc_type = _infer_doc_type(filename, raw_text)
        stored_documents.append(
            {
                "file_path": str(dest_path),
                "document_type": doc_type,
                "file_hash": hashlib.sha256(data).hexdigest(),
                "filename": filename,
                "doc_type": doc_type,
                "raw_text": raw_text,
            }
        )
    return stored_documents


def _build_banker_request(detail: Dict[str, Any]) -> BankerRequest:
    return BankerRequest(
        id=str(detail["case_id"]),
        status=_map_status(detail["status"], detail.get("decision")),
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        client_id=str(detail["user_id"]),
        summary=detail.get("summary"),
        auto_decision=detail.get("auto_decision"),
        auto_decision_confidence=float(detail["auto_decision_confidence"]) if detail.get("auto_decision_confidence") is not None else None,
        auto_review_required=detail.get("auto_review_required"),
        amount=float(detail["loan_amount"]),
        duration_months=int(detail["loan_duration"]),
        monthly_income=float(detail["monthly_income"]),
        other_income=float(detail.get("other_income") or 0),
        monthly_charges=float(detail["monthly_charges"]),
        employment_type=detail.get("employment_type"),
        contract_type=detail.get("contract_type"),
        seniority_years=detail.get("seniority_years"),
        marital_status=detail.get("marital_status"),
        number_of_children=detail.get("number_of_children"),
        spouse_employed=detail.get("spouse_employed"),
        housing_status=detail.get("housing_status"),
        is_primary_holder=detail.get("is_primary_holder"),
        documents=_map_documents(detail.get("documents", [])),
        agents=_map_agent_outputs(detail.get("agent_outputs", [])),
        comments=_map_comments(detail.get("comments", [])),
        decision=_map_decision(detail.get("decision")),
        loan=_map_loan(detail.get("loan")),
        installments=_map_installments(detail.get("installments", [])),
        payments=_map_payments(detail.get("payments", [])),
        payment_behavior_summary=_map_payment_summary(detail.get("payment_behavior_summary")),
    )


# --- Auth ---------------------------------------------------------------------
@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    email = body.email.strip().lower()
    password = body.password.strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    user = fetch_user_by_email(email)
    if not user or user.get("password_hash") != password:
        raise HTTPException(status_code=401, detail="Mot de passe incorrect")

    role_value = user.get("role")
    if role_value not in ("CLIENT", "BANKER"):
        raise HTTPException(status_code=403, detail="User role not allowed")

    role = "banker" if role_value == "BANKER" else "client"
    user_id = int(user.get("user_id"))
    token = f"token:{role}:{user_id}:{uuid4()}"
    return LoginResponse(token=token, role=role, user_id=str(user_id))


@router.post("/auth/logout")
def logout(user: Dict = Depends(get_current_user)):
    if (user.get("role") or "").lower() == "banker":
        clear_agent_sessions_for_banker(int(user.get("user_id")))
    return {"status": "ok"}


# --- Client -------------------------------------------------------------------
@router.post("/client/credit-requests", response_model=CreditRequest)
def create_credit_request(body: CreditRequestCreate, user=Depends(get_current_user), background_tasks: BackgroundTasks = None):
    _require_role(user, "client")
    payload_data = body.model_dump()
    documents_payloads = [
        {"doc_type": _infer_doc_type(name), "raw_text": "", "filename": name}
        for name in (payload_data.get("documents") or [])
    ]
    if documents_payloads:
        payload_data = {**payload_data, "documents_payloads": documents_payloads}

    created = create_credit_request_db(user["user_id"], payload_data, None)
    case_id = int(created["case_id"])
    orchestration = run_orchestrator({**payload_data, "case_id": case_id, "user_id": user["user_id"]})
    save_orchestration(case_id, orchestration)
    if sync_credit_case_to_qdrant and background_tasks is not None:
        background_tasks.add_task(sync_credit_case_to_qdrant, case_id)
    banker_detail = fetch_case_detail(case_id)
    if banker_detail:
        _hydrate_documents(banker_detail)
        _prime_agent_sessions(banker_detail)
    detail = fetch_case_detail_for_client(created["case_id"], user["user_id"])
    if not detail:
        raise HTTPException(status_code=500, detail="Failed to create request")
    decision_info = _map_decision(detail.get("decision"))
    return CreditRequest(
        id=str(detail["case_id"]),
        status=_map_status(detail["status"], detail.get("decision")),
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        client_id=str(detail["user_id"]),
        summary=detail.get("summary") or "Dossier créé",
        customer_explanation=None,
        agents=_map_agent_outputs(detail.get("agent_outputs", [])),
        decision=decision_info,
        comments=_map_comments(detail.get("comments", [])),
        auto_decision=detail.get("auto_decision"),
        auto_decision_confidence=float(detail["auto_decision_confidence"]) if detail.get("auto_decision_confidence") is not None else None,
        auto_review_required=detail.get("auto_review_required"),
        loan=_map_loan(detail.get("loan")),
        installments=_map_installments(detail.get("installments", [])),
        payments=_map_payments(detail.get("payments", [])),
        payment_behavior_summary=_map_payment_summary(detail.get("payment_behavior_summary")),
    )


@router.post("/client/credit-requests/upload", response_model=CreditRequest)
async def create_credit_request_upload(
    user=Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
    payload: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
):
    _require_role(user, "client")
    payload_data = _parse_json_field(payload)
    if not isinstance(payload_data, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    payload_for_db = {**payload_data, "documents": []}
    created = create_credit_request_db(user["user_id"], payload_for_db, None)
    case_id = int(created["case_id"])

    stored_documents = await _store_files(case_id, files)

    if stored_documents:
        add_case_documents(case_id, stored_documents)
        documents_payloads = [
            {
                "doc_type": doc.get("document_type"),
                "raw_text": doc.get("raw_text") or "",
                "filename": doc.get("filename"),
            }
            for doc in stored_documents
        ]
        payload_data = {
            **payload_data,
            "documents_payloads": documents_payloads,
            "documents": [doc.get("filename") for doc in stored_documents],
        }
    orchestration = run_orchestrator({**payload_data, "case_id": case_id, "user_id": user["user_id"]})
    save_orchestration(case_id, orchestration)
    if sync_credit_case_to_qdrant and background_tasks is not None:
        background_tasks.add_task(sync_credit_case_to_qdrant, case_id)
    banker_detail = fetch_case_detail(case_id)
    if banker_detail:
        _hydrate_documents(banker_detail)
        _prime_agent_sessions(banker_detail)

    detail = fetch_case_detail_for_client(created["case_id"], user["user_id"])
    if not detail:
        raise HTTPException(status_code=500, detail="Failed to create request")
    decision_info = _map_decision(detail.get("decision"))
    return CreditRequest(
        id=str(detail["case_id"]),
        status=_map_status(detail["status"], detail.get("decision")),
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        client_id=str(detail["user_id"]),
        summary=detail.get("summary") or "Dossier créé",
        customer_explanation=None,
        agents=_map_agent_outputs(detail.get("agent_outputs", [])),
        decision=decision_info,
        comments=_map_comments(detail.get("comments", [])),
        auto_decision=detail.get("auto_decision"),
        auto_decision_confidence=float(detail["auto_decision_confidence"]) if detail.get("auto_decision_confidence") is not None else None,
        auto_review_required=detail.get("auto_review_required"),
        loan=_map_loan(detail.get("loan")),
        installments=_map_installments(detail.get("installments", [])),
        payments=_map_payments(detail.get("payments", [])),
        payment_behavior_summary=_map_payment_summary(detail.get("payment_behavior_summary")),
    )


@router.get("/client/credit-requests/{req_id}", response_model=CreditRequest)
def get_credit_request(req_id: str, user=Depends(get_current_user)):
    _require_role(user, "client")
    detail = fetch_case_detail_for_client(int(req_id), user["user_id"])
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    return CreditRequest(
        id=str(detail["case_id"]),
        status=_map_status(detail["status"], detail.get("decision")),
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        client_id=str(detail["user_id"]),
        summary=detail.get("summary") or "Dossier en cours",
        customer_explanation=None,
        agents=_map_agent_outputs(detail.get("agent_outputs", [])),
        decision=_map_decision(detail.get("decision")),
        comments=_map_comments(detail.get("comments", [])),
        auto_decision=detail.get("auto_decision"),
        auto_decision_confidence=float(detail["auto_decision_confidence"]) if detail.get("auto_decision_confidence") is not None else None,
        auto_review_required=detail.get("auto_review_required"),
        loan=_map_loan(detail.get("loan")),
        installments=_map_installments(detail.get("installments", [])),
        payments=_map_payments(detail.get("payments", [])),
        payment_behavior_summary=_map_payment_summary(detail.get("payment_behavior_summary")),
    )


@router.post("/client/credit-requests/{req_id}/resubmit", response_model=CreditRequest)
async def resubmit_credit_request(
    req_id: str,
    user=Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
    payload: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
):
    _require_role(user, "client")
    payload_data = _parse_json_field(payload)
    if not isinstance(payload_data, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    stored_documents: List[Dict[str, str]] = []
    if files:
        stored_documents = await _store_files(int(req_id), files)
        if stored_documents:
            add_case_documents(int(req_id), stored_documents)

    documents_payloads = [
        {
            "doc_type": doc.get("document_type"),
            "raw_text": doc.get("raw_text") or "",
            "filename": doc.get("filename"),
        }
        for doc in stored_documents
    ]

    payload_data = {
        **payload_data,
        "documents": payload_data.get("documents") or [doc.get("filename") for doc in stored_documents],
        "documents_payloads": documents_payloads,
    }

    updated = resubmit_credit_request_db(int(req_id), int(user["user_id"]), payload_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Credit request not found")

    orchestration = run_orchestrator({**payload_data, "case_id": int(req_id), "user_id": user["user_id"]})
    save_orchestration(int(req_id), orchestration)
    if sync_credit_case_to_qdrant and background_tasks is not None:
        background_tasks.add_task(sync_credit_case_to_qdrant, int(req_id))
    banker_detail = fetch_case_detail(int(req_id))
    if banker_detail:
        _hydrate_documents(banker_detail)
        _prime_agent_sessions(banker_detail)

    detail = fetch_case_detail_for_client(int(req_id), user["user_id"])
    if not detail:
        raise HTTPException(status_code=500, detail="Failed to resubmit request")
    decision_info = _map_decision(detail.get("decision"))
    return CreditRequest(
        id=str(detail["case_id"]),
        status=_map_status(detail["status"], detail.get("decision")),
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        client_id=str(detail["user_id"]),
        summary=detail.get("summary") or "Dossier mis à jour",
        customer_explanation=None,
        agents=_map_agent_outputs(detail.get("agent_outputs", [])),
        decision=decision_info,
        comments=_map_comments(detail.get("comments", [])),
        auto_decision=detail.get("auto_decision"),
        auto_decision_confidence=float(detail["auto_decision_confidence"]) if detail.get("auto_decision_confidence") is not None else None,
        auto_review_required=detail.get("auto_review_required"),
        loan=_map_loan(detail.get("loan")),
        installments=_map_installments(detail.get("installments", [])),
        payments=_map_payments(detail.get("payments", [])),
        payment_behavior_summary=_map_payment_summary(detail.get("payment_behavior_summary")),
    )


@router.get("/client/credit-requests", response_model=list[CreditRequest])
def list_client_requests(user=Depends(get_current_user)):
    _require_role(user, "client")
    records = list_cases_for_client(user["user_id"])
    results: List[CreditRequest] = []
    for detail in records:
        if not detail:
            continue
        results.append(
            CreditRequest(
                id=str(detail["case_id"]),
                status=_map_status(detail["status"], detail.get("decision")),
                created_at=detail["created_at"],
                updated_at=detail["updated_at"],
                client_id=str(detail["user_id"]),
                summary=detail.get("summary"),
                customer_explanation=None,
                agents=_map_agent_outputs(detail.get("agent_outputs", [])),
                decision=_map_decision(detail.get("decision")),
                comments=_map_comments(detail.get("comments", [])),
                auto_decision=detail.get("auto_decision"),
                auto_decision_confidence=float(detail["auto_decision_confidence"]) if detail.get("auto_decision_confidence") is not None else None,
                auto_review_required=detail.get("auto_review_required"),
                loan=_map_loan(detail.get("loan")),
                installments=_map_installments(detail.get("installments", [])),
                payments=_map_payments(detail.get("payments", [])),
                payment_behavior_summary=_map_payment_summary(detail.get("payment_behavior_summary")),
            )
        )
    return results


# --- Banker -------------------------------------------------------------------
@router.get("/banker/credit-requests", response_model=list[BankerRequest])
def list_requests(status: Optional[str] = None, user: Dict = Depends(get_current_user)):
    _require_role(user, "banker")
    records = list_cases_for_banker(status_filter=status)
    results: List[BankerRequest] = []
    for detail in records:
        if not detail:
            continue
        results.append(_build_banker_request(detail))
    return results


@router.get("/banker/credit-requests/{req_id}", response_model=BankerRequest)
def get_request_detail(req_id: str, user: Dict = Depends(get_current_user)):
    _require_role(user, "banker")
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    _hydrate_documents(detail)
    return _build_banker_request(detail)


@router.get("/files/{case_id}/{filename}")
def download_file(case_id: int, filename: str, user: Dict = Depends(get_current_user)):
    case_detail: Optional[Dict[str, Any]] = None
    if user.get("role") == "client":
        case_detail = fetch_case_detail_for_client(case_id, user["user_id"])
    else:
        case_detail = fetch_case_detail(case_id)
    if not case_detail:
        raise HTTPException(status_code=404, detail="Case not found")

    safe_name = _safe_filename(filename)
    documents = case_detail.get("documents", [])
    match = None
    for doc in documents:
        path = str(doc.get("file_path") or "")
        if os.path.basename(path) == safe_name:
            match = path
            break
    if not match or not os.path.exists(match):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(match, filename=safe_name)


@router.post("/banker/credit-requests/{req_id}/documents", response_model=BankerRequest)
async def upload_case_documents(
    req_id: str,
    user: Dict = Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
    files: Optional[List[UploadFile]] = File(default=None),
):
    _require_role(user, "banker")
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")

    stored_documents = await _store_files(int(req_id), files)
    if not stored_documents:
        raise HTTPException(status_code=400, detail="No files uploaded")

    add_case_documents(int(req_id), stored_documents)

    refreshed = fetch_case_detail(int(req_id))
    if refreshed:
        _hydrate_documents(refreshed)
        orchestration = run_orchestrator(refreshed)
        save_orchestration(int(req_id), orchestration)
        if sync_credit_case_to_qdrant and background_tasks is not None:
            background_tasks.add_task(sync_credit_case_to_qdrant, int(req_id))
        _prime_agent_sessions(refreshed, int(user.get("user_id")))

    final_detail = fetch_case_detail(int(req_id))
    if not final_detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    return _build_banker_request(final_detail)


@router.post("/banker/credit-requests/{req_id}/decision")
def submit_decision(req_id: str, body: DecisionCreate, user: Dict = Depends(get_current_user), background_tasks: BackgroundTasks = None):
    _require_role(user, "banker")
    decision_row = upsert_decision(int(req_id), body.decision, body.note, int(user.get("user_id")))
    if not decision_row:
        raise HTTPException(status_code=404, detail="Credit request not found")
    if sync_credit_case_to_qdrant and background_tasks is not None:
        background_tasks.add_task(sync_credit_case_to_qdrant, int(req_id))
    return {"status": _map_status("DECIDED", decision_row), "note": body.note}


@router.post("/banker/credit-requests/{req_id}/payments", response_model=PaymentInfo)
def create_payment(req_id: str, body: PaymentCreate, user: Dict = Depends(get_current_user), background_tasks: BackgroundTasks = None):
    """
    Record a payment for the case's loan, update payment_behavior_summary, and sync to Qdrant.

    This is banker-only and best-effort; it is meant to simulate/record payments for the existing loan.
    """
    _require_role(user, "banker")
    payment = create_payment_for_case(
        int(req_id),
        payment_date=body.payment_date,
        amount=body.amount,
        channel=body.channel,
        status=body.status,
        installment_id=body.installment_id,
    )
    if not payment:
        raise HTTPException(status_code=400, detail="No loan found for this case (approve the case first)")
    if sync_credit_case_to_qdrant and background_tasks is not None:
        background_tasks.add_task(sync_credit_case_to_qdrant, int(req_id))
    return PaymentInfo(
        payment_id=int(payment["payment_id"]),
        loan_id=int(payment["loan_id"]),
        installment_id=payment.get("installment_id"),
        payment_date=payment["payment_date"],
        amount=float(payment["amount"]),
        channel=str(payment["channel"]),
        status=str(payment["status"]),
        is_reversal=bool(payment["is_reversal"]),
        reversal_of=payment.get("reversal_of"),
        created_at=payment["created_at"],
    )


@router.post("/banker/credit-requests/{req_id}/comments")
def add_comment(req_id: str, body: CommentCreate, user=Depends(get_current_user)):
    _require_role(user, "banker")
    comment = add_comment_db(int(req_id), int(user.get("user_id")), body.message, True)
    return comment


@router.get("/banker/credit-requests/{req_id}/agent-chat/{agent_name}", response_model=AgentChatResponse)
def get_agent_chat(req_id: str, agent_name: str, user: Dict = Depends(get_current_user)):
    _require_role(user, "banker")
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    agent_name = agent_name.lower().strip()
    banker_id = int(user.get("user_id"))
    session = get_agent_session(int(req_id), agent_name, banker_id)
    snapshot = session.get("snapshot_json") if session else _build_chat_snapshot(detail)
    messages_payload = (session.get("messages_json") or []) if session else []
    messages: List[AgentChatMessage] = [
        AgentChatMessage(**msg) for msg in messages_payload if isinstance(msg, dict)
    ]
    if not messages:
        initial_payload = build_initial_agent_reply(agent_name, snapshot)
        initial_message = AgentChatMessage(
            role="agent",
            content=str(initial_payload.get("summary", "")),
            structured_output=initial_payload.get("structured_output"),
            created_at=datetime.now(timezone.utc),
        )
        messages = [initial_message]
        upsert_agent_session(int(req_id), agent_name, snapshot, [m.model_dump() for m in messages], banker_id)
    return AgentChatResponse(agent_name=agent_name, messages=messages)


@router.post("/banker/credit-requests/{req_id}/agent-chat", response_model=AgentChatResponse)
def post_agent_chat(req_id: str, body: AgentChatRequest, user: Dict = Depends(get_current_user)):
    _require_role(user, "banker")
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")

    agent_name = body.agent_name.lower().strip()
    if not agent_name:
        raise HTTPException(status_code=400, detail="agent_name is required")

    banker_id = int(user.get("user_id"))
    session = get_agent_session(int(req_id), agent_name, banker_id)
    if session:
        messages_payload = session.get("messages_json") or []
        snapshot = session.get("snapshot_json") or _build_chat_snapshot(detail)
    else:
        messages_payload = []
        snapshot = _build_chat_snapshot(detail)

    messages: List[AgentChatMessage] = [
        AgentChatMessage(**msg) for msg in messages_payload if isinstance(msg, dict)
    ]
    if not messages:
        initial_payload = build_initial_agent_reply(agent_name, snapshot)
        initial_message = AgentChatMessage(
            role="agent",
            content=str(initial_payload.get("summary", "")),
            structured_output=initial_payload.get("structured_output"),
            created_at=datetime.now(timezone.utc),
        )
        messages.append(initial_message)
    user_message = AgentChatMessage(role="banker", content=body.message, created_at=datetime.now(timezone.utc))
    messages.append(user_message)

    history_payload = [m.model_dump() for m in messages]
    reply_payload = generate_agent_reply(agent_name, snapshot, history_payload)
    assistant_message = AgentChatMessage(
        role="agent",
        content=str(reply_payload.get("summary", "")),
        structured_output=reply_payload.get("structured_output"),
        created_at=datetime.now(timezone.utc),
    )
    messages.append(assistant_message)

    if len(messages) > 50:
        messages = messages[-50:]

    upsert_agent_session(int(req_id), agent_name, snapshot, [m.model_dump() for m in messages], banker_id)
    return AgentChatResponse(agent_name=agent_name, messages=messages)


@router.post("/banker/credit-requests/{req_id}/rerun")
def rerun_agents(req_id: str, user: Dict = Depends(get_current_user), background_tasks: BackgroundTasks = None):
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    result = run_orchestrator(detail)
    save_orchestration(int(req_id), result)
    if sync_credit_case_to_qdrant and background_tasks is not None:
        background_tasks.add_task(sync_credit_case_to_qdrant, int(req_id))
    refreshed = fetch_case_detail(int(req_id))
    if refreshed:
        _hydrate_documents(refreshed)
        _prime_agent_sessions(refreshed, int(user.get("user_id")))
    agents = result.get("agents") or None
    return {"status": "ok", "agents": agents}


@router.post("/banker/credit-requests/{req_id}/decision-suggestion")
def suggest_decision_note(req_id: str, body: Dict[str, Any] = None, _: Dict = Depends(get_current_user)):
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    snapshot = _build_chat_snapshot(detail)
    decision_mode = (body or {}).get("decision") or "reject"
    reply = generate_agent_reply("explanation", snapshot, [])
    raw = str(reply.get("summary") or "").strip()
    cleaned = " ".join(raw.split())

    def _fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "—"
        return f"{value}{suffix}"

    amount = _fmt(snapshot.get("amount") if isinstance(snapshot, dict) else None, " €")
    duration = _fmt(snapshot.get("duration_months") if isinstance(snapshot, dict) else None, " mois")
    income = _fmt(snapshot.get("monthly_income") if isinstance(snapshot, dict) else None, " €")
    charges = _fmt(snapshot.get("monthly_charges") if isinstance(snapshot, dict) else None, " €")
    employment = _fmt(snapshot.get("employment_type") if isinstance(snapshot, dict) else None)
    contract = _fmt(snapshot.get("contract_type") if isinstance(snapshot, dict) else None)
    seniority = _fmt(snapshot.get("seniority_years") if isinstance(snapshot, dict) else None, " ans")

    expl = snapshot.get("agents", {}).get("explanation", {}) if isinstance(snapshot, dict) else {}
    explanations = expl.get("explanations", {}) if isinstance(expl, dict) else {}
    internal = explanations.get("internal_explanation", {}) if isinstance(explanations, dict) else {}
    key_factors = internal.get("key_factors") or []
    supporting = internal.get("supporting_signals") or []

    parts: List[str] = []
    if decision_mode == "review":
        parts.append(
            "Votre dossier est en cours de revue. Merci de mettre à jour les informations ci-dessous ou d'ajouter les documents demandés pour poursuivre l'analyse."
        )
    if cleaned:
        parts.append(cleaned)
    parts.append(
        "Détails du dossier: revenu mensuel "
        + income
        + ", charges mensuelles "
        + charges
        + ", emploi "
        + employment
        + ", contrat "
        + contract
        + ", ancienneté "
        + seniority
        + ", montant demandé "
        + amount
        + " sur "
        + duration
        + "."
    )
    if isinstance(key_factors, list) and key_factors:
        parts.append("Facteurs clés: " + ", ".join([str(x) for x in key_factors]))
    if isinstance(supporting, list) and supporting:
        parts.append("Signaux de support: " + ", ".join([str(x) for x in supporting]))

    if not cleaned and decision_mode != "review":
        parts.insert(
            0,
            "Votre demande est en cours de revue. Des incohérences ou des éléments manquants doivent être clarifiés avant décision.",
        )
    if decision_mode == "review":
        parts.append(
            "Recommandations: vérifier le montant et la durée du prêt, confirmer les revenus et charges, et joindre les pièces manquantes (ex: justificatif de revenus, contrat, relevés bancaires)."
        )
    note = " ".join([p for p in parts if p])
    return {"note": note}
