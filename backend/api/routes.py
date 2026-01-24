from datetime import datetime
from uuid import uuid4
from typing import Dict, Optional, List, Any

from fastapi import APIRouter, HTTPException, Depends

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
)


router = APIRouter(prefix="/api")


def _require_role(user: Dict[str, Any], role: str) -> None:
    if user.get("role") != role:
        raise HTTPException(status_code=403, detail="Forbidden")


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
        elif name == "behavior":
            bundle.behavior = result
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


# --- Auth ---------------------------------------------------------------------
@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    email = body.email.strip().lower()
    password = body.password.strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    user = fetch_user_by_email(email)
    if not user or user.get("password_hash") != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    role_value = user.get("role")
    if role_value not in ("CLIENT", "BANKER"):
        raise HTTPException(status_code=403, detail="User role not allowed")

    role = "banker" if role_value == "BANKER" else "client"
    user_id = int(user.get("user_id"))
    token = f"token:{role}:{user_id}:{uuid4()}"
    return LoginResponse(token=token, role=role, user_id=str(user_id))


# --- Client -------------------------------------------------------------------
@router.post("/client/credit-requests", response_model=CreditRequest)
def create_credit_request(body: CreditRequestCreate, user=Depends(get_current_user)):
    _require_role(user, "client")
    orchestration = run_orchestrator(body.model_dump())
    created = create_credit_request_db(user["user_id"], body.model_dump(), orchestration)
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
        results.append(
            BankerRequest(
                id=str(detail["case_id"]),
                status=_map_status(detail["status"], detail.get("decision")),
                created_at=detail["created_at"],
                updated_at=detail["updated_at"],
                client_id=str(detail["user_id"]),
                summary=detail.get("summary"),
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
            )
        )
    return results


@router.get("/banker/credit-requests/{req_id}", response_model=BankerRequest)
def get_request_detail(req_id: str, user: Dict = Depends(get_current_user)):
    _require_role(user, "banker")
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    return BankerRequest(
        id=str(detail["case_id"]),
        status=_map_status(detail["status"], detail.get("decision")),
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        client_id=str(detail["user_id"]),
        summary=detail.get("summary"),
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
    )


@router.post("/banker/credit-requests/{req_id}/decision")
def submit_decision(req_id: str, body: DecisionCreate, user: Dict = Depends(get_current_user)):
    _require_role(user, "banker")
    decision_row = upsert_decision(int(req_id), body.decision, body.note, int(user.get("user_id")))
    if not decision_row:
        raise HTTPException(status_code=404, detail="Credit request not found")
    return {"status": _map_status("DECIDED", decision_row), "note": body.note}


@router.post("/banker/credit-requests/{req_id}/comments")
def add_comment(req_id: str, body: CommentCreate, user=Depends(get_current_user)):
    _require_role(user, "banker")
    comment = add_comment_db(int(req_id), int(user.get("user_id")), body.message, True)
    return comment


@router.post("/banker/credit-requests/{req_id}/rerun")
def rerun_agents(req_id: str, _: Dict = Depends(get_current_user)):
    detail = fetch_case_detail(int(req_id))
    if not detail:
        raise HTTPException(status_code=404, detail="Credit request not found")
    result = run_orchestrator(detail)
    agents = result.get("agents") or None
    return {"status": "ok", "agents": agents}
