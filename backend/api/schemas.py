from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field


Role = Literal["client", "banker"]
Decision = Literal["approve", "reject", "review"]
Status = Literal["pending", "in_review", "approved", "rejected"]


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: Role
    user_id: str


class CreditRequestCreate(BaseModel):
    amount: float
    duration_months: int
    monthly_income: float
    monthly_charges: float
    employment_type: str
    contract_type: str
    seniority_years: int
    family_status: str
    documents: List[str] = Field(default_factory=list)

    other_income: Optional[float] = 0.0
    marital_status: Optional[str] = None
    number_of_children: Optional[int] = 0
    spouse_employed: Optional[bool] = None
    housing_status: Optional[str] = None
    is_primary_holder: Optional[bool] = True

    telemetry: Optional[Dict[str, Any]] = None
    documents_payloads: Optional[List[Dict[str, Any]]] = None
    document_texts: Optional[Dict[str, str]] = None
    transaction_flags: Optional[List[str]] = None
    image_flags: Optional[List[str]] = None
    free_text: Optional[List[str]] = None
    declared_profile: Optional[Dict[str, Any]] = None


class AgentResult(BaseModel):
    name: Optional[str] = None
    score: Optional[float] = None
    flags: Optional[List[str]] = None
    explanations: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None


class AgentBundle(BaseModel):
    document: Optional[AgentResult] = None
    similarity: Optional[AgentResult] = None
    fraud: Optional[AgentResult] = None
    explanation: Optional[AgentResult] = None
    behavior: Optional[AgentResult] = None
    image: Optional[AgentResult] = None


class DocumentInfo(BaseModel):
    document_id: int
    document_type: str
    file_path: str
    file_hash: str
    uploaded_at: datetime


class DecisionInfo(BaseModel):
    decision: Decision
    confidence: Optional[float] = None
    reason_codes: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None


class AgentChatMessage(BaseModel):
    role: Literal["banker", "agent"]
    content: str
    created_at: datetime
    structured_output: Optional[Dict[str, Any]] = None


class AgentChatRequest(BaseModel):
    agent_name: str
    message: str


class AgentChatResponse(BaseModel):
    agent_name: str
    messages: List[AgentChatMessage]


class DecisionCreate(BaseModel):
    decision: Decision
    note: Optional[str] = None


class CommentCreate(BaseModel):
    message: str


class Comment(BaseModel):
    author_id: str
    message: str
    created_at: datetime
    is_public: Optional[bool] = None


class CreditRequest(BaseModel):
    id: str
    status: Status
    created_at: datetime
    updated_at: datetime
    client_id: str
    summary: Optional[str] = None
    customer_explanation: Optional[str] = None
    agents: Optional[AgentBundle] = None
    decision: Optional[DecisionInfo] = None
    comments: List[Comment] = Field(default_factory=list)
    auto_decision: Optional[str] = None
    auto_decision_confidence: Optional[float] = None
    auto_review_required: Optional[bool] = None


class BankerRequest(BaseModel):
    id: str
    status: Status
    created_at: datetime
    updated_at: datetime
    client_id: str
    summary: Optional[str] = None
    amount: Optional[float] = None
    duration_months: Optional[int] = None
    monthly_income: Optional[float] = None
    other_income: Optional[float] = None
    monthly_charges: Optional[float] = None
    employment_type: Optional[str] = None
    contract_type: Optional[str] = None
    seniority_years: Optional[int] = None
    marital_status: Optional[str] = None
    number_of_children: Optional[int] = None
    spouse_employed: Optional[bool] = None
    housing_status: Optional[str] = None
    is_primary_holder: Optional[bool] = None
    documents: List[DocumentInfo] = Field(default_factory=list)
    agents: Optional[AgentBundle] = None
    comments: List[Comment] = Field(default_factory=list)
    decision: Optional[DecisionInfo] = None
    auto_decision: Optional[str] = None
    auto_decision_confidence: Optional[float] = None
    auto_review_required: Optional[bool] = None
