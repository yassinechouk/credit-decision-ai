from fastapi import APIRouter
from api.schemas import CreditRequest
from core.orchestrator import run_orchestrator

router = APIRouter()


@router.post("/credit/decision")
def credit_decision(request: CreditRequest):
    return run_orchestrator(request)
