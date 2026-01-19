from pydantic import BaseModel
from typing import List


class CreditRequest(BaseModel):
    case_id: str
    monthly_income: float
    requested_amount: float
    documents: List[str]
