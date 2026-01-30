import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from agents.similarity_agent import (  # type: ignore
    QDRANT_CREDIT_CASE_PAYLOAD_SCHEMA,
    _normalize_payload_credit_case,
)


def test_qdrant_payload_schema_contains_expected_core_fields():
    # Core fields used by the current dataset + similarity formatting.
    for key in (
        "case_id",
        "loan_amount",
        "loan_duration",
        "monthly_income",
        "monthly_charges",
        "employment_type",
        "contract_type",
        "defaulted",
        "fraud_flag",
    ):
        assert key in QDRANT_CREDIT_CASE_PAYLOAD_SCHEMA


def test_normalize_payload_credit_case_casts_types_and_keeps_extra_fields():
    record = {
        "case_id": "1014",
        "loan_amount": "5000.0",
        "loan_duration": "24",
        "monthly_income": "3000",
        "other_income": "0",
        "monthly_charges": "800",
        "employment_type": 123,  # should become "123"
        "contract_type": "permanent",
        "seniority_years": "5",
        "marital_status": "single",
        "number_of_children": "0",
        "spouse_employed": "1",
        "housing_status": "owner",
        "is_primary_holder": "0",
        "defaulted": "0",
        "fraud_flag": "1",
        "some_new_field": {"x": 1},
    }

    payload = _normalize_payload_credit_case(record)

    assert payload["case_id"] == 1014
    assert payload["loan_amount"] == 5000.0
    assert payload["loan_duration"] == 24
    assert payload["monthly_income"] == 3000.0
    assert payload["monthly_charges"] == 800.0
    assert payload["employment_type"] == "123"
    assert payload["seniority_years"] == 5

    assert payload["defaulted"] is False
    assert payload["fraud_flag"] is True
    assert payload["spouse_employed"] is True
    assert payload["is_primary_holder"] is False

    # Extra fields remain untouched.
    assert payload["some_new_field"] == {"x": 1}
