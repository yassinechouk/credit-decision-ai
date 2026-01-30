import sys
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


import services.vector_sync as vector_sync  # type: ignore


class _FakeEmbedder:
    def embed_query(self, _text: str):
        # Deterministic vector size.
        return [0.1, 0.2, 0.3]


class _FakeQdrant:
    def __init__(self):
        self.created = False
        self.upserts = []

    def collection_exists(self, _name: str) -> bool:
        return self.created

    def create_collection(self, **_kwargs):
        self.created = True
        return True

    def upsert(self, collection_name: str, points):
        self.upserts.append((collection_name, points))
        return True


def test_sync_credit_case_to_qdrant_success(monkeypatch: pytest.MonkeyPatch):
    fake_qdrant = _FakeQdrant()

    def _fake_get_deps():
        return vector_sync._Deps(qdrant_client=fake_qdrant, embedder=_FakeEmbedder())

    monkeypatch.setattr(vector_sync, "_get_deps", _fake_get_deps)

    def _fake_fetch(case_id: int):
        assert case_id == 123
        return {
            "case_id": 123,
            "user_id": 77,
            "status": "SUBMITTED",
            "decision": None,
            "loan_amount": 5000,
            "loan_duration": 24,
            "updated_at": "2026-01-29T00:00:00Z",
            "monthly_income": 3000,
            "other_income": 0,
            "monthly_charges": 800,
            "employment_type": "employee",
            "contract_type": "permanent",
            "seniority_years": 5,
            "marital_status": "single",
            "number_of_children": 0,
            "spouse_employed": True,
            "housing_status": "owner",
            "is_primary_holder": True,
            "defaulted": False,
            "loan": {"status": "ACTIVE"},
            "payment_behavior_summary": {
                "on_time_rate": 0.9,
                "late_installments": 1,
                "missed_installments": 0,
                "avg_days_late": 2.5,
                "max_days_late": 7,
                "last_payment_date": "2026-01-28",
            },
        }

    monkeypatch.setattr(vector_sync, "fetch_case_vector_sync", _fake_fetch)

    ok = vector_sync.sync_credit_case_to_qdrant(123)
    assert ok is True
    assert len(fake_qdrant.upserts) == 1

    collection_name, points = fake_qdrant.upserts[0]
    assert collection_name == vector_sync.QDRANT_COLLECTION_NAME
    # We don't assert PointStruct type here; we only validate payload content shape and vector names.
    if hasattr(points[0], "payload"):
        payload = points[0].payload
        vectors = points[0].vector
    else:
        payload = points[0]["payload"]
        vectors = points[0]["vector"]
    assert payload["case_id"] == 123
    assert payload["user_id"] == 77
    assert payload["case_status"] == "pending"
    assert "profile" in vectors


def test_sync_credit_case_to_qdrant_non_blocking_on_missing_deps(monkeypatch: pytest.MonkeyPatch):
    def _fake_get_deps():
        return vector_sync._Deps(qdrant_client=None, embedder=None)

    monkeypatch.setattr(vector_sync, "_get_deps", _fake_get_deps)
    ok = vector_sync.sync_credit_case_to_qdrant(999)
    assert ok is False
