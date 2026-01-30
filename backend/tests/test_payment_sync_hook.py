import sys
from pathlib import Path
from datetime import date

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def test_create_payment_endpoint_schedules_qdrant_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.routes as routes  # type: ignore
    from api.schemas import PaymentCreate  # type: ignore

    scheduled = []

    class _BG:
        def add_task(self, fn, *args, **kwargs):
            scheduled.append((fn, args, kwargs))

    def _fake_create_payment_for_case(case_id: int, payment_date: date, amount: float, channel: str, status: str, installment_id=None):
        assert case_id == 1014
        return {
            "payment_id": 1,
            "loan_id": 10,
            "installment_id": installment_id,
            "payment_date": payment_date,
            "amount": amount,
            "channel": channel,
            "status": status,
            "is_reversal": False,
            "reversal_of": None,
            "created_at": "2026-01-29T00:00:00Z",
        }

    def _fake_sync(case_id: int) -> bool:
        return True

    monkeypatch.setattr(routes, "create_payment_for_case", _fake_create_payment_for_case)
    monkeypatch.setattr(routes, "sync_credit_case_to_qdrant", _fake_sync)

    body = PaymentCreate(payment_date=date(2026, 1, 29), amount=100.0, channel="bank_transfer", status="COMPLETED")
    bg = _BG()
    user = {"role": "banker", "user_id": 1}

    res = routes.create_payment("1014", body, user=user, background_tasks=bg)
    assert res.payment_id == 1
    assert len(scheduled) == 1
    assert scheduled[0][0] == _fake_sync
    assert scheduled[0][1] == (1014,)

