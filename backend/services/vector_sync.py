"""
Postgres -> Qdrant synchronization for credit cases.

Goal:
- Keep Qdrant vectors + payload in sync with the source of truth (Postgres).
- Best-effort: failures must NOT break the business API flow.

This module is designed to work with the current database model and existing frontend forms
(no new form fields required).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import time
import threading

from qdrant_client import QdrantClient

from core.db import fetch_case_vector_sync


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "credit_dataset")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

try:
    QDRANT_TIMEOUT_SEC = float(os.getenv("QDRANT_TIMEOUT_SEC", "2.5"))
except ValueError:
    QDRANT_TIMEOUT_SEC = 2.5
try:
    QDRANT_RETRY_COUNT = int(os.getenv("QDRANT_RETRY_COUNT", "2"))
except ValueError:
    QDRANT_RETRY_COUNT = 2
try:
    EMBEDDING_TIMEOUT_SEC = float(os.getenv("EMBEDDING_TIMEOUT_SEC", "3.0"))
except ValueError:
    EMBEDDING_TIMEOUT_SEC = 3.0
try:
    EMBEDDING_RETRY_COUNT = int(os.getenv("EMBEDDING_RETRY_COUNT", "1"))
except ValueError:
    EMBEDDING_RETRY_COUNT = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embed_with_timeout(embedder, text: str, timeout_sec: float) -> Optional[list[float]]:
    result: Dict[str, Any] = {"value": None, "error": None}

    def _run() -> None:
        try:
            result["value"] = embedder.embed_query(text)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        return None
    if result.get("error") is not None:
        return None
    return result.get("value")


def _embed_with_retry(embedder, text: str, timeout_sec: float, retries: int) -> Optional[list[float]]:
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        vector = _embed_with_timeout(embedder, text, timeout_sec)
        if vector:
            return vector
        if attempt < attempts - 1:
            time.sleep(0.1 * (attempt + 1))
    return None


def _map_case_status(db_status: Optional[str], decision_value: Optional[str]) -> str:
    """
    Align with API/front statuses (pending/in_review/approved/rejected).
    Based on backend/api/routes.py::_map_status, but duplicated here to avoid circular imports.
    """
    decision_value = (decision_value or "").upper().strip() or None
    if decision_value:
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


def _build_profile_text(row: Dict[str, Any]) -> str:
    case_status = _map_case_status(row.get("status"), row.get("decision"))
    loan = row.get("loan") or {}

    return (
        "Credit case profile:\n"
        f"- case_id: {row.get('case_id')}\n"
        f"- user_id: {row.get('user_id')}\n"
        f"- case_status: {case_status}\n"
        f"- loan_amount: {row.get('loan_amount')}\n"
        f"- loan_duration: {row.get('loan_duration')}\n"
        f"- monthly_income: {row.get('monthly_income')}\n"
        f"- other_income: {row.get('other_income')}\n"
        f"- monthly_charges: {row.get('monthly_charges')}\n"
        f"- employment_type: {row.get('employment_type')}\n"
        f"- contract_type: {row.get('contract_type')}\n"
        f"- seniority_years: {row.get('seniority_years')}\n"
        f"- marital_status: {row.get('marital_status')}\n"
        f"- number_of_children: {row.get('number_of_children')}\n"
        f"- housing_status: {row.get('housing_status')}\n"
        f"- loan_status: {loan.get('status')}\n"
        f"- defaulted: {row.get('defaulted')}\n"
    )


def _build_payment_text(row: Dict[str, Any]) -> str:
    payment = row.get("payment_behavior_summary") or {}
    if not isinstance(payment, dict) or not payment:
        return ""
    return (
        "Payment behavior:\n"
        f"- on_time_rate: {payment.get('on_time_rate')}\n"
        f"- late_installments: {payment.get('late_installments')}\n"
        f"- missed_installments: {payment.get('missed_installments')}\n"
        f"- avg_days_late: {payment.get('avg_days_late')}\n"
        f"- max_days_late: {payment.get('max_days_late')}\n"
        f"- last_payment_date: {payment.get('last_payment_date')}\n"
    )


def _build_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payment = row.get("payment_behavior_summary") or {}
    loan = row.get("loan") or {}
    decision_value = row.get("decision")
    case_status = _map_case_status(row.get("status"), decision_value)

    defaulted = bool(row.get("defaulted"))

    updated_at = row.get("updated_at")
    if isinstance(updated_at, datetime):
        updated_at = updated_at.astimezone(timezone.utc).isoformat()

    last_payment_date = payment.get("last_payment_date")
    if isinstance(last_payment_date, datetime):
        last_payment_date = last_payment_date.astimezone(timezone.utc).isoformat()

    return {
        "case_id": int(row["case_id"]),
        "user_id": int(row["user_id"]),
        "case_status": case_status,
        "loan_amount": float(row.get("loan_amount") or 0),
        "loan_duration": int(row.get("loan_duration") or 0),
        "monthly_income": float(row.get("monthly_income") or 0),
        "other_income": float(row.get("other_income") or 0),
        "monthly_charges": float(row.get("monthly_charges") or 0),
        "employment_type": str(row.get("employment_type") or ""),
        "contract_type": str(row.get("contract_type") or ""),
        "seniority_years": int(row.get("seniority_years") or 0),
        "marital_status": str(row.get("marital_status") or ""),
        "number_of_children": int(row.get("number_of_children") or 0),
        "spouse_employed": row.get("spouse_employed"),
        "housing_status": str(row.get("housing_status") or ""),
        "is_primary_holder": row.get("is_primary_holder"),
        "defaulted": defaulted,
        "loan_status": loan.get("status"),
        "late_installments": int(payment.get("late_installments") or 0),
        "missed_installments": int(payment.get("missed_installments") or 0),
        "on_time_rate": float(payment.get("on_time_rate") or 0),
        "avg_days_late": float(payment.get("avg_days_late") or 0),
        "max_days_late": int(payment.get("max_days_late") or 0),
        "last_payment_date": last_payment_date,
        "updated_at": updated_at,
        "synced_at": _now_iso(),
    }


@dataclass
class _Deps:
    qdrant_client: Optional[QdrantClient]
    embedder: Any


_deps_singleton: Optional[_Deps] = None


def _get_deps() -> _Deps:
    global _deps_singleton
    if _deps_singleton is not None:
        return _deps_singleton

    qdrant_client: Optional[QdrantClient] = None
    if QDRANT_URL:
        try:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        except Exception:
            qdrant_client = None

    embedder = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception:
        embedder = None

    _deps_singleton = _Deps(qdrant_client=qdrant_client, embedder=embedder)
    return _deps_singleton


def _ensure_collection_best_effort(client: QdrantClient, vector_size: int) -> None:
    """
    Minimal collection ensure for sync. We keep it best-effort and non-destructive.
    SimilarityAgentAI also ensures collection at startup, so this is a safety net.
    """
    try:
        if client.collection_exists(QDRANT_COLLECTION_NAME):
            return
    except Exception:
        return
    try:
        from qdrant_client.http.models import Distance, HnswConfigDiff, VectorParams

        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "profile": VectorParams(size=vector_size, distance=Distance.COSINE),
                "payment": VectorParams(size=vector_size, distance=Distance.COSINE),
            },
            on_disk_payload=True,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )
    except Exception:
        return


def sync_credit_case_to_qdrant(case_id: int) -> bool:
    """
    Fetches the credit case from Postgres, generates an embedding, and upserts into Qdrant.

    Returns True on success, False on any failure (never raises).
    """
    deps = _get_deps()
    if not deps.qdrant_client or not deps.embedder:
        print(f"[WARN] Qdrant sync skipped (missing deps) for case_id={case_id}")
        return False

    row = fetch_case_vector_sync(int(case_id))
    if not row:
        print(f"[WARN] Qdrant sync skipped (case not found) for case_id={case_id}")
        return False

    profile_text = _build_profile_text(row)
    payment_text = _build_payment_text(row)
    profile_vector = _embed_with_retry(deps.embedder, profile_text, EMBEDDING_TIMEOUT_SEC, EMBEDDING_RETRY_COUNT)
    if not profile_vector:
        print(f"[WARN] Qdrant sync failed (embedding profile) for case_id={case_id}")
        return False

    vectors = {"profile": profile_vector}
    if payment_text:
        payment_vector = _embed_with_retry(deps.embedder, payment_text, EMBEDDING_TIMEOUT_SEC, EMBEDDING_RETRY_COUNT)
        if payment_vector:
            vectors["payment"] = payment_vector

    try:
        _ensure_collection_best_effort(deps.qdrant_client, len(profile_vector))
        payload = _build_payload(row)
        try:
            from qdrant_client.http.models import PointStruct

            points = [PointStruct(id=int(case_id), vector=vectors, payload=payload)]
        except Exception:
            points = [{"id": int(case_id), "vector": vectors, "payload": payload}]
        attempts = max(1, QDRANT_RETRY_COUNT + 1)
        for attempt in range(attempts):
            try:
                deps.qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=points,
                    timeout=QDRANT_TIMEOUT_SEC,
                )
                return True
            except TypeError:
                # Some client stubs or older clients don't accept timeout.
                try:
                    deps.qdrant_client.upsert(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points=points,
                    )
                    return True
                except Exception as exc:
                    if attempt < attempts - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    print(f"[WARN] Qdrant sync failed (upsert) for case_id={case_id}: {exc}")
                    return False
            except Exception as exc:
                if attempt < attempts - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"[WARN] Qdrant sync failed (upsert) for case_id={case_id}: {exc}")
                return False
    except Exception as exc:
        print(f"[WARN] Qdrant sync failed (upsert) for case_id={case_id}: {exc}")
        return False


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True, type=int)
    args = parser.parse_args()
    ok = sync_credit_case_to_qdrant(args.case_id)
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    _main()
