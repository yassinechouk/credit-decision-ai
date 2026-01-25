import os
import json
import hashlib
from typing import Optional, Dict, Any, List

import psycopg2
from psycopg2.extras import RealDictCursor


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _get_db_params() -> Dict[str, str]:
    return {
        "host": os.getenv("DB_HOST", "postgres"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "credit"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }


def _connect():
    return psycopg2.connect(**_get_db_params())


def init_db() -> None:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    ALTER TABLE credit_cases
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE credit_cases
                    ADD COLUMN IF NOT EXISTS summary TEXT
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE credit_cases
                    ADD COLUMN IF NOT EXISTS auto_decision TEXT
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE credit_cases
                    ADD COLUMN IF NOT EXISTS auto_decision_confidence NUMERIC
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE credit_cases
                    ADD COLUMN IF NOT EXISTS auto_review_required BOOLEAN
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE decisions
                    ADD COLUMN IF NOT EXISTS note TEXT
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS comments (
                        comment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        case_id BIGINT NOT NULL REFERENCES credit_cases(case_id) ON DELETE CASCADE,
                        author_id BIGINT NOT NULL REFERENCES users(user_id),
                        message TEXT NOT NULL,
                        is_public BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        session_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        case_id BIGINT NOT NULL REFERENCES credit_cases(case_id) ON DELETE CASCADE,
                        agent_name TEXT NOT NULL,
                        snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        messages_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (case_id, agent_name)
                    )
                    """
                )
                cur.execute(
                    """
                    SELECT setval(
                        pg_get_serial_sequence('credit_cases', 'case_id'),
                        COALESCE((SELECT MAX(case_id) FROM credit_cases), 1),
                        (SELECT MAX(case_id) IS NOT NULL FROM credit_cases)
                    )
                    """
                )
    finally:
        conn.close()


def fetch_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_id, email, password_hash, role
                FROM users
                WHERE lower(email) = lower(%s)
                LIMIT 1
                """,
                (email,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def fetch_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_id, email, role
                FROM users
                WHERE user_id = %s
                """,
                (user_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def create_credit_request(
    user_id: int,
    payload: Dict[str, Any],
    orchestration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO credit_cases (user_id, status, loan_amount, loan_duration, summary)
                    VALUES (%s, 'SUBMITTED', %s, %s, %s)
                    RETURNING case_id, created_at, updated_at
                    """,
                    (
                        user_id,
                        payload.get("amount"),
                        payload.get("duration_months"),
                        (orchestration or {}).get("summary") if orchestration else None,
                    ),
                )
                case_row = cur.fetchone()
                case_id = case_row["case_id"]

                cur.execute(
                    """
                    INSERT INTO financial_profile (
                        case_id,
                        monthly_income,
                        other_income,
                        monthly_charges,
                        employment_type,
                        contract_type,
                        seniority_years,
                        marital_status,
                        number_of_children,
                        spouse_employed,
                        housing_status,
                        is_primary_holder
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        case_id,
                        payload.get("monthly_income"),
                        payload.get("other_income", 0),
                        payload.get("monthly_charges"),
                        payload.get("employment_type"),
                        _normalize_contract_type(payload.get("contract_type")),
                        payload.get("seniority_years"),
                        payload.get("family_status"),
                        payload.get("number_of_children", 0),
                        payload.get("spouse_employed"),
                        payload.get("housing_status"),
                        payload.get("is_primary_holder"),
                    ),
                )

                _insert_documents(cur, case_id, payload.get("documents") or [])

                if orchestration:
                    agents = (orchestration or {}).get("agents") or {}
                    for agent_name, output in agents.items():
                        if agent_name not in {"document", "similarity", "behavior", "fraud", "image"}:
                            continue
                        cur.execute(
                            """
                            INSERT INTO agent_outputs (case_id, agent_name, output_json)
                            VALUES (%s, %s, %s::jsonb)
                            """,
                            (case_id, agent_name, json.dumps(output)),
                        )

                return {
                    "case_id": case_id,
                    "created_at": case_row["created_at"],
                    "updated_at": case_row["updated_at"],
                }
    finally:
        conn.close()


def _normalize_contract_type(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if not normalized:
        return None
    mapping = {
        "cdi": "permanent",
        "permanent": "permanent",
        "permanentcontract": "permanent",
        "cdd": "temporary",
        "temporary": "temporary",
        "interim": "temporary",
        "interimaire": "temporary",
        "freelance": "none",
        "independent": "none",
        "contractor": "none",
        "consultant": "none",
        "none": "none",
    }
    return mapping.get(normalized, normalized)


def _insert_documents(cur, case_id: int, documents: List[Any]) -> None:
    for doc in documents or []:
        if isinstance(doc, dict):
            file_path = str(doc.get("file_path") or doc.get("path") or "")
            if not file_path:
                continue
            doc_type = str(doc.get("document_type") or doc.get("doc_type") or "uploaded")
            doc_hash = str(doc.get("file_hash") or "")
            if not doc_hash:
                doc_hash = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
        else:
            file_path = str(doc)
            doc_type = "uploaded"
            doc_hash = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
        cur.execute(
            """
            INSERT INTO documents (case_id, document_type, file_path, file_hash)
            VALUES (%s, %s, %s, %s)
            """,
            (case_id, doc_type, file_path, doc_hash),
        )


def add_case_documents(case_id: int, documents: List[Dict[str, Any]]) -> None:
    if not documents:
        return
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                _insert_documents(cur, case_id, documents)
                cur.execute(
                    """
                    UPDATE credit_cases
                    SET updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
    finally:
        conn.close()


def save_orchestration(case_id: int, orchestration: Dict[str, Any]) -> None:
    if not orchestration:
        return
    conn = _connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                summary = orchestration.get("summary")
                decision_payload = orchestration.get("decision") or {}
                auto_decision = decision_payload.get("decision") if isinstance(decision_payload, dict) else None
                auto_confidence = decision_payload.get("decision_confidence") if isinstance(decision_payload, dict) else None
                auto_review_required = decision_payload.get("human_review_required") if isinstance(decision_payload, dict) else None
                if summary is not None:
                    cur.execute(
                        """
                        UPDATE credit_cases
                        SET summary = %s,
                            auto_decision = %s,
                            auto_decision_confidence = %s,
                            auto_review_required = %s,
                            updated_at = NOW(),
                            status = 'UNDER_REVIEW'
                        WHERE case_id = %s
                        """,
                        (summary, auto_decision, auto_confidence, auto_review_required, case_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE credit_cases
                        SET auto_decision = %s,
                            auto_decision_confidence = %s,
                            auto_review_required = %s,
                            updated_at = NOW(),
                            status = 'UNDER_REVIEW'
                        WHERE case_id = %s
                        """,
                        (auto_decision, auto_confidence, auto_review_required, case_id),
                    )

                cur.execute(
                    """
                    DELETE FROM agent_outputs
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )

                agents = orchestration.get("agents") or {}
                for agent_name, output in agents.items():
                    if agent_name not in {"document", "similarity", "behavior", "fraud", "image"}:
                        continue
                    cur.execute(
                        """
                        INSERT INTO agent_outputs (case_id, agent_name, output_json)
                        VALUES (%s, %s, %s::jsonb)
                        """,
                        (case_id, agent_name, json.dumps(output)),
                    )
    finally:
        conn.close()


def get_agent_session(case_id: int, agent_name: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT session_id, case_id, agent_name, snapshot_json, messages_json, updated_at
                FROM agent_sessions
                WHERE case_id = %s AND agent_name = %s
                """,
                (case_id, agent_name),
            )
            return cur.fetchone()
    finally:
        conn.close()


def upsert_agent_session(case_id: int, agent_name: str, snapshot: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_sessions (case_id, agent_name, snapshot_json, messages_json)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (case_id, agent_name)
                    DO UPDATE SET
                        snapshot_json = EXCLUDED.snapshot_json,
                        messages_json = EXCLUDED.messages_json,
                        updated_at = NOW()
                    """,
                    (case_id, agent_name, _json_dumps(snapshot), _json_dumps(messages)),
                )
    finally:
        conn.close()


def ensure_agent_session_snapshot(case_id: int, agent_name: str, snapshot: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_sessions (case_id, agent_name, snapshot_json)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (case_id, agent_name)
                    DO UPDATE SET
                        snapshot_json = EXCLUDED.snapshot_json,
                        updated_at = NOW()
                    """,
                    (case_id, agent_name, _json_dumps(snapshot)),
                )
    finally:
        conn.close()


def list_cases_for_banker(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if status_filter == "pending":
                cur.execute(
                    """
                    SELECT case_id
                    FROM credit_cases
                    WHERE status IN ('SUBMITTED', 'UNDER_REVIEW')
                    ORDER BY created_at DESC
                    """
                )
            elif status_filter == "decided":
                cur.execute(
                    """
                    SELECT case_id
                    FROM credit_cases
                    WHERE status = 'DECIDED'
                    ORDER BY created_at DESC
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT case_id
                    FROM credit_cases
                    ORDER BY created_at DESC
                    """
                )
            case_ids = [row["case_id"] for row in cur.fetchall()]
        return [fetch_case_detail(case_id) for case_id in case_ids if case_id is not None]
    finally:
        conn.close()


def list_cases_for_client(user_id: int) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT case_id
                FROM credit_cases
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            case_ids = [row["case_id"] for row in cur.fetchall()]
        return [fetch_case_detail_for_client(case_id, user_id) for case_id in case_ids if case_id is not None]
    finally:
        conn.close()


def fetch_case_detail(case_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.case_id,
                    c.user_id,
                    c.status,
                    c.loan_amount,
                    c.loan_duration,
                    c.summary,
                    c.auto_decision,
                    c.auto_decision_confidence,
                    c.auto_review_required,
                    c.created_at,
                    c.updated_at,
                    f.monthly_income,
                    f.other_income,
                    f.monthly_charges,
                    f.employment_type,
                    f.contract_type,
                    f.seniority_years,
                    f.marital_status,
                    f.number_of_children,
                    f.spouse_employed,
                    f.housing_status,
                    f.is_primary_holder
                FROM credit_cases c
                JOIN financial_profile f ON f.case_id = c.case_id
                WHERE c.case_id = %s
                """,
                (case_id,),
            )
            base = cur.fetchone()
            if not base:
                return None

            cur.execute(
                """
                SELECT document_id, document_type, file_path, file_hash, uploaded_at
                FROM documents
                WHERE case_id = %s
                ORDER BY document_id ASC
                """,
                (case_id,),
            )
            documents = cur.fetchall()

            cur.execute(
                """
                SELECT agent_name, output_json, created_at
                FROM agent_outputs
                WHERE case_id = %s
                ORDER BY output_id ASC
                """,
                (case_id,),
            )
            agents = cur.fetchall()

            cur.execute(
                """
                SELECT decision, confidence, reason_codes, note, decided_by, decided_at
                FROM decisions
                WHERE case_id = %s
                """,
                (case_id,),
            )
            decision = cur.fetchone()

            cur.execute(
                """
                SELECT comment_id, author_id, message, is_public, created_at
                FROM comments
                WHERE case_id = %s
                ORDER BY created_at ASC
                """,
                (case_id,),
            )
            comments = cur.fetchall()

            base["documents"] = documents
            base["agent_outputs"] = agents
            base["decision"] = decision
            base["comments"] = comments
            return base
    finally:
        conn.close()


def fetch_case_detail_for_client(case_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT case_id
                FROM credit_cases
                WHERE case_id = %s AND user_id = %s
                """,
                (case_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                return None
        detail = fetch_case_detail(case_id)
        if detail:
            detail["comments"] = [c for c in detail.get("comments", []) if c.get("is_public")]
        return detail
    finally:
        conn.close()


def upsert_decision(case_id: int, decision: str, note: Optional[str], banker_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    decision_upper = decision.upper()
    status = "UNDER_REVIEW" if decision_upper == "REVIEW" else "DECIDED"
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT 1 FROM credit_cases WHERE case_id = %s
                    """,
                    (case_id,),
                )
                if not cur.fetchone():
                    return None
                cur.execute(
                    """
                    INSERT INTO decisions (case_id, decision, note, decided_by)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (case_id)
                    DO UPDATE SET
                        decision = EXCLUDED.decision,
                        note = EXCLUDED.note,
                        decided_by = EXCLUDED.decided_by,
                        decided_at = NOW()
                    RETURNING decision, confidence, reason_codes, note, decided_by, decided_at
                    """,
                    (case_id, decision_upper, note, banker_id),
                )
                decision_row = cur.fetchone()
                cur.execute(
                    """
                    UPDATE credit_cases
                    SET status = %s, updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (status, case_id),
                )
                return decision_row
    finally:
        conn.close()


def add_comment(case_id: int, author_id: int, message: str, is_public: bool = True) -> Dict[str, Any]:
    conn = _connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO comments (case_id, author_id, message, is_public)
                    VALUES (%s, %s, %s, %s)
                    RETURNING comment_id, author_id, message, is_public, created_at
                    """,
                    (case_id, author_id, message, is_public),
                )
                comment = cur.fetchone()
                cur.execute(
                    """
                    UPDATE credit_cases
                    SET updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
                return comment
    finally:
        conn.close()
