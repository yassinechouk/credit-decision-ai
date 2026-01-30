import os
import json
import hashlib
import time
from datetime import date
from typing import Optional, Dict, Any, List, Tuple

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


def _host_candidates(primary: str) -> List[str]:
    candidates: List[str] = []
    if primary:
        candidates.append(primary)
    if primary != "postgres":
        candidates.append("postgres")
    candidates.extend(["host.docker.internal", "localhost", "127.0.0.1"])
    seen = set()
    deduped: List[str] = []
    for host in candidates:
        if host and host not in seen:
            deduped.append(host)
            seen.add(host)
    return deduped


def _connect():
    params = _get_db_params()
    retries = int(os.getenv("DB_CONNECT_RETRIES", "5"))
    delay = float(os.getenv("DB_CONNECT_DELAY", "1.0"))
    last_exc: Optional[Exception] = None
    for host in _host_candidates(params.get("host") or "postgres"):
        params["host"] = host
        for attempt in range(retries):
            try:
                return psycopg2.connect(**params)
            except psycopg2.OperationalError as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = (
                    "could not translate host name" in msg
                    or "name or service not known" in msg
                    or "temporary failure in name resolution" in msg
                    or "could not connect to server" in msg
                    or "connection refused" in msg
                )
                if not transient or attempt == retries - 1:
                    break
                time.sleep(delay)
        # try next host candidate
    if last_exc:
        raise last_exc
    raise psycopg2.OperationalError("Unable to connect to database")


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
                        banker_id BIGINT,
                        snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        messages_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (case_id, agent_name, banker_id)
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE agent_sessions
                    ADD COLUMN IF NOT EXISTS banker_id BIGINT
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE agent_sessions
                    DROP CONSTRAINT IF EXISTS agent_sessions_case_id_agent_name_key
                    """
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint
                            WHERE conname = 'agent_sessions_case_id_agent_name_banker_id_key'
                        ) THEN
                            ALTER TABLE agent_sessions
                            ADD CONSTRAINT agent_sessions_case_id_agent_name_banker_id_key
                            UNIQUE (case_id, agent_name, banker_id);
                        END IF;
                    END$$;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS loans (
                        loan_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        case_id BIGINT UNIQUE REFERENCES credit_cases(case_id) ON DELETE SET NULL,
                        principal_amount NUMERIC(14,2) NOT NULL CHECK (principal_amount > 0),
                        interest_rate NUMERIC(5,4) NOT NULL CHECK (interest_rate >= 0),
                        term_months INTEGER NOT NULL CHECK (term_months > 0),
                        status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'CLOSED', 'DEFAULTED', 'CANCELLED')),
                        approved_at TIMESTAMPTZ,
                        start_date DATE,
                        end_date DATE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS installments (
                        installment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        loan_id BIGINT NOT NULL REFERENCES loans(loan_id) ON DELETE CASCADE,
                        installment_number INTEGER NOT NULL CHECK (installment_number > 0),
                        due_date DATE NOT NULL,
                        amount_due NUMERIC(14,2) NOT NULL CHECK (amount_due >= 0),
                        status TEXT NOT NULL CHECK (status IN ('PENDING', 'PAID', 'LATE', 'MISSED')),
                        amount_paid NUMERIC(14,2) NOT NULL DEFAULT 0 CHECK (amount_paid >= 0),
                        paid_at DATE,
                        days_late INTEGER,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (loan_id, installment_number)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS payments (
                        payment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        loan_id BIGINT NOT NULL REFERENCES loans(loan_id) ON DELETE CASCADE,
                        installment_id BIGINT REFERENCES installments(installment_id) ON DELETE SET NULL,
                        payment_date DATE NOT NULL,
                        amount NUMERIC(14,2) NOT NULL,
                        channel TEXT NOT NULL CHECK (channel IN ('bank_transfer', 'card', 'cash', 'direct_debit', 'mobile')),
                        status TEXT NOT NULL CHECK (status IN ('COMPLETED', 'PENDING', 'FAILED', 'REVERSED')),
                        is_reversal BOOLEAN NOT NULL DEFAULT FALSE,
                        reversal_of BIGINT REFERENCES payments(payment_id) ON DELETE SET NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS payment_behavior_summary (
                        summary_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        user_id BIGINT NOT NULL UNIQUE REFERENCES users(user_id) ON DELETE CASCADE,
                        total_loans INTEGER NOT NULL DEFAULT 0,
                        total_installments INTEGER NOT NULL DEFAULT 0,
                        on_time_installments INTEGER NOT NULL DEFAULT 0,
                        late_installments INTEGER NOT NULL DEFAULT 0,
                        missed_installments INTEGER NOT NULL DEFAULT 0,
                        on_time_rate NUMERIC(5,4) NOT NULL DEFAULT 0,
                        avg_days_late NUMERIC(6,2) NOT NULL DEFAULT 0,
                        max_days_late INTEGER NOT NULL DEFAULT 0,
                        avg_payment_amount NUMERIC(14,2) NOT NULL DEFAULT 0,
                        last_payment_date DATE,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_loans_user_id ON loans(user_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_installments_loan_id ON installments(loan_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_payments_loan_id ON payments(loan_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_payments_installment_id ON payments(installment_id)
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
                        if agent_name not in {"document", "similarity", "behavior", "fraud", "image", "decision", "explanation"}:
                            continue
                        cur.execute(
                            """
                            INSERT INTO agent_outputs (case_id, agent_name, output_json)
                            VALUES (%s, %s, %s::jsonb)
                            """,
                            (case_id, agent_name, _json_dumps(output)),
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


def resubmit_credit_request_db(case_id: int, user_id: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT case_id
                    FROM credit_cases
                    WHERE case_id = %s AND user_id = %s
                    """,
                    (case_id, user_id),
                )
                if not cur.fetchone():
                    return None

                cur.execute(
                    """
                    UPDATE credit_cases
                    SET loan_amount = %s,
                        loan_duration = %s,
                        summary = NULL,
                        auto_decision = NULL,
                        auto_decision_confidence = NULL,
                        auto_review_required = NULL,
                        status = 'SUBMITTED',
                        updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (
                        payload.get("amount"),
                        payload.get("duration_months"),
                        case_id,
                    ),
                )

                cur.execute(
                    """
                    UPDATE financial_profile
                    SET monthly_income = %s,
                        other_income = %s,
                        monthly_charges = %s,
                        employment_type = %s,
                        contract_type = %s,
                        seniority_years = %s,
                        marital_status = %s,
                        number_of_children = %s,
                        spouse_employed = %s,
                        housing_status = %s,
                        is_primary_holder = %s
                    WHERE case_id = %s
                    """,
                    (
                        payload.get("monthly_income"),
                        payload.get("other_income", 0),
                        payload.get("monthly_charges"),
                        payload.get("employment_type"),
                        _normalize_contract_type(payload.get("contract_type")),
                        payload.get("seniority_years"),
                        payload.get("family_status") or payload.get("marital_status"),
                        payload.get("number_of_children", 0),
                        payload.get("spouse_employed"),
                        payload.get("housing_status"),
                        payload.get("is_primary_holder"),
                        case_id,
                    ),
                )

                cur.execute(
                    """
                    DELETE FROM decisions
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
                cur.execute(
                    """
                    DELETE FROM agent_outputs
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
                cur.execute(
                    """
                    DELETE FROM agent_sessions
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )

                documents = payload.get("documents") or []
                if documents:
                    _insert_documents(cur, case_id, documents)

                return {
                    "case_id": case_id,
                }
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
                    if agent_name not in {"document", "similarity", "behavior", "fraud", "image", "decision", "explanation"}:
                        continue
                    cur.execute(
                        """
                        INSERT INTO agent_outputs (case_id, agent_name, output_json)
                        VALUES (%s, %s, %s::jsonb)
                        """,
                        (case_id, agent_name, _json_dumps(output)),
                    )
    finally:
        conn.close()


def get_agent_session(case_id: int, agent_name: str, banker_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT session_id, case_id, agent_name, banker_id, snapshot_json, messages_json, updated_at
                FROM agent_sessions
                WHERE case_id = %s AND agent_name = %s AND banker_id = %s
                """,
                (case_id, agent_name, banker_id),
            )
            return cur.fetchone()
    finally:
        conn.close()


def upsert_agent_session(
    case_id: int,
    agent_name: str,
    snapshot: Dict[str, Any],
    messages: List[Dict[str, Any]],
    banker_id: int,
) -> None:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_sessions (case_id, agent_name, banker_id, snapshot_json, messages_json)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (case_id, agent_name, banker_id)
                    DO UPDATE SET
                        snapshot_json = EXCLUDED.snapshot_json,
                        messages_json = EXCLUDED.messages_json,
                        updated_at = NOW()
                    """,
                    (case_id, agent_name, banker_id, _json_dumps(snapshot), _json_dumps(messages)),
                )
    finally:
        conn.close()


def ensure_agent_session_snapshot(case_id: int, agent_name: str, snapshot: Dict[str, Any], banker_id: int) -> None:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_sessions (case_id, agent_name, banker_id, snapshot_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (case_id, agent_name, banker_id)
                    DO UPDATE SET
                        snapshot_json = EXCLUDED.snapshot_json,
                        updated_at = NOW()
                    """,
                    (case_id, agent_name, banker_id, _json_dumps(snapshot)),
                )
    finally:
        conn.close()


def clear_agent_sessions_for_banker(banker_id: int) -> int:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM agent_sessions
                    WHERE banker_id = %s
                    """,
                    (banker_id,),
                )
                return cur.rowcount
    finally:
        conn.close()


def clear_agent_sessions() -> int:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM agent_sessions
                    """
                )
                return cur.rowcount
    finally:
        conn.close()


def list_cases_for_banker(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if status_filter == "pending":
                where_clause = "WHERE c.status IN ('SUBMITTED', 'UNDER_REVIEW')"
                params: tuple = ()
            elif status_filter == "decided":
                where_clause = "WHERE c.status = 'DECIDED'"
                params = ()
            else:
                where_clause = ""
                params = ()

            cur.execute(
                f"""
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
                    f.is_primary_holder,
                    d.decision AS decision_value,
                    d.confidence AS decision_confidence,
                    d.reason_codes AS decision_reason_codes,
                    d.note AS decision_note,
                    d.decided_by AS decision_decided_by,
                    d.decided_at AS decision_decided_at
                FROM credit_cases c
                JOIN financial_profile f ON f.case_id = c.case_id
                LEFT JOIN decisions d ON d.case_id = c.case_id
                {where_clause}
                ORDER BY c.created_at DESC
                """,
                params,
            )
            rows = cur.fetchall()

            results: List[Dict[str, Any]] = []
            for row in rows:
                if row.get("decision_value") is not None:
                    row["decision"] = {
                        "decision": row.pop("decision_value"),
                        "confidence": row.pop("decision_confidence"),
                        "reason_codes": row.pop("decision_reason_codes"),
                        "note": row.pop("decision_note"),
                        "decided_by": row.pop("decision_decided_by"),
                        "decided_at": row.pop("decision_decided_at"),
                    }
                else:
                    row.pop("decision_value", None)
                    row.pop("decision_confidence", None)
                    row.pop("decision_reason_codes", None)
                    row.pop("decision_note", None)
                    row.pop("decision_decided_by", None)
                    row.pop("decision_decided_at", None)
                    row["decision"] = None

                row["documents"] = []
                row["agent_outputs"] = []
                row["comments"] = []
                results.append(row)

            return results
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
        return [fetch_case_overview_for_client(case_id, user_id) for case_id in case_ids if case_id is not None]
    finally:
        conn.close()


def fetch_case_overview(case_id: int) -> Optional[Dict[str, Any]]:
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
                SELECT decision, confidence, reason_codes, note, decided_by, decided_at
                FROM decisions
                WHERE case_id = %s
                """,
                (case_id,),
            )
            decision = cur.fetchone()

            base["documents"] = []
            base["agent_outputs"] = []
            base["comments"] = []
            base["decision"] = decision
            return base
    finally:
        conn.close()


def fetch_case_overview_for_client(case_id: int, user_id: int) -> Optional[Dict[str, Any]]:
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
        detail = fetch_case_overview(case_id)
        return detail
    finally:
        conn.close()


def fetch_case_detail(case_id: int, include_payments: bool = True) -> Optional[Dict[str, Any]]:
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

            loan = None
            installments: List[Dict[str, Any]] = []
            payments: List[Dict[str, Any]] = []
            payment_summary = None
            if include_payments:
                cur.execute(
                    """
                    SELECT loan_id, user_id, case_id, principal_amount, interest_rate, term_months,
                           status, approved_at, start_date, end_date, created_at
                    FROM loans
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
                loan = cur.fetchone()

                if loan and loan.get("loan_id") is not None:
                    loan_id = int(loan["loan_id"])
                    cur.execute(
                        """
                        SELECT installment_id, loan_id, installment_number, due_date, amount_due,
                               status, amount_paid, paid_at, days_late, created_at
                        FROM installments
                        WHERE loan_id = %s
                        ORDER BY installment_number ASC
                        """,
                        (loan_id,),
                    )
                    installments = cur.fetchall()

                    cur.execute(
                        """
                        SELECT payment_id, loan_id, installment_id, payment_date, amount, channel,
                               status, is_reversal, reversal_of, created_at
                        FROM payments
                        WHERE loan_id = %s
                        ORDER BY payment_date ASC, payment_id ASC
                        """,
                        (loan_id,),
                    )
                    payments = cur.fetchall()

                cur.execute(
                    """
                    SELECT summary_id, user_id, total_loans, total_installments, on_time_installments,
                           late_installments, missed_installments, on_time_rate, avg_days_late,
                           max_days_late, avg_payment_amount, last_payment_date, updated_at
                    FROM payment_behavior_summary
                    WHERE user_id = %s
                    """,
                    (base["user_id"],),
                )
                payment_summary = cur.fetchone()

            base["documents"] = documents
            base["agent_outputs"] = agents
            base["decision"] = decision
            base["comments"] = comments
            if include_payments:
                base["loan"] = loan
                base["installments"] = installments
                base["payments"] = payments
                base["payment_behavior_summary"] = payment_summary
            return base
    finally:
        conn.close()


def fetch_case_vector_sync(case_id: int) -> Optional[Dict[str, Any]]:
    """
    Lightweight fetch for Postgres -> Qdrant sync.

    Returns only the fields needed to build payload + embedding text.
    """
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
                SELECT decision
                FROM decisions
                WHERE case_id = %s
                """,
                (case_id,),
            )
            decision = cur.fetchone()
            base["decision"] = (decision or {}).get("decision")

            cur.execute(
                """
                SELECT loan_id, status
                FROM loans
                WHERE case_id = %s
                """,
                (case_id,),
            )
            loan = cur.fetchone()
            base["loan"] = loan

            cur.execute(
                """
                SELECT summary_id, user_id, total_installments, on_time_installments,
                       late_installments, missed_installments, on_time_rate, avg_days_late,
                       max_days_late, last_payment_date, updated_at
                FROM payment_behavior_summary
                WHERE user_id = %s
                """,
                (base["user_id"],),
            )
            base["payment_behavior_summary"] = cur.fetchone()

            # Derive defaulted from loan status if available.
            loan_status = (loan or {}).get("status")
            base["defaulted"] = str(loan_status or "").upper() == "DEFAULTED"

            return base
    finally:
        conn.close()


def fetch_case_detail_for_client(case_id: int, user_id: int, include_payments: bool = True) -> Optional[Dict[str, Any]]:
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
        detail = fetch_case_detail(case_id, include_payments=include_payments)
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
                    SELECT user_id FROM credit_cases WHERE case_id = %s
                    """,
                    (case_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                user_id = int(row.get("user_id"))
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
                if decision_upper == "APPROVE":
                    _ensure_loan_for_case(cur, case_id)
                    # Loan/instalments creation impacts payment context: keep summary synced.
                    _recompute_payment_behavior_summary_tx(cur, user_id)
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


def _add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    # February
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return 29 if is_leap else 28


def _ensure_loan_for_case(cur, case_id: int) -> Optional[int]:
    cur.execute(
        """
        SELECT loan_id FROM loans WHERE case_id = %s
        """,
        (case_id,),
    )
    row = cur.fetchone()
    if row:
        return int(row["loan_id"]) if isinstance(row, dict) else int(row[0])

    cur.execute(
        """
        SELECT user_id, loan_amount, loan_duration
        FROM credit_cases
        WHERE case_id = %s
        """,
        (case_id,),
    )
    case_row = cur.fetchone()
    if not case_row:
        return None
    user_id = case_row["user_id"]
    principal = float(case_row["loan_amount"])
    term_months = int(case_row["loan_duration"])
    interest_rate = 0.05

    today = date.today()
    end_date = _add_months(today, term_months)

    cur.execute(
        """
        INSERT INTO loans (
            user_id,
            case_id,
            principal_amount,
            interest_rate,
            term_months,
            status,
            approved_at,
            start_date,
            end_date
        )
        VALUES (%s, %s, %s, %s, %s, 'ACTIVE', NOW(), %s, %s)
        RETURNING loan_id
        """,
        (user_id, case_id, principal, interest_rate, term_months, today, end_date),
    )
    loan_id = cur.fetchone()["loan_id"]

    total_with_interest = principal * (1 + interest_rate * (term_months / 12.0))
    monthly_amount = round(total_with_interest / term_months, 2)
    for n in range(1, term_months + 1):
        due_date = _add_months(today, n)
        cur.execute(
            """
            INSERT INTO installments (
                loan_id, installment_number, due_date, amount_due, status
            )
            VALUES (%s, %s, %s, %s, 'PENDING')
            """,
            (loan_id, n, due_date, monthly_amount),
        )

    return int(loan_id)


def fetch_loan_by_case(case_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT loan_id, user_id, case_id, principal_amount, interest_rate, term_months,
                       status, approved_at, start_date, end_date, created_at
                FROM loans
                WHERE case_id = %s
                """,
                (case_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def fetch_installments_by_loan(loan_id: int) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT installment_id, loan_id, installment_number, due_date, amount_due,
                       status, amount_paid, paid_at, days_late, created_at
                FROM installments
                WHERE loan_id = %s
                ORDER BY installment_number ASC
                """,
                (loan_id,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def fetch_payments_by_loan(loan_id: int) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT payment_id, loan_id, installment_id, payment_date, amount, channel,
                       status, is_reversal, reversal_of, created_at
                FROM payments
                WHERE loan_id = %s
                ORDER BY payment_date ASC, payment_id ASC
                """,
                (loan_id,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def fetch_payment_behavior_summary(user_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT summary_id, user_id, total_loans, total_installments, on_time_installments,
                       late_installments, missed_installments, on_time_rate, avg_days_late,
                       max_days_late, avg_payment_amount, last_payment_date, updated_at
                FROM payment_behavior_summary
                WHERE user_id = %s
                """,
                (user_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def fetch_payment_context(user_id: int, case_id: Optional[int] = None) -> Dict[str, Any]:
    loan = None
    installments: List[Dict[str, Any]] = []
    payments: List[Dict[str, Any]] = []
    if case_id is not None:
        loan = fetch_loan_by_case(case_id)
        if loan and loan.get("loan_id") is not None:
            loan_id = int(loan["loan_id"])
            installments = fetch_installments_by_loan(loan_id)
            payments = fetch_payments_by_loan(loan_id)
    summary = fetch_payment_behavior_summary(user_id)
    return {
        "loan": loan,
        "installments": installments,
        "payments": payments,
        "payment_behavior_summary": summary,
    }


def _recompute_payment_behavior_summary_tx(cur, user_id: int) -> Optional[Dict[str, Any]]:
    """
    Recompute and upsert payment_behavior_summary inside an existing transaction.

    This keeps "late/missed/on_time" derived from installments and payments in sync.
    """
    cur.execute("SELECT COUNT(*) AS total_loans FROM loans WHERE user_id = %s", (user_id,))
    total_loans = int((cur.fetchone() or {}).get("total_loans") or 0)

    cur.execute(
        """
        SELECT
            COUNT(*) AS total_installments,
            SUM(CASE WHEN i.status = 'PAID' AND COALESCE(i.days_late, 0) <= 0 THEN 1 ELSE 0 END) AS on_time_installments,
            SUM(CASE WHEN i.status = 'MISSED' THEN 1 ELSE 0 END) AS missed_installments,
            SUM(CASE WHEN (i.status = 'LATE') OR (i.status = 'PAID' AND COALESCE(i.days_late, 0) > 0) THEN 1 ELSE 0 END) AS late_installments,
            AVG(CASE WHEN COALESCE(i.days_late, 0) > 0 THEN i.days_late END) AS avg_days_late,
            MAX(COALESCE(i.days_late, 0)) AS max_days_late
        FROM installments i
        JOIN loans l ON l.loan_id = i.loan_id
        WHERE l.user_id = %s
        """,
        (user_id,),
    )
    inst = cur.fetchone() or {}
    total_installments = int(inst.get("total_installments") or 0)
    on_time_installments = int(inst.get("on_time_installments") or 0)
    late_installments = int(inst.get("late_installments") or 0)
    missed_installments = int(inst.get("missed_installments") or 0)
    avg_days_late = float(inst.get("avg_days_late") or 0)
    max_days_late = int(inst.get("max_days_late") or 0)
    on_time_rate = float(on_time_installments / total_installments) if total_installments > 0 else 0.0

    cur.execute(
        """
        SELECT
            AVG(p.amount) AS avg_payment_amount,
            MAX(p.payment_date) AS last_payment_date
        FROM payments p
        JOIN loans l ON l.loan_id = p.loan_id
        WHERE l.user_id = %s
          AND p.status = 'COMPLETED'
          AND p.is_reversal = FALSE
        """,
        (user_id,),
    )
    pay = cur.fetchone() or {}
    avg_payment_amount = float(pay.get("avg_payment_amount") or 0)
    last_payment_date = pay.get("last_payment_date")

    cur.execute(
        """
        INSERT INTO payment_behavior_summary (
            user_id,
            total_loans,
            total_installments,
            on_time_installments,
            late_installments,
            missed_installments,
            on_time_rate,
            avg_days_late,
            max_days_late,
            avg_payment_amount,
            last_payment_date,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE SET
            total_loans = EXCLUDED.total_loans,
            total_installments = EXCLUDED.total_installments,
            on_time_installments = EXCLUDED.on_time_installments,
            late_installments = EXCLUDED.late_installments,
            missed_installments = EXCLUDED.missed_installments,
            on_time_rate = EXCLUDED.on_time_rate,
            avg_days_late = EXCLUDED.avg_days_late,
            max_days_late = EXCLUDED.max_days_late,
            avg_payment_amount = EXCLUDED.avg_payment_amount,
            last_payment_date = EXCLUDED.last_payment_date,
            updated_at = NOW()
        RETURNING summary_id, user_id, total_loans, total_installments, on_time_installments,
                  late_installments, missed_installments, on_time_rate, avg_days_late,
                  max_days_late, avg_payment_amount, last_payment_date, updated_at
        """,
        (
            user_id,
            total_loans,
            total_installments,
            on_time_installments,
            late_installments,
            missed_installments,
            on_time_rate,
            avg_days_late,
            max_days_late,
            avg_payment_amount,
            last_payment_date,
        ),
    )
    return cur.fetchone()


def recompute_payment_behavior_summary(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Public wrapper to recompute payment_behavior_summary for a user.
    """
    conn = _connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _recompute_payment_behavior_summary_tx(cur, user_id)
    finally:
        conn.close()


def create_payment_for_case(
    case_id: int,
    payment_date: date,
    amount: float,
    channel: str,
    status: str,
    installment_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Insert a payment for a case's loan and recompute payment_behavior_summary.

    This is intentionally simple: it updates one installment when provided (or the next due),
    then recomputes the user's aggregated payment summary.
    """
    conn = _connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT loan_id, user_id
                    FROM loans
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
                loan = cur.fetchone()
                if not loan:
                    return None
                loan_id = int(loan["loan_id"])
                user_id = int(loan["user_id"])

                # Pick an installment if not specified.
                if installment_id is None:
                    cur.execute(
                        """
                        SELECT installment_id
                        FROM installments
                        WHERE loan_id = %s AND status IN ('PENDING', 'LATE', 'MISSED')
                        ORDER BY installment_number ASC
                        LIMIT 1
                        """,
                        (loan_id,),
                    )
                    row = cur.fetchone()
                    installment_id = int(row["installment_id"]) if row and row.get("installment_id") is not None else None

                # Validate installment belongs to the loan.
                installment_row = None
                if installment_id is not None:
                    cur.execute(
                        """
                        SELECT installment_id, due_date, amount_due, amount_paid
                        FROM installments
                        WHERE installment_id = %s AND loan_id = %s
                        FOR UPDATE
                        """,
                        (installment_id, loan_id),
                    )
                    installment_row = cur.fetchone()
                    if not installment_row:
                        installment_id = None

                cur.execute(
                    """
                    INSERT INTO payments (
                        loan_id, installment_id, payment_date, amount, channel, status, is_reversal
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, FALSE)
                    RETURNING payment_id, loan_id, installment_id, payment_date, amount, channel,
                              status, is_reversal, reversal_of, created_at
                    """,
                    (loan_id, installment_id, payment_date, amount, channel, status),
                )
                payment = cur.fetchone()

                # Best-effort update of installment stats.
                if installment_row and installment_id is not None:
                    due_date = installment_row.get("due_date")
                    amount_due = float(installment_row.get("amount_due") or 0)
                    amount_paid = float(installment_row.get("amount_paid") or 0)
                    new_paid = amount_paid + float(amount)
                    new_status = "PAID" if new_paid >= amount_due and amount_due > 0 else "PENDING"
                    cur.execute(
                        """
                        UPDATE installments
                        SET amount_paid = %s,
                            paid_at = %s,
                            days_late = GREATEST(0, (%s::date - due_date)),
                            status = %s
                        WHERE installment_id = %s
                        """,
                        (new_paid, payment_date, payment_date, new_status, installment_id),
                    )

                _recompute_payment_behavior_summary_tx(cur, user_id)
                cur.execute(
                    """
                    UPDATE credit_cases
                    SET updated_at = NOW()
                    WHERE case_id = %s
                    """,
                    (case_id,),
                )
                return payment
    finally:
        conn.close()
