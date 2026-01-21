import json
import os
import random
import sys
from decimal import Decimal

import psycopg2


def get_db_params():
    params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    missing = [k for k, v in params.items() if not v]
    if missing:
        raise RuntimeError(f"Missing DB env vars: {', '.join(missing)}")
    return params


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def insert_users(cur, num_clients):
    client_ids = []
    for i in range(1, num_clients + 1):
        cur.execute(
            """
            INSERT INTO users (email, password_hash, role)
            VALUES (%s, %s, 'CLIENT')
            RETURNING user_id
            """,
            (f"client{i}@test.com", "hashed-password"),
        )
        client_ids.append(cur.fetchone()[0])
    cur.execute(
        """
        INSERT INTO users (email, password_hash, role)
        VALUES (%s, %s, 'BANKER')
        RETURNING user_id
        """,
        ("banker1@test.com", "hashed-password"),
    )
    banker_id = cur.fetchone()[0]
    return client_ids, banker_id


def compute_decision_flags(record):
    fraud_flag = bool(record.get("fraud_flag", False))
    defaulted = bool(record.get("defaulted", False))
    if fraud_flag:
        decision = "REJECT"
        confidence = Decimal("0.3")
    elif defaulted:
        decision = "APPROVE"
        confidence = Decimal("0.6")
    else:
        decision = "APPROVE"
        confidence = Decimal("0.85")
    return decision, confidence, fraud_flag, defaulted


def seed_case(cur, record, client_ids, banker_id):
    decision, confidence, fraud_flag, defaulted = compute_decision_flags(record)
    user_id = random.choice(client_ids)
    case_id = record.get("case_id")
    loan_amount = Decimal(str(record.get("loan_amount", 0)))
    loan_duration = int(record.get("loan_duration", 0))

    cur.execute(
        """
        INSERT INTO credit_cases (case_id, user_id, status, loan_amount, loan_duration)
        OVERRIDING SYSTEM VALUE
        VALUES (%s, %s, 'DECIDED', %s, %s)
        RETURNING case_id
        """,
        (case_id, user_id, loan_amount, loan_duration),
    )
    new_case_id = cur.fetchone()[0]

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
            new_case_id,
            Decimal(str(record.get("monthly_income", 0))),
            Decimal(str(record.get("other_income", 0))),
            Decimal(str(record.get("monthly_charges", 0))),
            record.get("employment_type"),
            record.get("contract_type"),
            record.get("seniority_years"),
            record.get("marital_status"),
            record.get("number_of_children", 0),
            record.get("spouse_employed"),
            record.get("housing_status"),
            record.get("is_primary_holder"),
        ),
    )

    cur.execute(
        """
        INSERT INTO decisions (
            case_id,
            decision,
            confidence,
            reason_codes,
            decided_by
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            new_case_id,
            decision,
            confidence,
            json.dumps(
                {
                    "fraud_flag": fraud_flag,
                    "defaulted": defaulted,
                    "credit_outcome": record.get("credit_outcome"),
                    "decision_year": record.get("decision_year"),
                }
            ),
            banker_id,
        ),
    )


def main():
    dataset_path = os.getenv(
        "DATASET_PATH",
        os.path.join(os.path.dirname(__file__), "data", "synthetic", "credit_dataset.json"),
    )
    dataset = load_dataset(dataset_path)
    if not isinstance(dataset, list) or not dataset:
        raise RuntimeError("Dataset is empty or invalid")

    params = get_db_params()
    conn = psycopg2.connect(**params)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            client_ids, banker_id = insert_users(cur, num_clients=len(dataset))
            for record in dataset:
                seed_case(cur, record, client_ids, banker_id)
        conn.commit()
        print("Database seeding completed successfully")
    except Exception as exc:
        conn.rollback()
        print(f"Seeding failed: {exc}", file=sys.stderr)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
