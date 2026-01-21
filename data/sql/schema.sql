-- Credit Decision System Schema

CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('CLIENT', 'BANKER')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS credit_cases (
    case_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('DRAFT', 'SUBMITTED', 'UNDER_REVIEW', 'DECIDED')),
    loan_amount NUMERIC(14,2) NOT NULL CHECK (loan_amount > 0),
    loan_duration INTEGER NOT NULL CHECK (loan_duration > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS financial_profile (
    case_id BIGINT PRIMARY KEY REFERENCES credit_cases(case_id) ON DELETE CASCADE,
    monthly_income NUMERIC(14,2) NOT NULL CHECK (monthly_income >= 0),
    other_income NUMERIC(14,2) NOT NULL DEFAULT 0 CHECK (other_income >= 0),
    monthly_charges NUMERIC(14,2) NOT NULL CHECK (monthly_charges >= 0),
    employment_type TEXT NOT NULL CHECK (employment_type IN ('employee', 'freelancer', 'self_employed', 'unemployed')),
    contract_type TEXT CHECK (contract_type IN ('permanent', 'temporary', 'none') OR contract_type IS NULL),
    seniority_years INTEGER,
    marital_status TEXT CHECK (marital_status IN ('single', 'married') OR marital_status IS NULL),
    number_of_children INTEGER NOT NULL DEFAULT 0 CHECK (number_of_children >= 0),
    spouse_employed BOOLEAN,
    housing_status TEXT CHECK (housing_status IN ('rent', 'owner', 'family') OR housing_status IS NULL),
    is_primary_holder BOOLEAN
);

CREATE TABLE IF NOT EXISTS documents (
    document_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id BIGINT NOT NULL REFERENCES credit_cases(case_id) ON DELETE CASCADE,
    document_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_outputs (
    output_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id BIGINT NOT NULL REFERENCES credit_cases(case_id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL CHECK (agent_name IN ('document', 'image', 'behavior', 'similarity', 'fraud')),
    output_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id BIGINT NOT NULL UNIQUE REFERENCES credit_cases(case_id) ON DELETE CASCADE,
    decision TEXT NOT NULL CHECK (decision IN ('APPROVE', 'REJECT', 'REVIEW')),
    confidence NUMERIC,
    reason_codes JSONB,
    decided_by BIGINT NOT NULL REFERENCES users(user_id),
    decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
