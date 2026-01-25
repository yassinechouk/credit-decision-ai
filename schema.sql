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
    summary TEXT,
    auto_decision TEXT,
    auto_decision_confidence NUMERIC,
    auto_review_required BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    note TEXT,
    decided_by BIGINT NOT NULL REFERENCES users(user_id),
    decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id BIGINT NOT NULL REFERENCES credit_cases(case_id) ON DELETE CASCADE,
    author_id BIGINT NOT NULL REFERENCES users(user_id),
    message TEXT NOT NULL,
    is_public BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Loans created from approved credit cases
CREATE TABLE IF NOT EXISTS loans (
    loan_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id BIGINT UNIQUE REFERENCES credit_cases(case_id) ON DELETE SET NULL,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    lender_name TEXT,
    product_type TEXT,
    purpose TEXT,
    start_date DATE NOT NULL,
    end_date DATE,
    principal NUMERIC(14,2) NOT NULL CHECK (principal > 0),
    interest_rate NUMERIC(6,3) CHECK (interest_rate >= 0),
    term_months INTEGER CHECK (term_months > 0),
    currency TEXT DEFAULT 'USD',
    status TEXT NOT NULL CHECK (status IN ('active', 'closed', 'defaulted', 'restructured')),
    closed_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Expected installments (scheduled tranches)
CREATE TABLE IF NOT EXISTS installments (
    installment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    loan_id BIGINT NOT NULL REFERENCES loans(loan_id) ON DELETE CASCADE,
    sequence_no INTEGER NOT NULL CHECK (sequence_no > 0),
    due_date DATE NOT NULL,
    amount_due NUMERIC(14,2) NOT NULL CHECK (amount_due >= 0),
    principal_due NUMERIC(14,2) DEFAULT 0 CHECK (principal_due >= 0),
    interest_due NUMERIC(14,2) DEFAULT 0 CHECK (interest_due >= 0),
    fees_due NUMERIC(14,2) DEFAULT 0 CHECK (fees_due >= 0),
    status TEXT NOT NULL CHECK (status IN ('due', 'paid', 'late', 'partial', 'waived')),
    paid_amount NUMERIC(14,2) DEFAULT 0 CHECK (paid_amount >= 0),
    paid_date DATE,
    days_late INTEGER DEFAULT 0 CHECK (days_late >= 0),
    is_restructured BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Actual payments made by the client
CREATE TABLE IF NOT EXISTS payments (
    payment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    loan_id BIGINT NOT NULL REFERENCES loans(loan_id) ON DELETE CASCADE,
    installment_id BIGINT REFERENCES installments(installment_id) ON DELETE SET NULL,
    payment_date DATE NOT NULL,
    amount_paid NUMERIC(14,2) NOT NULL CHECK (amount_paid > 0),
    payment_method TEXT,
    payment_channel TEXT,
    status TEXT NOT NULL CHECK (status IN ('received', 'reversed', 'failed')),
    is_reversal BOOLEAN NOT NULL DEFAULT FALSE,
    external_ref TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Aggregated payment behavior for scoring
CREATE TABLE IF NOT EXISTS payment_behavior_summary (
    summary_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    loan_id BIGINT REFERENCES loans(loan_id) ON DELETE CASCADE,
    period_start DATE,
    period_end DATE,
    total_installments INTEGER DEFAULT 0 CHECK (total_installments >= 0),
    paid_installments INTEGER DEFAULT 0 CHECK (paid_installments >= 0),
    missed_installments INTEGER DEFAULT 0 CHECK (missed_installments >= 0),
    partial_installments INTEGER DEFAULT 0 CHECK (partial_installments >= 0),
    on_time_rate NUMERIC(5,2) CHECK (on_time_rate >= 0 AND on_time_rate <= 100),
    avg_days_late NUMERIC(6,2) CHECK (avg_days_late >= 0),
    max_days_late INTEGER DEFAULT 0 CHECK (max_days_late >= 0),
    total_amount_due NUMERIC(14,2) DEFAULT 0 CHECK (total_amount_due >= 0),
    total_amount_paid NUMERIC(14,2) DEFAULT 0 CHECK (total_amount_paid >= 0),
    outstanding_amount NUMERIC(14,2) DEFAULT 0 CHECK (outstanding_amount >= 0),
    last_payment_date DATE,
    last_missed_date DATE,
    behavior_score NUMERIC(6,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_loans_user_status ON loans(user_id, status);
CREATE INDEX IF NOT EXISTS idx_installments_loan_due ON installments(loan_id, due_date);
CREATE INDEX IF NOT EXISTS idx_payments_loan_date ON payments(loan_id, payment_date);
CREATE INDEX IF NOT EXISTS idx_behavior_summary_user ON payment_behavior_summary(user_id);
