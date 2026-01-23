export type Role = "client" | "banker";

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  role: Role;
  user_id: string;
}

export interface CreditRequestCreate {
  amount: number;
  duration_months: number;
  monthly_income: number;
  monthly_charges: number;
  employment_type: string;
  contract_type: string;
  seniority_years: number;
  family_status: string;
  documents: string[];
}

export interface CreditRequest {
  id: string;
  status: "pending" | "in_review" | "approved" | "rejected";
  created_at: string;
  updated_at: string;
  client_id: string;
  summary?: string;
  customer_explanation?: string;
  agents?: AgentBundle;
}

export interface AgentResult {
  name: string;
  score?: number;
  flags?: string[];
  explanations?: {
    flag_explanations?: Record<string, string>;
    internal_explanation?: unknown;
    customer_explanation?: unknown;
    global_summary?: string;
  };
  confidence?: number;
}

export interface AgentBundle {
  document?: AgentResult;
  similarity?: AgentResult;
  fraud?: AgentResult;
  explanation?: AgentResult;
}

export interface CommentCreate {
  message: string;
}

export interface DecisionCreate {
  decision: "approve" | "reject" | "review";
  note?: string;
}
