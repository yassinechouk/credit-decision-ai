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
  decision?: DecisionInfo;
  comments?: Comment[];
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
  behavior?: AgentResult;
}

export interface CommentCreate {
  message: string;
}

export interface DecisionCreate {
  decision: "approve" | "reject" | "review";
  note?: string;
}

export interface DocumentInfo {
  document_id: number;
  document_type: string;
  file_path: string;
  file_hash: string;
  uploaded_at: string;
}

export interface Comment {
  author_id: string;
  message: string;
  created_at: string;
  is_public?: boolean;
}

export interface DecisionInfo {
  decision: "approve" | "reject" | "review";
  confidence?: number;
  reason_codes?: Record<string, unknown>;
  note?: string;
  decided_by?: string;
  decided_at?: string;
}

export interface BankerRequest {
  id: string;
  status: "pending" | "in_review" | "approved" | "rejected";
  created_at: string;
  updated_at: string;
  client_id: string;
  summary?: string;
  amount?: number;
  duration_months?: number;
  monthly_income?: number;
  other_income?: number;
  monthly_charges?: number;
  employment_type?: string;
  contract_type?: string;
  seniority_years?: number;
  marital_status?: string;
  number_of_children?: number;
  spouse_employed?: boolean;
  housing_status?: string;
  is_primary_holder?: boolean;
  documents?: DocumentInfo[];
  agents?: AgentBundle;
  comments?: Comment[];
  decision?: DecisionInfo;
}
