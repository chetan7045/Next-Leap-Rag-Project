/**
 * Types mirroring the FastAPI OpenAPI contract.
 *
 * Kept deliberately hand-written and in sync with `backend/app/models/chat.py`.
 * The backend is the single source of truth for answer safety: the UI only ever
 * renders what the API returns and never derives an answer of its own.
 */

export type ResponseType =
  | "ANSWER"
  | "REFUSAL"
  | "CLARIFICATION"
  | "NO_CONTEXT"
  | "ERROR";

export type RefusalReason =
  | "ADVICE_REQUEST"
  | "PERFORMANCE_PROMISE"
  | "FORECAST_REQUEST"
  | "PII_DETECTED"
  | "OUT_OF_SCOPE"
  | "AMBIGUOUS_SCHEME"
  | "LOW_CONFIDENCE"
  | "NO_INDEX"
  | "LLM_UNAVAILABLE"
  | "LLM_ERROR"
  | "VALIDATION_FAILED";

export type CitationStatus = "VERIFIED" | "NOT_APPLICABLE" | "MISSING";

export type QueryType =
  | "FACTUAL"
  | "ADVICE"
  | "PERFORMANCE"
  | "OUT_OF_SCOPE"
  | "PII_RISK"
  | "UNKNOWN"
  | "CLARIFICATION";

export type SourceType = "AMC_OFFICIAL" | "AMFI" | "SEBI" | "REFERENCE";

export type DocumentType =
  | "SCHEME_PAGE"
  | "FACTSHEET"
  | "SID"
  | "KIM"
  | "FAQ"
  | "TAX_GUIDE"
  | "STATEMENT_GUIDE"
  | "OTHER";

export interface SourceCitation {
  title: string;
  url: string;
  scheme_id: string;
  scheme_name: string;
  source_type: SourceType;
  source_type_label: string;
  document_type?: DocumentType | null;
  /** 1 - cosine distance, filled in by the backend. */
  relevance?: number | null;
  last_updated?: string | null;
}

export interface RetrievalDebug {
  chunks_used: number;
  confidence: number;
  threshold: number;
  candidates: number;
  detected_scheme?: string | null;
  source_ids: string[];
  retrieval_ms: number;
  llm_ms: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceCitation[];
  last_updated: string | null;
  query_type: QueryType;
  answer_type: ResponseType;
  refusal_reason: RefusalReason | null;
  confidence: number;
  citation_status: CitationStatus;
  /** Legacy booleans kept in sync by the backend. Prefer `answer_type`. */
  refusal: boolean;
  clarification: boolean;
  clarification_options: Array<Record<string, string>>;
  retrieval: RetrievalDebug | null;
  disclaimer: string;
  request_id: string;
}

export interface ChatRequest {
  message: string;
  scheme_id?: string | null;
  conversation_id?: string | null;
}

export interface SchemeOut {
  id: string;
  name: string;
  plan?: string | null;
  amc?: string | null;
  categories: string[];
  aliases: string[];
  source_count: number;
  indexed: boolean;
}

export interface SchemeListResponse {
  schemes: SchemeOut[];
  count: number;
}

export interface SourceOut {
  id: string;
  title: string;
  url: string;
  scheme_id: string;
  scheme_name: string;
  amc: string;
  source_type: SourceType;
  source_type_label: string;
  document_type: DocumentType;
  publisher?: string | null;
  last_updated?: string | null;
  published_at?: string | null;
  retrieved_at?: string | null;
  chunk_count: number;
  indexed: boolean;
}

export interface SourceListResponse {
  sources: SourceOut[];
  count: number;
  schemes_indexed: number;
}

export interface LLMHealth {
  provider: string;
  model: string;
  /** A credential is present. This does NOT prove the model can generate. */
  configured: boolean;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  chroma_mode: string;
  index_ready: boolean;
  llm: LLMHealth;
}
