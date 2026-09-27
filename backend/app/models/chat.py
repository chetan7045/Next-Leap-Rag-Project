"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import (
    CitationStatus,
    DocumentType,
    QueryType,
    RefusalReason,
    ResponseType,
    SourceType,
)


class ChatRequest(BaseModel):
    """Validated chat input."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(
        min_length=1,
        max_length=1000,
        description="A factual question about an HDFC Mutual Fund scheme.",
    )
    scheme_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional scheme filter, e.g. HDFC_LARGE_CAP.",
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Opaque client-side id used for request correlation only. Not persisted.",
    )

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain at least one non-whitespace character")
        return value.strip()

    @field_validator("scheme_id")
    @classmethod
    def _normalise_scheme_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned or None


class SourceCitation(BaseModel):
    """A citation built by the backend from retrieved chunk metadata."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    scheme_id: str
    scheme_name: str
    source_type: SourceType
    source_type_label: str
    document_type: DocumentType | None = None
    authority_level: int = Field(exclude=True)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0, description="1 - cosine distance.")
    last_updated: str | None = None


class RetrievalDebug(BaseModel):
    """Development-only diagnostics. Omitted unless DEBUG_RAG=true."""

    model_config = ConfigDict(extra="forbid")

    chunks_used: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    candidates: int = Field(default=0, ge=0)
    detected_scheme: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    llm_ms: float = Field(default=0.0, ge=0.0)


class ChatResponse(BaseModel):
    """Structured answer. Never contains chain-of-thought or internal prompts."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    last_updated: str = "Date not available"
    query_type: QueryType = QueryType.FACTUAL
    answer_type: ResponseType = ResponseType.ANSWER
    refusal_reason: RefusalReason | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_status: CitationStatus = CitationStatus.NOT_APPLICABLE
    # Retained for backwards compatibility and always derived from ``answer_type``.
    refusal: bool = False
    clarification: bool = False
    clarification_options: list[dict[str, str]] = Field(default_factory=list)
    retrieval: RetrievalDebug | None = None
    disclaimer: str = ""
    request_id: str = ""

    @model_validator(mode="after")
    def _sync_legacy_flags(self) -> "ChatResponse":
        """Keep ``refusal``/``clarification`` consistent with ``answer_type``.

        Deriving them here means no call site can return a refusal without the flag,
        or a clarification that still claims to be a refusal.
        """
        if self.answer_type in {ResponseType.REFUSAL, ResponseType.NO_CONTEXT, ResponseType.ERROR}:
            object.__setattr__(self, "refusal", True)
        elif self.answer_type is ResponseType.CLARIFICATION:
            object.__setattr__(self, "clarification", True)
        return self


class SchemeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    plan: str = "Direct Growth"
    amc: str = "HDFC Mutual Fund"
    categories: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)
    #: True when the scheme has at least one chunk in the vector index. The frontend
    #: uses this for the "N of M schemes indexed" badge and to warn on a cold index.
    indexed: bool = False


class SchemeListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemes: list[SchemeOut]
    count: int


class SourceOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str
    scheme_id: str
    scheme_name: str
    amc: str
    source_type: SourceType
    source_type_label: str
    document_type: DocumentType
    publisher: str | None = None
    last_updated: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    indexed: bool = False


class SourceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceOut]
    count: int
    schemes_indexed: int = Field(default=0, ge=0)


class LLMHealth(BaseModel):
    """Configuration state of the generation provider.

    ``configured`` only proves a credential is present. It does not prove the model can
    generate: a project with depleted credits is configured but unusable, which is why
    the chat response reports ``LLM_ERROR`` rather than a 5xx.

    The provider name and model ID are intentionally absent. ``/health`` is a public,
    unauthenticated endpoint, so it reports only whether the backend is usable, not
    which model backs it.
    """

    model_config = ConfigDict(extra="forbid")

    configured: bool


class HealthResponse(BaseModel):
    """Deliberately cheap: no LLM call, no embedding work."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    version: str
    environment: str
    chroma_mode: str
    index_ready: bool
    llm: LLMHealth


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    request_id: str | None = None
    context: dict[str, Any] | None = None
