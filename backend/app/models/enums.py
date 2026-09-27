"""Shared enumerations for the RAG domain."""

from __future__ import annotations

from enum import StrEnum


class QueryType(StrEnum):
    """Pre-LLM classification of a user question."""

    FACTUAL = "FACTUAL"
    ADVICE = "ADVICE"
    PERFORMANCE = "PERFORMANCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    PII_RISK = "PII_RISK"
    UNKNOWN = "UNKNOWN"
    CLARIFICATION = "CLARIFICATION"


class SourceType(StrEnum):
    """Provenance tier of a document. Never inferred — always declared in sources.json."""

    AMC_OFFICIAL = "AMC_OFFICIAL"
    AMFI = "AMFI"
    SEBI = "SEBI"
    REFERENCE = "REFERENCE"

    @property
    def authority_level(self) -> int:
        return _AUTHORITY[self]

    @property
    def ui_label(self) -> str:
        return _UI_LABEL[self]


_AUTHORITY: dict[SourceType, int] = {
    SourceType.AMC_OFFICIAL: 1,
    SourceType.AMFI: 2,
    SourceType.SEBI: 2,
    SourceType.REFERENCE: 3,
}

_UI_LABEL: dict[SourceType, str] = {
    SourceType.AMC_OFFICIAL: "HDFC Mutual Fund",
    SourceType.AMFI: "AMFI",
    SourceType.SEBI: "SEBI",
    SourceType.REFERENCE: "Reference source",
}


class DocumentType(StrEnum):
    SCHEME_PAGE = "SCHEME_PAGE"
    FACTSHEET = "FACTSHEET"
    SID = "SID"
    KIM = "KIM"
    FAQ = "FAQ"
    TAX_GUIDE = "TAX_GUIDE"
    STATEMENT_GUIDE = "STATEMENT_GUIDE"
    OTHER = "OTHER"


class ChromaMode(StrEnum):
    PERSISTENT = "persistent"
    EPHEMERAL = "ephemeral"


class ResponseType(StrEnum):
    """What kind of answer the user got. The frontend switches its UI on this.

    ``refusal``/``clarification`` booleans remain in the payload for backwards
    compatibility, but they cannot express the difference between "I will not give
    investment advice" and "I could not find that in the sources" — this can.
    """

    ANSWER = "ANSWER"
    REFUSAL = "REFUSAL"
    CLARIFICATION = "CLARIFICATION"
    NO_CONTEXT = "NO_CONTEXT"
    ERROR = "ERROR"


class CitationStatus(StrEnum):
    """Whether the attached sources actually support the answer."""

    VERIFIED = "VERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING = "MISSING"


class RefusalReason(StrEnum):
    """Machine-readable cause for a non-answer."""

    ADVICE_REQUEST = "ADVICE_REQUEST"
    PERFORMANCE_PROMISE = "PERFORMANCE_PROMISE"
    FORECAST_REQUEST = "FORECAST_REQUEST"
    PII_DETECTED = "PII_DETECTED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS_SCHEME = "AMBIGUOUS_SCHEME"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_INDEX = "NO_INDEX"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_ERROR = "LLM_ERROR"
    VALIDATION_FAILED = "VALIDATION_FAILED"
