"""Retrieval-domain models: documents, chunks, and search results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import DocumentType, SourceType


class SourceConfig(BaseModel):
    """One entry of ``sources.json`` — the only place source URLs are defined."""

    model_config = ConfigDict(extra="ignore")

    id: str
    scheme_id: str
    scheme_name: str
    amc: str = "HDFC Mutual Fund"
    url: str
    source_type: SourceType = SourceType.REFERENCE
    document_type: DocumentType = DocumentType.SCHEME_PAGE
    title: str | None = None
    publisher: str | None = None
    plan: str = "Direct Growth"
    loader: str = "static"
    enabled: bool = True
    topics: list[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Full provenance record attached to every chunk."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_id: str
    source_url: str
    source_title: str
    scheme_id: str
    scheme_name: str
    amc: str
    source_type: SourceType
    document_type: DocumentType
    authority_level: int
    published_at: str | None = None
    last_updated_at: str | None = None
    retrieved_at: str
    content_hash: str
    publisher: str | None = None
    loader: str = "static"

    @property
    def effective_updated(self) -> str | None:
        """Source date only. Retrieval time is never substituted for it."""
        return self.last_updated_at or self.published_at


class Document(BaseModel):
    """A fetched, cleaned, normalised document ready for chunking."""

    model_config = ConfigDict(extra="forbid")

    metadata: DocumentMetadata
    text: str
    sections: list["Section"] = Field(default_factory=list)
    tables: list["TableBlock"] = Field(default_factory=list)


class Section(BaseModel):
    """A heading and the content that belongs to it.

    ``blocks`` holds extractor block objects. They are typed loosely to keep this
    module free of a dependency on the ingestion package; the chunker only needs
    each block's ``.text``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    heading: str
    level: int = 2
    blocks: list[Any] = Field(default_factory=list)


class TableBlock(BaseModel):
    """A table preserved as a header row plus aligned data rows."""

    model_config = ConfigDict(extra="forbid")

    caption: str | None = None
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_text(self) -> str:
        lines: list[str] = []
        if self.caption:
            lines.append(self.caption)
        if self.header:
            lines.append(" | ".join(self.header))
        for row in self.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)


class Chunk(BaseModel):
    """A retrievable unit. One chunk = one coherent factual context."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    heading_path: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    topics: list[str] = Field(default_factory=list)
    metadata: DocumentMetadata


class SearchHit(BaseModel):
    """A chunk returned by Chroma plus its similarity."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    ordinal: int
    heading_path: list[str] = Field(default_factory=list)
    metadata: DocumentMetadata


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[SearchHit] = Field(default_factory=list)
    confidence: float = 0.0
    candidates_considered: int = 0
    scheme_id: str | None = None
    scheme_ambiguous: bool = False


Document.model_rebuild()
