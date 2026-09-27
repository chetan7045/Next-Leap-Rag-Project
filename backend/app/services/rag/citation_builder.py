"""Citation construction — backend-owned, never LLM-generated.

A citation may only be produced from metadata that came back out of Chroma, and its
URL must belong to the configured source set. This is the mechanism that makes it
impossible for a prompt-injected string to become a link in the UI.
"""

from __future__ import annotations

from app.core.errors import CitationValidationError
from app.core.logging import get_logger
from app.models.chat import SourceCitation
from app.models.retrieval import SearchHit
from app.services.ingestion.metadata import resolve_effective_date, source_priority
from app.services.ingestion.registry import SourceRegistry

logger = get_logger("app.services.rag.citation_builder")

DATE_UNAVAILABLE = "Date not available"


class CitationBuilder:
    """Builds, de-duplicates, and validates citations from retrieved hits."""

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry

    def build(
        self,
        hits: list[SearchHit],
        *,
        limit: int = 3,
        used_only: bool = True,
    ) -> list[SourceCitation]:
        """Return up to ``limit`` citations, most authoritative and most relevant first."""
        if not hits:
            return []

        allowed_urls = self._registry.allowed_urls()
        by_document: dict[str, tuple[float, SearchHit]] = {}

        for hit in hits:
            url = hit.metadata.source_url
            # Hard rule: only configured source URLs may ever be returned.
            if url not in allowed_urls:
                logger.warning("Dropped citation with unconfigured URL for chunk %s", hit.chunk_id)
                continue
            existing = by_document.get(hit.document_id)
            if existing is None or hit.score > existing[0]:
                by_document[hit.document_id] = (hit.score, hit)

        ordered = sorted(
            by_document.values(),
            key=lambda item: (source_priority(item[1].metadata), item[0]),
            reverse=True,
        )

        citations: list[SourceCitation] = []
        for score, hit in ordered:
            meta = hit.metadata
            citations.append(
                SourceCitation(
                    title=meta.source_title or meta.scheme_name,
                    url=meta.source_url,
                    scheme_id=meta.scheme_id,
                    scheme_name=meta.scheme_name,
                    source_type=meta.source_type,
                    source_type_label=meta.source_type.ui_label,
                    document_type=meta.document_type,
                    authority_level=meta.authority_level,
                    relevance=round(score, 3),
                    last_updated=resolve_effective_date(meta),
                )
            )
            if len(citations) >= limit:
                break
        return citations

    @staticmethod
    def resolve_last_updated(citations: list[SourceCitation]) -> str:
        """Most recent source date across the cited documents, or an honest 'not available'."""
        dates = [c.last_updated for c in citations if c.last_updated]
        if not dates:
            return DATE_UNAVAILABLE
        return max(dates)

    def validate(self, answer: str, citations: list[SourceCitation]) -> None:
        """Raise when the answer cannot be presented as a verified, sourced fact.

        Rules: a non-empty answer, at least one citation, and every citation URL present
        in the configured source set.
        """
        if not answer or not answer.strip():
            raise CitationValidationError("Generated answer was empty.")
        if not citations:
            raise CitationValidationError("No valid source was available for the generated answer.")
        allowed = self._registry.allowed_urls()
        for citation in citations:
            if citation.url not in allowed:
                raise CitationValidationError(
                    f"Citation URL is not part of the configured source set: {citation.url}"
                )
