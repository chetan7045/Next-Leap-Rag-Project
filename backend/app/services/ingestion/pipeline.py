"""Ingestion pipeline: sources.json → ChromaDB.

Versioning contract (see ARCHITECTURE.md §2):
  * unchanged content_hash  → skip entirely, no re-embedding
  * changed   content_hash  → delete the document's chunks, then re-chunk and upsert
  * with ``--prune``        → delete documents no longer present in sources.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.errors import IngestionError
from app.core.logging import get_logger
from app.models.retrieval import Chunk, Document, DocumentMetadata, Section, SourceConfig
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.ingestion.cleaner import (
    blocks_to_text,
    clean_extraction,
    normalise_text,
    strip_leading_navigation,
)
from app.services.ingestion.extractor import Block, ExtractionResult, extract
from app.services.ingestion.loader import DocumentLoader, FetchedDocument
from app.services.ingestion.metadata import build_metadata
from app.services.ingestion.registry import SourceRegistry, get_registry
from app.services.rag.chroma_service import ChromaService
from app.services.ingestion.chunker import chunk_document, estimate_tokens, summarise_chunks

logger = get_logger("app.services.ingestion.pipeline")


@dataclass(slots=True)
class DocumentOutcome:
    source_id: str
    scheme_id: str
    url: str
    status: str  # indexed | unchanged | replaced | failed | skipped | empty
    document_id: str = ""
    chunks: int = 0
    reason: str = ""
    content_hash: str = ""
    tokens: int = 0
    title: str = ""


@dataclass(slots=True)
class IngestionReport:
    started_at: datetime
    finished_at: datetime | None = None
    documents: list[DocumentOutcome] = field(default_factory=list)
    pruned: int = 0

    @property
    def processed(self) -> int:
        return len(self.documents)

    @property
    def indexed(self) -> int:
        return sum(1 for d in self.documents if d.status in {"indexed", "replaced"})

    @property
    def unchanged(self) -> int:
        return sum(1 for d in self.documents if d.status == "unchanged")

    @property
    def failed(self) -> int:
        return sum(1 for d in self.documents if d.status == "failed")

    @property
    def total_chunks(self) -> int:
        return sum(d.chunks for d in self.documents)

    @property
    def total_tokens(self) -> int:
        return sum(d.tokens for d in self.documents)

    def summary(self) -> str:
        return (
            f"sources processed: {self.processed} | indexed: {self.indexed} | "
            f"unchanged: {self.unchanged} | failed: {self.failed} | "
            f"chunks created: {self.total_chunks} | pruned: {self.pruned}"
        )


def _document_from_extraction(
    source: SourceConfig,
    result: ExtractionResult,
    fetched: FetchedDocument,
    settings: Settings,
) -> Document:
    """Clean the extraction and shape it into a Document with heading sections."""
    cleaned = clean_extraction(result)
    cleaned.blocks = strip_leading_navigation(cleaned.blocks)
    text = normalise_text(blocks_to_text(cleaned.blocks))
    if len(text) < settings.ingestion_min_content_chars:
        raise IngestionError(
            f"Extracted content for {source.url} was too short "
            f"({len(text)} chars); the page is probably client-rendered."
        )

    metadata: DocumentMetadata = build_metadata(
        source,
        title=cleaned.title or source.scheme_name,
        text=text,
        retrieved_at=fetched.retrieved_at,
        date_hint=cleaned.published_hint,
        settings=settings,
    )

    sections: list[Section] = []
    heading = ""
    blocks: list[Block] = []
    for block in cleaned.blocks:
        if block.kind == "heading":
            if blocks:
                sections.append(Section(heading=heading, level=2, blocks=blocks))
            heading = block.text
            blocks = [block]
        else:
            blocks.append(block)
    if blocks:
        sections.append(Section(heading=heading, level=2, blocks=blocks))

    return Document(metadata=metadata, text=text, sections=sections)


class IngestionPipeline:
    """Runs the offline ingestion flow."""

    def __init__(
        self,
        settings: Settings,
        registry: SourceRegistry | None = None,
        chroma: ChromaService | None = None,
        embeddings: EmbeddingService | None = None,
        loader: DocumentLoader | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry or get_registry(settings)
        self._chroma = chroma or ChromaService(settings)
        self._embeddings = embeddings or EmbeddingService.get_instance(settings)
        self._loader = loader or DocumentLoader(settings, allowed_urls=self._registry.allowed_urls())

    # --- Raw archive (optional, for auditability) -----------------------
    def _archive(self, source: SourceConfig, fetched: FetchedDocument) -> None:
        try:
            self._settings.raw_dir.mkdir(parents=True, exist_ok=True)
            path = Path(self._settings.raw_dir) / f"{source.id}.html"
            path.write_text(fetched.html, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - best effort only
            logger.warning("Could not archive raw HTML for %s: %s", source.id, exc)

    # --- Main entry point ------------------------------------------------
    def run(self, *, force: bool = False, prune: bool = False, source_ids: list[str] | None = None) -> IngestionReport:
        report = IngestionReport(started_at=datetime.now(UTC))

        sources = self._registry.sources(enabled_only=True)
        if source_ids:
            wanted = set(source_ids)
            sources = [s for s in sources if s.id in wanted]
        if not sources:
            raise IngestionError("No enabled sources are configured; nothing to ingest.")

        self._chroma.initialize()
        known_hashes = self._chroma.get_document_hashes()
        logger.info("Index currently holds %d chunks across %d documents", self._chroma.count(), len(known_hashes))

        for source in sources:
            outcome = self._ingest_one(source, known_hashes, force=force)
            report.documents.append(outcome)
            if outcome.status == "indexed" or outcome.status == "replaced":
                known_hashes[outcome.document_id] = outcome.content_hash

        if prune:
            report.pruned = self._prune({self._document_id_for(s) for s in sources})

        report.finished_at = datetime.now(UTC)
        return report

    @staticmethod
    def _document_id_for(source: SourceConfig) -> str:
        from app.services.ingestion.metadata import make_document_id

        return make_document_id(source.scheme_id, source.url)

    def _ingest_one(
        self, source: SourceConfig, known_hashes: dict[str, str], *, force: bool
    ) -> DocumentOutcome:
        document_id = self._document_id_for(source)
        try:
            fetched = self._loader.fetch(source.url)
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort the run
            logger.error("Fetch failed for %s: %s", source.id, exc)
            return DocumentOutcome(
                source_id=source.id,
                scheme_id=source.scheme_id,
                url=source.url,
                status="failed",
                reason=str(exc)[:200],
            )

        self._archive(source, fetched)

        try:
            extraction = extract(fetched.html, base_title=source.scheme_name)
            document = _document_from_extraction(source, extraction, fetched, self._settings)
        except Exception as exc:  # noqa: BLE001
            logger.error("Extraction failed for %s: %s", source.id, exc)
            return DocumentOutcome(
                source_id=source.id,
                scheme_id=source.scheme_id,
                url=source.url,
                status="empty",
                reason=str(exc)[:200],
            )

        metadata = document.metadata
        previous_hash = known_hashes.get(metadata.document_id)
        if not force and previous_hash == metadata.content_hash:
            return DocumentOutcome(
                source_id=source.id,
                scheme_id=source.scheme_id,
                url=source.url,
                status="unchanged",
                document_id=metadata.document_id,
                content_hash=metadata.content_hash,
                title=metadata.source_title,
            )

        chunks = chunk_document(document, self._settings)
        if not chunks:
            return DocumentOutcome(
                source_id=source.id,
                scheme_id=source.scheme_id,
                url=source.url,
                status="empty",
                document_id=metadata.document_id,
                content_hash=metadata.content_hash,
                reason="No chunks produced from the extracted content.",
            )

        replaced = previous_hash is not None
        if replaced:
            self._chroma.delete_document(metadata.document_id)

        self._embed_and_store(chunks)
        stats = summarise_chunks(chunks)
        return DocumentOutcome(
            source_id=source.id,
            scheme_id=source.scheme_id,
            url=source.url,
            status="replaced" if replaced else "indexed",
            document_id=metadata.document_id,
            chunks=len(chunks),
            content_hash=metadata.content_hash,
            tokens=stats["tokens"],
            title=metadata.source_title,
        )

    def _embed_and_store(self, chunks: list[Chunk]) -> None:
        vectors = self._embeddings.embed_documents([c.text for c in chunks])
        if len(vectors) != len(chunks):  # pragma: no cover - defensive
            raise IngestionError("Embedding count did not match chunk count.")
        self._chroma.upsert_chunks(chunks, vectors)

    def _prune(self, keep_document_ids: set[str]) -> int:
        removed = 0
        for document_id in self._chroma.document_ids():
            if document_id not in keep_document_ids:
                removed += self._chroma.delete_document(document_id)
        if removed:
            logger.info("Pruned %d stale chunks", removed)
        return removed

    # --- Startup helper --------------------------------------------------
    def ingest_if_empty(self) -> IngestionReport | None:
        """Used only when ``AUTO_INGEST_ON_STARTUP=true`` and the index is empty."""
        self._chroma.initialize()
        if self._chroma.count() > 0:
            logger.info("Startup ingestion skipped: index already holds %d chunks", self._chroma.count())
            return None
        logger.info("Startup ingestion enabled and index is empty; ingesting configured sources.")
        try:
            return self.run()
        except Exception as exc:  # noqa: BLE001 - never block startup
            logger.error("Startup ingestion failed: %s", exc)
            return None


def _token_estimate(text: str) -> int:  # pragma: no cover - re-export convenience
    return estimate_tokens(text)
