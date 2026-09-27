"""ChromaDB access layer.

Owns client/collection lifecycle, metadata projection, deterministic ids, and the
document-level upsert/delete primitives that make ingestion idempotent.
"""

from __future__ import annotations

import threading
from typing import Any

from app.core.config import Settings
from app.core.errors import ConfigurationError, RetrievalError
from app.core.logging import get_logger
from app.models.retrieval import Chunk, DocumentMetadata, SearchHit

logger = get_logger("app.services.chroma")

# Chroma metadata values must be primitives. Anything else is stringified.
_MAX_METADATA_VALUE_LEN = 1200


def _flatten_metadata(metadata: DocumentMetadata, *, ordinal: int, heading_path: list[str]) -> dict[str, Any]:
    return {
        "document_id": metadata.document_id,
        "source_id": metadata.source_id,
        "source_url": metadata.source_url,
        "source_title": metadata.source_title[:300],
        "scheme_id": metadata.scheme_id,
        "scheme_name": metadata.scheme_name,
        "amc": metadata.amc,
        "source_type": str(metadata.source_type),
        "source_type_label": metadata.source_type.ui_label,
        "document_type": str(metadata.document_type),
        "authority_level": int(metadata.authority_level),
        "published_at": metadata.published_at or "",
        "last_updated_at": metadata.last_updated_at or "",
        "retrieved_at": metadata.retrieved_at,
        "content_hash": metadata.content_hash,
        "publisher": metadata.publisher or "",
        "loader": metadata.loader,
        "ordinal": int(ordinal),
        "heading_path": " > ".join(heading_path)[:_MAX_METADATA_VALUE_LEN],
    }


def _rehydrate(payload: dict[str, Any]) -> DocumentMetadata:
    from app.models.enums import DocumentType, SourceType

    return DocumentMetadata(
        document_id=str(payload.get("document_id", "")),
        source_id=str(payload.get("source_id", "")),
        source_url=str(payload.get("source_url", "")),
        source_title=str(payload.get("source_title", "")),
        scheme_id=str(payload.get("scheme_id", "")),
        scheme_name=str(payload.get("scheme_name", "")),
        amc=str(payload.get("amc", "HDFC Mutual Fund")),
        source_type=SourceType(payload.get("source_type", "REFERENCE")),
        document_type=DocumentType(payload.get("document_type", "OTHER")),
        authority_level=int(payload.get("authority_level", 3)),
        published_at=payload.get("published_at") or None,
        last_updated_at=payload.get("last_updated_at") or None,
        retrieved_at=str(payload.get("retrieved_at", "")),
        content_hash=str(payload.get("content_hash", "")),
        publisher=payload.get("publisher") or None,
        loader=str(payload.get("loader", "static")),
    )


class ChromaService:
    """Thin, well-scoped wrapper around Chroma."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._collection: Any | None = None
        self._lock = threading.Lock()

    # --- Lifecycle -------------------------------------------------------
    def initialize(self) -> None:
        """Create the client and collection once, readying the persist directory."""
        if self._collection is not None:
            return
        with self._lock:
            if self._collection is not None:
                return
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ConfigurationError("chromadb is not installed.") from exc

            persist_path = self._settings.chroma_path
            if persist_path is not None:
                persist_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
                )
                mode = f"persistent at {persist_path}"
            else:
                self._client = chromadb.EphemeralClient(
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
                )
                mode = "ephemeral (in-memory; data does not survive restarts)"

            # embedding_function=None: vectors are always supplied explicitly by
            # EmbeddingService. Without this, Chroma lazily downloads its own ONNX
            # model and silently embeds text with a different pipeline.
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection,
                metadata={"hnsw:space": "cosine", "schema_version": "1"},
                embedding_function=None,
            )
            logger.info("Chroma collection '%s' ready (%s)", self._settings.chroma_collection, mode)

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self.initialize()
        assert self._collection is not None
        return self._collection

    @property
    def is_initialized(self) -> bool:
        return self._collection is not None

    # --- Ingestion primitives -------------------------------------------
    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:  # noqa: BLE001 - surfaced as RetrievalError
            raise RetrievalError(f"Chroma count failed: {exc}") from exc

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> int:
        """Insert or replace chunks by their deterministic id.

        ``vectors`` must be supplied: Chroma must never embed text itself.
        """
        if not chunks:
            return 0
        if vectors is None or len(vectors) != len(chunks):
            raise RetrievalError(
                "upsert_chunks requires one pre-computed embedding per chunk "
                "(all-MiniLM-L6-v2); refusing to let Chroma embed implicitly."
            )
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                _flatten_metadata(
                    chunk.metadata,
                    ordinal=chunk.ordinal,
                    heading_path=chunk.heading_path,
                )
            )
        try:
            self.collection.upsert(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Chroma upsert failed: {exc}") from exc
        return len(ids)

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to a document."""
        try:
            existing = self.collection.get(where={"document_id": document_id}, include=[])
            ids = existing.get("ids", []) or []
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Chroma delete failed: {exc}") from exc

    def get_document_hashes(self) -> dict[str, str]:
        """Map ``document_id -> content_hash`` for change detection."""
        try:
            data = self.collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Chroma read failed: {exc}") from exc
        hashes: dict[str, str] = {}
        metadatas = data.get("metadatas") or []
        for metadata in metadatas:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id", ""))
            if not document_id:
                continue
            hashes.setdefault(document_id, str(metadata.get("content_hash", "")))
        return hashes

    def document_ids(self) -> set[str]:
        try:
            data = self.collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Chroma read failed: {exc}") from exc
        result: set[str] = set()
        for metadata in data.get("metadatas") or []:
            if metadata and metadata.get("document_id"):
                result.add(str(metadata["document_id"]))
        return result

    def distinct(self, field: str) -> set[str]:
        try:
            data = self.collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Chroma read failed: {exc}") from exc
        return {
            str(metadata[field])
            for metadata in (data.get("metadatas") or [])
            if metadata and metadata.get(field)
        }

    # --- Query primitive -------------------------------------------------
    def query(
        self,
        embedding: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Similarity search returning hydrated hits with ``score = 1 - distance``."""
        if not embedding:
            raise RetrievalError("Cannot query Chroma with an empty embedding.")
        try:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=max(1, top_k),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Chroma query failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits: list[SearchHit] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))
            heading_path = [
                part.strip() for part in str(metadata.get("heading_path", "")).split(">") if part.strip()
            ]
            hits.append(
                SearchHit(
                    chunk_id=str(ids[index]) if index < len(ids) else "",
                    document_id=str(metadata.get("document_id", "")),
                    text=str(document or ""),
                    score=round(score, 4),
                    ordinal=int(metadata.get("ordinal", index) or 0),
                    heading_path=heading_path,
                    metadata=_rehydrate(metadata),
                )
            )
        return hits

    def reset(self) -> None:
        """Delete and recreate the collection. Used by ``scripts/rebuild_index.py``."""
        if self._client is None:
            self.initialize()
        assert self._client is not None
        try:
            self._client.delete_collection(self._settings.chroma_collection)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = None
        self.initialize()
        logger.info("Chroma collection '%s' reset", self._settings.chroma_collection)
