"""Retriever: query embedding → Chroma search → confidence gate."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.retrieval import RetrievalResult, SearchHit
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.rag.chroma_service import ChromaService
from app.services.rag.query_normalizer import normalize_for_retrieval

logger = get_logger("app.services.rag.retriever")


@dataclass(slots=True)
class RetrievalTiming:
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0


class Retriever:
    """Vector retrieval with scheme metadata filtering and a similarity gate."""

    def __init__(
        self,
        settings: Settings,
        chroma: ChromaService,
        embeddings: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._chroma = chroma
        self._embeddings = embeddings

    def retrieve(
        self,
        query: str,
        *,
        scheme_id: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> tuple[RetrievalResult, RetrievalTiming]:
        """Return ranked hits for ``query`` plus per-stage timings."""
        query = query.strip()
        if not query:
            return RetrievalResult(), RetrievalTiming()

        k = top_k or self._settings.top_k
        limit = self._settings.similarity_threshold if threshold is None else threshold
        candidates = max(k, self._settings.retrieve_candidates)

        where = {"scheme_id": scheme_id} if scheme_id else None

        # The corpus and the encoder are English-only, so a Hinglish or Devanagari
        # question can match nothing (or match the wrong chunk confidently). When
        # the question is not already English, search a translated variant too and
        # merge. English questions take the single-query path below unchanged.
        normalized = normalize_for_retrieval(query)

        start = time.perf_counter()
        embeddings = [self._embeddings.embed_query(query)]
        if normalized:
            embeddings.append(self._embeddings.embed_query(normalized))
        embedding_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        hits: list[SearchHit] = []
        for embedding in embeddings:
            hits = self._merge_hits(hits, self._chroma.query(embedding, top_k=candidates, where=where))
        search_ms = (time.perf_counter() - start) * 1000

        kept = self._apply_threshold(hits, limit)[:k]
        confidence = kept[0].score if kept else 0.0

        result = RetrievalResult(
            hits=kept,
            confidence=round(confidence, 4),
            candidates_considered=len(hits),
            scheme_id=scheme_id,
        )
        timing = RetrievalTiming(
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
            total_ms=round(embedding_ms + search_ms, 2),
        )
        logger.info(
            "Retrieval: candidates=%d kept=%d confidence=%.3f scheme=%s in %.0fms",
            len(hits),
            len(kept),
            confidence,
            scheme_id or "-",
            timing.total_ms,
        )
        return result, timing

    @staticmethod
    def _merge_hits(*groups: list[SearchHit]) -> list[SearchHit]:
        """Merge ranked hit lists, keeping each chunk's best score.

        The same chunk can be returned by both the original and the translated
        query, so results are de-duplicated on ``chunk_id``. Text is not a safe
        key here: two chunks from different schemes can legitimately share
        identical boilerplate, and merging them would drop a real hit.
        """
        best: dict[str, SearchHit] = {}
        for group in groups:
            for hit in group:
                key = hit.chunk_id or hit.text
                existing = best.get(key)
                if existing is None or hit.score > existing.score:
                    best[key] = hit
        return sorted(best.values(), key=lambda h: h.score, reverse=True)

    @staticmethod
    def _apply_threshold(hits: list[SearchHit], threshold: float) -> list[SearchHit]:
        return [hit for hit in hits if hit.score >= threshold]

    def is_confident(self, result: RetrievalResult, threshold: float | None = None) -> bool:
        limit = self._settings.similarity_threshold if threshold is None else threshold
        return bool(result.hits) and result.hits[0].score >= limit
