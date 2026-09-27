"""RAG orchestration.

The single entry point for answering a question:

    validation → safety classification → scheme resolution → embedding → Chroma
    → confidence gate → context → LLM → citations → response validation

Deterministic refusals (advice, performance, PII) and deterministic "not found"
answers never reach the LLM.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.errors import LLMProviderError
from app.core.logging import get_logger, set_query_ref
from app.models.chat import ChatResponse, SourceCitation
from app.models.enums import (
    CitationStatus,
    QueryType,
    RefusalReason,
    ResponseType,
)
from app.models.retrieval import RetrievalResult
from app.services.ingestion.registry import SourceRegistry
from app.services.llm.base import LLMProvider, LLMRequest
from app.services.rag.citation_builder import DATE_UNAVAILABLE, CitationBuilder
from app.services.rag.context_builder import BuiltContext, build_context
from app.services.rag.prompt import DISCLAIMER, SYSTEM_PROMPT
from app.services.rag.retriever import Retriever
from app.services.rag.validator import validate_and_repair
from app.services.safety.classifier import classify
from app.services.safety.scheme_resolver import resolve_scheme

logger = get_logger("app.services.rag.service")

#: Maps a pre-LLM classification onto the machine-readable reason sent to the client.
_REFUSAL_REASONS: dict[QueryType, RefusalReason] = {
    QueryType.ADVICE: RefusalReason.ADVICE_REQUEST,
    QueryType.PERFORMANCE: RefusalReason.PERFORMANCE_PROMISE,
    QueryType.OUT_OF_SCOPE: RefusalReason.OUT_OF_SCOPE,
    QueryType.PII_RISK: RefusalReason.PII_DETECTED,
    QueryType.CLARIFICATION: RefusalReason.AMBIGUOUS_SCHEME,
}

NOT_FOUND_ANSWER = "I couldn't find that information in the available sources."
UNSUPPORTED_ANSWER = (
    "I couldn't find a sufficiently supported answer in the available sources."
)
GENERATION_ERROR_ANSWER = "We couldn't generate an answer right now. Please try again."
NO_INDEX_ANSWER = (
    "The source index is empty right now. Please run the ingestion step, then try again."
)
UNVERIFIED_REFUSAL = (
    "This assistant shares facts from indexed public sources. It doesn't provide investment "
    "advice, recommendations, or return predictions."
)


@dataclass(slots=True)
class RAGMetrics:
    """Per-request timings and counts. Carries no query text and no PII."""

    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    chunks_considered: int = 0
    chunks_used: int = 0
    confidence: float = 0.0
    threshold: float = 0.0
    detected_scheme: str | None = None
    source_ids: list[str] = field(default_factory=list)
    query_type: str = QueryType.FACTUAL
    llm_called: bool = False
    problems: list[str] = field(default_factory=list)


class RAGService:
    """Answers a validated :class:`ChatRequest` into a :class:`ChatResponse`."""

    def __init__(
        self,
        settings: Settings,
        registry: SourceRegistry,
        retriever: Retriever,
        llm: LLMProvider,
        chroma: Any,  # ChromaService, typed loosely to avoid a circular import
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._retriever = retriever
        self._llm = llm
        self._chroma = chroma
        self._citations = CitationBuilder(registry)

    # --- Public API ------------------------------------------------------
    async def answer(self, message: str, scheme_id: str | None = None) -> tuple[ChatResponse, RAGMetrics]:
        """Answer one question. Never raises for expected failure modes."""
        started = time.perf_counter()
        metrics = RAGMetrics(threshold=self._settings.similarity_threshold)

        # A short, non-reversible reference for correlating logs. The message itself is
        # never logged or stored.
        set_query_ref(hashlib.sha256(message.encode("utf-8")).hexdigest()[:10])

        response = await self._answer_inner(message, scheme_id, metrics)
        metrics.total_ms = round((time.perf_counter() - started) * 1000, 2)
        response.disclaimer = DISCLAIMER
        return response, metrics

    async def _answer_inner(
        self, message: str, scheme_id: str | None, metrics: RAGMetrics
    ) -> ChatResponse:
        # 1. Safety classification — deterministic, before any retrieval or LLM work.
        classification = classify(message)
        metrics.query_type = str(classification.query_type)
        if classification.query_type is not QueryType.FACTUAL and classification.canned_answer:
            logger.info("Refused query as %s (%s)", classification.query_type, classification.reason)
            return ChatResponse(
                answer=classification.canned_answer,
                sources=[],
                last_updated=DATE_UNAVAILABLE,
                query_type=classification.query_type,
                answer_type=ResponseType.REFUSAL,
                refusal_reason=_REFUSAL_REASONS.get(
                    classification.query_type, RefusalReason.OUT_OF_SCOPE
                ),
            )

        # 2. Index readiness.
        try:
            if self._chroma.count() == 0:
                logger.warning("Chroma index is empty; answering without retrieval.")
                return ChatResponse(
                    answer=NO_INDEX_ANSWER,
                    sources=[],
                    last_updated=DATE_UNAVAILABLE,
                    query_type=classification.query_type,
                    answer_type=ResponseType.NO_CONTEXT,
                    refusal_reason=RefusalReason.NO_INDEX,
                )
        except Exception as exc:  # noqa: BLE001 - never fail closed on a health check
            logger.error("Index readiness check failed: %s", exc)

        # 3. Provisional retrieval (unfiltered) so the scheme resolver has a signal.
        provisional, timing = self._retriever.retrieve(message)
        metrics.retrieval_ms += timing.total_ms
        metrics.chunks_considered = provisional.candidates_considered
        metrics.confidence = provisional.confidence

        # 4. Scheme resolution — explicit selection wins, then aliases, then consensus.
        resolution = resolve_scheme(message, self._registry, scheme_id, provisional)
        metrics.detected_scheme = resolution.scheme_id
        if resolution.ambiguous:
            return self._clarification_response(resolution, classification.query_type)

        # 5. Scheme-scoped retrieval when a scheme is known.
        result = provisional
        if resolution.scheme_id:
            scoped, scoped_timing = self._retriever.retrieve(
                message, scheme_id=resolution.scheme_id
            )
            metrics.retrieval_ms += scoped_timing.total_ms
            result = scoped
            if not scoped.hits:
                # A scheme filter must not silently discard the only evidence we have.
                logger.info(
                    "Scheme %s filter returned nothing; falling back to unfiltered hits.",
                    resolution.scheme_id,
                )
                result = RetrievalResult(
                    hits=provisional.hits,
                    confidence=provisional.confidence,
                    candidates_considered=provisional.candidates_considered,
                    scheme_id=resolution.scheme_id,
                )
            result.scheme_id = resolution.scheme_id

        metrics.confidence = result.confidence

        # 6. Confidence gate — the LLM is never asked to guess from weak evidence.
        if not result.hits or result.hits[0].score < self._settings.similarity_threshold:
            logger.info(
                "Below confidence threshold (%.3f < %.3f); returning not-found.",
                result.confidence,
                self._settings.similarity_threshold,
            )
            return ChatResponse(
                answer=NOT_FOUND_ANSWER,
                sources=[],
                last_updated=DATE_UNAVAILABLE,
                query_type=classification.query_type,
                answer_type=ResponseType.NO_CONTEXT,
                refusal_reason=RefusalReason.LOW_CONFIDENCE,
                confidence=round(result.confidence, 3),
            )

        # 7. Context construction.
        context = build_context(result, message, self._settings)
        metrics.chunks_used = len(context.used_hits)
        metrics.source_ids = list(context.source_ids)

        # 8. LLM generation.
        if not self._llm.is_configured():
            logger.error("LLM provider is not configured; cannot generate a grounded answer.")
            return ChatResponse(
                answer=GENERATION_ERROR_ANSWER,
                sources=[],
                last_updated=DATE_UNAVAILABLE,
                query_type=classification.query_type,
                answer_type=ResponseType.ERROR,
                refusal_reason=RefusalReason.LLM_UNAVAILABLE,
            )

        request = LLMRequest(
            system_prompt=SYSTEM_PROMPT,
            question=self._compose_question(
                message, resolution.scheme.name if resolution.scheme else None
            ),
            context=context.text,
            max_output_tokens=self._settings.llm_max_output_tokens,
            temperature=self._settings.llm_temperature,
        )

        llm_started = time.perf_counter()
        metrics.llm_called = True
        try:
            raw = await self._llm.generate_answer(request)
        except LLMProviderError as exc:
            metrics.llm_ms = round((time.perf_counter() - llm_started) * 1000, 2)
            logger.error("LLM generation failed: %s", exc.detail)
            return ChatResponse(
                answer=exc.public_message,
                sources=[],
                last_updated=DATE_UNAVAILABLE,
                query_type=classification.query_type,
                answer_type=ResponseType.ERROR,
                refusal_reason=RefusalReason.LLM_ERROR,
            )
        metrics.llm_ms = round((time.perf_counter() - llm_started) * 1000, 2)

        # 9. Validation and repair.
        validated = validate_and_repair(raw)
        metrics.problems.extend(validated.problems)
        if not validated.acceptable:
            logger.warning("Answer rejected by validation: %s", validated.problems)
            return ChatResponse(
                answer=UNSUPPORTED_ANSWER,
                sources=[],
                last_updated=DATE_UNAVAILABLE,
                query_type=classification.query_type,
                answer_type=ResponseType.NO_CONTEXT,
                refusal_reason=RefusalReason.VALIDATION_FAILED,
            )

        # 10. Backend-generated citations.
        citations = self._citations.build(context.used_hits, limit=3)
        try:
            self._citations.validate(validated.text, citations)
        except Exception as exc:  # noqa: BLE001
            logger.error("Citation validation failed: %s", exc)
            return ChatResponse(
                answer=UNSUPPORTED_ANSWER,
                sources=[],
                last_updated=DATE_UNAVAILABLE,
                query_type=classification.query_type,
                answer_type=ResponseType.NO_CONTEXT,
                refusal_reason=RefusalReason.VALIDATION_FAILED,
            )

        last_updated = self._citations.resolve_last_updated(citations)
        return ChatResponse(
            answer=validated.text,
            sources=citations,
            last_updated=last_updated,
            query_type=classification.query_type,
            answer_type=ResponseType.ANSWER,
            citation_status=CitationStatus.VERIFIED,
            confidence=round(result.confidence, 3),
        )

    # --- Helpers ---------------------------------------------------------
    @staticmethod
    def _compose_question(message: str, scheme_name: str | None) -> str:
        """State the resolved scheme explicitly so the model cannot hedge across funds."""
        if not scheme_name:
            return message
        return f'{message}\n\n(Resolved scheme for this question: {scheme_name}.)'

    def _clarification_response(self, resolution, query_type: QueryType) -> ChatResponse:
        options = [
            {"id": scheme.id, "name": scheme.name}
            for scheme in self._registry.schemes()
            if scheme.id in set(resolution.candidates) or resolution.candidates == ()
        ]
        names = ", ".join(option["name"] for option in options) or "the available schemes"
        return ChatResponse(
            answer=(
                f"Which HDFC Mutual Fund scheme would you like that for? "
                f"Available schemes: {names}."
            ),
            sources=[],
            last_updated=DATE_UNAVAILABLE,
            query_type=QueryType.CLARIFICATION,
            answer_type=ResponseType.CLARIFICATION,
            refusal_reason=RefusalReason.AMBIGUOUS_SCHEME,
            clarification_options=options,
        )
