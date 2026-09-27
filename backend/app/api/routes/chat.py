"""POST /api/v1/chat — the only route that reaches retrieval and the LLM.

Deliberately thin: validate, delegate to :class:`RAGService`, serialise. No retrieval
logic, no prompting, and no chunking happen here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.logging import get_logger, log_event
from app.dependencies import get_services
from app.dependencies.services import ServiceContainer
from app.models.chat import ChatRequest, ChatResponse, ErrorResponse, RetrievalDebug
from app.models.enums import QueryType

logger = get_logger("app.api.chat")

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Ask a factual question about an indexed HDFC Mutual Fund scheme",
)
async def chat(
    payload: ChatRequest,
    request: Request,
    services: ServiceContainer = Depends(get_services),
) -> ChatResponse:
    if payload.scheme_id and not services.registry.has_scheme(payload.scheme_id):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail=f"Unknown scheme_id '{payload.scheme_id}'.",
        )

    response, metrics = await services.rag.answer(payload.message, payload.scheme_id)

    log_event(
        logger,
        20,
        "chat request",
        request_id=getattr(request.state, "request_id", "-"),
        query_type=str(metrics.query_type),
        scheme=metrics.detected_scheme,
        chunks_considered=metrics.chunks_considered,
        chunks_used=metrics.chunks_used,
        confidence=metrics.confidence,
        llm_called=metrics.llm_called,
        sources=metrics.source_ids,
        retrieval_ms=metrics.retrieval_ms,
        llm_ms=metrics.llm_ms,
        total_ms=metrics.total_ms,
        cited=len(response.sources),
        problems=metrics.problems,
    )

    response.request_id = getattr(request.state, "request_id", "")

    # Retrieval diagnostics are development-only.
    if services.settings.debug_rag and not services.settings.is_production:
        response.retrieval = RetrievalDebug(
            chunks_used=metrics.chunks_used,
            confidence=metrics.confidence,
            threshold=metrics.threshold,
            candidates=metrics.chunks_considered,
            detected_scheme=metrics.detected_scheme,
            source_ids=metrics.source_ids,
            retrieval_ms=metrics.retrieval_ms,
            llm_ms=metrics.llm_ms,
        )
    else:
        response.retrieval = None

    if response.query_type is QueryType.CLARIFICATION:
        response.last_updated = "Date not available"
    return response
