"""GET /health — cheap liveness. No LLM call, no embedding work."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_services
from app.dependencies.services import ServiceContainer
from app.models.chat import HealthResponse, LLMHealth

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and configuration health")
async def health(services: ServiceContainer = Depends(get_services)) -> HealthResponse:
    """Public, unauthenticated liveness check.

    Reports only whether the deployment is usable. The LLM provider name and model ID
    are not returned: this endpoint needs no credential, so it should not help anyone
    fingerprint the stack behind it.
    """
    return HealthResponse(
        status="ok",
        service=services.settings.service_name,
        version="1.0.0",
        environment=services.settings.environment,
        chroma_mode=services.settings.chroma_mode,
        index_ready=services.index_ready(),
        llm=LLMHealth(configured=services.llm.is_configured()),
    )
