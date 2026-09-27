"""GET /api/v1/schemes — the configured scheme list.

The frontend drives its selector and its "N schemes indexed" badge from this endpoint
rather than hard-coding scheme data.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_services
from app.dependencies.services import ServiceContainer
from app.models.chat import SchemeListResponse, SchemeOut

router = APIRouter(prefix="/api/v1/schemes", tags=["schemes"])


@router.get("", response_model=SchemeListResponse, summary="List configured schemes")
async def list_schemes(services: ServiceContainer = Depends(get_services)) -> SchemeListResponse:
    schemes = services.registry.schemes()
    try:
        indexed_schemes = services.chroma.distinct("scheme_id")
    except Exception:  # noqa: BLE001 - a cold index must not break this endpoint
        indexed_schemes = set()

    return SchemeListResponse(
        schemes=[
            SchemeOut(
                id=scheme.id,
                name=scheme.name,
                plan=scheme.plan,
                amc=scheme.amc,
                categories=scheme.categories,
                aliases=scheme.aliases,
                source_count=len(services.registry.sources_for_scheme(scheme.id)),
                indexed=scheme.id in indexed_schemes,
            )
            for scheme in schemes
        ],
        count=len(schemes),
    )


@router.get("/indexed", response_model=SchemeListResponse, summary="List schemes present in the vector index")
async def list_indexed_schemes(services: ServiceContainer = Depends(get_services)) -> SchemeListResponse:
    """Schemes that actually have chunks in Chroma, for an index-accurate count."""
    indexed = services.chroma.distinct("scheme_id")
    known = {s.id for s in services.registry.schemes()}
    return SchemeListResponse(
        schemes=[
            SchemeOut(
                id=scheme.id,
                name=scheme.name,
                plan=scheme.plan,
                amc=scheme.amc,
                categories=scheme.categories,
                source_count=len(services.registry.sources_for_scheme(scheme.id)),
                indexed=scheme.id in indexed,
            )
            for scheme in services.registry.schemes()
            if scheme.id in indexed and scheme.id in known
        ],
        count=len(indexed & known),
    )
