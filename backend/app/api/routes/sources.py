"""GET /api/v1/sources — indexed source provenance.

Exposes titles, scheme, source type, document type, URL, and dates. Never exposes
embeddings, vectors, internal prompts, or configuration secrets.
"""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends

from app.dependencies import get_services
from app.dependencies.services import ServiceContainer
from app.models.chat import SourceListResponse, SourceOut
from app.models.retrieval import DocumentMetadata
from app.services.ingestion.metadata import resolve_effective_date

router = APIRouter(prefix="/api/v1/sources", tags=["sources"])


@router.get("", response_model=SourceListResponse, summary="List configured and indexed sources")
async def list_sources(services: ServiceContainer = Depends(get_services)) -> SourceListResponse:
    try:
        indexed_documents = services.chroma.document_ids()
        indexed_schemes = services.chroma.distinct("scheme_id")
    except Exception:  # noqa: BLE001 - empty index is a valid state
        indexed_documents, indexed_schemes = set(), set()

    records: dict[str, DocumentMetadata] = {}
    chunk_counts: Counter[str] = Counter()
    try:
        raw = services.chroma.collection.get(include=["metadatas"])
        from app.services.rag.chroma_service import _rehydrate

        for metadata in raw.get("metadatas") or []:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id", ""))
            if not document_id:
                continue
            chunk_counts[document_id] += 1
            if document_id not in records:
                records[document_id] = _rehydrate(metadata)
    except Exception:  # noqa: BLE001
        records, chunk_counts = {}, Counter()

    outputs: list[SourceOut] = []
    for source in services.registry.sources():
        from app.services.ingestion.metadata import make_document_id

        document_id = make_document_id(source.scheme_id, source.url)
        indexed_record = records.get(document_id)
        outputs.append(
            SourceOut(
                id=source.id,
                title=indexed_record.source_title if indexed_record else (source.title or source.scheme_name),
                url=source.url,
                scheme_id=source.scheme_id,
                scheme_name=source.scheme_name,
                amc=source.amc,
                source_type=source.source_type,
                source_type_label=source.source_type.ui_label,
                document_type=source.document_type,
                publisher=source.publisher,
                last_updated=resolve_effective_date(indexed_record) if indexed_record else None,
                published_at=indexed_record.published_at if indexed_record else None,
                retrieved_at=indexed_record.retrieved_at if indexed_record else None,
                chunk_count=chunk_counts.get(document_id, 0),
                indexed=document_id in indexed_documents,
            )
        )

    return SourceListResponse(
        sources=outputs,
        count=len(outputs),
        schemes_indexed=len(indexed_schemes),
    )
