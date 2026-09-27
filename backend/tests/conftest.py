"""Shared pytest fixtures.

Every test runs against an ephemeral Chroma collection, a temporary data directory and
the offline ``fake`` LLM provider. No test may reach the network or the developer's
real index, and no test may spend Gemini quota.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
for path in (str(BACKEND_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.config import Settings, reset_settings_cache  # noqa: E402
from app.models.enums import DocumentType, SourceType  # noqa: E402
from app.models.retrieval import DocumentMetadata  # noqa: E402
from app.models.source import SourceRegistryFile  # noqa: E402
from app.services.embeddings.embedding_service import EmbeddingService  # noqa: E402
from app.services.llm.fake import FakeLLMProvider  # noqa: E402
from app.services.ingestion.registry import SourceRegistry  # noqa: E402
from app.services.rag.chroma_service import ChromaService  # noqa: E402

_COLLECTION_COUNTER = itertools.count(1)

EMBEDDING_DIMENSION = 384


@pytest.fixture(scope="session", autouse=True)
def _deterministic_hash_seed() -> None:
    import random

    random.seed(20260925)


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    """Ephemeral settings rooted in a temp directory."""
    reset_settings_cache()
    created = Settings(
        environment="test",
        llm_provider="fake",
        llm_model="fake-echo",
        chroma_mode="ephemeral",
        # Chroma's in-memory client is process-wide, so each test gets a unique
        # collection name; otherwise chunks seeded by one test leak into the next.
        chroma_collection=f"test_collection_{next(_COLLECTION_COUNTER)}",
        data_dir=tmp_path,
        similarity_threshold=0.45,
        log_level="WARNING",
    )
    yield created
    reset_settings_cache()


@pytest.fixture
def fake_llm(settings: Settings) -> FakeLLMProvider:
    return FakeLLMProvider(settings)


@pytest.fixture
def chroma(settings: Settings) -> Iterator[ChromaService]:
    service = ChromaService(settings)
    service.initialize()
    yield service


@pytest.fixture(scope="session")
def embeddings() -> EmbeddingService:
    """Real MiniLM encoder, loaded once for the session.

    Tests assert on 384-dimensional vectors, so the actual model is used rather than a
    stub. The download is cached under ``backend/.cache/huggingface``.
    """
    created = Settings(environment="test", chroma_mode="ephemeral")
    return EmbeddingService.get_instance(created)


@pytest.fixture
def registry_file() -> SourceRegistryFile:
    """The validated ``sources.json`` file model."""
    raw = (REPO_ROOT / "sources.json").read_text(encoding="utf-8")
    return SourceRegistryFile.model_validate_json(raw)


@pytest.fixture
def registry(registry_file: SourceRegistryFile) -> SourceRegistry:
    """The runtime registry the services depend on."""
    return SourceRegistry(registry_file, str(REPO_ROOT / "sources.json"))


def build_document(
    *,
    scheme_id: str = "HDFC_LARGE_CAP",
    scheme_name: str = "HDFC Large Cap Fund",
    source_url: str = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    source_title: str = "HDFC Large Cap Fund Direct Growth - NAV, Mutual Fund Performance & Portfolio",
    source_type: SourceType = SourceType.REFERENCE,
    authority_level: int = 3,
    document_type: DocumentType = DocumentType.SCHEME_PAGE,
    source_date: str = "2026-09-25",
    chunk_index: int = 0,
) -> DocumentMetadata:
    """Build chunk metadata for a known fact."""
    del chunk_index  # chunk position lives on the Chunk, not on provenance metadata
    return DocumentMetadata(
        document_id=f"doc-{scheme_id}",
        source_id=f"src-{scheme_id}",
        source_url=source_url,
        source_title=source_title,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        amc="HDFC Mutual Fund",
        source_type=source_type,
        document_type=document_type,
        authority_level=authority_level,
        published_at=source_date,
        last_updated_at=source_date,
        retrieved_at=source_date,
        content_hash=f"hash-{scheme_id}",
        publisher="Groww",
    )


@pytest.fixture
def make_chunk():
    from app.models.retrieval import Chunk

    def _factory(
        text: str,
        metadata: DocumentMetadata,
        ordinal: int = 0,
        chunk_id: str | None = None,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id or f"{metadata.scheme_id}-{ordinal}",
            document_id=metadata.document_id,
            ordinal=ordinal,
            text=text,
            heading_path=list(metadata.heading_path) if hasattr(metadata, "heading_path") else [],
            token_estimate=len(text.split()),
            metadata=metadata,
        )

    return _factory


@pytest.fixture
def seeded_chroma(chroma: ChromaService, embeddings: EmbeddingService, make_chunk):
    """Return a callable that indexes a set of facts and returns their total count."""

    def _seed(facts: dict[str, list[str]]) -> int:
        total = 0
        for scheme_id, texts in facts.items():
            chunks = []
            for index, text in enumerate(texts):
                metadata = build_document(
                    scheme_id=scheme_id,
                    scheme_name=SCHEME_NAMES.get(scheme_id, scheme_id),
                    source_url=SOURCE_URLS[scheme_id],
                    source_title=f"{SCHEME_NAMES.get(scheme_id, scheme_id)} - NAV, Mutual Fund Performance & Portfolio",
                    chunk_index=index,
                )
                chunks.append(make_chunk(text, metadata, ordinal=index))
            vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
            chroma.upsert_chunks(chunks, vectors)
            total += len(chunks)
        return total

    return _seed


SCHEME_NAMES = {
    "HDFC_LARGE_CAP": "HDFC Large Cap Fund",
    "HDFC_FLEXI_CAP": "HDFC Flexi Cap Fund",
    "HDFC_ELSS": "HDFC ELSS Tax Saver Fund",
    "HDFC_SMALL_CAP": "HDFC Small Cap Fund",
    "HDFC_BALANCED_ADVANTAGE": "HDFC Balanced Advantage Fund",
}

SOURCE_URLS = {
    "HDFC_LARGE_CAP": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "HDFC_FLEXI_CAP": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    "HDFC_ELSS": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "HDFC_SMALL_CAP": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "HDFC_BALANCED_ADVANTAGE": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
}
