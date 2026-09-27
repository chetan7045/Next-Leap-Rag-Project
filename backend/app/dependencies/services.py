"""Service singletons.

One embedding model, one Chroma client/collection, and one LLM client per process.
Nothing expensive is constructed per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.ingestion.registry import SourceRegistry, get_registry
from app.services.llm.base import LLMProvider
from app.services.llm.factory import build_llm_provider
from app.services.rag.chroma_service import ChromaService
from app.services.rag.retriever import Retriever
from app.services.rag.service import RAGService

logger = get_logger("app.dependencies")


@dataclass(slots=True)
class ServiceContainer:
    """Everything the routes may use."""

    settings: Settings
    registry: SourceRegistry
    embeddings: EmbeddingService
    chroma: ChromaService
    llm: LLMProvider
    retriever: Retriever
    rag: RAGService
    started_at: float

    def index_ready(self) -> bool:
        try:
            return self.chroma.is_initialized and self.chroma.count() > 0
        except Exception:  # noqa: BLE001 - health must never raise
            return False


_container: ServiceContainer | None = None


def build_container(settings: Settings) -> ServiceContainer:
    """Construct the container. The embedding model is loaded lazily on first use."""
    import time

    registry = get_registry(settings)
    embeddings = EmbeddingService.get_instance(settings)
    chroma = ChromaService(settings)
    chroma.initialize()
    llm = build_llm_provider(settings)
    retriever = Retriever(settings, chroma, embeddings)
    rag = RAGService(settings, registry, retriever, llm, chroma)

    logger.info(
        "Services ready: schemes=%d sources=%d collection=%s mode=%s",
        registry.scheme_count(),
        len(registry.sources()),
        settings.chroma_collection,
        settings.chroma_mode,
    )
    return ServiceContainer(
        settings=settings,
        registry=registry,
        embeddings=embeddings,
        chroma=chroma,
        llm=llm,
        retriever=retriever,
        rag=rag,
        started_at=time.time(),
    )


def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = build_container(get_settings())
    return _container


def set_container(container: ServiceContainer | None) -> None:
    """Inject or clear the container (used by tests)."""
    global _container
    _container = container


def reset_container() -> None:
    set_container(None)
    EmbeddingService.reset_instance()


@lru_cache(maxsize=1)
def get_registry_singleton() -> SourceRegistry:
    return get_registry()


def container_as_dict(container: ServiceContainer) -> dict[str, Any]:  # pragma: no cover - debug helper
    return {
        "environment": container.settings.environment,
        "provider": container.llm.name,
        "model": container.settings.llm_model,
        "chroma_mode": container.settings.chroma_mode,
        "embedding_model": container.settings.embedding_model,
    }
