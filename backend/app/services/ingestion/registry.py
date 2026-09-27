"""Source registry: loads ``sources.json`` and exposes schemes + allowed sources.

The registry is the only component aware of which URLs may be ingested. Every layer
downstream (loader, chunker, citations) works from registry objects, so the corpus
can grow by editing configuration alone.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.models.enums import SourceType
from app.models.retrieval import SourceConfig
from app.models.source import SchemeRecord, SourceRegistryFile

logger = get_logger("app.services.ingestion.registry")


class SourceRegistry:
    """Validated view over ``sources.json``."""

    def __init__(self, file_model: SourceRegistryFile, path: str) -> None:
        self._file = file_model
        self._path = path
        self._schemes: dict[str, SchemeRecord] = {s.id: s for s in file_model.schemes}
        self._sources: dict[str, SourceConfig] = {s.id: s for s in file_model.sources}
        self._validate()

    # --- Loading ---------------------------------------------------------
    @classmethod
    def load(cls, settings: Settings) -> "SourceRegistry":
        path = settings.sources_path
        if not path.exists():
            raise ConfigurationError(f"Source configuration not found at {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Source configuration is not valid JSON: {exc}") from exc
        model = SourceRegistryFile.model_validate(raw)
        return cls(model, str(path))

    def _validate(self) -> None:
        if not self._schemes:
            raise ConfigurationError("Source configuration defines no schemes")
        if not self._sources:
            raise ConfigurationError("Source configuration defines no sources")

        seen_urls: set[str] = set()
        for source in self._sources.values():
            if source.scheme_id not in self._schemes:
                raise ConfigurationError(
                    f"Source '{source.id}' references unknown scheme_id '{source.scheme_id}'"
                )
            if source.url in seen_urls:
                raise ConfigurationError(f"Duplicate source URL in configuration: {source.url}")
            seen_urls.add(source.url)
        for scheme in self._schemes.values():
            if not any(s.scheme_id == scheme.id for s in self._sources.values()):
                logger.warning("Scheme %s has no configured source; it will not be retrievable.", scheme.id)

    # --- Queries ---------------------------------------------------------
    @property
    def version(self) -> str:
        return self._file.version

    @property
    def amc(self) -> str:
        return self._file.amc

    def schemes(self) -> list[SchemeRecord]:
        return list(self._schemes.values())

    def scheme(self, scheme_id: str) -> SchemeRecord | None:
        return self._schemes.get(scheme_id.upper())

    def has_scheme(self, scheme_id: str) -> bool:
        return scheme_id.upper() in self._schemes

    def sources(self, *, enabled_only: bool = False) -> list[SourceConfig]:
        items = list(self._sources.values())
        if enabled_only:
            items = [s for s in items if s.enabled]
        return items

    def source(self, source_id: str) -> SourceConfig | None:
        return self._sources.get(source_id)

    def sources_for_scheme(self, scheme_id: str) -> list[SourceConfig]:
        return [s for s in self._sources.values() if s.scheme_id == scheme_id.upper()]

    def allowed_urls(self) -> set[str]:
        return {s.url for s in self._sources.values() if s.enabled}

    def scheme_count(self) -> int:
        return len(self._schemes)

    def best_source_type_for(self, scheme_id: str) -> SourceType:
        sources = self.sources_for_scheme(scheme_id)
        if not sources:
            return SourceType.REFERENCE
        return min((s.source_type for s in sources), key=lambda t: t.authority_level)

    def has_only_reference_sources(self) -> bool:
        return all(s.source_type is SourceType.REFERENCE for s in self._sources.values())


@lru_cache(maxsize=4)
def _cached_registry(path: str, mtime: float) -> SourceRegistry:  # noqa: ARG001 - mtime in cache key
    return SourceRegistry.load(_settings_for_path(path))


def _settings_for_path(path: str) -> Settings:  # pragma: no cover - helper
    from app.core.config import get_settings

    settings = get_settings()
    settings.sources_file = path
    return settings


def get_registry(settings: Settings | None = None) -> SourceRegistry:
    """Return the registry for the configured sources file (cached per file mtime)."""
    from app.core.config import get_settings

    active = settings or get_settings()
    path = active.sources_path
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _cached_registry(str(path), mtime)
