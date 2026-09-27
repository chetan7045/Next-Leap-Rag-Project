"""LLM provider factory.

Adding a provider means implementing :class:`LLMProvider` and adding one branch here.
No RAG, route, or frontend code changes.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.services.llm.base import LLMProvider, LLMRequest
from app.services.llm.fake import FakeLLMProvider
from app.services.llm.gemini import GeminiProvider

logger = get_logger("app.services.llm.factory")

_REGISTRY: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    # Offline provider for tests and local development only. build_llm_provider()
    # rejects it when ENVIRONMENT=production.
    "fake": FakeLLMProvider,
}


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured provider. Raises on an unknown provider name."""
    name = settings.llm_provider.lower().strip()
    provider_cls = _REGISTRY.get(name)
    if provider_cls is None:
        raise ConfigurationError(
            f"Unknown LLM provider {name!r}. Available: {sorted(_REGISTRY)}"
        )
    if name == "fake" and settings.is_production:
        raise ConfigurationError(
            "The 'fake' LLM provider exists only for tests and local development "
            "and must not be used when ENVIRONMENT=production."
        )
    provider = provider_cls(settings)
    if not provider.is_configured():
        logger.error("LLM provider '%s' is configured but its credentials are missing.", name)
    else:
        logger.info("LLM configuration: loaded (provider=%s model=%s)", name, settings.llm_model)
    return provider


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["LLMProvider", "LLMRequest", "build_llm_provider", "available_providers"]
