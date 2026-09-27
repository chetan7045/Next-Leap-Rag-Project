"""LLM provider abstraction.

The rest of the application depends on :class:`LLMProvider` only. Swapping Gemini for
another provider means adding a subclass and a factory branch — no RAG code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class LLMRequest:
    """Everything a provider needs. Providers never see the original user message."""

    system_prompt: str
    question: str
    context: str
    max_output_tokens: int = 512
    temperature: float = 0.1


class LLMProvider(ABC):
    """Contract every LLM backend implements."""

    name: str = "base"

    @abstractmethod
    async def generate_answer(self, request: LLMRequest) -> str:
        """Return a grounded answer, or raise ``LLMProviderError``."""

    @abstractmethod
    def is_configured(self) -> bool:
        """True when the provider has everything it needs to run."""

    @property
    def model_name(self) -> str:  # pragma: no cover - trivial
        return self.name

    async def health(self) -> bool:
        """Cheap readiness probe that never spends meaningful quota."""
        return self.is_configured()
