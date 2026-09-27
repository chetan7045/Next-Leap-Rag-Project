"""Deterministic offline provider for tests and local development.

This exists so the suite can assert two things that matter for the product:

1. Refusal paths (advice, PII, ambiguous scheme, low confidence) return **without**
   ever calling the LLM.
2. Answer shaping, citation attachment and the post-generation validator behave the
   same way in tests as in production.

It is never a production option: :meth:`FakeLLMProvider.is_configured` returns False
unless the environment is local/development/test, and the factory refuses to build it
in production.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import LLMProviderError
from app.services.llm.base import LLMProvider, LLMRequest

_ALLOWED_ENVIRONMENTS = {"local", "development", "test"}

#: Returned when a test does not override :attr:`FakeLLMProvider.response`.
DEFAULT_RESPONSE = (
    "HDFC Large Cap Fund Direct Growth lists a minimum SIP of ₹100.\n\n"
    "Source: HDFC Large Cap Fund Direct Growth - NAV, Mutual Fund Performance & Portfolio"
)


class FakeLLMProvider(LLMProvider):
    """Records every request so tests can assert the LLM was (not) called."""

    name = "fake"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.response: str | None = None
        self.error: Exception | None = None
        self.calls: list[LLMRequest] = []

    @property
    def model_name(self) -> str:
        return "fake-echo"

    def is_configured(self) -> bool:
        return self._settings.environment in _ALLOWED_ENVIRONMENTS

    async def generate_answer(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        # Echo the first fact line out of the supplied context so grounding checks
        # behave realistically without a network call.
        for line in request.context.splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith(("#", "SOURCE")):
                return f"{candidate}\n\nSource: see the cited scheme page."
        raise LLMProviderError("FakeLLMProvider received an empty context.")

    async def health(self) -> bool:
        return self.is_configured()

    def reset(self) -> None:
        self.calls.clear()
        self.response = None
        self.error = None


__all__ = ["FakeLLMProvider", "DEFAULT_RESPONSE"]
