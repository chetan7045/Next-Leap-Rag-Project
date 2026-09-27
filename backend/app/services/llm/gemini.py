"""Gemini provider.

API-surface choice: this is a single-turn, system-instructed text completion, so the
provider uses ``client.aio.models.generate_content`` with ``system_instruction``. The
newer Interactions API targets multi-step/agentic work and adds an abstraction this
use case does not need. The choice is isolated here so switching later touches no RAG
code.

The API key is read from settings and used only in this process. It is never logged,
returned, or exposed to the frontend.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.core.config import Settings
from app.core.errors import LLMProviderError
from app.core.logging import get_logger, redact
from app.services.llm.base import LLMProvider, LLMRequest

if TYPE_CHECKING:  # pragma: no cover
    from google import genai

logger = get_logger("app.services.llm.gemini")


class GeminiProvider(LLMProvider):
    """Google Gemini implementation of :class:`LLMProvider`."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._model = settings.llm_model

    # --- Configuration ---------------------------------------------------
    def is_configured(self) -> bool:
        return bool(self._settings.gemini_api_key.strip())

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.is_configured():
            raise LLMProviderError(
                "LLM provider 'gemini' is configured but GEMINI_API_KEY is missing."
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMProviderError("google-genai is not installed on the server.") from exc

        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        logger.info("LLM configuration: loaded (provider=%s model=%s)", self.name, self._model)
        return self._client

    # --- Generation ------------------------------------------------------
    async def generate_answer(self, request: LLMRequest) -> str:
        client = self._get_client()

        user_prompt = (
            f"{request.context}\n\n"
            f"<<<QUESTION>>>\n{request.question}\n<<<END QUESTION>>>\n\n"
            "Answer the question using only the SOURCE blocks above."
        )

        config = self._build_config(request)

        last_error: Exception | None = None
        for attempt in range(1, self._settings.llm_max_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self._model,
                        contents=user_prompt,
                        config=config,
                    ),
                    timeout=self._settings.llm_timeout_seconds,
                )
                text = self._extract_text(response)
                if not text:
                    raise LLMProviderError("Gemini returned an empty response.")
                return text.strip()
            except LLMProviderError as exc:
                if "empty response" not in exc.detail:
                    raise
                last_error = exc
            except asyncio.TimeoutError as exc:
                last_error = LLMProviderError(f"Gemini request timed out after {self._settings.llm_timeout_seconds}s")
                logger.warning("LLM attempt %d timed out", attempt)
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                mapped = self._map_error(exc)
                if mapped.retryable and attempt < self._settings.llm_max_attempts:
                    last_error = mapped
                    await asyncio.sleep(min(2.0 * attempt, 4.0))
                    continue
                raise mapped from exc

        raise last_error or LLMProviderError("Gemini generation failed.")

    def _build_config(self, request: LLMRequest) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            # No tools: grounding, search, and code execution are intentionally disabled
            # so the model can only use the supplied context.
            tools=None,
            candidate_count=1,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        parts: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                value = getattr(part, "text", None)
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts).strip()

    def _map_error(self, exc: Exception) -> LLMProviderError:
        """Translate SDK errors into safe, actionable provider errors."""
        raw_status = getattr(exc, "code", None)
        status = getattr(raw_status, "value", raw_status)
        try:
            status_int = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_int = None
        name = type(exc).__name__
        message = redact(str(exc)) or name
        detail_text = message.lower()

        if status_int in (401, 403) or "api key" in detail_text or "permission_denied" in detail_text:
            logger.error("Gemini rejected the API key (status=%s).", status_int)
            return LLMProviderError(
                "Gemini rejected the configured API key.",
                public_message="We couldn't generate an answer right now. Please try again.",
            )
        if status_int == 402 or "resource_exhausted" in name.lower() or "billing" in detail_text:
            # Quota/billing is an operational issue, not a code issue.
            logger.error("Gemini reported no remaining quota or credits (status=%s).", status_int)
            return LLMProviderError(
                "Gemini reported exhausted quota or depleted credits for the configured project.",
                public_message="We couldn't generate an answer right now. Please try again.",
            )
        if status_int == 429 or "rate limit" in detail_text or "resource_exhausted" in name.lower():
            return LLMProviderError("Gemini rate limit reached.", retryable=True)
        if status_int == 404 or "not_found" in name.lower():
            logger.error("Gemini model %s was not found.", self._model)
            return LLMProviderError(
                f"Configured Gemini model {self._model!r} is unavailable.",
                public_message="We couldn't generate an answer right now. Please try again.",
            )
        if status_int is not None and status_int >= 500:
            return LLMProviderError(f"Gemini server error (status={status_int}).", retryable=True)
        logger.error("Gemini call failed: %s %s", name, message[:300])
        return LLMProviderError(f"Gemini call failed: {name}", retryable=False)
