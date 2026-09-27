"""Domain exceptions and their HTTP mapping.

Routes convert these into safe, user-facing messages. Raw tracebacks never reach a client.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, handled failures."""

    status_code: int = 500
    public_message: str = "Something went wrong. Please try again."

    def __init__(self, detail: str | None = None, *, public_message: str | None = None) -> None:
        super().__init__(detail or self.__class__.__name__)
        self.detail = detail or self.__class__.__name__
        if public_message:
            self.public_message = public_message

    def to_public(self) -> str:
        return self.public_message


class ConfigurationError(AppError):
    status_code = 500
    public_message = "The service is not configured correctly."


class SourceFetchError(AppError):
    """A configured source could not be downloaded or was not usable."""

    status_code = 502
    public_message = "A configured source could not be retrieved."


class IngestionError(AppError):
    """The ingestion pipeline could not complete."""

    status_code = 500
    public_message = "Source ingestion failed."


class RetrievalError(AppError):
    """Vector retrieval failed."""

    status_code = 503
    public_message = "Retrieval is temporarily unavailable."


class LLMProviderError(AppError):
    """The configured LLM provider failed, was unreachable, or is not configured."""

    status_code = 503
    public_message = "We couldn't generate an answer right now. Please try again."

    def __init__(
        self,
        detail: str | None = None,
        *,
        public_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(detail, public_message=public_message)
        self.retryable = retryable


class CitationValidationError(AppError):
    """An answer could not be tied to at least one valid indexed source."""

    status_code = 200
    public_message = "I couldn't find a sufficiently supported answer in the available sources."


class UnsafeRequestError(AppError):
    """The request was refused by a safety control (advice, performance, PII)."""

    status_code = 200

    def __init__(self, message: str) -> None:
        super().__init__(message, public_message=message)
