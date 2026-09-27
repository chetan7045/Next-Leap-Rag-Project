"""Structured logging with secret redaction and request-scoped context."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from typing import Any

_SENSITIVE_KEYS = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|credential|cookie)", re.IGNORECASE
)
# Gemini keys and Google-style tokens, redacted even if they appear inside free text.
_KEY_LIKE = re.compile(r"\b(?:AIza|AQ)[A-Za-z0-9_\-]{10,}")

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_query_ref: contextvars.ContextVar[str] = contextvars.ContextVar("query_ref", default="-")

#: Literal secrets registered at startup. Pattern matching alone is not enough: a key
#: that does not look like a key would otherwise reach a log line or a traceback.
_literal_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register a literal secret so it is scrubbed from every rendered log line."""
    candidate = (value or "").strip()
    if len(candidate) >= 8:
        _literal_secrets.add(candidate)


def scrub(text: str) -> str:
    """Mask registered secrets and key-shaped strings in already-rendered output."""
    for secret in _literal_secrets:
        if secret in text:
            text = text.replace(secret, "***redacted***")
    return _KEY_LIKE.sub("***redacted***", text)


def redact(value: Any) -> Any:
    """Recursively mask secret-looking values before they reach a log sink."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_KEYS.search(key):
                out[key] = "***redacted***"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _KEY_LIKE.sub("***redacted***", value)
    return value


def set_request_id(value: str) -> None:
    _request_id.set(value)


def set_query_ref(value: str) -> None:
    """Store a non-reversible short reference for a query, never the query text."""
    _query_ref.set(value)


class ScrubbingFormatter(logging.Formatter):
    """Scrub the fully rendered record, which also covers args and tracebacks.

    Redacting the message alone would miss a key embedded in an exception raised by a
    third-party SDK, or interpolated into a ``%s`` argument.
    """

    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


class JsonFormatter(ScrubbingFormatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": scrub(record.getMessage()),
            "request_id": _request_id.get(),
        }
        query_ref = _query_ref.get()
        if query_ref != "-":
            payload["query_ref"] = query_ref
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload.update(redact(extra))
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            ScrubbingFormatter(
                fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "sentence_transformers",
        "httpx._client",
        "google",
        "google_genai",
        "google.generativeai",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Log an event with a redacted structured context payload."""
    logger.log(level, message, extra={"context": context})
