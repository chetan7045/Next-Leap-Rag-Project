"""Document loader: fetch configured public URLs safely.

Security posture:
  * only URLs from ``sources.json`` reach this module (the registry is the allow-list),
  * every URL is re-validated against SSRF rules before the request is made,
  * responses are size-capped, timeout-bounded, and retried with backoff,
  * unsupported content types are rejected rather than parsed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.core.config import Settings
from app.core.errors import SourceFetchError
from app.core.logging import get_logger
from app.core.security import validate_ingest_url

logger = get_logger("app.services.ingestion.loader")

ALLOWED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/xml",
    "application/xml",
)
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class FetchedDocument:
    url: str
    html: str
    content_type: str
    status_code: int
    retrieved_at: datetime
    bytes_len: int
    etag: str | None = None
    last_modified: str | None = None


class DocumentLoader:
    """HTTP loader for configured sources."""

    def __init__(self, settings: Settings, allowed_urls: set[str] | None = None) -> None:
        self._settings = settings
        self._allowed_urls = allowed_urls

    def _assert_allowed(self, url: str) -> None:
        if self._allowed_urls is not None and url not in self._allowed_urls:
            raise SourceFetchError(f"URL is not in the configured source allow-list: {url}")

    def fetch(self, url: str) -> FetchedDocument:
        """Download one configured URL with retries. Raises on any failure."""
        self._assert_allowed(url)
        try:
            safe_url = validate_ingest_url(url)
        except ValueError as exc:
            raise SourceFetchError(f"Refused to fetch {url}: {exc}") from exc

        settings = self._settings
        headers = {
            "User-Agent": settings.ingestion_user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-IN,en;q=0.9",
        }
        last_error: Exception | None = None

        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(settings.ingestion_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        ) as client:
            for attempt in range(1, settings.ingestion_max_attempts + 1):
                try:
                    with client.stream("GET", safe_url, headers=headers) as response:
                        if response.status_code in RETRYABLE_STATUS:
                            raise SourceFetchError(
                                f"HTTP {response.status_code} for {url}"
                            )
                        if response.status_code >= 400:
                            raise SourceFetchError(
                                f"HTTP {response.status_code} for {url}", public_message="A source returned an error."
                            )

                        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                        if content_type and not content_type.startswith(ALLOWED_CONTENT_TYPES):
                            raise SourceFetchError(
                                f"Unsupported content type {content_type!r} for {url}"
                            )

                        body, truncated = self._read_capped(response, url)
                        if truncated:
                            logger.warning("Response for %s hit the size cap; using the truncated body.", url)

                        return FetchedDocument(
                            url=safe_url,
                            html=body,
                            content_type=content_type or "text/html",
                            status_code=response.status_code,
                            retrieved_at=datetime.now(UTC),
                            bytes_len=len(body),
                            etag=response.headers.get("etag"),
                            last_modified=response.headers.get("last-modified"),
                        )
                except httpx.TimeoutException as exc:
                    last_error = exc
                except httpx.HTTPError as exc:
                    last_error = exc
                except SourceFetchError as exc:
                    last_error = exc
                    if attempt >= settings.ingestion_max_attempts:
                        break
                if attempt < settings.ingestion_max_attempts:
                    time.sleep(min(2.0 * attempt, 5.0))

        raise SourceFetchError(f"Failed to fetch {url}: {type(last_error).__name__ if last_error else 'unknown'}")

    def _read_capped(self, response: httpx.Response, url: str) -> tuple[str, bool]:
        limit = self._settings.ingestion_max_bytes
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            total += len(chunk)
            if total >= limit:
                break
        return b"".join(chunks).decode("utf-8", errors="replace"), total >= limit

    def fetch_all(self, urls: list[str]) -> dict[str, FetchedDocument | Exception]:
        """Fetch several URLs, collecting per-URL outcomes instead of aborting."""
        results: dict[str, FetchedDocument | Exception] = {}
        for url in urls:
            try:
                results[url] = self.fetch(url)
                logger.info("Fetched %s", url)
            except SourceFetchError as exc:
                results[url] = exc
                logger.error("Fetch failed for %s: %s", url, exc.detail)
        return results
