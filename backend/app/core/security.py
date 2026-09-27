"""HTTP security controls: CORS, payload limits, request identity, URL allow-list."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.core.config import Settings
from app.core.logging import get_logger, set_request_id

logger = get_logger("app.core.security")

_ALLOWED_URL_SCHEMES = {"https"}
_ALLOWED_URL_PORTS = {443, 8443}
# Hosts that must never be reached, regardless of configuration.
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
    "instance-data",
}
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localdomain", ".cluster.local")


class PayloadTooLargeError(Exception):
    pass


def build_cors_origins(settings: Settings) -> list[str]:
    """Explicit origin list. Never a wildcard in production."""
    origins = [origin.rstrip("/") for origin in settings.allowed_origin_list if origin.strip()]
    if settings.is_production and any(origin == "*" for origin in origins):
        # Fail closed rather than shipping a public, credential-leaking API.
        logger.warning("Wildcard CORS origin rejected in production; falling back to an empty allow-list.")
        return []
    return origins


def normalize_origin(origin: str) -> str:
    parsed = urlparse(origin.strip())
    if not parsed.scheme or not parsed.netloc:
        return origin.strip().rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def new_request_id() -> str:
    return uuid4().hex[:12]


def _is_public_address(host: str) -> bool:
    """True when the host resolves only to public addresses."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return False
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def validate_ingest_url(url: str) -> str:
    """Validate a configured ingestion URL against SSRF and scheme rules.

    Only URLs present in ``sources.json`` ever reach this function; it is the second
    line of defence, not the allow-list itself.

    Raises:
        ValueError: when the URL is not safe to fetch.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"URL scheme must be https, got {parsed.scheme or 'none'!r}")
    if not parsed.hostname:
        raise ValueError("URL is missing a hostname")
    if parsed.username or parsed.password:
        raise ValueError("URL must not embed credentials")
    if parsed.port and parsed.port not in _ALLOWED_URL_PORTS:
        raise ValueError(f"URL port {parsed.port} is not permitted")

    host = parsed.hostname.lower()
    if host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise ValueError(f"Host {host} is not permitted")
    try:
        if ipaddress.ip_address(host):
            raise ValueError("Direct IP literals are not permitted as ingestion sources")
    except ValueError as exc:
        if "Direct IP literals" in str(exc):
            raise
    if not _is_public_address(host):
        raise ValueError(f"Host {host} does not resolve exclusively to public addresses")
    return parsed.geturl()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach baseline security headers and a per-request id."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or new_request_id()
        set_request_id(request_id)
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies before they are parsed."""

    def __init__(self, app, max_bytes: int) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method in {"POST", "PUT", "PATCH"}:
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > self._max_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Request payload is too large."},
                        )
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})
        return await call_next(request)
