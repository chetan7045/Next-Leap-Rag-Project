"""Security controls.

The threat model is small and specific: an SSRF via the ingestion loader, a leaked
secret in a log line or an API response, and a source that claims more authority than
the registry allows.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest

from app.core.config import Settings
from app.core.errors import SourceFetchError
from app.core.logging import (
    configure_logging,
    get_logger,
    log_event,
    redact,
    register_secret,
)
from app.core.security import build_cors_origin_regex, build_cors_origins, normalize_origin, validate_ingest_url
from app.models.enums import SourceType
from app.services.ingestion.loader import DocumentLoader
from app.services.ingestion.metadata import build_metadata

SECRET = "AIzaSyFAKEKEY_abcdefghijklmnopqrstuvwxyz123456"


# --- SSRF ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/admin",
        "http://localhost/admin",
        "https://169.254.169.254/latest/meta-data/",
        "https://10.0.0.1/internal",
        "https://192.168.1.1/router",
        "https://[::1]/admin",
        "https://metadata.google.internal/computeMetadata/v1/",
    ],
)
def test_private_and_metadata_addresses_are_refused(url: str):
    with pytest.raises(Exception):
        validate_ingest_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "https://www.hdfcmf.com/fund/equity",
    ],
)
def test_public_https_urls_are_allowed(url: str, monkeypatch):
    # DNS is stubbed so the test asserts the URL policy, not the sandbox's resolver.
    monkeypatch.setattr(
        "app.core.security._is_public_address",
        lambda host: True,
    )
    validate_ingest_url(url)


def test_non_https_schemes_are_refused():
    with pytest.raises(Exception):
        validate_ingest_url("http://groww.in/mutual-funds/x")


def test_disallowed_ports_are_refused():
    for url in ("https://groww.in:8080/x", "https://groww.in:22/x"):
        with pytest.raises(Exception):
            validate_ingest_url(url)


def test_loader_enforces_the_allow_list(settings: Settings, registry):
    loader = DocumentLoader(settings, allowed_urls=registry.allowed_urls())

    with pytest.raises(SourceFetchError):
        loader.fetch("https://evil.example.com/x")


# --- Secrets ------------------------------------------------------------------
def test_logs_never_contain_the_api_key(caplog: pytest.LogCaptureFixture, settings: Settings):
    created = Settings(
        environment="test",
        chroma_mode="ephemeral",
        chroma_collection="log_leak_check",
        gemini_api_key=SECRET,
    )
    configure_logging("DEBUG")
    logger = get_logger("test.secret")

    with caplog.at_level(logging.DEBUG):
        logger.info(
            "model=%s key=%s auth=%s",
            created.llm_model,
            created.gemini_api_key,
            f"Bearer {SECRET}",
        )
        logger.debug("full settings: %s", created.model_dump_json())

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert SECRET not in rendered
    assert "AIza" not in rendered
    assert created.gemini_api_key == SECRET, "redaction must not mutate the real value"


def test_settings_dump_never_contains_the_key():
    """Defense in depth: the key is excluded from any serialisation of settings."""
    created = Settings(environment="test", gemini_api_key=SECRET)

    assert created.gemini_api_key == SECRET
    assert SECRET not in created.model_dump_json()
    assert SECRET not in json.dumps(created.model_dump(mode="json"))
    assert SECRET not in repr(created)
    assert "gemini_api_key" not in created.model_dump()


def test_health_payload_has_no_credentials():
    from app.models.chat import HealthResponse, LLMHealth

    body = HealthResponse(
        status="ok",
        service="hdfc-mutual-fund-rag-api",
        version="1.0.0",
        environment="test",
        chroma_mode="ephemeral",
        index_ready=False,
        llm=LLMHealth(configured=True),
    )
    rendered = body.model_dump_json()

    assert "AIza" not in rendered
    assert "api_key" not in rendered.lower()
    # The new LLM block must carry no credential field of any kind.
    assert "AQ." not in rendered


# --- CORS ---------------------------------------------------------------------
def test_cors_allows_configured_origins_only():
    created = Settings(environment="test", allowed_origins="http://localhost:3000,https://facts.example.com")
    allowed = build_cors_origins(created)

    assert "http://localhost:3000" in allowed
    assert "https://facts.example.com" in allowed
    assert len(allowed) == 2


def test_cors_never_widens_to_every_origin():
    created = Settings(environment="test", allowed_origins="https://facts.example.com")
    allowed = build_cors_origins(created)

    assert "*" not in allowed
    assert allowed == ["https://facts.example.com"]


def test_origin_normalisation_strips_a_trailing_slash():
    assert normalize_origin("http://localhost:3000/") == "http://localhost:3000"


# --- CORS origin suffixes -----------------------------------------------------
def test_cors_suffix_regex_is_absent_when_unconfigured():
    """Without a suffix the exact-match allow-list is the only rule."""
    assert build_cors_origin_regex(Settings(environment="test")) is None


def test_cors_suffix_matches_only_https_subdomains():
    """Render serves every service from <service>.onrender.com, so a deployment can
    accept its own frontend without the operator knowing the generated hostname."""
    settings = Settings(environment="production", allowed_origin_suffixes=".onrender.com")
    pattern = build_cors_origin_regex(settings)
    assert pattern is not None

    assert re.match(pattern, "https://niva-web.onrender.com")
    assert re.match(pattern, "https://niva-api.onrender.com")


@pytest.mark.parametrize(
    "origin",
    [
        "https://niva-web.evil.com",  # different domain entirely
        "https://niva-web.onrender.com.evil.com",  # suffix appears but is not the end
        "http://niva-web.onrender.com",  # plaintext must not be accepted
        "https://onrender.com",  # bare suffix is not a subdomain of itself
        "https://niva-web.onrender.com:443",  # ports are not part of an Origin
    ],
)
def test_cors_suffix_rejects_lookalike_origins(origin: str):
    settings = Settings(environment="production", allowed_origin_suffixes="onrender.com")
    pattern = build_cors_origin_regex(settings)
    assert not re.match(pattern, origin)


def test_cors_suffix_ignores_entries_that_are_not_hostname_suffixes():
    """A scheme, path, port or wildcard in the config must not widen the match."""
    created = Settings(
        environment="production",
        allowed_origin_suffixes="https://*.onrender.com,*evil,.example.com/../etc",
    )
    assert created.allowed_origin_suffix_list == []


# --- Source integrity ---------------------------------------------------------
def test_registry_never_declares_official_sources(registry):
    for source in registry.sources():
        assert source.source_type is SourceType.REFERENCE
        assert source.source_type is not SourceType.AMC_OFFICIAL


def test_metadata_authority_follows_the_configured_source(settings: Settings, registry):
    source = registry.source("hdfc-large-cap-scheme-page")

    metadata = build_metadata(
        source,
        title="T",
        text="Min. for SIP ₹100",
        retrieved_at=datetime(2026, 9, 25, tzinfo=timezone.utc),
        settings=settings,
    )

    assert metadata.authority_level == source.source_type.authority_level
    assert metadata.authority_level == 3


def test_every_configured_url_is_https_and_allow_listed(registry):
    """Structural checks only: validate_ingest_url resolves DNS, which must not be
    required for the suite to pass."""
    for url in registry.allowed_urls():
        assert url.startswith("https://")
        assert url in registry.allowed_urls()
        parsed = urlparse(url)
        assert parsed.hostname and not parsed.username and not parsed.password


# --- Secret redaction ---------------------------------------------------------
#: Synthetic, key-shaped stand-in so the scrubber is exercised. Never a real key.
_SECRET = "AQ." + "NotARealKey" * 3 + "x"


def test_registered_secret_is_scrubbed_from_plain_text_logs(capsys):
    """The configured key is scrubbed even when it never reaches a log call site."""
    register_secret(_SECRET)
    configure_logging("INFO", json_output=False)

    logging.getLogger("app.test").info("connecting with key %s", _SECRET)

    out = capsys.readouterr().out
    assert _SECRET not in out
    assert "***redacted***" in out


def test_registered_secret_is_scrubbed_from_json_logs(capsys):
    register_secret(_SECRET)
    configure_logging("INFO", json_output=True)

    log_event(
        logging.getLogger("app.test"),
        logging.INFO,
        "llm call",
        endpoint="https://generativelanguage.googleapis.com",
        api_key=_SECRET,
    )

    out = capsys.readouterr().out
    assert _SECRET not in out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["api_key"] == "***redacted***"


def test_secret_in_an_exception_traceback_is_scrubbed(capsys):
    """A third-party SDK can raise an error containing the key in its message."""
    register_secret(_SECRET)
    configure_logging("INFO", json_output=False)
    logger = logging.getLogger("app.test")

    try:
        raise RuntimeError(f"gemini rejected key {_SECRET}")
    except RuntimeError:
        logger.exception("llm call failed")

    out = capsys.readouterr().out
    assert _SECRET not in out
    assert "gemini rejected key ***redacted***" in out


def test_unregistered_key_shaped_strings_are_still_scrubbed():
    assert redact(f"token AIza{'B' * 30}") == "token ***redacted***"
    assert redact({"password": "hunter2"}) == {"password": "***redacted***"}
