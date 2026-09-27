"""HTTP contract tests.

The frontend is written against these responses, so the shape of every endpoint is part
of the public contract. These tests use the real ASGI app with the offline LLM provider.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies.services import ServiceContainer, set_container
from app.main import create_app
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.llm.fake import FakeLLMProvider
from app.services.rag.chroma_service import ChromaService
from app.services.rag.retriever import Retriever
from app.services.rag.service import RAGService


@pytest.fixture
def container(settings, registry, chroma, embeddings, fake_llm) -> ServiceContainer:
    retriever = Retriever(settings, chroma, embeddings)
    built = ServiceContainer(
        settings=settings,
        registry=registry,
        embeddings=embeddings,
        chroma=chroma,
        llm=fake_llm,
        retriever=retriever,
        rag=RAGService(settings, registry, retriever, fake_llm, chroma),
        started_at=0.0,
    )
    set_container(built)
    yield built
    set_container(None)


@pytest.fixture
async def client(container) -> AsyncClient:
    app = create_app()
    # raise_app_exceptions=False so the app's own exception handler is exercised and
    # the client sees the real 500 response instead of the raised error.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def test_health_reports_ok(client: AsyncClient):
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"]
    assert "gemini_api_key" not in body


async def test_health_reports_llm_configuration_without_the_key(client: AsyncClient):
    """The frontend needs to know a credential is present, must never see the credential
    itself, and must not learn which model is behind the API."""
    body = (await client.get("/health")).json()

    assert body["llm"]["configured"] is True
    assert "key" not in json.dumps(body["llm"]).lower().replace('"llm"', "")
    assert "provider" not in body["llm"]
    assert "model" not in body["llm"]


async def test_health_does_not_leak_the_model_name(client: AsyncClient):
    """/health is unauthenticated, so it must not fingerprint the stack."""
    body = (await client.get("/health")).json()

    assert "gemini" not in json.dumps(body).lower()
    assert "gemini" not in json.dumps(body["llm"]).lower()


async def test_health_never_calls_the_llm(client: AsyncClient, container):
    await client.get("/health")

    assert container.llm.calls == []


async def test_schemes_endpoint_lists_every_configured_scheme(client: AsyncClient):
    response = await client.get("/api/v1/schemes")

    assert response.status_code == 200
    body = response.json()
    assert len(body["schemes"]) == 5
    ids = {scheme["id"] for scheme in body["schemes"]}
    assert "HDFC_LARGE_CAP" in ids
    assert "HDFC_ELSS" in ids
    for scheme in body["schemes"]:
        assert scheme["name"]
        assert isinstance(scheme["indexed"], bool)


async def test_indexed_schemes_endpoint_returns_200(client: AsyncClient):
    """Regression: this route referenced an undefined local and returned 500.

    It is part of the public API, so any caller got a generic 500 instead of the
    scheme list. The error handler hid the cause, which is how it survived.
    """
    response = await client.get("/api/v1/schemes/indexed")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["schemes"])


async def test_indexed_schemes_only_lists_schemes_present_in_the_index(
    client: AsyncClient, seeded_chroma
):
    """The list is driven by the vector index, not by the registry alone."""
    empty = (await client.get("/api/v1/schemes/indexed")).json()
    assert empty["schemes"] == []
    assert empty["count"] == 0

    seeded_chroma({"HDFC_ELSS": ["Lock-in period is 3 years."]})

    seeded = (await client.get("/api/v1/schemes/indexed")).json()
    assert [scheme["id"] for scheme in seeded["schemes"]] == ["HDFC_ELSS"]
    assert seeded["count"] == 1
    assert seeded["schemes"][0]["indexed"] is True


async def test_sources_endpoint_never_claims_official_provenance(client: AsyncClient):
    response = await client.get("/api/v1/sources")

    assert response.status_code == 200
    for source in response.json()["sources"]:
        assert source["source_type"] != "AMC_OFFICIAL"
        assert source["url"].startswith("https://")


async def test_chat_rejects_an_empty_message(client: AsyncClient):
    response = await client.post("/api/v1/chat", json={"message": ""})

    assert response.status_code == 422
    assert "detail" in response.json()


async def test_chat_rejects_an_overlong_message(client: AsyncClient):
    response = await client.post("/api/v1/chat", json={"message": "x" * 5000})

    assert response.status_code == 422


async def test_chat_refuses_advice_with_a_typed_body(client: AsyncClient, container):
    response = await client.post("/api/v1/chat", json={"message": "Should I invest in HDFC ELSS?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_type"] == "REFUSAL"
    assert body["refusal"] is True
    assert body["refusal_reason"] == "ADVICE_REQUEST"
    assert body["sources"] == []
    assert body["disclaimer"]
    assert container.llm.calls == []


async def test_chat_returns_a_structured_clarification(
    client: AsyncClient, container, seeded_chroma
):
    seeded_chroma(
        {
            "HDFC_LARGE_CAP": ["Exit load: 1% if redeemed within 1 year, nil thereafter."],
            "HDFC_ELSS": ["Exit load: nil. Lock-in period of 3 years applies."],
        }
    )

    response = await client.post("/api/v1/chat", json={"message": "What is the exit load?"})

    body = response.json()
    assert body["answer_type"] == "CLARIFICATION"
    assert body["clarification"] is True
    assert body["clarification_options"]
    assert container.llm.calls == []


async def test_chat_answers_with_backend_citations(
    client: AsyncClient, container, seeded_chroma
):
    seeded_chroma(
        {
            "HDFC_LARGE_CAP": [
                "HDFC Large Cap Fund Direct Growth All | Min. for SIP ₹100 "
                "Expense ratio 1.03% NAV ₹1,189.08"
            ]
        }
    )
    container.llm.response = "The minimum SIP for HDFC Large Cap Fund is ₹100."

    response = await client.post(
        "/api/v1/chat", json={"message": "What is the minimum SIP for HDFC Large Cap Fund?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_type"] == "ANSWER"
    assert body["citation_status"] == "VERIFIED"
    assert body["sources"]
    assert body["sources"][0]["url"].startswith("https://")
    assert body["last_updated"]
    assert len(container.llm.calls) == 1


async def test_chat_accepts_an_explicit_scheme_selection(
    client: AsyncClient, container, seeded_chroma
):
    seeded_chroma({"HDFC_ELSS": ["ELSS • 3Y Lock-in. Minimum SIP ₹500."]})

    response = await client.post(
        "/api/v1/chat",
        json={"message": "What is the lock-in period?", "scheme_id": "HDFC_ELSS"},
    )

    assert response.status_code == 200
    assert response.json()["answer_type"] in {"ANSWER", "NO_CONTEXT"}


async def test_chat_rejects_an_unknown_scheme_id(client: AsyncClient):
    response = await client.post(
        "/api/v1/chat", json={"message": "What is the NAV?", "scheme_id": "NOT_A_SCHEME"}
    )

    assert response.status_code in {400, 422}


async def test_error_body_never_leaks_a_traceback(client: AsyncClient, container, monkeypatch):
    """An unexpected crash must surface as a safe 500, not a traceback or the key."""

    async def boom(*args, **kwargs):
        raise RuntimeError("internal detail: /etc/passwd AIzaSySECRET")

    monkeypatch.setattr(container.rag, "answer", boom, raising=True)
    response = await client.post("/api/v1/chat", json={"message": "What is the NAV?"})

    assert response.status_code >= 400
    body = response.text
    assert "AIzaSySECRET" not in body
    assert "Traceback" not in body
    assert "/etc/passwd" not in body


async def test_security_headers_are_present(client: AsyncClient):
    response = await client.get("/health")

    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options")
    assert "X-Request-Id" in response.headers


async def test_oversized_payload_is_rejected_before_routing(client: AsyncClient):
    response = await client.post("/api/v1/chat", content=b"{\"message\": \"" + b"a" * 200_000)

    assert response.status_code in {413, 422}
