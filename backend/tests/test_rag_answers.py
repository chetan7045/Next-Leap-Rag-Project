"""The happy path: a factual question must produce a grounded, cited answer.

Also covers the guarantees that make the answers trustworthy — sources come from the
backend, dates come from the source (never from "now"), and the Gemini key never leaks.
"""

from __future__ import annotations

import pytest

from app.models.enums import CitationStatus, RefusalReason, ResponseType, SourceType
from app.services.llm.fake import FakeLLMProvider
from app.services.rag.retriever import Retriever
from app.services.rag.service import RAGService

LARGE_CAP_FACTS = {
    "HDFC_LARGE_CAP": [
        "HDFC Large Cap Fund Direct Growth All | NAV: 25 Sep '26 ₹1,189.08 "
        "Min. for SIP ₹100 Fund size (AUM) ₹39,933.37 Cr Expense ratio 1.03% Rating 4",
        "Exit load: 1% if units are redeemed within 1 year of allotment, nil thereafter.",
        "Benchmark: NIFTY 100 Total Return Index.",
    ]
}


@pytest.fixture
def rag(settings, registry, chroma, embeddings, fake_llm) -> RAGService:
    retriever = Retriever(settings, chroma, embeddings)
    return RAGService(settings, registry, retriever, fake_llm, chroma)


async def test_factual_question_answers_with_citations(
    rag: RAGService, seeded_chroma, fake_llm: FakeLLMProvider
):
    seeded_chroma(LARGE_CAP_FACTS)
    fake_llm.response = "The minimum SIP for HDFC Large Cap Fund Direct Growth is ₹100."

    response, metrics = await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    assert response.answer_type is ResponseType.ANSWER
    assert response.citation_status is CitationStatus.VERIFIED
    assert "₹100" in response.answer
    assert response.sources, "a grounded answer must carry at least one source"
    assert metrics.llm_called is True
    assert len(fake_llm.calls) == 1


async def test_citations_are_backend_generated(rag: RAGService, seeded_chroma, fake_llm):
    seeded_chroma(LARGE_CAP_FACTS)
    # The model tries to cite something the backend never gave it.
    fake_llm.response = (
        "The minimum SIP is ₹100.\n\n"
        "Source: https://example.com/not-a-real-source\n"
        "Source: https://sebi.gov.in/whatever"
    )

    response, _ = await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    urls = {source.url for source in response.sources}
    assert urls, "backend must still attach the real source"
    assert "https://example.com/not-a-real-source" not in urls
    assert "https://sebi.gov.in/whatever" not in urls
    for source in response.sources:
        assert source.source_type is SourceType.REFERENCE
        assert source.url.startswith("https://groww.in/")


async def test_source_date_comes_from_the_source_not_today(
    rag: RAGService, seeded_chroma, fake_llm
):
    seeded_chroma(LARGE_CAP_FACTS)
    fake_llm.response = "The minimum SIP for HDFC Large Cap Fund is ₹100."

    response, _ = await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    assert response.last_updated
    assert "2026" in response.last_updated
    for source in response.sources:
        assert source.url and source.title
        assert source.source_type is SourceType.REFERENCE


async def test_every_source_carries_provenance(rag: RAGService, seeded_chroma, fake_llm):
    seeded_chroma(LARGE_CAP_FACTS)
    fake_llm.response = "The minimum SIP for HDFC Large Cap Fund is ₹100."

    response, _ = await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    for source in response.sources:
        assert source.scheme_id
        assert source.scheme_name
        assert source.source_type is SourceType.REFERENCE
        assert source.authority_level == 3
        assert source.url.startswith("https://")
        assert source.title.strip()


async def test_prompt_forbids_advice_even_if_evidence_looks_supportive(
    rag: RAGService, seeded_chroma, fake_llm: FakeLLMProvider
):
    """A factual question must not become advice just because the model volunteers it."""
    seeded_chroma(LARGE_CAP_FACTS)
    fake_llm.response = "You should definitely invest in HDFC Large Cap Fund; it is the best choice."

    response, _ = await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    assert response.answer_type is ResponseType.NO_CONTEXT
    assert response.refusal_reason is RefusalReason.VALIDATION_FAILED
    assert response.sources == []
    assert fake_llm.calls, "the validator runs after generation, so one call is expected"


async def test_context_contains_the_question_and_sources(
    rag: RAGService, seeded_chroma, fake_llm: FakeLLMProvider
):
    seeded_chroma(LARGE_CAP_FACTS)

    await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    request = fake_llm.calls[0]
    assert "minimum sip" in request.question.lower()
    assert "HDFC Large Cap Fund" in request.question
    assert request.context.strip()
    assert "SOURCE" in request.context.upper()
    assert "groww.in" in request.context


async def test_llm_failure_degrades_to_a_typed_error(
    rag: RAGService, seeded_chroma, fake_llm: FakeLLMProvider
):
    from app.core.errors import LLMProviderError

    seeded_chroma(LARGE_CAP_FACTS)
    fake_llm.error = LLMProviderError("upstream 500", public_message="Something went wrong.")

    response, metrics = await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    assert response.answer_type is ResponseType.ERROR
    assert response.refusal_reason is RefusalReason.LLM_ERROR
    assert response.sources == []
    assert metrics.llm_called is True


async def test_answer_never_echoes_the_api_key(rag: RAGService, seeded_chroma, settings):
    from app.core.config import Settings

    secret = "AIzaSyTESTKEY_must_never_appear_in_output"
    created = Settings(
        environment="test",
        chroma_mode="ephemeral",
        chroma_collection=f"leak_check_{secret[:6]}",
        gemini_api_key=secret,
    )
    assert created.gemini_api_key == secret

    seeded_chroma(LARGE_CAP_FACTS)
    response, _ = await rag.answer("What is the minimum SIP for HDFC Large Cap Fund?")

    payload = response.model_dump_json()
    assert secret not in payload
    assert "AIza" not in payload
