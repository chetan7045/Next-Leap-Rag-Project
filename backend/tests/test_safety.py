"""Safety guarantees.

The product contract is that advice, forecasts, PII and ambiguous-scheme questions are
answered deterministically **without calling the LLM**. Every test here asserts both the
response shape and ``fake_llm.calls == []``.
"""

from __future__ import annotations

import pytest

from app.models.chat import ChatResponse
from app.models.enums import CitationStatus, RefusalReason, ResponseType
from app.services.llm.fake import FakeLLMProvider
from app.services.rag.retriever import Retriever
from app.services.rag.service import RAGService


@pytest.fixture
def rag(settings, registry, chroma, embeddings, fake_llm) -> RAGService:
    retriever = Retriever(settings, chroma, embeddings)
    return RAGService(settings, registry, retriever, fake_llm, chroma)


async def test_advice_is_refused_without_llm(rag: RAGService, fake_llm: FakeLLMProvider):
    response, metrics = await rag.answer("Should I invest in HDFC Large Cap Fund?")

    assert response.refusal_reason is RefusalReason.ADVICE_REQUEST
    assert response.answer_type is ResponseType.REFUSAL
    assert response.sources == []
    assert fake_llm.calls == []
    assert metrics.llm_called is False


async def test_performance_promise_is_refused(rag: RAGService, fake_llm: FakeLLMProvider):
    response, _ = await rag.answer("Which HDFC fund will give me the highest return?")

    assert response.refusal_reason is RefusalReason.PERFORMANCE_PROMISE
    assert fake_llm.calls == []


async def test_forecast_is_refused(rag: RAGService, fake_llm: FakeLLMProvider):
    response, _ = await rag.answer("What will HDFC Small Cap Fund's NAV be in 2030?")

    assert response.refusal_reason is RefusalReason.PERFORMANCE_PROMISE
    assert fake_llm.calls == []


async def test_pii_is_refused_without_llm(rag: RAGService, fake_llm: FakeLLMProvider):
    response, _ = await rag.answer(
        "My PAN is ABCDE1234F and my phone is 9876543210. "
        "What is the minimum SIP for HDFC Large Cap Fund?"
    )

    assert response.refusal_reason is RefusalReason.PII_DETECTED
    assert response.answer_type is ResponseType.REFUSAL
    assert "ABCDE1234F" not in response.answer
    assert "9876543210" not in response.answer
    assert fake_llm.calls == []


async def test_ambiguous_scheme_asks_for_clarification(
    rag: RAGService, seeded_chroma, fake_llm: FakeLLMProvider
):
    """Several schemes match equally well, so the bot must ask rather than guess."""
    seeded_chroma(
        {
            "HDFC_LARGE_CAP": ["Exit load: 1% if redeemed within 1 year, nil thereafter."],
            "HDFC_ELSS": ["Exit load: nil. Lock-in period of 3 years applies."],
        }
    )

    response, _ = await rag.answer("What is the exit load?")

    assert response.answer_type is ResponseType.CLARIFICATION
    assert response.refusal_reason is RefusalReason.AMBIGUOUS_SCHEME
    assert response.clarification is True
    assert response.sources == []
    assert fake_llm.calls == []


async def test_clarification_lists_the_supported_schemes(
    rag: RAGService, seeded_chroma, fake_llm: FakeLLMProvider
):
    seeded_chroma(
        {
            "HDFC_LARGE_CAP": ["Exit load: 1% if redeemed within 1 year, nil thereafter."],
            "HDFC_ELSS": ["Exit load: nil. Lock-in period of 3 years applies."],
        }
    )

    response, _ = await rag.answer("What is the exit load?")

    options = {option["id"] for option in response.clarification_options}
    assert options
    assert options <= {scheme.id for scheme in rag._registry.schemes()}
    assert "?" in response.answer


async def test_empty_index_returns_no_index_status(rag: RAGService, fake_llm: FakeLLMProvider):
    response, _ = await rag.answer("What is the expense ratio of HDFC Large Cap Fund?")

    assert response.answer_type is ResponseType.NO_CONTEXT
    assert response.refusal_reason is RefusalReason.NO_INDEX
    assert response.sources == []
    assert fake_llm.calls == []


async def test_every_refusal_carries_the_disclaimer(rag: RAGService):
    response, _ = await rag.answer("Should I buy HDFC Flexi Cap Fund?")

    assert response.disclaimer
    assert "not" in response.disclaimer.lower()


async def test_citations_are_never_fabricated(rag: RAGService, fake_llm: FakeLLMProvider):
    """A refusal must not carry sources, even if retrieval would have found some."""
    fake_llm.response = "HDFC Large Cap Fund will definitely outperform every other fund."

    response, _ = await rag.answer("Tell me if HDFC Large Cap Fund is the best fund to buy")

    assert response.answer_type is ResponseType.REFUSAL
    assert response.sources == []
    assert response.confidence == 0.0
    assert fake_llm.calls == []


async def test_low_confidence_avoids_the_llm(
    rag: RAGService, seeded_chroma, fake_llm: FakeLLMProvider
):
    """Evidence exists, but nothing resembles the question: no LLM call."""
    seeded_chroma({"HDFC_LARGE_CAP": ["Minimum SIP amount is ₹100 per month."]})

    response, _ = await rag.answer("What is the polar ice cap melting rate?")

    assert response.answer_type is ResponseType.NO_CONTEXT
    assert response.refusal_reason is RefusalReason.LOW_CONFIDENCE
    assert response.sources == []
    assert fake_llm.calls == []


def test_legacy_flags_are_derived_not_set_by_hand() -> None:
    """``refusal``/``clarification`` must always agree with ``answer_type``."""
    refusal = ChatResponse(answer="No advice.", answer_type=ResponseType.REFUSAL)
    assert refusal.refusal is True
    assert refusal.clarification is False
    assert refusal.citation_status is CitationStatus.NOT_APPLICABLE

    clarification = ChatResponse(answer="Which scheme?", answer_type=ResponseType.CLARIFICATION)
    assert clarification.clarification is True
    assert clarification.refusal is False

    answer = ChatResponse(answer="Min. SIP is ₹100.", answer_type=ResponseType.ANSWER)
    assert answer.refusal is False
    assert answer.clarification is False
