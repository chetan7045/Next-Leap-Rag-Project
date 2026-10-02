"""Two guarantees that were previously missing:

1. A visible answer is never cut off partway through a sentence.
   ``gemini-3.8-flash`` is a thinking model and ``max_output_tokens`` is a budget
   shared between its internal reasoning and the answer, so a small budget spent
   the whole allowance on thinking and returned a sentence sliced mid-word.

2. The assistant answers in the language the user used.
   Detection is generic — script block plus high-frequency function words — so
   it holds for questions that are not in any of these tests.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.enums import DocumentType, SourceType
from app.models.retrieval import DocumentMetadata, SearchHit

from app.core.config import Settings
from app.core.errors import LLMProviderError
from app.services.llm.base import LLMRequest
from app.services.llm.gemini import (
    INCOMPLETE_ANSWER_MESSAGE,
    MAX_OUTPUT_TOKENS_CEILING,
    GeminiProvider,
)
from app.services.rag.language import (
    ENGLISH,
    HINGLISH,
    HINDI_SCRIPT,
    detect_language_style,
    language_instruction,
)
from app.services.rag.prompt import SYSTEM_PROMPT, build_system_prompt
from app.services.rag.query_normalizer import (
    devanagari_to_roman,
    normalize_for_retrieval,
)
from app.services.rag.retriever import Retriever
from app.services.rag.validator import validate_and_repair
from app.services.safety import guardrails

# --- Output budget ----------------------------------------------------


def test_default_output_budget_leaves_room_for_thinking() -> None:
    """A thinking model needs its budget split between reasoning and the answer."""
    field_default = Settings.model_fields["llm_max_output_tokens"].default
    assert field_default >= 2048


def test_env_example_does_not_reintroduce_the_starved_budget() -> None:
    """The example env is what new deploys copy, so it must not pin a tiny budget."""
    example = (Path(__file__).resolve().parents[2] / "backend" / ".env.example").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^LLM_MAX_OUTPUT_TOKENS=(\d+)$", example, re.M)
    assert match, "LLM_MAX_OUTPUT_TOKENS must be documented in .env.example"
    assert int(match.group(1)) >= 2048


def test_llm_request_default_budget_is_not_starved() -> None:
    assert LLMRequest(system_prompt="", question="q", context="c").max_output_tokens >= 2048


# --- Gemini truncation handling ---------------------------------------


class _FakeResponse:
    """Minimal stand-in for a google-genai response."""

    def __init__(self, text: str, finish_reason: Any) -> None:
        self.text = text
        self.candidates = [
            SimpleNamespace(
                finish_reason=finish_reason,
                content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
            )
        ]


class _FakeModels:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.configs: list[Any] = []

    async def generate_content(self, *, model: str, contents: str, config: Any) -> _FakeResponse:
        self.configs.append(config)
        return self._responses.pop(0)


class _FakeGemini:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.aio = SimpleNamespace(models=_FakeModels(responses))


def _provider(settings: Settings, responses: list[_FakeResponse]) -> GeminiProvider:
    provider = GeminiProvider(settings)
    provider._client = _FakeGemini(responses)  # bypasses the API key check
    return provider


def _request(budget: int = 2048) -> LLMRequest:
    return LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        question="Who is the fund manager?",
        context="<<<SOURCE 1>>>\nDhruv Muchhal is the fund manager.\nSOURCE>>>",
        max_output_tokens=budget,
    )


def test_finish_reason_normalises_plain_string_and_enum() -> None:
    assert GeminiProvider._finish_reason(_FakeResponse("x", "STOP")) == "STOP"
    assert GeminiProvider._finish_reason(_FakeResponse("x", "MAX_TOKENS")) == "MAX_TOKENS"

    enum_like = SimpleNamespace(value="MAX_TOKENS")
    assert GeminiProvider._finish_reason(_FakeResponse("x", enum_like)) == "MAX_TOKENS"
    assert GeminiProvider._was_truncated(_FakeResponse("x", enum_like)) is True
    assert GeminiProvider._was_truncated(_FakeResponse("x", "STOP")) is False


async def test_truncated_response_escalates_budget_and_retries(settings: Settings) -> None:
    """The reported bug: a truncated answer must be retried, never returned."""
    provider = _provider(
        settings,
        [
            _FakeResponse("Dhruv Muchhal is the current fund manager of HDFC Small Cap", "MAX_TOKENS"),
            _FakeResponse(
                "Dhruv Muchhal is the current fund manager of HDFC Small Cap Fund.",
                "STOP",
            ),
        ],
    )

    answer = await provider.generate_answer(_request())

    assert answer.endswith("Fund.")
    configs = provider._client.aio.models.configs
    assert len(configs) == 2
    assert configs[0].max_output_tokens == 2048
    assert configs[1].max_output_tokens == 4096, "budget must grow on retry"


async def test_complete_response_is_returned_without_retry(settings: Settings) -> None:
    provider = _provider(settings, [_FakeResponse("Dhruv Muchhal is the fund manager.", "STOP")])

    assert await provider.generate_answer(_request()) == "Dhruv Muchhal is the fund manager."
    assert len(provider._client.aio.models.configs) == 1


async def test_persistent_truncation_fails_instead_of_returning_cut_off_text(
    settings: Settings,
) -> None:
    """Never show a mid-sentence answer, even after the budget has been raised."""
    provider = _provider(
        settings,
        [_FakeResponse("Dhruv Muchhal is the current fund manager of HDFC Small", "MAX_TOKENS")] * 4,
    )

    with pytest.raises(LLMProviderError) as excinfo:
        await provider.generate_answer(_request())

    assert excinfo.value.public_message == INCOMPLETE_ANSWER_MESSAGE
    assert "Dhruv" not in excinfo.value.public_message


async def test_escalation_never_exceeds_the_ceiling(settings: Settings) -> None:
    provider = _provider(
        settings,
        [_FakeResponse("Dhruv Muchhal is the current fund manager of HDFC Small Cap", "MAX_TOKENS")] * 6,
    )

    with pytest.raises(LLMProviderError):
        await provider.generate_answer(_request(budget=MAX_OUTPUT_TOKENS_CEILING))

    for cfg in provider._client.aio.models.configs:
        assert cfg.max_output_tokens <= MAX_OUTPUT_TOKENS_CEILING


# --- Truncation is visible --------------------------------------------


def test_validator_flags_an_answer_with_no_sentence_ending() -> None:
    """The truncation that shipped silently must now be logged."""
    validated = validate_and_repair("For HDFC Small Cap Fund, Dhruv Muchhal is listed as the")

    assert "incomplete_final_sentence" in validated.problems
    assert validated.acceptable, "the text is kept; we do not disguise the truncation"


def test_validator_does_not_flag_complete_answers() -> None:
    validated = validate_and_repair("Dhruv Muchhal is the fund manager of HDFC Small Cap Fund.")

    assert "incomplete_final_sentence" not in validated.problems
    assert validated.problems == []


def test_absence_answers_are_not_flagged_as_truncated() -> None:
    validated = validate_and_repair("I couldn't find that information in the available sources.")

    assert validated.reports_absence is True
    assert "incomplete_final_sentence" not in validated.problems


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("Dhruv Muchhal is the fund manager.", False),
        ('She said "the expense ratio is 1.03%".', False),
        ("Dhruv Muchhal is the fund manager of HDFC Small Cap Fund", True),
        ("Total expense ratio is 1.03%", True),
        # Devanagari answers end with a danda, which is a proper full stop.
        ("ध्रुव मुचल फंड मैनेजर हैं।", False),
        ("ध्रुव मुचल फंड मैनेजर हैं", True),
    ],
)
def test_ends_mid_sentence(text: str, expected: bool) -> None:
    assert guardrails.ends_mid_sentence(text) is expected


def test_devanagari_truncation_uses_a_danda() -> None:
    """A clipped Hindi answer must be closed with Hindi punctuation."""
    closed = guardrails.truncate_to_sentences("ध्रुव मुचल हैं। यह अनुपूर्ण", max_sentences=1)

    assert closed.endswith("।")
    assert guardrails.ends_mid_sentence(closed) is False


def test_english_truncation_keeps_a_full_stop() -> None:
    closed = guardrails.truncate_to_sentences("Dhruv Muchhal is the manager. This is cut", max_sentences=1)

    assert closed.endswith(".")


# --- Language detection ------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Who is the fund manager of HDFC Small Cap Fund?",
        "What is the expense ratio of HDFC Flexi Cap Fund?",
        "Is there a lock-in period?",
        "hdfc small cap fund",
        "NAV as of 25 Sep 2026?",
    ],
)
def test_english_questions_stay_english(question: str) -> None:
    assert detect_language_style(question) == ENGLISH


@pytest.mark.parametrize(
    "question",
    [
        "HDFC Small Cap Fund ka manager kaun hai?",
        "HDFC small cap fund ka manager kaun hai bhai?",
        "expense ratio kya hai",
        " batao HDFC Flexi Cap ka lock-in period kitna hai",
        "HDFC Large Cap aur HDFC Small Cap mein se kiska expense ratio kam hai",
        "fund manager kaun hai",
    ],
)
def test_hinglish_questions_are_detected(question: str) -> None:
    assert detect_language_style(question) == HINGLISH


@pytest.mark.parametrize(
    "question",
    [
        "एचडीएफसी स्मॉल कैप फंड का फंड मैनेजर कौन है?",
        "HDFC Flexi Cap का expense ratio क्या है?",
        "क्या इसमें lock-in period है?",
    ],
)
def test_devanagari_questions_are_detected(question: str) -> None:
    assert detect_language_style(question) == HINDI_SCRIPT


def test_single_strong_hinglish_marker_is_enough() -> None:
    assert detect_language_style("kaun?") == HINGLISH


def test_english_words_shared_with_hindi_need_two_occurrences() -> None:
    """A lone ambiguous token must not flip a plain English question."""
    assert detect_language_style("What is the expense ratio?") == ENGLISH


def test_language_instruction_matches_the_detected_style() -> None:
    assert "Devanagari" in language_instruction(HINDI_SCRIPT)
    assert "Hinglish" in language_instruction(HINGLISH)
    assert "English" in language_instruction(ENGLISH)
    # Roman-script Hindi must never be answered in Devanagari.
    assert "Do not switch to Devanagari" in language_instruction(HINGLISH)


# --- Prompt wiring -----------------------------------------------------


def test_built_prompt_carries_the_language_section() -> None:
    assert "Hinglish" in build_system_prompt("HDFC Small Cap Fund ka manager kaun hai?")
    assert "Devanagari" in build_system_prompt("HDFC Small Cap Fund का manager कौन है?")
    assert "English" in build_system_prompt("Who manages HDFC Small Cap Fund?")


def test_built_prompt_keeps_completeness_and_absolute_rules() -> None:
    """Language guidance must not weaken grounding, completeness or citation rules."""
    prompt = build_system_prompt("HDFC Small Cap Fund ka manager kaun hai?")

    assert prompt.startswith(SYSTEM_PROMPT)
    assert "COMPLETENESS" in prompt
    assert "never end mid-word" in prompt
    assert "Use only the retrieved SOURCE blocks." in prompt
    assert "Never output a URL" in prompt
    assert "never relaxes an absolute rule" in prompt


# --- Retrieval normalisation -------------------------------------------
# The corpus and the encoder are English-only, so without translation a
# Hinglish or Devanagari question cannot reach the chunk holding the answer.


@pytest.mark.parametrize(
    "question",
    [
        "Who is the fund manager of HDFC Small Cap Fund?",
        "manager of small cap hdfc funds?",
        "What is the expense ratio of HDFC Flexi Cap Fund?",
        "Is there a lock-in period?",
    ],
)
def test_english_queries_are_left_untouched(question: str) -> None:
    """English must embed exactly as written, so the existing path is unchanged."""
    assert normalize_for_retrieval(question) == ""


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("HDFC Small Cap Fund ka manager kaun hai?", "hdfc small cap fund manager"),
        ("HDFC small cap fund ka manager kaun hai bhai?", "hdfc small cap fund manager"),
        ("expense ratio kya hai", "expense ratio"),
        ("lock-in period kya hai", "lock-in period"),
        ("HDFC Flexi Cap ka expense ratio kitna hai?", "hdfc flexi cap expense ratio"),
        ("एचडीएफसी स्मॉल कैप फंड का फंड मैनेजर कौन है?", "hdfc small cap fund manager"),
        ("एचडीएफसी फ्लेक्सी कैप का expense ratio क्या है?", "hdfc flexi cap expense ratio"),
    ],
)
def test_non_english_queries_normalise_to_english_terms(question: str, expected: str) -> None:
    assert normalize_for_retrieval(question) == expected


def test_normalisation_drops_hindi_function_words() -> None:
    normalized = normalize_for_retrieval("HDFC Small Cap Fund ka manager kaun hai bhai?")

    for word in ("ka", "kaun", "hai", "bhai"):
        assert word not in normalized.split()


def test_normalisation_never_empties_a_question() -> None:
    """An empty retrieval query would silently break recall."""
    for question in ("कौन", "kaun", "क्या", "bhai"):
        assert normalize_for_retrieval(question).strip()


def test_transliteration_produces_ascii() -> None:
    roman = devanagari_to_roman("स्मॉल कैप")

    assert roman.isascii()
    assert "sm" in roman


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        source_id="src-1",
        source_url="https://example.invalid/x",
        source_title="t",
        scheme_id="HDFC_SMALL_CAP",
        scheme_name="HDFC Small Cap Fund",
        amc="HDFC Mutual Fund",
        source_type=SourceType.REFERENCE,
        document_type=DocumentType.SCHEME_PAGE,
        authority_level=3,
        retrieved_at="2026-09-25",
        content_hash="abc123",
    )


def test_merge_keeps_best_score_and_drops_duplicates() -> None:
    meta = _metadata()

    def hit(text: str, score: float, chunk_id: str, ordinal: int) -> SearchHit:
        return SearchHit(
            text=text,
            score=score,
            chunk_id=chunk_id,
            document_id="doc-1",
            ordinal=ordinal,
            metadata=meta,
        )

    merged = Retriever._merge_hits(
        [hit("alpha", 0.4, "c-alpha", 0), hit("beta", 0.9, "c-beta", 1)],
        [hit("alpha", 0.7, "c-alpha", 0), hit("gamma", 0.5, "c-gamma", 2)],
    )

    assert [h.text for h in merged] == ["beta", "alpha", "gamma"]
    assert merged[1].score == 0.7, "the higher of the two scores must win"


def test_merge_keeps_distinct_chunks_that_share_identical_text() -> None:
    """Two schemes can hold identical boilerplate; merging on text would lose one."""
    meta = _metadata()

    def hit(score: float, chunk_id: str, ordinal: int) -> SearchHit:
        return SearchHit(
            text="identical boilerplate",
            score=score,
            chunk_id=chunk_id,
            document_id="doc-1",
            ordinal=ordinal,
            metadata=meta,
        )

    merged = Retriever._merge_hits(
        [hit(0.6, "c-small-cap", 0)],
        [hit(0.8, "c-large-cap", 1)],
    )

    assert [h.chunk_id for h in merged] == ["c-large-cap", "c-small-cap"]