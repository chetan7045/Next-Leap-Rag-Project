"""Post-generation response validation and repair.

The system prompt is the main control; this is the practical MVP backstop. It removes
model-generated URLs, enforces the length contract, and rejects recommendation or
performance-prediction phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.services.safety import guardrails

logger = get_logger("app.services.rag.validator")

MAX_SENTENCES = 3


@dataclass(slots=True)
class ValidatedAnswer:
    text: str
    acceptable: bool
    problems: list[str] = field(default_factory=list)
    repaired: bool = False
    reports_absence: bool = False


def validate_and_repair(raw_text: str, *, max_sentences: int = MAX_SENTENCES) -> ValidatedAnswer:
    """Clean the model's text and decide whether it may be shown as a sourced answer."""
    problems: list[str] = []
    text = (raw_text or "").strip()

    if not text:
        return ValidatedAnswer(text="", acceptable=False, problems=["empty_answer"])

    # 1. The model must never produce links; citations are attached by the backend.
    if guardrails.contains_url(text):
        text = guardrails.sanitize(text)
        problems.append("model_generated_url_removed")

    text = " ".join(text.split())
    if not text:
        return ValidatedAnswer(text="", acceptable=False, problems=["empty_after_sanitize"])

    # 2. Report-absence is a valid answer and is never repaired away.
    if guardrails.reports_not_found(text):
        return ValidatedAnswer(
            text=guardrails.truncate_to_sentences(text, max_sentences),
            acceptable=True,
            problems=problems,
            repaired=bool(problems),
            reports_absence=True,
        )

    # 3. Hard refusals: these must not reach a user as a sourced fact.
    recommendation = guardrails.find_recommendation_language(text)
    prediction = guardrails.find_performance_prediction(text)
    if recommendation or prediction:
        problems.append("recommendation_language" if recommendation else "performance_prediction")
        logger.warning("Rejected generated answer (%s)", problems[-1])
        return ValidatedAnswer(text=text, acceptable=False, problems=problems)

    # 4. Enforce the length contract without cutting a sentence mid-way.
    if guardrails.sentence_count(text) > max_sentences:
        text = guardrails.truncate_to_sentences(text, max_sentences)
        problems.append("length_truncated")
    text = guardrails.clip(text)

    return ValidatedAnswer(
        text=text,
        acceptable=True,
        problems=problems,
        repaired=bool(problems),
    )
