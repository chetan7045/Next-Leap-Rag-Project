"""Guardrails applied *after* generation.

The system prompt is the primary control; this module is the practical MVP backstop.
It catches the failure modes that matter: an empty answer, a URL the backend never
issued, or recommendation/performance phrasing leaking into the response.
"""

from __future__ import annotations

import re

RECOMMENDATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "you should invest", "you should definitely invest", "you really ought to buy".
    # The short window keeps legitimate factual phrasing out of the net.
    re.compile(
        r"\byou\s+(?:should|ought\s+to|need\s+to|have\s+to|must)\b[^.!?]{0,25}?"
        r"\b(?:invest|buy|allocate|put\s+money|switch|sell|redeem|purchase)\b",
        re.I,
    ),
    re.compile(r"\b(?:recommend|suggest)\s+(?:the\s+|investing\s+in\s+)?(?:hdfc|this|a)\s+(?:fund|scheme)\b", re.I),
    re.compile(r"\b(?:best|top)\s+(?:performing\s+)?(?:fund|scheme)\s+to\s+invest\b", re.I),
    re.compile(
        r"\b(?:is|are)\s+(?:the\s+|a\s+|an\s+)?(?:best|good|safe|great|ideal|top)\s+"
        r"(?:choice|fund|scheme|investment|option|pick)\b",
        re.I,
    ),
    re.compile(r"\b(?:best|top|safest)\s+(?:fund|scheme|mutual\s+fund)\b", re.I),
    re.compile(r"\bideal\s+for\s+(?:you|your)\b", re.I),
    re.compile(r"\bworth\s+investing\b", re.I),
    re.compile(r"\bconsider\s+investing\b", re.I),
    re.compile(r"\bwould\s+be\s+a\s+(?:good|smart)\s+(?:investment|choice)\b", re.I),
)

PERFORMANCE_PREDICTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwill\s+(?:give|generate|earn|return|deliver|yield)\b", re.I),
    re.compile(r"\bexpected\s+returns?\s+(?:of|is|are)\b", re.I),
    re.compile(r"\b(?:is|are)\s+expected\s+to\s+(?:grow|rise|increase|deliver)\b", re.I),
    re.compile(r"\bprojected?\s+returns?\b", re.I),
    re.compile(r"\b(?:likely|probable)\s+to\s+outperform\b", re.I),
    re.compile(r"\boutperform\s+the\s+(?:other|rest)\s+funds?\b", re.I),
    re.compile(r"\bin\s+the\s+(?:next|coming)\s+\d+\s+years?\b.{0,40}\b(?:return|grow)\b", re.I),
)

# Anything that looks like a URL must not come from the model.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.I)

# Hedge phrases that indicate the model is reporting absence of information.
NOT_FOUND_PHRASES: tuple[str, ...] = (
    "not found in the available sources",
    "not available in the retrieved context",
    "not present in the retrieved context",
    "not mentioned in the available sources",
    "couldn't find that information",
    "does not specify",
    "not specified in the retrieved",
    "i don't have that information",
    "i do not have that information",
)

MAX_ANSWER_SENTENCES = 4
MAX_ANSWER_CHARS = 900

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


def contains_url(text: str) -> bool:
    return bool(URL_PATTERN.search(text))


def find_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def find_recommendation_language(text: str) -> str | None:
    for pattern in RECOMMENDATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def find_performance_prediction(text: str) -> str | None:
    for pattern in PERFORMANCE_PREDICTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def reports_not_found(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in NOT_FOUND_PHRASES)


# Punctuation that can end a sentence. A visible answer that lacks all of them
# has most likely been cut off partway through by the token budget. Devanagari
# answers end with a danda, so it counts as a proper terminator.
_SENTENCE_END_CHARS = frozenset(".!?\"')]}।॥")

# Devanagari danda (। / ॥) is the Devanagari full stop.
_SENTENCE_TERMINATORS = (".", "!", "?", "।", "॥")


def ends_mid_sentence(text: str) -> bool:
    """True when the text carries no sentence-ending punctuation at all."""
    stripped = text.strip()
    if not stripped:
        return False
    return stripped[-1] not in _SENTENCE_END_CHARS


def sentence_terminator(text: str) -> str:
    """Return the full stop that matches the script the answer is written in."""
    return "।" if any("\u0900" <= ch <= "\u097f" for ch in text) else "."


def sentence_count(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT.split(text.strip()) if s])


def truncate_to_sentences(text: str, max_sentences: int = MAX_ANSWER_SENTENCES) -> str:
    """Enforce the length contract without cutting a sentence in half."""
    parts = [p for p in _SENTENCE_SPLIT.split(text.strip()) if p]
    if len(parts) <= max_sentences:
        return " ".join(parts)
    kept = parts[:max_sentences]
    joined = " ".join(kept)
    if joined.endswith(_SENTENCE_TERMINATORS):
        return joined
    return joined.rstrip(",;: ") + sentence_terminator(joined)


def clip(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    boundary = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if boundary > max_chars // 2:
        return clipped[: boundary + 1]
    return clipped.rstrip() + "..."


def sanitize(text: str) -> str:
    """Remove model-generated links. Citations are attached by the backend instead."""
    if not contains_url(text):
        return text
    cleaned = URL_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned).strip()


def validate_answer_text(text: str) -> list[str]:
    """Return a list of problems. Empty list means the answer passed."""
    problems: list[str] = []
    stripped = text.strip()
    if not stripped:
        problems.append("empty_answer")
    if len(stripped) > MAX_ANSWER_CHARS * 2:
        problems.append("answer_too_long")
    if contains_url(stripped):
        problems.append("model_generated_url")
    if find_recommendation_language(stripped):
        problems.append("recommendation_language")
    if find_performance_prediction(stripped):
        problems.append("performance_prediction")
    return problems
