"""Generic language and register detection for the user's question.

The assistant should answer in the language the user wrote in. Detection keys off
two signals only — which script the characters belong to, and the presence of
high-frequency function words — so it works for any question without containing
any scheme, topic, or example-specific logic.

Three styles are recognised:

``english``
    Latin script with no Hindi signal.
``hinglish``
    Latin script containing romanised Hindi ("HDFC small cap fund ka manager
    kaun hai?"). Answered in natural Hinglish, never transliterated into
    Devanagari.
``hindi_script``
    Written in Devanagari. Answered in Devanagari Hindi.

Finance terminology such as "Fund Manager", "NAV" and "Expense Ratio" is kept in
English in every style, because that is how the terms are written in the indexed
sources and translating them makes them harder to match against a source.
"""

from __future__ import annotations

import re

ENGLISH = "english"
HINGLISH = "hinglish"
HINDI_SCRIPT = "hindi_script"

# Unicode block for Devanagari, the script Hindi and Marathi are written in.
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")

# A single unambiguous romanised Hindi word is enough to set the register.
_STRONG_MARKERS = frozenset(
    {
        "kaun", "kya", "kyun", "kyu", "kyon", "kaise", "kahan", "kab",
        "batao", "bataiye", "batae", "bataye", "batana", "bhai", "yaar", "arre",
        "chahiye", "chahye", "mera", "meri", "mere", "mujhe", "tera", "tumhara",
        "nahi", "nhi", "hua", "hue", "honi", "chalega", "chalte", "samjhao",
    }
)

# These are common English words too, so two or more are needed before the
# register is treated as Hinglish.
_WEAK_MARKERS = frozenset(
    {
        "ka", "ki", "ke", "ko", "hai", "hain", "ho", "hon", "tha", "thi", "the",
        "raha", "rahe", "rahi", "rah", "kar", "karna", "karo", "kardo", "kiya",
        "karke", "wala", "wali", "wale", "laga", "lage", "lagta", "lagti", "chalo",
        "dekho", "dekha", "jaldi", "thoda", "zyada", "kam", "aur", "apna", "khud",
        "iska", "iski", "iske", "isme", "uska", "uski", "kaha", "kitna", "kitne",
        "kiska", "kiski", "kahan", "jisme", "jiska", "liye", "waala", "waali",
    }
)

# Tokens are split on non-letters, so punctuation and digits never create a
# false match ("ka?" and "ka" are the same token).
_WORD = re.compile(r"[a-z]+")

# Devanagari needs only a couple of characters to be unambiguous; Latin script
# needs a meaningful share of the letters to count as Hindi.
_MIN_DEVANAGARI_CHARS = 2


def detect_language_style(text: str) -> str:
    """Return ``english``, ``hinglish`` or ``hindi_script`` for ``text``."""
    if not text:
        return ENGLISH

    devanagari = len(_DEVANAGARI.findall(text))
    if devanagari >= _MIN_DEVANAGARI_CHARS:
        return HINDI_SCRIPT

    tokens = _WORD.findall(text.lower())
    strong = sum(1 for t in tokens if t in _STRONG_MARKERS)
    if strong >= 1:
        return HINGLISH

    weak = sum(1 for t in tokens if t in _WEAK_MARKERS)
    if weak >= 2:
        return HINGLISH

    return ENGLISH


def language_instruction(style: str) -> str:
    """Prompt guidance that makes the model answer in the user's language."""
    if style == HINDI_SCRIPT:
        return (
            "LANGUAGE\n"
            "The user wrote in Hindi using the Devanagari script. Answer in natural, "
            "everyday Hindi written in the same Devanagari script. Keep established "
            "finance terms in English where that is how they are written in the sources "
            "(for example Fund Manager, NAV, Expense Ratio, Small Cap). Do not reply in "
            "English or in Romanised Hindi."
        )
    if style == HINGLISH:
        return (
            "LANGUAGE\n"
            "The user wrote in Hinglish (Hindi written in Roman script). Answer in natural "
            "everyday Hinglish that matches how they wrote, using the same Roman script. "
            "Keep established finance terms in English where that is how they are written "
            "in the sources (for example Fund Manager, NAV, Expense Ratio, Small Cap). "
            "Do not switch to Devanagari, and do not reply in formal or literary Hindi."
        )
    return (
        "LANGUAGE\n"
        "The user wrote in English. Answer in clear, plain English."
    )