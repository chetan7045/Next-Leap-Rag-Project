"""Turn a non-English question into an English retrieval query.

The corpus is English-only and the embedding model (``all-MiniLM-L6-v2``) is an
English encoder, so a Devanagari or Hinglish question embeds far away from the
chunks that actually contain the answer. Without this step such a question
retrieves nothing, or retrieves a confidently-wrong chunk, and the assistant
answers "I couldn't find that information" while the fact is sitting in the index.

Normalisation is deliberately generic. It is driven by:

* a Devanagari → English term map for finance vocabulary,
* a generic Devanagari → Roman transliterator for everything else,
* a Romanised-Hindi → English map plus a stop list of Hindi function words.

Nothing here is tied to a specific question, scheme, or topic: any question in
any of the three supported styles is reduced to the English terms the corpus
actually uses.

This affects **retrieval only**. The model still receives the user's original
message, so the answer can be given back in their own language.
"""

from __future__ import annotations

import re

from app.services.rag.language import ENGLISH, detect_language_style

# --- Devanagari --------------------------------------------------------

# Independent vowels and matras, as written in the Roman script.
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "आ": "aa",
}
_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॊ": "e",
    "ं": "n", "ँ": "n", "ः": "h", "ऽ": "",
}
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # Nukta forms.
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r",
    "ढ़": "rh", "फ़": "f", "य़": "y",
}
# Punctuation and digits.
_SIGNS = {
    "।": ".", "॥": ".", "॰": ".", "०": "0", "१": "1", "२": "2",
    "३": "3", "४": "4", "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    ",": ",", ".": ".", "-": "-", "/": " ", "(": " ", ")": " ", "?": " ",
    "!": " ", "%": "%", "₹": " ",
}
_VIRAMA = "्"

# Domain vocabulary the corpus is written with. Longest first so that
# "स्मॉल कैप" wins over "स्मॉल".
_DEVANAGARI_TERMS: tuple[tuple[str, str], ...] = (
    ("म्यूचुअल फंड", "mutual fund"),
    ("म्यूच्युअल फण्ड", "mutual fund"),
    ("स्मॉल कैप", "small cap"),
    ("स्माल कैप", "small cap"),
    ("लार्ज कैप", "large cap"),
    ("लार्ज कप", "large cap"),
    ("बड़े कैप", "large cap"),
    ("मिड कैप", "mid cap"),
    ("मध्यम कैप", "mid cap"),
    ("फ्लेक्सी कैप", "flexi cap"),
    ("फ्लेक्स कैप", "flexi cap"),
    ("फ्लेक्सि कैप", "flexi cap"),
    ("एक्सपेंस रेशियो", "expense ratio"),
    ("एक्सपेंस रेट", "expense ratio"),
    ("खर्च अनुपात", "expense ratio"),
    ("फंड मैनेजर", "fund manager"),
    ("फण्ड मैनेजर", "fund manager"),
    ("लॉक इन", "lock in"),
    ("सीधी योजना", "direct growth"),
    ("डायरेक्ट ग्रोथ", "direct growth"),
    ("ग्रोथ ऑप्शन", "growth option"),
    ("स्वतंत्रता कोष", "aum"),
    ("एचडीएफसी", "hdfc"),
    ("मैनेजर", "manager"),
    ("फंड", "fund"),
    ("फण्ड", "fund"),
    ("कैप", "cap"),
    ("स्मॉल", "small"),
    ("लार्ज", "large"),
    ("मिड", "mid"),
    ("फ्लेक्सी", "flexi"),
    ("एनएवी", "nav"),
    ("सिप", "sip"),
    ("एआरएफ", "arf"),
    ("रिटर्न", "return"),
    ("अभी", "current"),
    ("कौन", "who"),
    ("क्या", "what"),
    ("कितना", "how much"),
    ("कितनी", "how much"),
    ("कब", "when"),
    ("कहाँ", "where"),
    ("क्यों", "why"),
    ("कैसे", "how"),
    ("बताओ", "tell"),
    ("बताइए", "tell"),
    ("तुलना", "compare"),
    ("अंतर", "difference"),
)

# --- Romanised Hindi ---------------------------------------------------

# Finance terms that arrive transliterated.
_ROMAN_TERMS: dict[str, str] = {
    "smoll": "small", "smal": "small", "saath": "with",
    "kaip": "cap", "kap": "cap", "cap": "cap",
    "manajar": "manager", "manager": "manager", "mngr": "manager",
    "funf": "fund", "fand": "fund", "fund": "fund",
    "expendse": "expense", "expense": "expense", "xpendse": "expense",
    "rasio": "ratio", "ratio": "ratio", "reashio": "ratio",
    "retren": "return", "riturn": "return", "return": "return",
    "flexi": "flexi", "flaxi": "flexi",
    "large": "large", "mid": "mid",
    "sip": "sip", "nav": "nav", "aum": "aum", "lock": "lock",
    "chalt": "current", "current": "current", "abhi": "current",
}

# Interrogatives and particles carry no retrieval signal once the question has
# been translated, and they actively push the embedding away from the chunk.
_HINDI_FUNCTION_WORDS = frozenset(
    {
        "ka", "ki", "ke", "ko", "se", "mein", "me", "par", "aur",
        "hai", "hain", "ho", "hu", "hota", "hoti", "tha", "thi", "the",
        "kaun", "kya", "kyun", "kyu", "kyon", "kahan", "kab", "kaise", "kitna",
        "kitni", "kitne", "kiska", "kiski", "kiska", "kaha",
        "batao", "bataiye", "batae", "bataye", "batana", "bata",
        "nahi", "nhi", "mat", "karna", "karo", "kardo", "kar", "kiya", "karke",
        "mera", "meri", "mere", "mujhe", "tera", "tumhara", "apna", "khud",
        "bhai", "yaar", "ji", "arre", "ya", "hi", "to", "phir", "jab",
        "kuch", "sab", "sabhi", "waala", "wali", "wale", "wala", "liye",
        "dhanyavad", "shukriya", "please", "plz", "suno", "sun",
        "chahiye", "chahye", "lagta", "lagti", "lage", "raha", "rahe", "rahi",
        "jaldi", "thoda", "zyada", "kam", "aur", "iska", "iski", "iske",
        "isme", "uska", "uski", "jiska", "jiski", "kisan", "kisi",
    }
)

_WORD_SPLIT = re.compile(r"(\s+)")
_TOKEN = re.compile(r"[a-z0-9%.\-]+")

# English interrogatives and function words. Safe to drop from the translated
# variant because the user's original query is still embedded on its own.
_ENGLISH_STOP_WORDS = frozenset(
    {
        "who", "whom", "what", "which", "when", "where", "why", "how",
        "much", "many", "is", "are", "was", "were", "am", "do", "does",
        "did", "of", "the", "a", "an", "and", "or", "in", "on", "for",
        "to", "with", "this", "that", "it", "be", "as", "at", "by",
    }
)


def _is_devanagari(char: str) -> bool:
    return "\u0900" <= char <= "\u097f"


def devanagari_to_roman(text: str) -> str:
    """Transliterate Devanagari to the Roman script.

    This is a pragmatic mapping, not a full ITRANS implementation. It exists so
    that words absent from the term map still contribute something searchable
    instead of being dropped entirely.
    """
    out: list[str] = []
    previous_was_consonant = False
    for char in text:
        if _is_devanagari(char):
            if char in _CONSONANTS:
                if previous_was_consonant:
                    # Inherent "a" after a consonant unless a matra follows.
                    out.append("a")
                out.append(_CONSONANTS[char])
                previous_was_consonant = True
            elif char == _VIRAMA:
                previous_was_consonant = False
            elif char in _MATRAS:
                out.append(_MATRAS[char])
                previous_was_consonant = False
            elif char in _VOWELS:
                if previous_was_consonant:
                    out.append("a")
                out.append(_VOWELS[char])
                previous_was_consonant = False
            elif char in _SIGNS:
                out.append(_SIGNS[char])
                previous_was_consonant = False
            else:
                previous_was_consonant = False
        else:
            if previous_was_consonant and (char.isalpha()):
                out.append("a")
            out.append(char)
            previous_was_consonant = False
    return "".join(out)


def _apply_devanagari_terms(text: str) -> str:
    result = text
    for term, replacement in _DEVANAGARI_TERMS:
        if term in result:
            result = result.replace(term, replacement)
    return result


def _dedupe(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for token in tokens:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            kept.append(token)
    return kept


def normalize_for_retrieval(text: str) -> str:
    """Return an English retrieval query, or ``""`` when none is needed.

    An empty string means the question is already English and should be embedded
    exactly as the user wrote it, which keeps existing retrieval behaviour for
    English questions completely untouched.
    """
    if not text.strip():
        return ""
    if detect_language_style(text) == ENGLISH:
        return ""

    working = text
    if any(_is_devanagari(c) for c in working):
        working = _apply_devanagari_terms(working)
        working = devanagari_to_roman(working)

    kept: list[str] = []
    for part in _WORD_SPLIT.split(working):
        if not part.strip():
            continue
        for token in _TOKEN.findall(part.lower()):
            if token in _HINDI_FUNCTION_WORDS:
                continue
            token = _ROMAN_TERMS.get(token, token)
            # Terms mapped out of Devanagari become English interrogatives
            # ("kaun" -> "who"); they carry no retrieval signal either.
            if token in _ENGLISH_STOP_WORDS:
                continue
            kept.append(token)

    normalized = " ".join(_dedupe(kept))
    # Never let normalisation empty the query: fall back to the ASCII that
    # survived, and if nothing did, transliterate whatever is left.
    if not normalized.strip():
        fallback = devanagari_to_roman(text)
        normalized = " ".join(_TOKEN.findall(fallback.lower()))
    return normalized.strip()