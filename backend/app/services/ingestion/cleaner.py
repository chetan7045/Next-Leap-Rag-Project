"""Content cleaning and normalisation.

Removes navigation, cookie banners, footers, repeated menus, and advertising noise
while preserving everything that can carry a financial fact: headings, labels, values,
list structure, tables, and footnotes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from app.services.ingestion.extractor import Block, ExtractionResult

# Whole-block junk: legal boilerplate, marketing, and app-store noise.
_JUNK_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:log\s*in|log\s*in\s*/\s*sign\s*up|sign\s*in|sign\s*up|download\s+app)\s*$", re.I),
    re.compile(r"^\s*(?:all\s+rights?\s+reserved|copyright|©.*?)\s*$", re.I),
    re.compile(r"^\s*(?:cookie|privacy\s+policy|terms\s+(?:of\s+use|and\s+conditions)|disclaimer)\s*$", re.I),
    re.compile(r"^\s*(?:follow\s+us|share\s+this|was\s+this\s+helpful)\b", re.I),
    re.compile(r"^\s*(?:invest\s+in\s+\d+\s+lumpsum|start\s+an\s+sip)\s*$", re.I),
    re.compile(r"^\s*(?:show\s+more|show\s+less|read\s+more|view\s+all|load\s+more)\s*$", re.I),
    re.compile(r"\bcookie\s+(?:consent|policy|settings|notice)\b", re.I),
    re.compile(r"\baccept\s+(?:all\s+)?cookies\b", re.I),
    re.compile(r"\bplease\s+enable\s+javascript\b", re.I),
    re.compile(r"\bbrowser\s+not\s+supported\b", re.I),
    re.compile(r"^\s*(?:home|back|next|previous|close|menu)\s*$", re.I),
    re.compile(r"^(?:page\s+)?\d+\s*(?:of|/)\s*\d+$", re.I),
)

# Sentences stripped from inside otherwise-valid paragraphs.
_JUNK_SENTENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:mutual\s+fund\s+investments?\s+are\s+subject\s+to\s+market\s+risks)"),
    # Legal boilerplate carries no scheme fact and only dilutes retrieval.
    re.compile(r"\ball\s+rights?\s+reserved\b", re.I),
    re.compile(r"\bcopyright(?:\s+\u00a9|\s+\(c\))?\b[^.]{0,60}", re.I),
    re.compile(r"\u00a9\s*[^.]{0,60}"),
    re.compile(r"\b(?:privacy\s+policy|terms\s+(?:of\s+use|and\s+conditions))\b[^.]{0,80}", re.I),
    re.compile(r"\bread\s+(?:all|the)\s+scheme\s+related\s+documents\b"),
    re.compile(r"\b(?:know\s+your\s+customer|kyc)\b.*\bnumber\b", re.I),
)

# Cross-sell carousels ("HDFC Value Fund Direct Plan Growth HDFC Ultra Short to Short
# Term Fund Direct Growth") are lists of unrelated scheme names carrying no fact. They
# embed close to any real query and crowd out genuine evidence, so they are dropped
# unless they also contain a value or prose.
_FUND_TOKEN = re.compile(r"\b(?:Fund|Plan|FoF|ETF|Scheme)\b")
# Lowercase words allowed inside a scheme name ("to", "of", "and", "tax", ...).
_NAME_CONNECTORS = frozenset(
    {
        "to", "of", "and", "the", "tax", "saver", "cap", "small", "large", "flexi",
        "balanced", "advantage", "equity", "value", "innovation", "focused", "ultra",
        "short", "term", "next", "index", "income", "debt", "hybrid", "retirement",
        "growth", "direct", "plan", "fund", "elss", "global", "savings", "banking",
        "consumer", "dividend", "midcap", "mid", "largecap", "momentum", "alpha",
        "pension", "children", "young", "women", "dynamic", "opportunities", " gilt",
    }
)


def _looks_like_scheme_names(text: str) -> bool:
    """True when a block is only a run of capitalised scheme names."""
    if len(_FUND_TOKEN.findall(text)) < 2:
        return False
    for word in re.findall(r"[A-Za-z][\w&.\-]*", text):
        if word[0].isupper():
            continue
        if word.lower() in _NAME_CONNECTORS:
            continue
        return False
    return True


_WHITESPACE = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def normalise_text(text: str) -> str:
    """Unicode + whitespace normalisation. Preserves digits, punctuation, and newlines."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    text = _CONTROL.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _is_bare_scheme_name(text: str) -> bool:
    """True for a single scheme name with no value, date or prose ("HDFC Small Cap Fund
    Direct Growth"). The page's own title is excluded by the caller."""
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if re.search(r"\d", stripped):
        return False
    if re.search(r"[.!?•|]", stripped):
        return False
    if _FUND_TOKEN.search(stripped) is None:
        return False
    for word in re.findall(r"[A-Za-z][\w&.\-]*", stripped):
        if word[0].isupper() or word.lower() in _NAME_CONNECTORS:
            continue
        return False
    return True


def _is_fund_name_list(text: str) -> bool:
    """True for a carousel of scheme names with no value, date or prose attached.

    These blocks embed close to any real query and crowd out genuine evidence, so they
    are dropped unless they carry a number (a value) or sentence punctuation (prose).
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 400:
        return False
    if re.search(r"\d", stripped):
        return False
    if re.search(r"[.!?]", stripped):
        return False
    return _looks_like_scheme_names(stripped)


def _is_junk(text: str) -> bool:
    if not text:
        return True
    if any(pattern.search(text) for pattern in _JUNK_BLOCK_PATTERNS):
        return True
    if _is_fund_name_list(text):
        return True
    if len(text) < 2:
        return True
    return False


def _strip_junk_sentences(text: str) -> str:
    for pattern in _JUNK_SENTENCE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def clean_extraction(result: ExtractionResult, *, min_paragraph_chars: int = 25) -> ExtractionResult:
    """Drop chrome blocks, strip boilerplate sentences, and de-duplicate."""
    kept: list[Block] = []
    seen_headings: set[str] = set()
    seen_paragraphs: set[str] = set()
    after_fact_label = False

    page_title = normalise_text(result.title).lower()

    for block in result.blocks:
        text = normalise_text(block.text)
        if not text or _is_junk(text):
            continue
        # Cross-sell carousels list other schemes by name only. They are chrome unless
        # the block is this page's own title.
        if text.lower() != page_title and _is_bare_scheme_name(text):
            continue

        if block.kind == "heading":
            key = text.lower()
            if key in seen_headings:
                continue
            seen_headings.add(key)
            kept.append(replace(block, text=text))
            after_fact_label = _FACT_LABEL.search(text) is not None
            continue

        if block.kind == "table":
            kept.append(replace(block, text=text))
            after_fact_label = False
            continue

        cleaned = _strip_junk_sentences(text)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen_paragraphs:
            continue
        if block.kind == "paragraph" and len(cleaned) < min_paragraph_chars:
            # Keep short factual lines ("Expense ratio: 1.16%") and bare fact labels
            # ("ELSS - 3Y Lock-in") — they are the payload of a facts assistant.
            if ":" not in cleaned and not _FACT_LABEL.search(cleaned) and not after_fact_label:
                continue
        seen_paragraphs.add(key)
        kept.append(replace(block, text=cleaned))

    kept = [b for b in kept if not _is_junk(b.text)]
    return replace(result, blocks=kept, title=normalise_text(result.title))


def blocks_to_text(blocks: list[Block]) -> str:
    return "\n\n".join(_dedupe_preserve_order([b.text for b in blocks if b.text]))


def estimate_char_density(text: str) -> float:
    """Fraction of characters that are alphanumeric — a cheap garbage detector."""
    if not text:
        return 0.0
    alnum = sum(1 for ch in text if ch.isalnum())
    return alnum / len(text)


# --- Navigation-run stripping ------------------------------------------------
# Site chrome often renders as a run of short, label-like items (menu entries)
# before any real prose. A navigation run is a leading sequence of such items with
# no sentence punctuation. Stripping only a *leading* run keeps legitimate fact
# lists further down the page intact.
_NAV_MIN_RUN = 3
_NAV_MAX_ITEM_WORDS = 10
_SENTENCE_END = (".", "!", "?", "…")
_FACT_MARKERS = ("₹", "Rs.", "%", "is set to", "as on", "as of")

# Short label-only blocks such as "Expense ratio" or "ELSS - 3Y Lock-in" are the
# vocabulary of a facts product, so they are kept even though they are terse.
_FACT_LABEL = re.compile(
    r"\b(?:expense\s*ratio|exit\s*load|entry\s*load|minimum\s*(?:sip|investment|lumpsum)|"
    r"min\.?\s*for\s*sip|lock[\s-]*in|benchmark|riskometer|rating|fund\s*size|aum|"
    r"direct\s*growth|plan|category|nav|tax|sip)\b",
    re.I,
)


_NAV_MEMBER_KINDS = {"list_item", "paragraph"}


def _is_nav_like(block: Block) -> bool:
    """True when a block looks like a menu entry rather than content.

    Deliberately strict: anything carrying fact vocabulary, a bullet separator, a
    digit-bearing value, or sentence punctuation is content and is never stripped.
    """
    if block.kind not in _NAV_MEMBER_KINDS:
        return False
    stripped = block.text.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped.endswith(_SENTENCE_END):
        return False
    if "•" in stripped or "|" in stripped:
        return False
    if re.search(r"\d", stripped):
        return False
    if any(marker in stripped for marker in _FACT_MARKERS):
        return False
    if _FACT_LABEL.search(stripped):
        return False
    return len(stripped.split()) <= _NAV_MAX_ITEM_WORDS


def strip_leading_navigation(blocks: list[Block], *, min_run: int = _NAV_MIN_RUN) -> list[Block]:
    """Drop a leading run of short label-like items that make up a nav menu."""
    index = 0
    while index < len(blocks) and _is_nav_like(blocks[index]):
        index += 1
    if index >= min_run:
        return blocks[index:]
    return blocks
