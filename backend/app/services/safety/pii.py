"""Lightweight PII detection for obvious financial/personal identifiers.

Deliberately conservative: it flags *shapes* that should never be forwarded to an
LLM. It never stores, logs, or forwards the matched values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern is anchored to a plausible label where possible to limit false positives.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")),
    ("bank_account", re.compile(r"(?:a/?c|account)\s*(?:no\.?|number|#)\s*[is:]?\s*[\dX]{9,18}\b", re.I)),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("otp", re.compile(r"\b(?:otp|one time pass ?word|verification code)\b\s*(?:is|:)?\s*\d{4,8}\b", re.I)),
    ("phone", re.compile(r"(?:\+91[-\s]?)?\b[6-9]\d{9}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("folio", re.compile(r"\b(?:folio|client)\s*(?:no\.?|number|#)\s*[is:]?\s*[A-Z0-9]{6,}\b", re.I)),
    ("demat", re.compile(r"\b(?:demat|dp)\s*(?:id|number|no\.?)\s*[is:]?\s*[A-Z0-9]{8,}\b", re.I)),
    ("password", re.compile(r"\b(?:password|passwd|pin|passcode)\b\s*(?:is|:|=)\s*\S+", re.I)),
    ("bearer", re.compile(r"\b(?:bearer|api[_-]?key|access[_-]?token|jwt)\s*[:=]?\s*[A-Za-z0-9_\-.]{12,}", re.I)),
    ("credentials", re.compile(r"\b(?:my\s+)?(?:pan|aadhaar|uidai|passport|dl|licen[cs]e)\s*(?:number|no\.?|#)?\s*(?:is|:)\s*[A-Za-z0-9]{4,}", re.I)),
)

# Phrases that indicate the *topic* without necessarily containing an identifier.
_PII_TOPIC = re.compile(
    r"\b(?:my\s+(?:pan|aadhaar|folio|account|demat)|bank\s+account\s+number|"
    r"update\s+my\s+(?:pan|aadhaar|address|email|phone)|"
    r"verify\s+my\s+(?:pan|aadhaar|otp))\b",
    re.I,
)

PII_NOTICE = (
    "Please don't enter PAN, Aadhaar, bank account, OTP, folio or other personal financial "
    "information here. You can ask your mutual fund question without sharing account details."
)


@dataclass(frozen=True, slots=True)
class PIIFinding:
    categories: tuple[str, ...]
    masked_preview: str


def mask(value: str) -> str:
    """Return a non-reversible hint: length only, never the value."""
    return f"<redacted:{len(value)} chars>"


def detect_pii(text: str) -> PIIFinding | None:
    """Return a finding when the text looks like it contains personal financial data."""
    if not text or not text.strip():
        return None

    categories: list[str] = [name for name, pattern in _PATTERNS if pattern.search(text)]
    if _PII_TOPIC.search(text):
        categories.append("financial_topic")

    if not categories:
        return None

    # A bare 10-digit run is ambiguous (could be an amount or a date fragment), so a
    # phone/card hit alone must be corroborated by another signal before we refuse.
    if categories == ["phone"] and len(text.strip()) < 40:
        return None

    return PIIFinding(categories=tuple(sorted(set(categories))), masked_preview=mask(text))


def has_pii(text: str) -> bool:
    return detect_pii(text) is not None
