"""Document metadata construction.

Every chunk carries a full provenance record, and content hashing makes ingestion
idempotent. Source dates are only ever taken from the document; retrieval time is
tracked separately and is never presented as the source's update date.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from app.core.config import Settings
from app.models.enums import DocumentType, SourceType
from app.models.retrieval import DocumentMetadata, SourceConfig

# Date shapes seen in fund documents/pages.
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"),
    re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b"),
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b"),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
)

_DATE_LABEL = re.compile(
    r"(?:as\s+on|as\s+of|updated\s+on|updated\s+as\s+on|last\s+updated|"
    r"data\s+as\s+on|nav\s+as\s+on|as\s+at)\s*[:\-]?\s*(.{6,32})",
    re.I,
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        ],
        start=1,
    )
}


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_document_id(scheme_id: str, url: str) -> str:
    """Deterministic per (scheme, normalised URL)."""
    normalised = url.strip().lower().rstrip("/")
    digest = hashlib.sha256(f"{scheme_id.upper()}|{normalised}".encode("utf-8")).hexdigest()
    return digest[:16]


def make_chunk_id(document_id: str, ordinal: int, text: str) -> str:
    """Deterministic per (document, position, content) so re-chunking is stable."""
    digest = hashlib.sha256(f"{document_id}|{ordinal}|{text}".encode("utf-8")).hexdigest()
    return digest[:32]


def normalise_date(value: str) -> str | None:
    """Convert a recognised date shape to ``YYYY-MM-DD``; return ``None`` if unclear."""
    if not value:
        return None
    text = " ".join(value.split()).strip(" .,;:")
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1)
        parsed = _to_iso(raw)
        if parsed:
            return parsed
    return None


def _to_iso(raw: str) -> str | None:
    text = raw.replace(",", " ").strip()
    parts = text.split()
    day = month = year = None
    for part in parts:
        token = part.strip(".").lower()
        if token[:3] in _MONTHS and len(token) >= 3:
            month = _MONTHS[token[:3]]
            for suffix in ("ber", "ary", "ust", "ary", "il", "une", "uly", "ust", "ptember", "ctober", "ovember", "ecember"):
                if token.startswith(suffix[:3]):
                    break
        elif token.isdigit():
            number = int(token)
            if len(token) == 4:
                year = number
            elif day is None:
                day = number
            elif month is None and 1 <= number <= 12:
                month = number
    # "Mar 31 2026" style where the first token was the month name.
    if month is None:
        for part in parts:
            token = part.strip(".").lower()
            if token[:3] in _MONTHS:
                month = _MONTHS[token[:3]]
                break
    if day is None:
        # ISO or dd/mm/yyyy
        nums = [p for p in text.replace("-", " ").replace("/", " ").split() if p.isdigit()]
        if len(nums) >= 3 and len(nums[0]) == 4:
            year, month, day = int(nums[0]), int(nums[1]), int(nums[2])
        elif len(nums) >= 3:
            day, month, year = int(nums[0]), int(nums[1]), int(nums[2])
    if not (year and month and day):
        return None
    try:
        return datetime(year, month, day, tzinfo=UTC).date().isoformat()
    except ValueError:
        return None


def extract_source_date(text: str, hint: str | None = None) -> str | None:
    """Find the document's own update date. Never falls back to today's date."""
    for candidate in (hint, None):
        if not candidate:
            continue
        match = _DATE_LABEL.search(candidate)
        if match:
            parsed = normalise_date(match.group(1))
            if parsed:
                return parsed
    if text:
        match = _DATE_LABEL.search(text)
        if match:
            parsed = normalise_date(match.group(1))
            if parsed:
                return parsed
    return None


def build_metadata(
    source: SourceConfig,
    *,
    title: str,
    text: str,
    retrieved_at: datetime,
    date_hint: str | None = None,
    settings: Settings | None = None,
) -> DocumentMetadata:
    """Assemble the provenance record for one ingested document."""
    # Scan the whole normalised document: an "as of" marker can sit deep in the page.
    source_date = extract_source_date(text, date_hint)
    document_id = make_document_id(source.scheme_id, source.url)
    return DocumentMetadata(
        document_id=document_id,
        source_id=source.id,
        source_url=source.url,
        source_title=(source.title or title or source.scheme_name).strip(),
        scheme_id=source.scheme_id,
        scheme_name=source.scheme_name,
        amc=source.amc,
        source_type=source.source_type,
        document_type=source.document_type or DocumentType.OTHER,
        authority_level=source.source_type.authority_level,
        published_at=source_date,
        last_updated_at=source_date,
        retrieved_at=retrieved_at.astimezone(UTC).isoformat(timespec="seconds"),
        content_hash=compute_content_hash(text),
        publisher=source.publisher,
        loader=source.loader,
    )


def resolve_effective_date(metadata: DocumentMetadata) -> str | None:
    return metadata.last_updated_at or metadata.published_at


def source_priority(metadata: DocumentMetadata) -> tuple[int, str]:
    """Sort key for ``sorted(..., key=source_priority, reverse=True)``.

    Authoritative sources outrank reference sources; within a tier the most recently
    updated document wins. An undated document (``""``) sorts last within its tier,
    so a dated authoritative document always beats an undated one.
    """
    return (metadata.authority_level, resolve_effective_date(metadata) or "")


def describe_source_type(source_type: SourceType) -> str:
    return source_type.ui_label
