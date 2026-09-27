"""Structure-aware recursive chunking.

Rules that matter for a facts product:
  * a heading never detaches from the fact it labels,
  * compact factual fields (expense ratio, exit load, minimum SIP, lock-in) stay in
    one small chunk so a single retrieval returns the whole fact,
  * tables keep their header row so column meanings survive the split,
  * no blind character slicing.
"""

from __future__ import annotations

import re

from app.core.config import Settings
from app.models.enums import DocumentType
from app.models.retrieval import Chunk, Document
from app.services.ingestion.extractor import Block
from app.services.ingestion.metadata import make_chunk_id

# Fields that should become their own small, self-contained chunk.
COMPACT_FACT_PATTERN = re.compile(
    r"\b(?:expense\s*ratio|exit\s*load|entry\s*load|minimum\s*(?:sip|investment|"
    r"additional|balance|amount)|lock[\s-]*in(?:\s*period)?|benchmark|"
    r"riskometer|risk\s*(?:ometer\s*)?rating|securities\s*transmitted|"
    r"direct\s*growth|plan\s*option|fund\s*category|category|aum|nav|"
    # Aggregator label forms: "Min. for SIP", "Min. for Lumpsum", "Lumpsum available".
    r"min\.?\s*(?:for|of)?\s*(?:sip|investment|lumpsum|amount)|"
    r"lumpsum\s*(?:available|amount)?|"
    r"fund\s*manager|tax\s*(?:status|benefit)|step[\s-]*up)\b",
    re.I,
)

# Heading text that is pure chrome and should not be used as a chunk's own subject.
_GENERIC_HEADINGS = {"overview", "details", "more", "about", "info", "key facts", "at a glance"}


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""
    return max(1, round(len(text) / 4)) if text.strip() else 0


def _is_compact_fact(text: str) -> bool:
    return bool(COMPACT_FACT_PATTERN.search(text))


def _heading_path(stack: list[tuple[int, str]]) -> list[str]:
    return [title for _, title in stack]


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _iter_sections(blocks: list[Block]) -> list[tuple[list[str], list[Block]]]:
    """Group blocks under their enclosing heading path."""
    sections: list[tuple[list[str], list[Block]]] = []
    stack: list[tuple[int, str]] = []
    current_path: list[str] = []
    current: list[Block] = []

    for block in blocks:
        if block.kind == "heading":
            if current:
                sections.append((current_path, current))
                current = []
            _push_heading(stack, block.level or 2, block.text)
            current_path = _heading_path(stack)
            continue
        current.append(block)

    if current:
        sections.append((current_path, current))
    return sections


def _pack(
    blocks: list[Block],
    heading_path: list[str],
    document: Document,
    settings: Settings,
    start_ordinal: int,
) -> list[Chunk]:
    """Pack ordered blocks into target-sized chunks with overlap, keeping the heading."""
    target = settings.chunk_target_tokens
    overlap = settings.chunk_overlap_tokens
    min_tokens = settings.chunk_min_tokens

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    ordinal = start_ordinal

    def flush(carry_overlap: bool) -> None:
        nonlocal buffer, buffer_tokens, ordinal
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if not body:
            buffer, buffer_tokens = [], 0
            return
        prefix = f"{' > '.join(heading_path)}\n" if heading_path else ""
        text = f"{prefix}{body}".strip()
        tokens = estimate_tokens(text)
        if tokens < min_tokens and chunks and ordinal > start_ordinal:
            # Too small to stand alone: fold into the previous chunk.
            previous = chunks[-1]
            merged = f"{previous.text}\n{body}"
            chunks[-1] = previous.model_copy(update={"text": merged, "token_estimate": estimate_tokens(merged)})
        else:
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.metadata.document_id, ordinal, text),
                    document_id=document.metadata.document_id,
                    ordinal=ordinal,
                    text=text,
                    heading_path=list(heading_path),
                    token_estimate=tokens,
                    topics=list(document.metadata.__dict__.get("topics", []) or []),
                    metadata=document.metadata,
                )
            )
            ordinal += 1

        if carry_overlap and overlap > 0 and buffer:
            tail_words = " ".join(buffer).split()
            tail = " ".join(tail_words[-overlap * 2 :])
            buffer = [tail] if tail else []
            buffer_tokens = estimate_tokens(tail)
        else:
            buffer, buffer_tokens = [], 0

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        block_tokens = estimate_tokens(text)

        # Oversized block (very long paragraph or wide table): split it on line
        # boundaries, keeping the heading prefix, so a fact is never orphaned.
        if block_tokens > target:
            flush(carry_overlap=True)
            for piece in _split_oversized(text, target):
                buffer = [piece]
                buffer_tokens = estimate_tokens(piece)
                flush(carry_overlap=True)
            continue

        if buffer_tokens + block_tokens > target and buffer:
            flush(carry_overlap=True)

        buffer.append(text)
        buffer_tokens += block_tokens

    flush(carry_overlap=False)
    return chunks


def _split_oversized(text: str, target_tokens: int) -> list[str]:
    limit = target_tokens * 4
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        line_len = len(line) + 1
        if size + line_len > limit and current:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += line_len
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_document(document: Document, settings: Settings) -> list[Chunk]:
    """Split a document into retrievable, heading-anchored chunks."""
    if not document.sections:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    compact_limit = settings.chunk_compact_token_limit

    flat_blocks = [block for section in document.sections for block in section.blocks]
    sections = drop_dangling_label_sections(_iter_sections(flat_blocks))

    scheme_name = document.metadata.scheme_name
    for heading_path, blocks in sections:
        # Every chunk names its own scheme. Aggregator headings are often generic
        # ("All", "Exit load"), so a bare fact such as "ELSS • 3Y Lock-in" would
        # otherwise lose the context needed to attribute it to a fund and would rank
        # poorly against a query that names the scheme.
        heading = " ".join(heading_path)
        subject = f"{scheme_name} {heading}".strip() if heading else scheme_name
        compact = _is_compact_fact(subject) or any(_is_compact_fact(b.text) for b in blocks[:2])

        if compact and len(blocks) <= 4:
            # Keep a compact factual block whole, even if slightly over target, so the
            # label, value, and any qualification stay together.
            body = "\n".join(b.text.strip() for b in blocks if b.text.strip())
            if body:
                text = f"{subject}\n{body}".strip() if subject else body
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(document.metadata.document_id, ordinal, text),
                        document_id=document.metadata.document_id,
                        ordinal=ordinal,
                        text=text,
                        heading_path=list(heading_path),
                        token_estimate=estimate_tokens(text),
                        topics=[],
                        metadata=document.metadata,
                    )
                )
                ordinal += 1
            continue

        produced = _pack(blocks, heading_path, document, settings, ordinal)
        if compact:
            for chunk in produced:
                chunk.topics = ["compact_fact"]
        chunks.extend(produced)
        ordinal += len(produced)

    if document.metadata.document_type in {DocumentType.FACTSHEET, DocumentType.SID, DocumentType.KIM}:
        for chunk in chunks:
            if chunk.token_estimate > compact_limit * 4:
                chunk.token_estimate = estimate_tokens(chunk.text)

    return [c for c in chunks if len(c.text.strip()) >= 20]


def summarise_chunks(chunks: list[Chunk]) -> dict[str, int]:
    if not chunks:
        return {"chunks": 0, "tokens": 0, "min": 0, "max": 0, "avg": 0}
    tokens = [c.token_estimate for c in chunks]
    return {
        "chunks": len(chunks),
        "tokens": sum(tokens),
        "min": min(tokens),
        "max": max(tokens),
        "avg": sum(tokens) // len(tokens),
    }


# --- Dangling-label filter ---------------------------------------------------
# Aggregator pages render a fact summary as a card ("Min. for SIP ₹100 ...") *and*
# repeat each label on its own ("Min. for SIP"). A heading section that contains only
# such a repeated label carries no fact, yet it embeds the query terms and can outrank
# the chunk that actually holds the value. Those sections are dropped.
_VALUE_PATTERN = re.compile(r"\d|₹|Rs\.|%")


def _section_text(blocks: list[Block]) -> str:
    return " ".join(block.text.strip() for block in blocks if block.text.strip())


def _is_dangling_label(heading: str, blocks: list[Block], seen_text: str) -> bool:
    text = _section_text(blocks)
    if not text:
        return False
    if _VALUE_PATTERN.search(text):
        return False
    if len(text) > 80 or not _is_compact_fact(f"{heading} {text}"):
        return False
    label = (heading or text).strip().lower()
    return bool(label) and label in seen_text


def drop_dangling_label_sections(sections: list[tuple[list[str], list[Block]]]) -> list[tuple[list[str], list[Block]]]:
    """Remove value-free label sections already present verbatim in earlier content."""
    kept: list[tuple[list[str], list[Block]]] = []
    accumulated: list[str] = []
    for heading_path, blocks in sections:
        heading = heading_path[-1] if heading_path else ""
        body = _section_text(blocks)
        if _is_dangling_label(heading, blocks, " ".join(accumulated).lower()):
            continue
        if body:
            accumulated.append(body.lower())
        kept.append((heading_path, blocks))
    return kept
