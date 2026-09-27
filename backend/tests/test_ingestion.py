"""Ingestion correctness: extraction, cleaning, chunking, provenance, loader guards.

These run against a small HTML fixture so they stay fast and deterministic. The
end-to-end behaviour against real pages is exercised by ``scripts/ingest.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.config import Settings
from app.models.enums import DocumentType, SourceType
from app.models.retrieval import Document, Section
from app.services.ingestion.chunker import (
    chunk_document,
    drop_dangling_label_sections,
    estimate_tokens,
)
from app.services.ingestion.cleaner import clean_extraction, normalise_text, strip_leading_navigation
from app.services.ingestion.extractor import Block, ExtractionResult, extract
from app.services.ingestion.loader import DocumentLoader
from app.services.ingestion.metadata import (
    build_metadata,
    compute_content_hash,
    extract_source_date,
    make_document_id,
)

SCHEME_HTML = """
<html><head><title>HDFC Large Cap Fund Direct Growth - NAV, Mutual Fund Performance</title></head>
<body>
  <nav><a href="/">Home</a><a href="/mf">Funds</a><a href="/login">Log in</a></nav>
  <div class="fyiel-accordion__content--fund-information">
    <h1>HDFC Large Cap Fund Direct Growth</h1>
    <p>HDFC Large Cap Fund Direct Growth is a Equity Mutual Fund Scheme launched by HDFC
       Mutual Fund. This scheme was made available to investors on 10 Dec 1999.</p>
    <p>NAV: 25 Sep '26 &#8377;1,189.08 Min. for SIP &#8377;100 Fund size (AUM) &#8377;39,933.37 Cr
       Expense ratio 1.03% Rating 4</p>
    <p>Exit load: 1% if units are redeemed within 1 year of allotment, nil thereafter.</p>
    <h2>Benchmark</h2>
    <p>NIFTY 100 Total Return Index</p>
    <h2>Investment Objective</h2>
    <p>The investment objective of the scheme is to generate long term capital appreciation
       from appreciation in the value of securities.</p>
  </div>
  <footer><p>&copy; Groww. All rights reserved.</p></footer>
</body></html>
"""

RETRIEVED_AT = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def extracted() -> ExtractionResult:
    return clean_extraction(extract(SCHEME_HTML))


def _all_text(result: ExtractionResult) -> str:
    return normalise_text(" ".join(block.text for block in result.blocks))


# --- Extraction ---------------------------------------------------------------
def test_extraction_keeps_the_core_facts(extracted: ExtractionResult):
    text = _all_text(extracted)

    assert "HDFC Large Cap Fund Direct Growth" in text
    assert "1,189.08" in text
    assert "Min. for SIP ₹100" in text
    assert "Expense ratio 1.03%" in text
    assert "Exit load: 1%" in text
    assert "NIFTY 100 Total Return Index" in text
    assert "10 Dec 1999" in text


def test_extraction_drops_navigation_and_footer(extracted: ExtractionResult):
    text = _all_text(extracted).lower()

    assert "log in" not in text
    assert "all rights reserved" not in text


def test_extraction_is_deterministic():
    first = extract(SCHEME_HTML)
    second = extract(SCHEME_HTML)

    assert [b.text for b in first.blocks] == [b.text for b in second.blocks]
    assert first.title == second.title


def test_empty_html_yields_no_blocks():
    assert extract("<html><body><div id='app'></div></body></html>").blocks == []


# --- Cleaning -----------------------------------------------------------------
def test_normalise_text_collapses_whitespace_and_drops_nul():
    """Newlines are preserved deliberately; horizontal runs collapse to one space."""
    assert normalise_text("  a \n\n b\x00  c ") == "a\n\nb c"
    assert normalise_text("a \t\u00a0 b") == "a b"


def test_strip_leading_navigation_removes_only_the_prefix():
    blocks = [
        Block(kind="paragraph", text="Home", level=None),
        Block(kind="paragraph", text="Funds", level=None),
        Block(kind="paragraph", text="Log in", level=None),
        Block(kind="paragraph", text="Min. for SIP ₹100"),
    ]

    assert strip_leading_navigation(blocks)[0].text == "Min. for SIP ₹100"


def test_a_single_short_leading_label_is_kept_as_content():
    """One short label may be a real heading, so a lone match is not stripped."""
    blocks = [
        Block(kind="paragraph", text="Overview", level=None),
        Block(kind="paragraph", text="Min. for SIP ₹100"),
    ]

    assert len(strip_leading_navigation(blocks)) == 2


# --- Chunking -----------------------------------------------------------------
def test_chunker_produces_bounded_chunks(settings: Settings, registry, extracted):
    source = registry.source("hdfc-large-cap-scheme-page")
    metadata = build_metadata(
        source,
        title=extracted.title,
        text=_all_text(extracted),
        retrieved_at=RETRIEVED_AT,
        settings=settings,
    )
    document = _document(extracted, metadata)
    chunks = chunk_document(document, settings)

    assert chunks, "a populated page must produce chunks"
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.token_estimate <= settings.chunk_target_tokens
        assert chunk.document_id == document.metadata.document_id
        assert estimate_tokens(chunk.text) > 0


def test_chunk_ids_are_unique_within_a_document(settings: Settings, registry, extracted):
    source = registry.source("hdfc-large-cap-scheme-page")
    metadata = build_metadata(
        source,
        title=extracted.title,
        text=_all_text(extracted),
        retrieved_at=RETRIEVED_AT,
        settings=settings,
    )
    chunks = chunk_document(_document(extracted, metadata), settings)

    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_dangling_label_sections_are_dropped():
    """A label repeated on its own must not become a value-free chunk."""
    sections = [
        (["Fund", "All"], [Block(kind="paragraph", text="Min. for SIP ₹100 Expense ratio 1.03%")]),
        (["Fund", "Min. for SIP"], [Block(kind="paragraph", text="Min. for SIP")]),
    ]

    assert len(drop_dangling_label_sections(sections)) == 1


def test_prose_sections_survive_the_dangling_label_filter():
    sections = [
        (["Fund", "About"], [Block(kind="paragraph", text="Min. for SIP ₹100")]),
        (
            ["Fund", "Investment Objective"],
            [Block(kind="paragraph", text="The objective is long term capital appreciation.")],
        ),
    ]

    assert len(drop_dangling_label_sections(sections)) == 2


def test_hyphenated_lock_in_is_a_retrievable_fact(settings: Settings, registry):
    """Regression: "Lock-in" never matched the compact-fact pattern, so the ELSS
    lock-in period was impossible to retrieve. The fact also has to name its scheme
    to survive the minimum chunk length."""
    document = _document_for(
        registry,
        "hdfc-elss-scheme-page",
        [Block(kind="paragraph", text="ELSS • 3Y Lock-in")],
    )

    texts = [c.text for c in chunk_document(document, settings)]

    assert any("Lock-in" in text for text in texts), texts
    assert any("HDFC ELSS" in text for text in texts if "Lock-in" in text)


def test_cross_sell_scheme_names_are_dropped_but_the_page_title_is_kept():
    """Regression: cross-sell carousels list other schemes by name only and were
    indexed as facts, crowding out real results."""
    result = ExtractionResult(
        title="HDFC Large Cap Fund Direct Growth",
        blocks=[
            Block(kind="heading", text="HDFC Large Cap Fund Direct Growth"),
            Block(kind="heading", text="HDFC Multi Asset Allocation Fund Direct Growth"),
            Block(kind="paragraph", text="HDFC Equity Savings Direct Plan Growth"),
            Block(kind="paragraph", text="The scheme invests in large cap equities."),
        ],
    )

    texts = [b.text for b in clean_extraction(result).blocks]

    assert "HDFC Large Cap Fund Direct Growth" in texts
    assert "HDFC Multi Asset Allocation Fund Direct Growth" not in texts
    assert "HDFC Equity Savings Direct Plan Growth" not in texts
    assert "The scheme invests in large cap equities." in texts


def test_a_bare_fact_value_survives_immediately_after_its_label():
    """Regression: the ELSS exit load renders as "Exit load" / "Nil". The value was
    demoted to a paragraph and then dropped as a short line, losing the fact."""
    result = ExtractionResult(
        title="HDFC ELSS Tax Saver Fund Direct Plan Growth",
        blocks=[
            Block(kind="heading", text="Exit load, stamp duty and tax"),
            Block(kind="heading", text="Exit load"),
            Block(kind="paragraph", text="Nil"),
            Block(kind="heading", text="Tax implication"),
        ],
    )

    cleaned = clean_extraction(FACT_VALUE_HTML)

    assert [b.text for b in cleaned.blocks][-3:] == ["Exit load", "Nil", "Tax implication"]


def test_a_fact_value_line_is_demoted_from_heading_to_content():
    """A value rendered on its own line must not be typed as a section heading, or the
    label above it ends up with no body to chunk."""
    html = """
    <html><body><h1>HDFC ELSS Tax Saver Fund Direct Plan Growth</h1>
    <div>Exit load</div><div>Nil</div>
    </body></html>
    """

    blocks = extract(html).blocks

    assert [b.kind for b in blocks if b.text in {"Exit load", "Nil"}] == [
        "heading",
        "paragraph",
    ]


FACT_VALUE_HTML = ExtractionResult(
    title="HDFC ELSS Tax Saver Fund Direct Plan Growth",
    blocks=[
        Block(kind="heading", text="Exit load, stamp duty and tax"),
        Block(kind="heading", text="Exit load"),
        Block(kind="paragraph", text="Nil"),
        Block(kind="heading", text="Tax implication"),
    ],
)


def _document_for(registry, source_id: str, blocks: list[Block]):
    """Wrap blocks in a real Document so chunker behaviour is exercised end to end."""
    source = registry.source(source_id)
    text = "\n".join(b.text for b in blocks)
    metadata = build_metadata(
        source,
        title=source.scheme_name,
        text=text,
        retrieved_at=datetime(2026, 9, 27, tzinfo=timezone.utc),
        date_hint=None,
    )
    return Document(
        metadata=metadata,
        sections=[Section(heading=source.scheme_name, blocks=blocks)],
        text=text,
    )


# --- Provenance ---------------------------------------------------------------
def test_metadata_never_invents_official_provenance(settings: Settings, registry):
    source = registry.source("hdfc-large-cap-scheme-page")
    metadata = build_metadata(
        source,
        title="HDFC Large Cap Fund Direct Growth",
        text="Min. for SIP ₹100 Expense ratio 1.03%",
        retrieved_at=RETRIEVED_AT,
        settings=settings,
    )

    assert metadata.source_type is SourceType.REFERENCE
    assert metadata.authority_level == 3
    assert metadata.amc == "HDFC Mutual Fund"
    assert metadata.document_type is DocumentType.SCHEME_PAGE
    assert metadata.source_url.startswith("https://")


def test_content_hash_and_document_id_are_stable(settings: Settings, registry):
    source = registry.source("hdfc-large-cap-scheme-page")
    kwargs = {
        "title": "HDFC Large Cap Fund Direct Growth",
        "text": "Min. for SIP ₹100",
        "retrieved_at": RETRIEVED_AT,
        "settings": settings,
    }

    first = build_metadata(source, **kwargs)
    second = build_metadata(source, **kwargs)

    assert first.content_hash == second.content_hash == compute_content_hash("Min. for SIP ₹100")
    assert first.document_id == second.document_id
    assert first.document_id == make_document_id(source.scheme_id, source.url)


def test_source_date_is_parsed_when_the_page_states_one():
    assert extract_source_date("NAV as of 25 Sep 2026 is ₹1,189.08") is not None


def test_source_date_is_none_when_absent():
    assert extract_source_date("No dates anywhere in this text.") is None


def test_source_date_is_never_substituted_with_today(settings: Settings, registry):
    """A page with no stated date must report no date — never today's date."""
    source = registry.source("hdfc-large-cap-scheme-page")
    metadata = build_metadata(
        source,
        title="HDFC Large Cap Fund Direct Growth",
        text="This page states no date at all.",
        retrieved_at=RETRIEVED_AT,
        settings=settings,
    )

    assert metadata.published_at is None
    assert metadata.effective_updated is None


def test_retrieval_time_is_never_used_as_the_source_date(settings: Settings, registry):
    source = registry.source("hdfc-large-cap-scheme-page")
    metadata = build_metadata(
        source,
        title="T",
        text="No date here.",
        retrieved_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        settings=settings,
    )

    assert metadata.retrieved_at.startswith("2020-01-01")
    assert metadata.effective_updated is None


# --- Loader -------------------------------------------------------------------
def test_loader_rejects_a_url_outside_the_allow_list(settings: Settings, registry):
    from app.core.errors import SourceFetchError

    loader = DocumentLoader(settings, allowed_urls=registry.allowed_urls())

    with pytest.raises(SourceFetchError):
        loader.fetch("https://evil.example.com/mutual-funds/hdfc-large-cap")


def test_loader_rejects_a_url_not_in_the_registry(settings: Settings, registry):
    from app.core.errors import SourceFetchError

    # No allow-list at all still refuses URLs the registry does not contain.
    loader = DocumentLoader(settings, allowed_urls=registry.allowed_urls())

    with pytest.raises(SourceFetchError):
        loader.fetch("http://127.0.0.1:8080/admin")


# --- Helpers ------------------------------------------------------------------
def _document(result: ExtractionResult, metadata):
    from app.models.retrieval import Document, Section

    return Document(
        metadata=metadata,
        text=_all_text(result),
        sections=[Section(heading=result.title, level=1, blocks=result.blocks)],
    )
