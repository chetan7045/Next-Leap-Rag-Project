"""HTML → structured content extraction.

Produces ordered blocks (headings, paragraphs, lists, tables) so the chunker can keep
a heading attached to the fact it labels. Nothing is discarded here; the cleaner owns
that decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

from app.models.retrieval import TableBlock

# Elements whose subtree is never meaningful scheme content.
_DROP_TAGS = (
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object",
    "embed", "form", "button", "select", "textarea", "input", "video", "audio",
)

# Class/id substrings that reliably mark chrome rather than content. Prefixes are
# matched loosely because sites use CSS-module names such as ``footerTopSection_x1``.
_CHROME_MARKERS = (
    "nav", "navbar", "menu", "header-", "footer", "sidebar", "cookie", "consent",
    "banner", "advert", "ads-", "promo", "popup", "modal", "breadcrumb", "pagination",
    "social", "share", "newsletter", "subscribe", "toast", "tooltip", "skiplink",
    "related-", "recommend", "trending", "searchbar", "login", "signup", "hiddentab",
    "showmore", "letterlinks", "sticky", "drawer", "hamburger", "screenreader",
    # Reference-site (aggregator) chrome wrappers.
    "dropdownui", "loggedout", "loggedin", "hoverdiv", "icongrid", "tablist",
    "vis-hidden", "a11y", "sr-only", "skip-to", "backtotop", "floating", "announcement",
)

_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "li", "tr", "br", "h1", "h2", "h3",
    "h4", "h5", "h6", "td", "th", "ul", "ol", "table", "blockquote", "pre",
}


@dataclass(slots=True)
class Block:
    kind: str  # heading | paragraph | list_item | table
    text: str
    level: int = 0
    table: TableBlock | None = None


@dataclass(slots=True)
class ExtractionResult:
    title: str
    blocks: list[Block] = field(default_factory=list)
    meta_description: str | None = None
    published_hint: str | None = None


def _marker_blob(tag: Tag) -> str:
    attrs = getattr(tag, "attrs", None)
    if not isinstance(attrs, dict):
        return ""
    return " ".join(
        filter(None, [str(attrs.get("class") or ""), str(attrs.get("id") or ""), str(attrs.get("role") or "")])
    ).lower()


def _is_chrome(tag: Tag) -> bool:
    blob = _marker_blob(tag)
    return bool(blob) and any(marker in blob for marker in _CHROME_MARKERS)


def _has_chrome_ancestor(tag: Tag, *, max_depth: int = 8) -> bool:
    """Chrome is usually a wrapper: ``<div class="footerTopSection">…<li>link</li>``.

    Checking only the element itself would let those links through, so ancestors are
    inspected too. Depth is bounded to keep this cheap on large pages.
    """
    depth = 0
    parent = tag.parent
    while parent is not None and depth < max_depth:
        if isinstance(parent, Tag) and _is_chrome(parent):
            return True
        parent = parent.parent
        depth += 1
    return False


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_table(tag: Tag) -> TableBlock | None:
    rows: list[list[str]] = []
    header: list[str] = []
    for tr in tag.find_all("tr"):
        cells = [_collapse(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if not cells:
            continue
        has_th = bool(tr.find_all("th"))
        if not rows and (has_th or len(cells) <= 3):
            header = cells
        else:
            rows.append(cells)
    if not header and not rows:
        return None
    if rows and not header and rows:
        # First data row often acts as the header for finance tables.
        header, rows = rows[0], rows[1:]
    caption_tag = tag.find("caption")
    caption = _collapse(caption_tag.get_text(" ", strip=True)) if caption_tag else None
    if len(rows) > 40:
        rows = rows[:40]
    return TableBlock(caption=caption, header=header, rows=rows)


def _extract_date_hint(soup: BeautifulSoup) -> str | None:
    candidates: list[str] = []
    for tag in soup.find_all(["time", "meta"]):
        if tag.name == "time":
            value = _collapse(tag.get_text(" ", strip=True))
            if value:
                candidates.append(value)
        else:
            name = str(tag.get("property") or tag.get("name") or "").lower()
            if any(token in name for token in ("published", "modified", "date", "updated")):
                value = tag.get("content")
                if value:
                    candidates.append(str(value).strip())
    for candidate in candidates:
        match = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", candidate)
        if match:
            return match.group(1)
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", candidate)
        if match:
            return match.group(1)
        match = re.search(r"\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b", candidate)
        if match:
            return match.group(1)
    return None


def extract(html: str, *, base_title: str | None = None) -> ExtractionResult:
    """Parse HTML into ordered, typed content blocks."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = _collapse(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        h1_text = _collapse(h1.get_text(" ", strip=True))
        if h1_text and (not title or len(h1_text) > len(title)):
            title = title or h1_text
    if not title:
        title = base_title or ""
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content") and len(str(og_title["content"])) > len(title):
        title = _collapse(str(og_title["content"]))

    description = None
    meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if meta_desc and meta_desc.get("content"):
        description = _collapse(str(meta_desc["content"]))

    published_hint = _extract_date_hint(soup)

    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[Block] = []
    seen_texts: set[str] = set()

    for element in root.find_all(list(_BLOCK_TAGS) + ["h1", "h2", "h3", "h4", "h5", "h6"], recursive=True):
        if not isinstance(element, Tag) or getattr(element, "attrs", None) is None:
            continue
        # Skip nodes living inside chrome or inside another element we already walked
        # as a leaf (e.g. a <p> nested in a table cell handled separately).
        if _is_chrome(element) or _has_chrome_ancestor(element):
            continue
        if element.find_parent("table") is not None and element.name != "tr":
            continue
        if element.find_parent(_DROP_TAGS) is not None:
            continue

        name = element.name.lower()
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            text = _collapse(element.get_text(" ", strip=True))
            if text and len(text) < 200 and text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                blocks.append(Block(kind="heading", text=text, level=int(name[1])))
            continue

        if name == "table":
            table = _extract_table(element)
            if table and len(table.header) + len(table.rows) >= 2:
                text = table.to_text()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    blocks.append(Block(kind="table", text=text, table=table))
            # Cells are skipped by the find_parent("table") guard above. The element is
            # never decomposed here: that would clear the attributes of tags still
            # pending in this iteration.
            continue

        if element.find(["table", "ul", "ol"]) is not None:
            continue

        if name == "li":
            text = _collapse(element.get_text(" ", strip=True))
            if text and text not in seen_texts:
                seen_texts.add(text)
                blocks.append(Block(kind="list_item", text=text))
            continue

        nested_blocks = element.find(list(_BLOCK_TAGS))
        if nested_blocks is not None:
            # A wrapper: normally skipped because its children carry the content.
            # Fund pages, however, put the whole fact summary (NAV, min SIP, AUM,
            # expense ratio, rating) in one wrapper whose children are label-only
            # fragments. Emit the wrapper's text as a fact card when it is dense
            # enough to be useful and short enough to stay coherent.
            text = _collapse(element.get_text(" ", strip=True))
            if 40 <= len(text) <= 700 and _looks_fact_dense(text):
                if text not in seen_texts:
                    seen_texts.add(text)
                    blocks.append(Block(kind="paragraph", text=text))
            continue

        text = _collapse(element.get_text(" ", strip=True))
        if not text or len(text) < 2:
            continue
        if text.lower() in seen_texts:
            continue
        seen_texts.add(text.lower())
        kind = "heading" if _looks_like_label(text) else "paragraph"
        if (
            kind == "heading"
            and blocks
            and blocks[-1].kind == "heading"
            and _is_fact_label(blocks[-1].text)
        ):
            # Aggregator pages render a fact as label then value on separate lines
            # ("Exit load" / "Nil"). The value looks like a label too, so it is
            # demoted to a paragraph to keep the label and its value together.
            kind = "paragraph"
        blocks.append(Block(kind=kind, text=text))

    return ExtractionResult(
        title=title,
        blocks=blocks,
        meta_description=description,
        published_hint=published_hint,
    )


_LABEL_RE = re.compile(r"^[A-Z][A-Za-z /&.\-]{2,44}:?$")

# Tokens that indicate a block carries scheme facts rather than prose.
_FACT_DENSITY_MARKERS = (
    "₹", "rs.", "%", "cr", "nav", "aum", "expense ratio", "exit load", "min. for sip",
    "minimum sip", "lumpsum", "benchmark", "rating", "risk", "lock-in", "lock in",
    "holdings", "category", "direct growth", "as on", "as of", "sector",
)


def _looks_fact_dense(text: str) -> bool:
    """True when a text block packs several labelled financial values."""
    lowered = text.lower()
    hits = sum(1 for marker in _FACT_DENSITY_MARKERS if marker in lowered)
    digit_hits = len(re.findall(r"\d", text))
    return hits >= 3 and digit_hits >= 3


#: Labels whose following line is the fact's value rather than a new section.
_FACT_LABEL_RE = re.compile(
    r"^(?:exit\s*load|entry\s*load|expense\s*ratio|benchmark|lock[\s-]*in(?:\s*period)?|"
    r"min\.?\s*(?:for|of)?\s*(?:sip|lumpsum|investment|amount)|"
    r"risk\s*(?:ometer)?|fund\s*category|plan\s*option|aum|nav|fund\s*manager|"
    r"tax\s*(?:status|benefit)|direct\s*growth|securities\s*transmitted)$",
    re.I,
)


def _is_fact_label(text: str) -> bool:
    """True for a label that owns the line beneath it ("Exit load" / "Nil")."""
    return bool(_FACT_LABEL_RE.match(text.strip()))


def _looks_like_label(text: str) -> bool:
    """Detect 'Expense ratio' style labels that render as their own block."""
    if len(text) > 45:
        return False
    if text.endswith((".", ",", ";")):
        return False
    return bool(_LABEL_RE.match(text))


def block_plain_text(blocks: list[Block]) -> str:
    return "\n\n".join(b.text for b in blocks if b.text)


def iter_strings(node: Tag) -> list[NavigableString]:  # pragma: no cover - helper
    return [n for n in node.descendants if isinstance(n, NavigableString)]
