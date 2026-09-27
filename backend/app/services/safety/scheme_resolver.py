"""Scheme resolution: map a natural-language question to a scheme_id.

Two signals are used, in order:
  1. an explicit client-side selection (``scheme_id``), then
  2. exact alias / name matching against the registry.

When a question names no scheme and the fact is scheme-specific, the pipeline asks
for clarification instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.retrieval import RetrievalResult
from app.models.source import SchemeRecord
from app.services.ingestion.registry import SourceRegistry

# Facts that only make sense for one specific scheme.
SCHEME_SPECIFIC_FACT = re.compile(
    r"\b(?:exit\s*load|expense\s*ratio|minimum\s*(?:sip|investment|investment\s*amount)|"
    r"lock[\s-]*in|benchmark|riskometer|nav|aum|fund\s*manager|direct\s*growth|"
    r"minimum\s*additional|securities\s*transmitted|minimum\s*balance)\b",
    re.I,
)

# Generic, scheme-independent questions that may be answered across the corpus.
GENERAL_QUESTION = re.compile(
    r"\b(?:compare|difference between|differences between|all\s+(?:hdfc\s+)?(?:funds|schemes)|"
    r"which\s+(?:amc|fund\s+house)|list\s+(?:the\s+)?(?:funds|schemes))\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class SchemeResolution:
    scheme_id: str | None
    scheme: SchemeRecord | None
    ambiguous: bool
    reason: str
    candidates: tuple[str, ...] = ()

    @property
    def needs_clarification(self) -> bool:
        return self.ambiguous


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_index(registry: SourceRegistry) -> list[tuple[str, str]]:
    """(alias, scheme_id) pairs ordered longest-first so specific aliases win."""
    pairs: list[tuple[str, str]] = []
    for scheme in registry.schemes():
        pairs.append((_normalise(scheme.name), scheme.id))
        for alias in scheme.aliases:
            pairs.append((_normalise(alias), scheme.id))
    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for alias, scheme_id in pairs:
        if alias and (alias, scheme_id) not in seen:
            seen.add((alias, scheme_id))
            unique.append((alias, scheme_id))
    return sorted(unique, key=lambda item: len(item[0]), reverse=True)


def resolve_scheme(
    query: str,
    registry: SourceRegistry,
    explicit_scheme_id: str | None = None,
    retrieved: RetrievalResult | None = None,
) -> SchemeResolution:
    """Resolve the target scheme for a question.

    Args:
        query: The user's question.
        registry: Configured schemes and aliases.
        explicit_scheme_id: Scheme selected by the client, if any.
        retrieved: Retrieval output, used as a tie-breaker when the query is vague.

    Returns:
        A :class:`SchemeResolution`. ``ambiguous=True`` means the caller must ask.
    """
    if explicit_scheme_id:
        scheme = registry.scheme(explicit_scheme_id)
        if scheme is None:
            return SchemeResolution(None, None, True, "unknown_scheme")
        return SchemeResolution(scheme.id, scheme, False, "explicit_selection")

    text = _normalise(query)
    if not text:
        return SchemeResolution(None, None, False, "empty_query")

    # 1. Alias / name match, longest alias first.
    for alias, scheme_id in _alias_index(registry):
        if alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
            scheme = registry.scheme(scheme_id)
            if scheme is not None:
                return SchemeResolution(scheme.id, scheme, False, "alias_match")

    # 2. A scheme-independent question: no filter, no clarification.
    if GENERAL_QUESTION.search(query):
        return SchemeResolution(None, None, False, "general_question")

    # 3. Retrieval tie-breaker: if the top hits agree on one scheme with real
    #    confidence, that is a confident identification, so answer.
    if retrieved is not None and retrieved.hits:
        top = retrieved.hits[0]
        runner_up = next(
            (h for h in retrieved.hits[1:] if h.metadata.scheme_id != top.metadata.scheme_id),
            None,
        )
        margin = top.score - (runner_up.score if runner_up else 0.0)
        if top.score >= 0.55 and margin >= 0.08:
            scheme = registry.scheme(top.metadata.scheme_id)
            if scheme is not None:
                return SchemeResolution(scheme.id, scheme, False, "retrieval_consensus")

    # 4. Scheme-specific fact with no scheme named: ask instead of guessing.
    if SCHEME_SPECIFIC_FACT.search(query):
        options = tuple(s.id for s in registry.schemes())
        return SchemeResolution(None, None, True, "ambiguous_scheme_specific_fact", options)

    # 5. Unknown. Let retrieval decide; no clarification demanded.
    return SchemeResolution(None, None, False, "unresolved")
