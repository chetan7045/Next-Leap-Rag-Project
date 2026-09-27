#!/usr/bin/env python
"""Developer retrieval debugger. Not exposed in any API route or the UI.

    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --scheme HDFC_ELSS
    python scripts/test_retrieval.py "what is the exit load" "minimum sip amount"
    python scripts/test_retrieval.py --all          # every question in evaluation/questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.services.embeddings.embedding_service import EmbeddingService  # noqa: E402
from app.services.ingestion.registry import get_registry  # noqa: E402
from app.services.rag.chroma_service import ChromaService  # noqa: E402
from app.services.rag.retriever import Retriever  # noqa: E402
from app.services.safety.scheme_resolver import resolve_scheme  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is the minimum SIP for HDFC Large Cap Fund?",
    "What is the exit load for HDFC ELSS Tax Saver Fund?",
    "What is the benchmark of HDFC Small Cap Fund?",
    "What is the lock-in period for HDFC ELSS?",
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the exit load?",
    "Should I invest in HDFC Large Cap Fund?",
    "Which of these funds will give the highest return?",
    "What is HDFC Small Cap Fund's NAV 10 years from now?",
]


def _questions_from_evaluation() -> list[str]:
    path = REPO_ROOT / "evaluation" / "questions.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = data.get("questions", data) if isinstance(data, dict) else data
    return [item["question"] for item in items if isinstance(item, dict) and item.get("question")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect retrieval results for a question.")
    parser.add_argument("questions", nargs="*", help="Questions to test.")
    parser.add_argument("--scheme", help="Force a scheme filter, e.g. HDFC_ELSS.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--all", action="store_true", help="Run the evaluation question set.")
    parser.add_argument("--full", action="store_true", help="Print full chunk text.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")

    questions = list(args.questions)
    if args.all:
        questions = _questions_from_evaluation() or DEFAULT_QUESTIONS
    if not questions:
        questions = DEFAULT_QUESTIONS

    registry = get_registry(settings)
    chroma = ChromaService(settings)
    chroma.initialize()
    total = chroma.count()
    if total == 0:
        print("The index is empty. Run: python scripts/ingest.py")
        return 1
    print(f"Collection '{settings.chroma_collection}' holds {total} chunks across "
          f"{len(chroma.document_ids())} documents (threshold={settings.similarity_threshold})\n")

    embeddings = EmbeddingService.get_instance(settings)
    retriever = Retriever(settings, chroma, embeddings)
    available = {s.id for s in registry.schemes()}

    for question in questions:
        print("=" * 100)
        print(f"QUERY: {question}")
        forced = args.scheme if args.scheme in available else None
        provisional, _ = retriever.retrieve(question)
        resolution = resolve_scheme(question, registry, forced, provisional)
        print(
            f"  scheme: {resolution.scheme_id or '-'}  (reason: {resolution.reason}"
            f"{', ambiguous' if resolution.ambiguous else ''})"
        )
        result, timing = retriever.retrieve(
            question, scheme_id=forced or resolution.scheme_id, top_k=args.top_k, threshold=args.threshold
        )
        print(
            f"  confidence: {result.confidence:.3f} | candidates: {result.candidates_considered} "
            f"| kept: {len(result.hits)} | {timing.total_ms:.0f}ms"
        )
        if not result.hits:
            print("  RESULT: below threshold — the LLM would not be called.\n")
            continue
        for index, hit in enumerate(result.hits, start=1):
            meta = hit.metadata
            print(f"  --- Result {index} ---")
            print(f"    Scheme : {meta.scheme_name} [{meta.scheme_id}]")
            print(f"    Score  : {hit.score:.4f}")
            print(f"    Source : {meta.source_title}")
            print(f"    Type   : {meta.source_type} (authority {meta.authority_level})")
            print(f"    Doc    : {meta.document_type} | chunk {hit.ordinal} | id {hit.chunk_id[:12]}")
            print(f"    URL    : {meta.source_url}")
            print(f"    Updated: {meta.last_updated_at or meta.published_at or 'not available'}")
            if hit.heading_path:
                print(f"    Section: {' > '.join(hit.heading_path)}")
            preview = hit.text if args.full else hit.text[:420].replace("\n", " | ")
            suffix = "" if args.full else ("..." if len(hit.text) > 420 else "")
            print(f"    Chunk  : {preview}{suffix}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
