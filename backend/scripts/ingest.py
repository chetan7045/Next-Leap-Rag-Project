#!/usr/bin/env python
"""Ingest the configured corpus into ChromaDB.

    python scripts/ingest.py                # add/update only what changed
    python scripts/ingest.py --force        # re-embed everything
    python scripts/ingest.py --prune        # also delete documents removed from sources.json
    python scripts/ingest.py --source ID    # ingest a single source (repeatable)

Ingestion is intentionally a local/admin CLI operation. There is no public endpoint
for it, so anonymous Internet users cannot trigger fetches or embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import IngestionError  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.services.embeddings.embedding_service import EmbeddingService  # noqa: E402
from app.services.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.services.ingestion.registry import get_registry  # noqa: E402
from app.services.rag.chroma_service import ChromaService  # noqa: E402

logger = get_logger("scripts.ingest")

_STATUS_ICON = {
    "indexed": "+",
    "replaced": "~",
    "unchanged": "=",
    "failed": "!",
    "empty": "?",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest configured sources into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Re-embed even when content is unchanged.")
    parser.add_argument("--prune", action="store_true", help="Delete documents no longer in sources.json.")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Restrict to one source id (repeatable).",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Ingestion starting (force=%s, prune=%s)", args.force, args.prune)

    registry = get_registry(settings)
    logger.info("Registry version %s: %d schemes, %d configured sources", registry.version, registry.scheme_count(), len(registry.sources()))

    chroma = ChromaService(settings)
    chroma.initialize()
    logger.info("Collection '%s' currently holds %d chunks", settings.chroma_collection, chroma.count())

    if not registry.has_only_reference_sources():
        logger.info("Corpus includes authoritative (AMC/AMFI/SEBI) sources.")

    EmbeddingService.get_instance(settings).load()
    logger.info("Embedding model ready: %s", settings.embedding_model)

    pipeline = IngestionPipeline(
        settings,
        registry=registry,
        chroma=chroma,
        embeddings=EmbeddingService.get_instance(settings),
    )

    try:
        report = pipeline.run(force=args.force, prune=args.prune, source_ids=args.sources)
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc.detail)
        return 1

    print("\n" + "=" * 78)
    print(f"{'SOURCE ID':<40} {'STATUS':<10} {'CHUNKS':>6}  TITLE")
    print("-" * 78)
    for outcome in report.documents:
        icon = _STATUS_ICON.get(outcome.status, "?")
        print(
            f"{icon} {outcome.source_id:<38} {outcome.status:<10} {outcome.chunks:>6}  "
            f"{(outcome.title or outcome.url)[:44]}"
        )
        if outcome.reason:
            print(f"    reason: {outcome.reason}")
    print("-" * 78)
    print(report.summary())
    print(f"collection now holds {chroma.count()} chunks")
    print("=" * 78 + "\n")

    if report.failed:
        logger.error("%d source(s) failed. Ingestion did not fully succeed.", report.failed)
        return 2
    if report.indexed == 0 and report.unchanged == 0:
        logger.error("No documents were indexed and none were unchanged — the corpus is empty.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
