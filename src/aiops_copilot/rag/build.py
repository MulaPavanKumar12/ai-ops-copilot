"""Build the incident knowledge base (post-mortems from a 30-day history + runbooks).

Usage::

    python -m aiops_copilot.rag.build
    python -m aiops_copilot.rag.build --days 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from aiops_copilot.core.config import PROJECT_ROOT, Settings, get_settings
from aiops_copilot.core.logging import setup_logging
from aiops_copilot.generator.dataset import DatasetConfig, generate_dataset
from aiops_copilot.rag.knowledge_base import KnowledgeBase, reports_from_history, runbooks_from_dir

logger = logging.getLogger("aiops_copilot.rag")

HISTORY_SEED = 404
RUNBOOK_DIR = PROJECT_ROOT / "knowledge_base" / "runbooks"


def kb_dir(settings: Settings) -> Path:
    """Where the knowledge base index lives."""
    return settings.data_dir / "kb"


def build_knowledge_base(settings: Settings, days: int = 30, incidents_per_day: float = 6.0) -> KnowledgeBase:
    """Generate the history dataset (if missing) and index reports + runbooks."""
    history_root = settings.data_dir / "history"
    run_name = f"history_{days}d"
    if not (history_root / run_name / "manifest.json").exists():
        generate_dataset(DatasetConfig(seed=HISTORY_SEED, start=datetime(2025, 11, 1, tzinfo=UTC),
                                       duration_hours=24 * days, incidents_per_day=incidents_per_day,
                                       output_dir=history_root, run_name=run_name))
    docs = reports_from_history(history_root / run_name) + runbooks_from_dir(RUNBOOK_DIR)
    kb = KnowledgeBase(kb_dir(settings), settings.vector_store, settings.embedding_backend)
    kb.build(docs)
    return kb


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30, help="Length of simulated incident history (7-180)")
    args = parser.parse_args(argv)
    if not 7 <= args.days <= 180:
        logger.error("--days must be between 7 and 180")
        return 2
    kb = build_knowledge_base(settings, args.days)
    hits = kb.search("failed SSH logins far above normal. Failed password for invalid user root from <ip>", 3)
    print(f"Knowledge base ready in {kb.directory}. Example search (brute force):")
    for h in hits:
        print(f"  {h.score:.3f}  {h.metadata.get('title')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
