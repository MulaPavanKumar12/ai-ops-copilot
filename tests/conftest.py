"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aiops_copilot.generator.dataset import DatasetConfig, DatasetResult, generate_dataset


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> DatasetResult:
    """A 12-hour dataset with enough incidents for every type to appear."""
    out: Path = tmp_path_factory.mktemp("synthetic")
    config = DatasetConfig(
        seed=7,
        start=datetime(2026, 1, 5, tzinfo=UTC),
        duration_hours=12,
        step_seconds=60,
        n_servers=6,
        incidents_per_day=24,  # 12 incidents in 12h -> all 10 types present
        output_dir=out,
        run_name="pytest_run",
    )
    return generate_dataset(config)


# ----------------------------------------------------------------------------- ML fixtures
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import pandas as pd  # noqa: E402

from aiops_copilot.core.config import Settings  # noqa: E402
from aiops_copilot.detection.hybrid import HybridDetector  # noqa: E402
from aiops_copilot.evaluation.run_detection import Split, ensure_split, load_split  # noqa: E402
from aiops_copilot.rag.build import build_knowledge_base, kb_dir  # noqa: E402
from aiops_copilot.rag.knowledge_base import KnowledgeBase  # noqa: E402


@dataclass
class Trained:
    """A small trained detector plus the held-out test split."""

    detector: HybridDetector
    test: pd.DataFrame
    test_logs: list[dict[str, Any]]
    test_incidents: list[dict[str, Any]]
    model_dir: Path


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    """Train on 36h, calibrate on 24h, test on 24h (no LSTM, to keep the suite fast)."""
    root = tmp_path_factory.mktemp("eval")
    splits = {
        "train": Split("train", 101, 36, datetime(2026, 1, 5, tzinfo=UTC)),
        "validation": Split("validation", 202, 24, datetime(2026, 1, 7, tzinfo=UTC)),
        "test": Split("test", 303, 24, datetime(2026, 1, 8, tzinfo=UTC)),
    }
    dirs = {name: ensure_split(s, root, 16.0) for name, s in splits.items()}
    train, _, _ = load_split(dirs["train"])
    val, _, _ = load_split(dirs["validation"])
    test, test_logs, test_incidents = load_split(dirs["test"])
    detector = HybridDetector(use_lstm=False).fit(train)
    detector.calibrate(val)
    model_dir = root / "models"
    detector.save(model_dir)
    return Trained(detector, test, test_logs, test_incidents, model_dir)


@pytest.fixture(scope="session")
def kb_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointing at a temporary knowledge base (in-memory vector store)."""
    root = tmp_path_factory.mktemp("kb")
    return Settings(_env_file=None, data_dir=root / "data", model_dir=root / "models",  # type: ignore[call-arg]
                    reports_dir=root / "reports", vector_store="memory", llm_provider="none")


@pytest.fixture(scope="session")
def knowledge_base(kb_settings: Settings) -> KnowledgeBase:
    """A knowledge base built from 8 days of simulated incident history plus the runbooks."""
    build_knowledge_base(kb_settings, days=8, incidents_per_day=8.0)
    return KnowledgeBase(kb_dir(kb_settings), "memory", "lsa").load()
