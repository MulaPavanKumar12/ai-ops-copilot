"""Build a labelled offline dataset and write it to disk.

Output layout (one folder per run)::

    data/synthetic/<run_name>/
        metrics.csv      one row per host per step, with is_anomaly / anomaly_type labels
        logs.jsonl       one JSON log record per line, with labels
        incidents.json   ground-truth incidents incl. root cause and suggested fix
        manifest.json    generation parameters + summary statistics

Files are streamed to disk step by step, so memory stays flat even for
multi-week datasets.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from aiops_copilot import __version__
from aiops_copilot.generator.fleet import build_fleet
from aiops_copilot.generator.incidents import BatchScheduler, IncidentTimeline
from aiops_copilot.generator.models import METRIC_NAMES
from aiops_copilot.generator.simulator import FleetSimulator

logger = logging.getLogger(__name__)

METRIC_COLUMNS: list[str] = [
    "timestamp", "host", "role", *METRIC_NAMES,
    "is_anomaly", "anomaly_type", "incident_id", "benign_event",
]
LATEST_POINTER = "LATEST"


@dataclass(frozen=True)
class DatasetConfig:
    """Parameters for one dataset run (validated on construction)."""

    seed: int = 42
    start: datetime = datetime(2026, 1, 5, tzinfo=UTC)
    duration_hours: float = 24.0
    step_seconds: int = 60
    n_servers: int = 6
    incidents_per_day: float = 14.0
    output_dir: Path = Path("data/synthetic")
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if not 0 < self.duration_hours <= 24 * 30:
            raise ValueError("duration_hours must be in (0, 720]")
        if not 10 <= self.step_seconds <= 3600:
            raise ValueError("step_seconds must be in [10, 3600]")
        if not 1 <= self.n_servers <= 60:
            raise ValueError("n_servers must be in [1, 60]")
        if not 0 <= self.incidents_per_day <= 200:
            raise ValueError("incidents_per_day must be in [0, 200]")
        if self.run_name is not None and not _is_safe_name(self.run_name):
            raise ValueError("run_name may only contain letters, digits, '-', '_' and '.'")

    @property
    def resolved_run_name(self) -> str:
        """Run folder name; deterministic from parameters when not given."""
        if self.run_name:
            return self.run_name
        hours = f"{self.duration_hours:g}h"
        return f"seed{self.seed}_{hours}_{self.n_servers}hosts"

    @property
    def n_incidents(self) -> int:
        """Number of incidents implied by the per-day rate."""
        return int(round(self.incidents_per_day * self.duration_hours / 24.0))


@dataclass
class DatasetResult:
    """Paths and statistics of a finished dataset run."""

    run_dir: Path
    metrics_path: Path
    logs_path: Path
    incidents_path: Path
    manifest_path: Path
    summary: dict[str, Any] = field(default_factory=dict)


def _is_safe_name(name: str) -> bool:
    """Reject path traversal and odd characters in user-supplied folder names."""
    return bool(name) and name not in {".", ".."} and all(
        c.isalnum() or c in "-_." for c in name
    )


def generate_dataset(config: DatasetConfig) -> DatasetResult:
    """Generate a labelled dataset and write it to ``config.output_dir``.

    Args:
        config: Generation parameters.

    Returns:
        A :class:`DatasetResult` with file paths and summary statistics.
    """
    t0 = time.perf_counter()
    start = config.start.replace(second=0, microsecond=0)
    start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    end = start + timedelta(hours=config.duration_hours)
    fleet = build_fleet(config.n_servers)

    scheduler_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    incidents = BatchScheduler(scheduler_rng).plan(fleet, start, end, config.n_incidents)
    timeline = IncidentTimeline(incidents)
    simulator = FleetSimulator(fleet, timeline, config.step_seconds, config.seed)

    run_dir = Path(config.output_dir) / config.resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    logs_path = run_dir / "logs.jsonl"
    incidents_path = run_dir / "incidents.json"
    manifest_path = run_dir / "manifest.json"

    n_steps = int((end - start).total_seconds() // config.step_seconds)
    metric_rows = metric_anomalies = log_rows = log_anomalies = 0
    rows_by_type: Counter[str] = Counter()
    logs_by_source: Counter[str] = Counter()
    logger.info("Generating %d steps x %d hosts with %d incidents -> %s",
                n_steps, len(fleet), len(incidents), run_dir)

    with metrics_path.open("w", newline="", encoding="utf-8") as mf, \
            logs_path.open("w", encoding="utf-8") as lf:
        writer = csv.DictWriter(mf, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for step in range(n_steps):
            ts = start + timedelta(seconds=step * config.step_seconds)
            result = simulator.step(ts)
            for sample in result.metrics:
                writer.writerow(sample.to_row())
                metric_rows += 1
                if sample.is_anomaly:
                    metric_anomalies += 1
                    rows_by_type[sample.anomaly_type] += 1
            for record in result.logs:
                lf.write(json.dumps(record.to_dict()) + "\n")
                log_rows += 1
                log_anomalies += record.is_anomaly
                logs_by_source[record.source] += 1
            if n_steps >= 10 and step % max(1, n_steps // 10) == 0:
                logger.debug("progress %d/%d", step, n_steps)

    incidents_path.write_text(
        json.dumps([i.to_dict() for i in incidents], indent=2), encoding="utf-8"
    )
    elapsed = time.perf_counter() - t0
    summary: dict[str, Any] = {
        "metric_rows": metric_rows,
        "metric_anomaly_rows": metric_anomalies,
        "metric_anomaly_ratio": round(metric_anomalies / max(1, metric_rows), 4),
        "log_rows": log_rows,
        "log_anomaly_rows": log_anomalies,
        "log_anomaly_ratio": round(log_anomalies / max(1, log_rows), 4),
        "incidents": len(incidents),
        "incidents_by_type": dict(Counter(i.incident_type.value for i in incidents)),
        "incidents_by_category": dict(Counter(i.category.value for i in incidents)),
        "anomaly_rows_by_type": dict(rows_by_type),
        "logs_by_source": dict(logs_by_source),
        "generation_seconds": round(elapsed, 2),
    }
    manifest = {
        "generator_version": __version__,
        "created_with": {
            "seed": config.seed,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "duration_hours": config.duration_hours,
            "step_seconds": config.step_seconds,
            "n_servers": config.n_servers,
            "incidents_per_day": config.incidents_per_day,
        },
        "hosts": [{"name": s.name, "role": s.role.value, "ip": s.ip} for s in fleet],
        "files": {
            "metrics": metrics_path.name,
            "logs": logs_path.name,
            "incidents": incidents_path.name,
        },
        "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (Path(config.output_dir) / LATEST_POINTER).write_text(run_dir.name, encoding="utf-8")
    logger.info("Done in %.2fs: %d metric rows (%.1f%% anomalous), %d log lines (%.1f%% anomalous)",
                elapsed, metric_rows, 100 * summary["metric_anomaly_ratio"],
                log_rows, 100 * summary["log_anomaly_ratio"])
    return DatasetResult(run_dir, metrics_path, logs_path, incidents_path, manifest_path, summary)


def resolve_run_dir(output_dir: Path, run_name: str | None = None) -> Path:
    """Return the folder of ``run_name`` or of the most recent run.

    Raises:
        FileNotFoundError: If no matching run exists.
        ValueError: If ``run_name`` is not a safe folder name.
    """
    if run_name is None:
        pointer = output_dir / LATEST_POINTER
        if not pointer.exists():
            raise FileNotFoundError(f"No dataset found in {output_dir}. Run `generate` first.")
        run_name = pointer.read_text(encoding="utf-8").strip()
    if not _is_safe_name(run_name):
        raise ValueError(f"invalid run name: {run_name!r}")
    run_dir = output_dir / run_name
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Run folder {run_dir} has no manifest.json")
    return run_dir
