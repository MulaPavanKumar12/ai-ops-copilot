"""Train, calibrate and evaluate every detector; save models and a results report.

Protocol (no leakage between splits, each split has a different random seed):

* **train**: 7 days, used to fit scalers and models (normal rows only).
* **validation**: 2 days, used only to choose alarm thresholds.
* **test**: 3 days, never seen before; every number in the report comes from here.

Usage::

    python -m aiops_copilot.evaluation.run_detection            # full run
    python -m aiops_copilot.evaluation.run_detection --quick    # smaller, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aiops_copilot.core.config import get_settings
from aiops_copilot.core.logging import setup_logging
from aiops_copilot.detection.hybrid import TORCH_AVAILABLE, HybridDetector
from aiops_copilot.evaluation.metrics import DetectorReport, evaluate_detector
from aiops_copilot.features.pipeline import load_metrics
from aiops_copilot.generator.dataset import DatasetConfig, generate_dataset

logger = logging.getLogger("aiops_copilot.evaluation")


@dataclass(frozen=True)
class Split:
    """One dataset split."""

    name: str
    seed: int
    hours: float
    start: datetime


SPLITS_FULL = (
    Split("train", 101, 24 * 7, datetime(2026, 1, 5, tzinfo=UTC)),
    Split("validation", 202, 24 * 2, datetime(2026, 1, 12, tzinfo=UTC)),
    Split("test", 303, 24 * 3, datetime(2026, 1, 14, tzinfo=UTC)),
)
SPLITS_QUICK = (
    Split("train", 101, 48, datetime(2026, 1, 5, tzinfo=UTC)),
    Split("validation", 202, 24, datetime(2026, 1, 7, tzinfo=UTC)),
    Split("test", 303, 24, datetime(2026, 1, 8, tzinfo=UTC)),
)


def ensure_split(split: Split, data_dir: Path, incidents_per_day: float) -> Path:
    """Generate a split if it does not exist yet and return its folder."""
    run_dir = data_dir / split.name
    if not (run_dir / "manifest.json").exists():
        generate_dataset(DatasetConfig(seed=split.seed, start=split.start, duration_hours=split.hours,
                                       incidents_per_day=incidents_per_day, output_dir=data_dir,
                                       run_name=split.name))
    return run_dir


def load_split(run_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (metrics frame, log records, ground-truth incidents) for a split."""
    frame = load_metrics(run_dir / "metrics.csv")
    with (run_dir / "logs.jsonl").open(encoding="utf-8") as fh:
        logs = [json.loads(line) for line in fh]
    incidents = json.loads((run_dir / "incidents.json").read_text(encoding="utf-8"))
    return frame, logs, incidents


def results_markdown(reports: list[DetectorReport], meta: dict[str, Any]) -> str:
    """Render the results table in Markdown."""
    lines = [
        "# Detection results (test split)",
        "",
        f"Test data: {meta['test_rows']:,} metric rows, {meta['test_incidents']} incidents, "
        f"{meta['test_days']:.0f} days, {meta['hosts']} hosts. Thresholds chosen on the validation split.",
        "",
        "| Detector | Precision | Recall | F1 | Incidents detected | Median time-to-detect | False alarms / day |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        ttd = r.events.median_time_to_detect_min
        lines.append(
            f"| {r.detector} | {r.point.precision:.3f} | {r.point.recall:.3f} | {r.point.f1:.3f} | "
            f"{r.events.detected}/{r.events.incidents} ({r.events.event_recall:.0%}) | "
            f"{'-' if ttd is None else f'{ttd:.1f} min'} | {r.events.false_alarms_per_day:.1f} |")
    lines += ["", "## Incident recall by type", "",
              "| Type | " + " | ".join(r.detector for r in reports) + " |",
              "|---|" + "---|" * len(reports)]
    types = sorted(reports[0].events.recall_by_type)
    for t in types:
        lines.append(f"| {t} | " + " | ".join(f"{r.events.recall_by_type.get(t, 0):.0%}" for r in reports) + " |")
    lines += ["", "Precision/recall/F1 are point-wise over (host, minute) rows. "
              "An incident is detected if any minute inside it is flagged."]
    return "\n".join(lines) + "\n"


def run(quick: bool = False, lstm_epochs: int = 8) -> dict[str, Any]:
    """Execute the full protocol and write models + reports. Returns the results dict."""
    settings = get_settings()
    data_dir = settings.data_dir / ("eval_quick" if quick else "eval")
    splits = SPLITS_QUICK if quick else SPLITS_FULL
    dirs = {s.name: ensure_split(s, data_dir, 14.0) for s in splits}

    train, _, _ = load_split(dirs["train"])
    val, val_logs, _ = load_split(dirs["validation"])
    test, test_logs, test_incidents = load_split(dirs["test"])

    t0 = time.perf_counter()
    detector = HybridDetector(use_lstm=TORCH_AVAILABLE)
    detector.fit(train, lstm_epochs=lstm_epochs)
    train_seconds = time.perf_counter() - t0

    # Calibrate the fused threshold with rule hits included, as in production.
    detector.calibrate(val)
    del val_logs  # validation logs are not needed to set metric thresholds

    t1 = time.perf_counter()
    result = detector.detect(test, test_logs)
    score_ms_per_row = (time.perf_counter() - t1) / len(test) * 1000

    reports: list[DetectorReport] = [
        evaluate_detector("Z-score baseline", test, result.flags["zscore"], test_incidents,
                          detector.zscore.threshold),
        evaluate_detector("Isolation Forest", test, result.flags["isolation_forest"], test_incidents,
                          detector.iforest.threshold),
    ]
    if "lstm_autoencoder" in result.flags:
        reports.append(evaluate_detector("LSTM autoencoder", test, result.flags["lstm_autoencoder"],
                                         test_incidents, detector.lstm.threshold))
    reports.append(evaluate_detector("Security log rules", test, result.flags["security_rules"],
                                     test_incidents))
    reports.append(evaluate_detector("Hybrid (z-score + IF + LSTM + rules)", test, result.final_flags, test_incidents,
                                     detector.fused_threshold))

    # Triage accuracy: does the suspected type of each detected incident match the truth?
    truth_rows = test.set_index(["host", "timestamp"])["anomaly_type"]
    matched = correct = 0
    for inc in result.incidents:
        window = truth_rows.loc[inc.host].loc[inc.start: inc.end]
        types = window[window != ""]
        if types.empty:
            continue
        matched += 1
        correct += int(types.mode().iloc[0] == inc.suspected_type.value)

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "quick": quick,
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "test_incidents": len(test_incidents), "hosts": int(test["host"].nunique()),
        "test_days": (test["timestamp"].max() - test["timestamp"].min()).total_seconds() / 86400,
        "train_seconds": round(train_seconds, 1),
        "scoring_ms_per_row": round(score_ms_per_row, 4),
        "detected_incidents": len(result.incidents),
        "triage_type_accuracy": round(correct / matched, 3) if matched else None,
        "torch_available": TORCH_AVAILABLE,
        "lstm_loss_history": [round(x, 5) for x in (detector.lstm.history if detector.lstm else [])],
    }
    out = {"meta": meta, "detectors": [r.to_dict() for r in reports]}
    detector.meta = {"trained_at": meta["generated_at"], "train_rows": len(train)}
    detector.save(settings.model_dir)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "detection_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (settings.reports_dir / "detection_results.md").write_text(results_markdown(reports, meta), encoding="utf-8")
    pd.DataFrame([i.to_dict() for i in result.incidents]).to_json(
        settings.reports_dir / "detected_incidents_test.json", orient="records", indent=2)
    logger.info("Models saved to %s; report written to %s", settings.model_dir, settings.reports_dir)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Small datasets for a fast smoke test")
    parser.add_argument("--lstm-epochs", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.lstm_epochs <= 100:
        logger.error("--lstm-epochs must be between 1 and 100")
        return 2
    out = run(quick=args.quick, lstm_epochs=args.lstm_epochs)
    print((settings.reports_dir / "detection_results.md").read_text(encoding="utf-8"))
    print(f"Triage type accuracy: {out['meta']['triage_type_accuracy']}")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
