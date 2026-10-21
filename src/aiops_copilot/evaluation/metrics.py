"""Evaluation metrics for anomaly detection.

Point-wise precision/recall/F1 alone is misleading for incidents: catching 3
of 60 minutes of a memory leak is a *detected incident* for an on-call
engineer, but scores 5% recall. So we report both:

* **Point-wise** P/R/F1 over metric rows.
* **Event-level**: an incident counts as detected if any row inside its window
  is flagged; *time-to-detect* is minutes from incident start to first flag.
  False alarms are flagged runs that overlap no incident, reported per day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PointMetrics:
    """Confusion counts and derived scores."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


@dataclass
class EventMetrics:
    """Incident-level detection quality."""

    incidents: int
    detected: int
    event_recall: float
    mean_time_to_detect_min: float | None
    median_time_to_detect_min: float | None
    false_alarm_events: int
    false_alarms_per_day: float
    recall_by_type: dict[str, float] = field(default_factory=dict)
    recall_by_category: dict[str, float] = field(default_factory=dict)


@dataclass
class DetectorReport:
    """Everything we report for one detector."""

    detector: str
    point: PointMetrics
    events: EventMetrics
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> PointMetrics:
    """Precision, recall and F1 for binary labels."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PointMetrics(tp, fp, fn, tn, round(precision, 4), round(recall, 4), round(f1, 4))


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as (start, end_exclusive) positions."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def event_metrics(frame: pd.DataFrame, y_pred: np.ndarray, incidents: list[dict[str, Any]]) -> EventMetrics:
    """Incident-level metrics.

    Args:
        frame: Host-sorted metrics frame with host, timestamp, is_anomaly, incident_id.
        y_pred: Boolean predictions aligned with ``frame``.
        incidents: Ground-truth incidents (``incidents.json`` records).
    """
    pred = np.asarray(y_pred, dtype=bool)
    ttd: list[float] = []
    detected_ids: set[str] = set()
    type_hits: dict[str, list[int]] = {}
    cat_hits: dict[str, list[int]] = {}
    inc_ids = frame["incident_id"].to_numpy()
    ts = frame["timestamp"]
    for inc in incidents:
        mask = inc_ids == inc["incident_id"]
        hit_rows = np.flatnonzero(mask & pred)
        hit = len(hit_rows) > 0
        type_hits.setdefault(inc["incident_type"], []).append(int(hit))
        cat_hits.setdefault(inc["category"], []).append(int(hit))
        if hit:
            detected_ids.add(inc["incident_id"])
            first = ts.iloc[hit_rows[0]]
            ttd.append((first - pd.Timestamp(inc["start"])).total_seconds() / 60.0)

    false_events = 0
    truth = frame["is_anomaly"].to_numpy().astype(bool)
    for _, idx in frame.groupby("host", sort=False).indices.items():
        idx = np.asarray(idx)
        for s, e in _runs(pred[idx]):
            # A run "touching" an incident (within 5 steps after it) is not a false alarm.
            lo, hi = max(0, s - 1), min(len(idx), e + 5)
            if not truth[idx[lo:hi]].any():
                false_events += 1
    days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1e-9)
    n = len(incidents)
    return EventMetrics(
        incidents=n,
        detected=len(detected_ids),
        event_recall=round(len(detected_ids) / n, 4) if n else 0.0,
        mean_time_to_detect_min=round(float(np.mean(ttd)), 2) if ttd else None,
        median_time_to_detect_min=round(float(np.median(ttd)), 2) if ttd else None,
        false_alarm_events=false_events,
        false_alarms_per_day=round(false_events / days, 2),
        recall_by_type={k: round(float(np.mean(v)), 3) for k, v in sorted(type_hits.items())},
        recall_by_category={k: round(float(np.mean(v)), 3) for k, v in sorted(cat_hits.items())},
    )


def evaluate_detector(name: str, frame: pd.DataFrame, y_pred: np.ndarray,
                      incidents: list[dict[str, Any]], threshold: float | None = None) -> DetectorReport:
    """Point + event metrics for one detector."""
    return DetectorReport(
        detector=name,
        point=point_metrics(frame["is_anomaly"].to_numpy(), y_pred),
        events=event_metrics(frame, y_pred, incidents),
        threshold=None if threshold is None else round(float(threshold), 6),
    )
