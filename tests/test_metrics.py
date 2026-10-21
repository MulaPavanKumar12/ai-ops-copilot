"""Evaluation metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from aiops_copilot.evaluation.metrics import event_metrics, point_metrics


def test_point_metrics_known_values() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = point_metrics(y_true, y_pred)
    assert m.precision == pytest.approx(2 / 3, abs=1e-3)
    assert m.recall == pytest.approx(2 / 3, abs=1e-3)
    assert m.f1 == pytest.approx(2 / 3, abs=1e-3)


def test_point_metrics_no_predictions_is_zero_not_nan() -> None:
    m = point_metrics(np.array([1, 0]), np.array([0, 0]))
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0


def test_event_metrics_detection_and_delay() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ts = [t0 + timedelta(minutes=i) for i in range(20)]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(ts, utc=True), "host": "web-01",
                          "is_anomaly": [0] * 5 + [1] * 5 + [0] * 10,
                          "incident_id": [None] * 5 + ["I1"] * 5 + [None] * 10})
    incidents = [{"incident_id": "I1", "incident_type": "cpu_spike", "category": "operational", "host": "web-01",
                  "start": ts[5].isoformat(), "end": ts[9].isoformat()}]
    pred = np.zeros(20, dtype=bool)
    pred[7] = True     # detected 2 minutes after start
    pred[15] = True    # one false alarm
    ev = event_metrics(frame, pred, incidents)
    assert ev.detected == 1 and ev.incidents == 1
    assert ev.median_time_to_detect_min == pytest.approx(2.0)
    assert ev.false_alarms_per_day > 0
