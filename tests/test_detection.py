"""Detectors, fusion, thresholds and security rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiops_copilot.detection.base import best_f1_threshold, smooth_scores
from aiops_copilot.detection.hybrid import TORCH_AVAILABLE, HybridDetector
from aiops_copilot.detection.security_rules import SecurityRuleEngine
from aiops_copilot.evaluation.metrics import evaluate_detector
from aiops_copilot.generator.models import IncidentType
from conftest import Trained

T0 = datetime(2026, 1, 1, tzinfo=UTC)


# ------------------------------------------------------------------ helpers
def test_best_f1_threshold_separates_clean_scores() -> None:
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.85])
    labels = np.array([0, 0, 0, 1, 1, 1])
    thr = best_f1_threshold(scores, labels)
    assert ((scores >= thr) == labels.astype(bool)).all()


def test_smoothing_does_not_leak_between_hosts() -> None:
    hosts = pd.Series(["a"] * 5 + ["b"] * 5)
    scores = np.array([0, 0, 0, 0, 9, 0, 0, 0, 0, 0], dtype=float)
    out = smooth_scores(scores, hosts, window=3)
    assert out[5:].max() == 0.0  # host b is untouched by host a's spike
    assert out[4] > 0


# ------------------------------------------------------------------ hybrid detector
def test_hybrid_detects_most_incidents(trained: Trained) -> None:
    result = trained.detector.detect(trained.test, trained.test_logs)
    report = evaluate_detector("hybrid", trained.test, result.final_flags, trained.test_incidents, 1.0)
    assert report.events.event_recall >= 0.8
    assert report.point.f1 >= 0.6
    assert report.events.false_alarms_per_day <= 20
    assert result.incidents, "expected grouped incidents"


def test_security_incidents_get_security_type(trained: Trained) -> None:
    result = trained.detector.detect(trained.test, trained.test_logs)
    with_findings = [i for i in result.incidents if i.findings]
    assert with_findings
    assert all(i.category.value == "security" for i in with_findings)


def test_save_load_gives_identical_scores(trained: Trained) -> None:
    loaded = HybridDetector.load(trained.model_dir)
    frame = trained.test.head(600)
    a = trained.detector.detect(frame, group=False).fused_score
    b = loaded.detect(frame, group=False).fused_score
    np.testing.assert_allclose(a, b, rtol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_lstm_autoencoder_scores_spike_higher(trained: Trained, tmp_path: Path) -> None:
    from aiops_copilot.detection.lstm_autoencoder import LSTMAutoencoderDetector
    from aiops_copilot.features.pipeline import build_features

    test = trained.test
    feats = build_features(test, trained.detector.scaler)
    normal = test["is_anomaly"].to_numpy() == 0
    lstm = LSTMAutoencoderDetector(epochs=2)
    lstm.fit(feats, test["host"], normal_mask=normal)
    scores = lstm.score(feats, test["host"])
    assert scores.shape == (len(test),) and np.isfinite(scores).all()
    assert np.median(scores[~normal]) > np.median(scores[normal])
    lstm.save(tmp_path / "lstm.pt")
    again = LSTMAutoencoderDetector.load(tmp_path / "lstm.pt").score(feats.head(300), test["host"].head(300))
    np.testing.assert_allclose(again, lstm.score(feats.head(300), test["host"].head(300)), rtol=1e-5)


# ------------------------------------------------------------------ security rules
def _log(i: int, message: str, host: str = "web-01") -> dict[str, object]:
    return {"timestamp": T0 + timedelta(seconds=10 * i), "host": host, "source": "sshd", "message": message}


def test_brute_force_needs_threshold_failures() -> None:
    engine = SecurityRuleEngine()
    fails = [_log(i, f"Failed password for invalid user admin from 203.0.113.7 port {4000 + i} ssh2")
             for i in range(14)]
    assert engine.process_many(fails) == []
    hit = engine.process(_log(15, "Failed password for root from 203.0.113.7 port 5000 ssh2"))
    assert [f.rule_id for f in hit] == ["BRUTE-001"]
    assert hit[0].incident_type == IncidentType.SSH_BRUTE_FORCE
    compromised = engine.process(_log(16, "Accepted password for root from 203.0.113.7 port 5001 ssh2"))
    assert any(f.rule_id == "BRUTE-002" and f.severity.value == "critical" for f in compromised)


def test_single_mistyped_password_is_not_an_attack() -> None:
    engine = SecurityRuleEngine()
    assert engine.process(_log(0, "Failed password for alice from 10.0.4.12 port 50122 ssh2")) == []
    assert engine.process(_log(1, "Accepted password for alice from 10.0.4.12 port 50123 ssh2")) == []


def test_exfiltration_only_to_external_addresses() -> None:
    engine = SecurityRuleEngine()
    internal = "flow 10.0.1.5:443 -> 10.0.9.9:5432 proto=tcp bytes_out=900000000"
    external = "flow 10.0.1.5:443 -> 198.51.100.23:443 proto=tcp bytes_out=900000000"
    assert not [f for f in engine.process(_log(0, internal)) if f.rule_id == "EXFIL-002"]
    hits = [f for f in engine.process(_log(1, external)) if f.rule_id == "EXFIL-002"]
    assert hits and hits[0].source_ip == "198.51.100.23"


def test_rules_never_crash_on_odd_input() -> None:
    engine = SecurityRuleEngine()
    for msg in ["", "x" * 10_000, "Failed password for  from 999.1.1.1", "-> 1.2.3:22 bytes_out=abc"]:
        engine.process({"timestamp": T0.isoformat(), "host": "h", "message": msg})
