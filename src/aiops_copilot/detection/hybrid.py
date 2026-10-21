"""The full detection system: features + Isolation Forest + LSTM AE + security rules.

Fusion design: each detector's score (z-score, Isolation Forest, LSTM) is divided by its own calibrated
threshold, so "1.0" means "at the alarm line" for every model. The fused score
is either the maximum ("any model is confident") or the mean ("the models agree
on average") of the normalised scores; the rule and its threshold are chosen on
validation data. Security rule hits raise an alarm directly, because
a signature match such as ``/etc/sudoers op=write`` needs no statistics.

PyTorch is optional at runtime: if it is not installed the system runs with
Isolation Forest + rules and logs a warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aiops_copilot.detection.base import best_f1_threshold, smooth_scores
from aiops_copilot.detection.grouping import DetectedIncident, group_incidents
from aiops_copilot.detection.isolation_forest import IsolationForestDetector
from aiops_copilot.detection.security_rules import SecurityFinding, SecurityRuleEngine
from aiops_copilot.detection.statistical import ZScoreDetector
from aiops_copilot.features.pipeline import Z_COLUMNS, HostScaler, build_features

logger = logging.getLogger(__name__)

try:  # optional dependency
    from aiops_copilot.detection.lstm_autoencoder import LSTMAutoencoderDetector

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False

SMOOTH_WINDOW = 3


@dataclass
class DetectionResult:
    """Per-row scores/flags for every detector, plus grouped incidents."""

    scores: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    fused_score: np.ndarray
    final_flags: np.ndarray
    findings: list[SecurityFinding]
    incidents: list[DetectedIncident]
    z: pd.DataFrame


def rule_flags(frame: pd.DataFrame, findings: list[SecurityFinding], step_seconds: int) -> np.ndarray:
    """Mark the metric rows (host, time bucket) where a security rule fired."""
    flags = np.zeros(len(frame), dtype=bool)
    if not findings:
        return flags
    keys = {(f.host, pd.Timestamp(f.timestamp).floor(f"{step_seconds}s")) for f in findings}
    ts = frame["timestamp"].dt.floor(f"{step_seconds}s")
    for i, (h, t) in enumerate(zip(frame["host"].to_numpy(), ts, strict=True)):
        if (h, t) in keys:
            flags[i] = True
    return flags


@dataclass
class HybridDetector:
    """Train, calibrate, run and persist the complete detector stack."""

    step_seconds: int = 60
    use_lstm: bool = True
    scaler: HostScaler = field(default_factory=HostScaler)
    zscore: ZScoreDetector = field(default_factory=ZScoreDetector)
    iforest: IsolationForestDetector = field(default_factory=IsolationForestDetector)
    lstm: Any = None
    fused_threshold: float = 1.0
    fusion: str = "max"
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- training
    def fit(self, train: pd.DataFrame, lstm_epochs: int = 8) -> HybridDetector:
        """Fit scaler and models on training metrics.

        Rows inside known (labelled) incidents are excluded from training, as a
        team would exclude periods covered by past incident tickets. If no
        labels exist, everything is used and the robust statistics absorb it.
        """
        normal = train["is_anomaly"].to_numpy() == 0 if "is_anomaly" in train else np.ones(len(train), bool)
        self.scaler = HostScaler().fit(train[normal])
        feats = build_features(train, self.scaler)
        self.iforest.fit(feats[normal])
        if self.use_lstm and TORCH_AVAILABLE:
            self.lstm = LSTMAutoencoderDetector(epochs=lstm_epochs)
            self.lstm.fit(feats, train["host"], normal_mask=normal)
        elif self.use_lstm:
            logger.warning("PyTorch not installed: LSTM autoencoder disabled")
            self.use_lstm = False
        return self

    def raw_scores(self, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        """Smoothed scores of each metric detector, and the feature frame."""
        feats = build_features(frame, self.scaler)
        hosts = frame["host"]
        scores = {
            "zscore": smooth_scores(self.zscore.score(feats), hosts, SMOOTH_WINDOW),
            "isolation_forest": smooth_scores(self.iforest.score(feats), hosts, SMOOTH_WINDOW),
        }
        if self.use_lstm and self.lstm is not None:
            scores["lstm_autoencoder"] = smooth_scores(self.lstm.score(feats, hosts), hosts, SMOOTH_WINDOW)
        return scores, feats

    def _fuse(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        parts = [scores["isolation_forest"] / max(self.iforest.threshold, 1e-9),
                 scores["zscore"] / max(self.zscore.threshold, 1e-9)]
        if "lstm_autoencoder" in scores and self.lstm is not None:
            parts.append(scores["lstm_autoencoder"] / max(self.lstm.threshold, 1e-9))
        if self.fusion == "mean":
            fused: np.ndarray = np.mean(parts, axis=0)
            return fused
        peak: np.ndarray = np.maximum.reduce(parts)
        return peak

    def calibrate(self, val: pd.DataFrame) -> dict[str, float]:
        """Set every alarm threshold to maximise F1 on a labelled validation set."""
        scores, _ = self.raw_scores(val)
        labels = val["is_anomaly"].to_numpy()
        self.zscore.threshold = best_f1_threshold(scores["zscore"], labels)
        self.iforest.threshold = best_f1_threshold(scores["isolation_forest"], labels)
        if "lstm_autoencoder" in scores:
            self.lstm.threshold = best_f1_threshold(scores["lstm_autoencoder"], labels)
        # Choose the fusion rule ("any model is confident" vs "models agree on average")
        # and its threshold on validation F1.
        best: tuple[float, str, float] = (-1.0, "max", 1.0)
        for fusion in ("max", "mean"):
            self.fusion = fusion
            fused = self._fuse(scores)
            thr = best_f1_threshold(fused, labels)
            pred = fused >= thr
            tp = float((pred & (labels == 1)).sum())
            f1 = 2 * tp / max(float(pred.sum() + (labels == 1).sum()), 1.0)
            if f1 > best[0]:
                best = (f1, fusion, thr)
        _, self.fusion, self.fused_threshold = best
        out = {"zscore": self.zscore.threshold, "isolation_forest": self.iforest.threshold,
               "fused": self.fused_threshold}
        if self.lstm is not None:
            out["lstm_autoencoder"] = self.lstm.threshold
        logger.info("Calibrated thresholds: %s (fusion=%s)", {k: round(v, 4) for k, v in out.items()},
                    self.fusion)
        return out

    # ------------------------------------------------------------ inference
    def detect(self, frame: pd.DataFrame, logs: list[dict[str, Any]] | None = None,
               rules: SecurityRuleEngine | None = None, group: bool = True) -> DetectionResult:
        """Run every detector on a host-sorted metrics frame (+ optional logs)."""
        scores, feats = self.raw_scores(frame)
        flags = {
            "zscore": scores["zscore"] >= self.zscore.threshold,
            "isolation_forest": scores["isolation_forest"] >= self.iforest.threshold,
        }
        if "lstm_autoencoder" in scores:
            flags["lstm_autoencoder"] = scores["lstm_autoencoder"] >= self.lstm.threshold
        engine = rules or SecurityRuleEngine()
        findings = engine.process_many(logs or [])
        flags["security_rules"] = rule_flags(frame, findings, self.step_seconds)
        fused = self._fuse(scores) / max(self.fused_threshold, 1e-9)
        final = (fused >= 1.0) | flags["security_rules"]
        z = feats[Z_COLUMNS]
        incidents = group_incidents(
            frame, final, fused, z,
            {k: v for k, v in flags.items() if k != "zscore"}, findings,
            step=timedelta(seconds=self.step_seconds),
        ) if group else []
        return DetectionResult(scores, flags, fused, final, findings, incidents, z)

    # ---------------------------------------------------------- persistence
    def save(self, model_dir: Path) -> None:
        """Write all model artefacts to ``model_dir``."""
        model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler.save(model_dir / "scaler.json")
        self.iforest.save(model_dir / "isolation_forest.joblib")
        if self.lstm is not None:
            self.lstm.save(model_dir / "lstm_autoencoder.pt")
        meta = {**self.meta, "step_seconds": self.step_seconds, "use_lstm": self.lstm is not None,
                "zscore_threshold": self.zscore.threshold, "fused_threshold": self.fused_threshold,
                "fusion": self.fusion}
        (model_dir / "detector_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path) -> HybridDetector:
        """Load artefacts written by :meth:`save`."""
        meta = json.loads((model_dir / "detector_meta.json").read_text(encoding="utf-8"))
        det = cls(step_seconds=int(meta["step_seconds"]), use_lstm=bool(meta["use_lstm"]))
        det.scaler = HostScaler.load(model_dir / "scaler.json")
        det.iforest = IsolationForestDetector.load(model_dir / "isolation_forest.joblib")
        det.zscore = ZScoreDetector(threshold=float(meta["zscore_threshold"]))
        det.fused_threshold = float(meta["fused_threshold"])
        det.fusion = str(meta.get("fusion", "max"))
        det.meta = meta
        if det.use_lstm:
            if TORCH_AVAILABLE and (model_dir / "lstm_autoencoder.pt").exists():
                det.lstm = LSTMAutoencoderDetector.load(model_dir / "lstm_autoencoder.pt")
            else:
                logger.warning("LSTM weights or PyTorch missing; running without the LSTM")
                det.use_lstm = False
        return det
