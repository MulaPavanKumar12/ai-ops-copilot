"""Isolation Forest detector.

Why Isolation Forest as the ML baseline:

* Unsupervised: needs no labelled incidents, which real teams rarely have.
* Anomalies are "few and different", so random splits isolate them in fewer
  steps; the average path length is the anomaly score. That intuition is easy
  to explain to stakeholders.
* Fast to train and score (sub-second on 50k rows), few hyper-parameters, and
  it handles many features without assuming a distribution.

Input: absolute values of the per-host seasonal z-scores, rolling means and
slopes. Taking the absolute value folds "too high" and "too low" onto the same
side, which in our experiments gave about +4 F1 points over raw signed features.

Limitation: it scores each time step independently, so it has no memory.
Slow ramps (memory leaks) are only caught once they become extreme. That is
the gap the LSTM autoencoder fills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from aiops_copilot.features.pipeline import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class IsolationForestDetector:
    """Wrapper around :class:`sklearn.ensemble.IsolationForest`.

    Attributes:
        n_estimators: Number of trees.
        max_samples: Rows sampled per tree (``"auto"`` = 256).
        random_state: Seed for reproducibility.
        threshold: Alert threshold on the score; set by calibration.
    """

    n_estimators: int = 300
    max_samples: int | str = 512
    random_state: int = 42
    threshold: float = 0.0
    name: str = "isolation_forest"
    model: IsolationForest | None = field(default=None, repr=False)

    def fit(self, features: pd.DataFrame) -> IsolationForestDetector:
        """Fit on (mostly normal) feature rows."""
        if features.empty:
            raise ValueError("cannot fit Isolation Forest on an empty frame")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(features[FEATURE_COLUMNS].abs().to_numpy())
        logger.info("Isolation Forest fitted on %d rows x %d features", len(features), len(FEATURE_COLUMNS))
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row; higher means more anomalous."""
        if self.model is None:
            raise RuntimeError("detector is not fitted")
        if features.empty:
            return np.zeros(0)
        scores: np.ndarray = -self.model.score_samples(features[FEATURE_COLUMNS].abs().to_numpy())
        return scores

    def save(self, path: Path) -> None:
        """Persist with joblib (only load files you created yourself)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Path) -> IsolationForestDetector:
        """Load a detector saved by :meth:`save`."""
        data = joblib.load(path)
        return cls(threshold=float(data["threshold"]), model=data["model"])
