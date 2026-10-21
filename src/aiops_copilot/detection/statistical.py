"""Z-score baseline detector.

Scores each sample by the largest absolute per-host robust z-score across all
metrics. Simple, explainable, and the bar every ML model must beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aiops_copilot.features.pipeline import Z_COLUMNS


@dataclass
class ZScoreDetector:
    """Max-|z| detector. ``threshold`` is in z units."""

    threshold: float = 4.0
    name: str = "zscore"

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row (higher = more anomalous)."""
        return features[Z_COLUMNS].abs().max(axis=1).to_numpy(dtype=float)
