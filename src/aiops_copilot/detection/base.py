"""Shared detector utilities: score smoothing and threshold calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_scores(scores: np.ndarray, hosts: pd.Series, window: int = 3) -> np.ndarray:
    """Rolling mean of scores per host.

    A one-sample blip is diluted while a real incident (several consecutive
    abnormal samples) keeps a high score. This trades ~1 step of detection
    delay for far fewer false alarms.
    """
    if window <= 1:
        return np.asarray(scores, dtype=float)
    series = pd.Series(np.asarray(scores, dtype=float), index=hosts.index)
    return (
        series.groupby(hosts.to_numpy(), sort=False)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(hosts.index)
        .to_numpy()
    )


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray, n_candidates: int = 400) -> float:
    """Pick the threshold that maximises F1 on a labelled validation set.

    Using a small labelled validation set to set *only* the alert threshold is
    standard practice: the model itself stays unsupervised, and the choice is
    then checked on a separate test set.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        return float(np.quantile(scores, 0.995))
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, n_candidates)))
    best_t, best_f1 = float(candidates[-1]), -1.0
    positives = labels.sum()
    for t in candidates:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / positives
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
