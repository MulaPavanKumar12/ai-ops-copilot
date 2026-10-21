"""LSTM autoencoder for time-series anomaly detection (PyTorch).

How it works: the model reads a sliding window of the last ``window`` steps of
per-host z-scored metrics, compresses it into a small latent vector, and tries
to reconstruct it. It is trained **only on normal windows**, so it learns what
normal temporal behaviour looks like. When the recent sequence is unusual
(a ramp, a change in correlation between metrics, a sustained shift), the
reconstruction error rises.

Why it complements Isolation Forest: it sees *sequences*, so it reacts to the
shape of a slow memory leak or disk fill, and to multi-metric patterns, before
any single value is extreme.

Scoring: the anomaly score at time *t* is the reconstruction error of the
last few steps of the window ending at *t*. Using the tail (not the whole
window) keeps the alarm aligned with the incident instead of lingering for a
full window after it ends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from aiops_copilot.features.pipeline import Z_COLUMNS

logger = logging.getLogger(__name__)

INPUT_CLIP = 10.0  # z-scores are clipped so a single huge value cannot dominate training


class _LSTMAE(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, n_features: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        latent = self.to_latent(h[-1])
        seq = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(seq)
        out: torch.Tensor = self.output(decoded)
        return out


def make_windows(z: np.ndarray, window: int) -> np.ndarray:
    """All overlapping windows of one host's series: shape (n-window+1, window, f)."""
    if len(z) < window:
        return np.zeros((0, window, z.shape[1]), dtype=np.float32)
    view = np.lib.stride_tricks.sliding_window_view(z, (window, z.shape[1]))[:, 0]
    return np.ascontiguousarray(view, dtype=np.float32)


@dataclass
class LSTMAutoencoderDetector:
    """Train/score wrapper. Scores rows of a host-sorted feature frame."""

    window: int = 20
    hidden: int = 32
    latent: int = 8
    epochs: int = 8
    batch_size: int = 256
    lr: float = 2e-3
    tail: int = 3
    stride: int = 2
    seed: int = 42
    threshold: float = 0.0
    name: str = "lstm_autoencoder"
    model: _LSTMAE | None = field(default=None, repr=False)
    history: list[float] = field(default_factory=list)

    def _host_arrays(self, features: pd.DataFrame, hosts: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        """(row positions, clipped z matrix) per host, in row order."""
        out = []
        z_all = features[Z_COLUMNS].to_numpy(dtype=np.float32).clip(-INPUT_CLIP, INPUT_CLIP)
        for _, pos in pd.Series(np.arange(len(hosts)), index=hosts.index).groupby(hosts.to_numpy(), sort=False):
            idx = pos.to_numpy()
            out.append((idx, z_all[idx]))
        return out

    def fit(self, features: pd.DataFrame, hosts: pd.Series,
            normal_mask: np.ndarray | None = None) -> LSTMAutoencoderDetector:
        """Train on windows that contain only normal rows.

        Args:
            features: Host-sorted feature frame.
            hosts: Host name per row (same index as ``features``).
            normal_mask: Boolean array, True for rows believed normal. Windows
                touching any non-normal row are excluded from training.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        mask = np.ones(len(features), dtype=bool) if normal_mask is None else np.asarray(normal_mask, bool)
        windows: list[np.ndarray] = []
        for idx, z in self._host_arrays(features, hosts):
            w = make_windows(z, self.window)
            if len(w) == 0:
                continue
            ok = make_windows(mask[idx].astype(np.float32)[:, None], self.window).min(axis=(1, 2)) > 0
            windows.append(w[ok][:: self.stride])
        if not windows or sum(len(w) for w in windows) == 0:
            raise ValueError("not enough normal data to build training windows")
        data = torch.from_numpy(np.concatenate(windows))
        logger.info("Training LSTM autoencoder on %d windows (%d steps x %d features)",
                    len(data), self.window, data.shape[2])

        self.model = _LSTMAE(data.shape[2], self.hidden, self.latent)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        generator = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            perm = torch.randperm(len(data), generator=generator)
            total = 0.0
            for start in range(0, len(data), self.batch_size):
                batch = data[perm[start:start + self.batch_size]]
                optimiser.zero_grad()
                loss = loss_fn(self.model(batch), batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                total += float(loss.detach()) * len(batch)
            self.history.append(total / len(data))
            logger.info("epoch %d/%d  loss=%.4f", epoch + 1, self.epochs, self.history[-1])
        return self

    @torch.no_grad()
    def score(self, features: pd.DataFrame, hosts: pd.Series) -> np.ndarray:
        """Score per row. The first ``window-1`` rows of each host get the score of
        the first full window (there is not enough history before that)."""
        if self.model is None:
            raise RuntimeError("detector is not fitted")
        self.model.eval()
        scores = np.zeros(len(features), dtype=float)
        for idx, z in self._host_arrays(features, hosts):
            w = make_windows(z, self.window)
            if len(w) == 0:
                continue
            errors = []
            for start in range(0, len(w), 2048):
                batch = torch.from_numpy(w[start:start + 2048])
                recon = self.model(batch)
                err = ((recon - batch) ** 2)[:, -self.tail:, :].mean(dim=(1, 2))
                errors.append(err.numpy())
            e = np.concatenate(errors)
            scores[idx[self.window - 1:]] = e
            scores[idx[: self.window - 1]] = e[0]
        return scores

    def save(self, path: Path) -> None:
        """Persist weights and hyper-parameters (safe ``state_dict`` format)."""
        if self.model is None:
            raise RuntimeError("detector is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": {"window": self.window, "hidden": self.hidden, "latent": self.latent,
                       "tail": self.tail, "threshold": self.threshold,
                       "n_features": len(Z_COLUMNS)},
        }, path)

    @classmethod
    def load(cls, path: Path) -> LSTMAutoencoderDetector:
        """Load a detector saved by :meth:`save` (weights only, no pickled code)."""
        data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        cfg = data["config"]
        det = cls(window=int(cfg["window"]), hidden=int(cfg["hidden"]), latent=int(cfg["latent"]),
                  tail=int(cfg["tail"]), threshold=float(cfg["threshold"]))
        det.model = _LSTMAE(int(cfg["n_features"]), det.hidden, det.latent)
        det.model.load_state_dict(data["state_dict"])
        det.model.eval()
        return det
