"""Feature engineering for metric-based anomaly detection.

Why these features:

* **Per-host, per-hour-of-day robust z-scores**. Hosts have very different
  "normal" (a DB sits at 70% memory, a web node at 55%), and traffic at 14:00
  is normally higher than at 02:00, so the expected value is the host's median
  *for that hour of the day*, and the scale is the host's robust spread
  (IQR) around those hourly medians. Median/IQR is used instead of mean/std
  because training data may contain unlabelled anomalies, and robust
  statistics are not dragged around by them.
* **Short rolling means** of the z-scores smooth single-sample noise and
  reduce false positives from one-off blips.
* **Slopes** of memory and disk catch slow ramps (leaks, filling disks) that
  never look extreme at a single point in time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aiops_copilot.generator.models import METRIC_NAMES

logger = logging.getLogger(__name__)

Z_COLUMNS: list[str] = [f"z_{m}" for m in METRIC_NAMES]
ROLL_COLUMNS: list[str] = [f"zroll_{m}" for m in METRIC_NAMES]
SLOPE_COLUMNS: list[str] = ["slope_mem_pct", "slope_disk_pct"]
FEATURE_COLUMNS: list[str] = Z_COLUMNS + ROLL_COLUMNS + SLOPE_COLUMNS

ROLL_WINDOW = 5
SLOPE_WINDOW = 10
Z_CLIP = 50.0
_MIN_SCALE: dict[str, float] = {  # floor for IQR so near-constant metrics don't explode
    "cpu_pct": 2.0, "mem_pct": 1.0, "disk_pct": 0.5, "latency_p95_ms": 5.0,
    "error_rate_pct": 0.2, "requests_per_sec": 5.0, "net_in_mbps": 1.0,
    "net_out_mbps": 2.0, "failed_logins": 1.0, "process_count": 2.0,
}


def load_metrics(path: Path | str) -> pd.DataFrame:
    """Load a ``metrics.csv`` file, parse timestamps and sort by host/time."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["anomaly_type"] = df["anomaly_type"].fillna("")
    df["incident_id"] = df["incident_id"].fillna("")
    if "benign_event" in df:
        df["benign_event"] = df["benign_event"].fillna("")
    return df.sort_values(["host", "timestamp"]).reset_index(drop=True)


Stats = dict[str, tuple[float, float]]  # metric -> (median, scale)


@dataclass
class HostScaler:
    """Robust seasonal scaling per host, with a per-role fallback.

    ``hourly[host][hour][metric]`` is the median for that hour of the day and
    ``scale[host][metric]`` the robust spread of values around those medians.
    The role-level fallback lets the model score a brand-new host of a known
    role without retraining.
    """

    hourly: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    scale: dict[str, dict[str, float]] = field(default_factory=dict)
    host_roles: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _fit_group(frame: pd.DataFrame) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
        hours = frame["timestamp"].dt.hour
        med = frame.groupby(hours)[list(METRIC_NAMES)].median()
        overall = frame[list(METRIC_NAMES)].median()
        hourly: dict[int, dict[str, float]] = {}
        for h in range(24):
            row = med.loc[h] if h in med.index else overall
            hourly[h] = {m: float(row[m]) for m in METRIC_NAMES}  # type: ignore[arg-type]
        expected = pd.DataFrame([hourly[int(h)] for h in hours], index=frame.index)
        resid = frame[list(METRIC_NAMES)] - expected
        scale = {}
        for m in METRIC_NAMES:
            iqr = float(resid[m].quantile(0.75) - resid[m].quantile(0.25))
            scale[m] = max(iqr / 1.349, _MIN_SCALE[m])  # IQR -> sigma-equivalent
        return hourly, scale

    def fit(self, df: pd.DataFrame) -> HostScaler:
        """Fit on (preferably normal-only) rows of a metrics frame."""
        if df.empty:
            raise ValueError("cannot fit scaler on an empty frame")
        for host, frame in df.groupby("host"):
            self.hourly[str(host)], self.scale[str(host)] = self._fit_group(frame)
            self.host_roles[str(host)] = str(frame["role"].iloc[0])
        for role, frame in df.groupby("role"):
            self.hourly[f"role:{role}"], self.scale[f"role:{role}"] = self._fit_group(frame)
        return self

    def _key(self, host: str, role: str) -> str:
        if host in self.hourly:
            return host
        if f"role:{role}" in self.hourly:
            return f"role:{role}"
        raise KeyError(f"no scaling statistics for host {host!r} or role {role!r}")

    def expected(self, host: str, role: str, hour: int) -> dict[str, float]:
        """Expected (median) value of each metric for this host at this hour."""
        return self.hourly[self._key(host, role)][int(hour)]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame of ``z_<metric>`` columns aligned with ``df``."""
        z = pd.DataFrame(index=df.index, columns=Z_COLUMNS, dtype=float)
        hours = df["timestamp"].dt.hour
        for group_key, idx in df.groupby(["host", "role"]).groups.items():
            host, role = cast(tuple[str, str], group_key)
            key = self._key(str(host), str(role))
            exp = pd.DataFrame([self.hourly[key][int(h)] for h in hours.loc[idx]], index=idx)
            for m in METRIC_NAMES:
                z.loc[idx, f"z_{m}"] = (df.loc[idx, m].astype(float) - exp[m]) / self.scale[key][m]
        return z.clip(-Z_CLIP, Z_CLIP)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {"hourly": {k: {str(h): v for h, v in d.items()} for k, d in self.hourly.items()},
                "scale": self.scale, "host_roles": self.host_roles}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HostScaler:
        """Inverse of :meth:`to_dict`."""
        hourly = {k: {int(h): {m: float(x) for m, x in v.items()} for h, v in d.items()}
                  for k, d in data["hourly"].items()}
        scale = {k: {m: float(x) for m, x in d.items()} for k, d in data["scale"].items()}
        return cls(hourly, scale, dict(data["host_roles"]))

    def save(self, path: Path) -> None:
        """Write to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HostScaler:
        """Read from JSON."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_features(df: pd.DataFrame, scaler: HostScaler) -> pd.DataFrame:
    """Compute the model feature matrix for a metrics frame.

    ``df`` must be sorted by host and timestamp (see :func:`load_metrics`).
    Rolling features are computed per host so hosts never leak into each other.

    Returns:
        A frame with :data:`FEATURE_COLUMNS`, aligned on ``df.index``.
    """
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=float)
    z = scaler.transform(df)
    feats = z.copy()
    grouped = z.groupby(df["host"], sort=False)
    rolled = grouped.rolling(ROLL_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    for m in METRIC_NAMES:
        feats[f"zroll_{m}"] = rolled[f"z_{m}"].reindex(df.index)
    for m in ("mem_pct", "disk_pct"):
        diff = z[f"z_{m}"].groupby(df["host"], sort=False).diff(SLOPE_WINDOW)
        feats[f"slope_{m}"] = (diff / SLOPE_WINDOW * 10.0).fillna(0.0)
    return feats[FEATURE_COLUMNS].astype(float).fillna(0.0)


def top_deviations(z_row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    """Return the ``k`` metrics with the largest absolute z-score in a row."""
    vals = {m: float(z_row[f"z_{m}"]) for m in METRIC_NAMES if f"z_{m}" in z_row}
    return sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
