"""Evidence access for root-cause analysis.

The RCA engines never touch storage directly. They ask an
:class:`EvidenceProvider` for metric summaries and log lines in a time window.
The batch evaluation backs it with DataFrames; the live worker backs it with
its rolling buffers. This keeps the LLM tools identical in both settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from aiops_copilot.features.pipeline import HostScaler
from aiops_copilot.generator.models import METRIC_NAMES

_LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
# Log fields the RCA may see. Ground-truth labels are deliberately excluded.
_LOG_FIELDS = ("timestamp", "host", "source", "level", "message")


class EvidenceProvider(Protocol):
    """What the RCA engines are allowed to look at."""

    def metrics_summary(self, host: str, start: datetime, end: datetime) -> dict[str, Any]: ...
    def logs(self, host: str, start: datetime, end: datetime, min_level: str = "WARNING",
             limit: int = 50) -> list[dict[str, Any]]: ...


def _as_utc(ts: datetime) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@dataclass
class FrameEvidence:
    """Evidence backed by a metrics DataFrame and a list of log dicts."""

    metrics: pd.DataFrame
    log_records: list[dict[str, Any]]
    scaler: HostScaler | None = None
    _logs_df: pd.DataFrame | None = field(default=None, repr=False)

    def _logs_frame(self) -> pd.DataFrame:
        if self._logs_df is None:
            df = pd.DataFrame(self.log_records, columns=list(_LOG_FIELDS)) if self.log_records \
                else pd.DataFrame(columns=list(_LOG_FIELDS))
            df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
            self._logs_df = df
        return self._logs_df

    def metrics_summary(self, host: str, start: datetime, end: datetime) -> dict[str, Any]:
        """Per-metric min/mean/max in the window, compared with the host's normal level."""
        m = self.metrics
        host_rows = m[m["host"] == host]
        if host_rows.empty:
            return {"error": f"unknown host {host!r}"}
        window = host_rows[(host_rows["timestamp"] >= _as_utc(start)) & (host_rows["timestamp"] < _as_utc(end))]
        if window.empty:
            return {"error": "no samples in that window"}
        before = host_rows[(host_rows["timestamp"] < _as_utc(start))
                           & (host_rows["timestamp"] >= _as_utc(start) - pd.Timedelta(hours=2))]
        role = str(host_rows["role"].iloc[0])
        out: dict[str, Any] = {"host": host, "role": role, "samples": int(len(window)), "metrics": {}}
        for metric in METRIC_NAMES:
            if self.scaler is not None:
                normal = self.scaler.expected(host, role, int(window["timestamp"].iloc[0].hour))[metric]
            elif not before.empty:
                normal = float(before[metric].median())
            else:
                normal = float(host_rows[metric].median())
            out["metrics"][metric] = {
                "min": round(float(window[metric].min()), 2), "mean": round(float(window[metric].mean()), 2),
                "max": round(float(window[metric].max()), 2), "normal": round(float(normal), 2),
            }
        return out

    def logs(self, host: str, start: datetime, end: datetime, min_level: str = "WARNING",
             limit: int = 50) -> list[dict[str, Any]]:
        """Log lines on ``host`` in the window at or above ``min_level`` (newest last)."""
        df = self._logs_frame()
        floor = _LEVELS.get(min_level.upper(), 2)
        sel = df[(df["host"] == host) & (df["ts"] >= _as_utc(start)) & (df["ts"] < _as_utc(end))
                 & (df["level"].map(lambda lv: _LEVELS.get(str(lv), 1)) >= floor)]
        return sel[list(_LOG_FIELDS)].head(max(1, min(limit, 500))).to_dict("records")  # type: ignore[return-value]


def window_with_margin(start: datetime, end: datetime, before_min: int = 10) -> tuple[datetime, datetime]:
    """Widen an incident window slightly so the build-up is visible."""
    return start - timedelta(minutes=before_min), end + timedelta(minutes=1)
