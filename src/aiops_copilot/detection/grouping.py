"""Turn per-sample alarms into incidents, and guess what kind of incident it is.

On-call engineers want one alert per problem, not one per data point. Flagged
samples on the same host are merged when the gap between them is small.
Each incident gets a *suspected type*: from a security rule if one fired,
otherwise from which metrics deviate most (a transparent rule table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from aiops_copilot.detection.security_rules import SecurityFinding
from aiops_copilot.features.pipeline import Z_COLUMNS
from aiops_copilot.generator.models import IncidentCategory, IncidentType, Severity

_SEVERITY_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
_SECURITY_TYPES = {IncidentType.SSH_BRUTE_FORCE, IncidentType.DATA_EXFILTRATION,
                   IncidentType.PRIVILEGE_ESCALATION, IncidentType.SUSPICIOUS_PROCESS}


@dataclass
class DetectedIncident:
    """An incident as seen by the detectors (no ground truth involved)."""

    host: str
    role: str
    start: datetime
    end: datetime
    peak_score: float
    detectors: list[str]
    suspected_type: IncidentType
    severity: Severity
    top_metrics: list[tuple[str, float]] = field(default_factory=list)
    findings: list[SecurityFinding] = field(default_factory=list)

    @property
    def category(self) -> IncidentCategory:
        """Security if the suspected type is an attack, else operational."""
        return IncidentCategory.SECURITY if self.suspected_type in _SECURITY_TYPES \
            else IncidentCategory.OPERATIONAL

    @property
    def key(self) -> str:
        """Stable identifier: host + type + start minute."""
        return f"{self.host}:{self.suspected_type.value}:{self.start:%Y%m%dT%H%M}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "key": self.key, "host": self.host, "role": self.role,
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "peak_score": round(self.peak_score, 4), "detectors": self.detectors,
            "suspected_type": self.suspected_type.value, "category": self.category.value,
            "severity": self.severity.value,
            "top_metrics": [{"metric": m, "z": round(z, 2)} for m, z in self.top_metrics],
            "findings": [f.to_dict() for f in self.findings[:20]],
        }


def guess_type(top: list[tuple[str, float]]) -> IncidentType:
    """Map the most deviating metrics to the most likely incident type."""
    zs = dict(top)

    def z(name: str) -> float:
        return zs.get(name, 0.0)

    if z("failed_logins") > 6:
        return IncidentType.SSH_BRUTE_FORCE
    if z("net_out_mbps") > 6 and z("requests_per_sec") < 3:
        return IncidentType.DATA_EXFILTRATION
    if z("requests_per_sec") > 4:
        return IncidentType.TRAFFIC_SURGE
    if z("disk_pct") > 4:
        return IncidentType.DISK_FILL
    if z("mem_pct") > 4:
        return IncidentType.MEMORY_LEAK
    if z("error_rate_pct") > 6:
        return IncidentType.ERROR_BURST
    if z("cpu_pct") > 4:
        return IncidentType.SUSPICIOUS_PROCESS if z("process_count") < 1.5 else IncidentType.CPU_SPIKE
    if z("latency_p95_ms") > 3:
        return IncidentType.LATENCY_DEGRADATION
    if z("process_count") > 3:
        return IncidentType.PRIVILEGE_ESCALATION
    return IncidentType.LATENCY_DEGRADATION


def _severity_for(incident_type: IncidentType, peak_norm: float) -> Severity:
    from aiops_copilot.generator.incidents import CATALOG

    base = CATALOG[incident_type].severity
    if peak_norm > 3 and base is not Severity.CRITICAL:
        return Severity(list(_SEVERITY_ORDER)[min(3, _SEVERITY_ORDER[base] + 1)])
    return base


def group_incidents(
    frame: pd.DataFrame,
    flags: np.ndarray,
    scores: np.ndarray,
    z: pd.DataFrame,
    detector_hits: dict[str, np.ndarray],
    findings: list[SecurityFinding],
    step: timedelta = timedelta(minutes=1),
    max_gap_steps: int = 5,
) -> list[DetectedIncident]:
    """Merge flagged rows into incidents.

    Args:
        frame: Host-sorted metrics frame (needs host, role, timestamp).
        flags: Final boolean alarm per row.
        scores: Fused score per row (normalised so 1.0 = threshold).
        z: Per-row z-scores (columns ``z_<metric>``), same index as ``frame``.
        detector_hits: Detector name -> boolean array, to record who fired.
        findings: Security findings, attached to overlapping incidents.
        step: Sampling interval.
        max_gap_steps: Merge alarms separated by at most this many steps.
    """
    incidents: list[DetectedIncident] = []
    flags = np.asarray(flags, dtype=bool)
    by_host: dict[str, list[SecurityFinding]] = {}
    for f in findings:
        by_host.setdefault(f.host, []).append(f)

    for host, idx in frame.groupby("host", sort=False).indices.items():
        idx = np.asarray(idx)
        host_flags = flags[idx]
        if not host_flags.any():
            continue
        ts = frame["timestamp"].to_numpy()[idx]
        flagged_pos = np.flatnonzero(host_flags)
        groups: list[list[int]] = [[int(flagged_pos[0])]]
        for p in flagged_pos[1:]:
            if p - groups[-1][-1] <= max_gap_steps:
                groups[-1].append(int(p))
            else:
                groups.append([int(p)])
        for g in groups:
            rows = idx[g]
            start = pd.Timestamp(ts[g[0]]).to_pydatetime()
            end = pd.Timestamp(ts[g[-1]]).to_pydatetime() + step
            peak_row = rows[int(np.argmax(scores[rows]))]
            mean_z = z.iloc[rows][Z_COLUMNS].mean()
            top = sorted(((c[2:], float(mean_z[c])) for c in Z_COLUMNS),
                         key=lambda kv: abs(kv[1]), reverse=True)[:4]
            hits = [name for name, arr in detector_hits.items() if np.asarray(arr)[rows].any()]
            sec = [f for f in by_host.get(str(host), [])
                   if start - step * 2 <= f.timestamp.replace(tzinfo=start.tzinfo) <= end + step]
            if sec:
                counts: dict[IncidentType, int] = {}
                for f in sec:
                    counts[f.incident_type] = counts.get(f.incident_type, 0) + 1
                suspected = max(counts, key=lambda t: counts[t])
            else:
                suspected = guess_type(top)
            severity = max((f.severity for f in sec), key=lambda s: _SEVERITY_ORDER[s], default=None) \
                or _severity_for(suspected, float(scores[peak_row]))
            incidents.append(DetectedIncident(
                host=str(host), role=str(frame["role"].iloc[rows[0]]), start=start, end=end,
                peak_score=float(scores[peak_row]), detectors=hits, suspected_type=suspected,
                severity=severity, top_metrics=top, findings=sec))
    incidents.sort(key=lambda i: i.start)
    return incidents
