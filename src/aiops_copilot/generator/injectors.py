"""Metric signatures of each incident type.

Each injector takes healthy metric values and returns modified values, given
how far into the incident we are (``progress`` in [0, 1]) and how strong it is
(``intensity`` in [0.6, 1.0]).

Design note: signatures deliberately differ in *shape*, not just size:
a CPU spike is a step change, a memory leak is a slow ramp (hard for
point-wise detectors such as Isolation Forest, easier for sequence models
such as an LSTM autoencoder), and privilege escalation barely moves metrics
at all (it must be caught from logs). That spread is what makes the
detector comparison in Phases 2-3 interesting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aiops_copilot.generator.models import PERCENT_METRICS, Incident, IncidentType

Values = dict[str, float]
Injector = Callable[[Values, float, float, np.random.Generator], Values]


def _ramp(progress: float, rise: float = 0.1) -> float:
    """Fast rise over the first ``rise`` fraction of the incident, then plateau."""
    return min(1.0, progress / rise) if rise > 0 else 1.0


def _cpu_spike(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    r = _ramp(p)
    v["cpu_pct"] += (88.0 + 8.0 * s - v["cpu_pct"]) * r
    v["latency_p95_ms"] *= 1.0 + 1.2 * s * r
    v["process_count"] += 6.0 * r
    return v


def _memory_leak(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    v["mem_pct"] += (97.0 - v["mem_pct"]) * (p**1.3) * s
    if p > 0.75:  # GC thrashing then OOM near the end
        late = (p - 0.75) / 0.25
        v["latency_p95_ms"] *= 1.0 + 2.0 * late * s
        v["error_rate_pct"] += 6.0 * late * s
        v["cpu_pct"] += 20.0 * late * s
    return v


def _disk_fill(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    v["disk_pct"] += (99.0 - v["disk_pct"]) * p * s
    if p > 0.8:
        v["error_rate_pct"] += 10.0 * (p - 0.8) / 0.2 * s
    return v


def _latency_degradation(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    r = _ramp(p, 0.2)
    v["latency_p95_ms"] *= 1.0 + 4.0 * s * r
    v["error_rate_pct"] += 1.5 * s * r
    v["requests_per_sec"] *= 1.0 - 0.15 * s * r
    return v


def _error_burst(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    r = _ramp(p, 0.05)
    v["error_rate_pct"] += (12.0 + 18.0 * s) * r * rng.uniform(0.8, 1.2)
    v["latency_p95_ms"] *= 1.0 + 0.6 * s * r
    return v


def _traffic_surge(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    r = _ramp(p, 0.15)
    factor = 1.0 + (2.0 + 2.5 * s) * r
    v["requests_per_sec"] *= factor
    v["net_in_mbps"] *= factor
    v["net_out_mbps"] *= 1.0 + (factor - 1.0) * 0.8
    v["cpu_pct"] += 30.0 * s * r
    v["latency_p95_ms"] *= 1.0 + 1.0 * s * r
    return v


def _ssh_brute_force(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    v["failed_logins"] += float(rng.poisson(40.0 + 80.0 * s))
    v["cpu_pct"] += 4.0 * s
    v["net_in_mbps"] += 1.5 * s
    v["process_count"] += float(rng.integers(3, 12))  # forked sshd children
    return v


def _data_exfiltration(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    r = _ramp(p, 0.1)
    v["net_out_mbps"] += (90.0 + 220.0 * s) * r * rng.uniform(0.85, 1.15)
    v["cpu_pct"] += 8.0 * s * r
    v["process_count"] += 2.0
    return v


def _privilege_escalation(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    # Intentionally subtle: a few sudo failures and extra processes only.
    v["failed_logins"] += float(rng.poisson(2.0))
    v["process_count"] += 4.0 + 3.0 * s
    v["cpu_pct"] += 2.0 * s
    return v


def _suspicious_process(v: Values, p: float, s: float, rng: np.random.Generator) -> Values:
    r = _ramp(p, 0.05)
    v["cpu_pct"] += (97.0 - v["cpu_pct"]) * r * (0.75 + 0.25 * s)
    v["process_count"] += 3.0 * r
    v["net_out_mbps"] += 2.0 * r
    v["latency_p95_ms"] *= 1.0 + 0.5 * s * r
    return v


INJECTORS: dict[IncidentType, Injector] = {
    IncidentType.CPU_SPIKE: _cpu_spike,
    IncidentType.MEMORY_LEAK: _memory_leak,
    IncidentType.DISK_FILL: _disk_fill,
    IncidentType.LATENCY_DEGRADATION: _latency_degradation,
    IncidentType.ERROR_BURST: _error_burst,
    IncidentType.TRAFFIC_SURGE: _traffic_surge,
    IncidentType.SSH_BRUTE_FORCE: _ssh_brute_force,
    IncidentType.DATA_EXFILTRATION: _data_exfiltration,
    IncidentType.PRIVILEGE_ESCALATION: _privilege_escalation,
    IncidentType.SUSPICIOUS_PROCESS: _suspicious_process,
}


def clip_values(values: Values) -> Values:
    """Keep metrics physically valid: percentages in [0, 100], others >= 0."""
    for name, value in values.items():
        upper = 100.0 if name in PERCENT_METRICS else float("inf")
        values[name] = float(min(upper, max(0.0, value)))
    values["failed_logins"] = float(round(values["failed_logins"]))
    values["process_count"] = float(round(values["process_count"]))
    return values


def apply_incident(
    values: Values, incident: Incident, progress: float, rng: np.random.Generator
) -> Values:
    """Apply the metric signature of ``incident`` to a copy of ``values``."""
    injector = INJECTORS[incident.incident_type]
    return injector(dict(values), progress, incident.intensity, rng)
