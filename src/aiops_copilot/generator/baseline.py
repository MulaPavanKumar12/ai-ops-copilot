"""Normal ("healthy") metric behaviour for each server.

Realistic synthetic data needs three properties, otherwise any detector looks
perfect and the evaluation is meaningless:

1. **Seasonality**: daily load curve (peak ~14:00 UTC, trough ~02:00) and a
   quieter weekend.
2. **Autocorrelation**: AR(1) noise, so a metric drifts smoothly instead of
   jumping randomly each step. This is what makes LSTM models useful later.
3. **Benign surprises**: heavy-tailed blips plus scheduled jobs (nightly DB
   backup, deploys, log rotation) that look unusual but are *not* incidents.
   These are the "hard negatives" that create realistic false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from aiops_copilot.generator.models import METRIC_NAMES, ServerProfile, ServerRole

#: Relative standard deviation of the AR(1) noise innovation, per metric.
_NOISE_SD: dict[str, float] = {
    "cpu_pct": 0.05,
    "mem_pct": 0.012,
    "disk_pct": 0.002,
    "latency_p95_ms": 0.06,
    "error_rate_pct": 0.20,
    "requests_per_sec": 0.04,
    "net_in_mbps": 0.06,
    "net_out_mbps": 0.06,
    "failed_logins": 0.0,
    "process_count": 0.01,
}
_AR_PHI: float = 0.8
_BLIP_PROBABILITY: float = 0.003


def load_factor(ts: datetime) -> float:
    """Daily/weekly traffic multiplier around 1.0 for timestamp ``ts``."""
    hour = ts.hour + ts.minute / 60.0
    daily = 1.0 + 0.35 * math.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
    weekend = 0.8 if ts.weekday() >= 5 else 1.0
    return daily * weekend


@dataclass(frozen=True)
class ScheduledJob:
    """A recurring, benign operation that perturbs metrics (a hard negative)."""

    name: str
    roles: frozenset[ServerRole]
    hour: int
    minute: int
    duration_min: int
    effects: dict[str, float]

    def is_running(self, ts: datetime) -> bool:
        """True if the job is running at ``ts`` (daily schedule)."""
        start = ts.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
        delta = (ts - start).total_seconds()
        return 0 <= delta < self.duration_min * 60


#: Benign scheduled jobs. Additive effects on top of the baseline.
SCHEDULED_JOBS: tuple[ScheduledJob, ...] = (
    ScheduledJob(
        name="nightly_backup",
        roles=frozenset({ServerRole.DB}),
        hour=2, minute=0, duration_min=20,
        effects={"net_out_mbps": 110.0, "cpu_pct": 15.0, "latency_p95_ms": 6.0},
    ),
    ScheduledJob(
        name="log_rotation",
        roles=frozenset(ServerRole),
        hour=0, minute=0, duration_min=3,
        effects={"cpu_pct": 8.0, "process_count": 4.0},
    ),
    ScheduledJob(
        name="deploy",
        roles=frozenset({ServerRole.API, ServerRole.WEB}),
        hour=15, minute=30, duration_min=4,
        effects={"latency_p95_ms": 60.0, "cpu_pct": 10.0, "error_rate_pct": 0.6},
    ),
)


@dataclass
class BaselineModel:
    """Stateful generator of healthy metrics for one server.

    Call :meth:`sample` once per time step, in chronological order.
    """

    profile: ServerProfile
    rng: np.random.Generator
    _ar_state: dict[str, float] = field(default_factory=dict)
    _disk_drift: float = 0.0

    def __post_init__(self) -> None:
        self._ar_state = {name: 0.0 for name in METRIC_NAMES}

    def _advance_noise(self) -> dict[str, float]:
        """Advance the AR(1) noise process and return relative perturbations."""
        for name in METRIC_NAMES:
            innovation = self.rng.normal(0.0, _NOISE_SD[name])
            self._ar_state[name] = _AR_PHI * self._ar_state[name] + innovation
        return dict(self._ar_state)

    def sample(self, ts: datetime, step_seconds: int) -> tuple[dict[str, float], list[str]]:
        """Generate healthy metric values at ``ts``.

        Args:
            ts: Timestamp of this step.
            step_seconds: Size of the simulation step, used for slow drifts.

        Returns:
            A tuple of (metric values, names of benign scheduled jobs running).
        """
        base = self.profile.baseline
        load = load_factor(ts)
        noise = self._advance_noise()
        self._disk_drift += 0.15 * step_seconds / 3600.0 / 24.0  # ~0.15 % per day

        values: dict[str, float] = {
            "cpu_pct": base["cpu_pct"] * load**0.9,
            "mem_pct": base["mem_pct"] + 4.0 * (load - 1.0),
            "disk_pct": base["disk_pct"] + self._disk_drift,
            "latency_p95_ms": base["latency_p95_ms"] * (1.0 + 0.3 * (load - 1.0)),
            "error_rate_pct": base["error_rate_pct"] * (0.8 + 0.2 * load),
            "requests_per_sec": base["requests_per_sec"] * load,
            "net_in_mbps": base["net_in_mbps"] * load,
            "net_out_mbps": base["net_out_mbps"] * load,
            "failed_logins": float(self.rng.poisson(base["failed_logins"])),
            "process_count": base["process_count"],
        }
        for name in METRIC_NAMES:
            values[name] *= 1.0 + noise[name]

        # Heavy-tailed one-off blip: a realistic source of false positives.
        if self.rng.random() < _BLIP_PROBABILITY:
            metric = self.rng.choice(["cpu_pct", "latency_p95_ms", "requests_per_sec"])
            values[str(metric)] *= 1.0 + self.rng.uniform(0.3, 0.7)

        running: list[str] = []
        for job in SCHEDULED_JOBS:
            if self.profile.role in job.roles and job.is_running(ts):
                running.append(job.name)
                for metric, delta in job.effects.items():
                    values[metric] += delta
        return values, running
