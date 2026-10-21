"""Live simulation state, independent of the web framework (easy to unit test).

Simulated time runs ``speedup`` times faster than wall-clock time so that a
30-minute incident plays out in ~2.5 real minutes during a demo.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from aiops_copilot.generator.fleet import build_fleet
from aiops_copilot.generator.incidents import IncidentTimeline, LiveScheduler
from aiops_copilot.generator.models import (
    Incident,
    IncidentType,
    LogRecord,
    MetricSample,
    ServerProfile,
)
from aiops_copilot.generator.simulator import FleetSimulator, StepResult

logger = logging.getLogger(__name__)


class RotatingJsonlWriter:
    """Append JSON lines to a file, rotating to ``<name>.1`` above a size limit."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.enabled = True
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._disable(exc)

    def _disable(self, exc: OSError) -> None:
        """Stop writing files but keep the simulator (and its API) running."""
        self.enabled = False
        logger.warning("Cannot write %s (%s); file output disabled, API still serves data",
                       self.path, exc)

    def write_many(self, records: list[dict[str, Any]]) -> None:
        """Append records, rotating first if the file is too large."""
        if not records or not self.enabled:
            return
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                backup = self.path.with_suffix(self.path.suffix + ".1")
                backup.unlink(missing_ok=True)
                self.path.rename(backup)
            with self.path.open("a", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            self._disable(exc)


@dataclass
class LiveConfig:
    """Runtime parameters of the live simulation."""

    n_servers: int = 6
    seed: int = 42
    tick_seconds: float = 5.0
    speedup: float = 12.0
    incident_probability: float = 0.02
    log_dir: Path = Path("data/live")
    log_max_mb: int = 50
    buffer_size: int = 5000

    @property
    def step_seconds(self) -> int:
        """Simulated seconds advanced per tick."""
        return max(1, int(round(self.tick_seconds * self.speedup)))


class LiveSimulation:
    """Holds the live fleet state and the most recent data in memory."""

    def __init__(self, config: LiveConfig, start: datetime | None = None) -> None:
        self.config = config
        self.fleet: list[ServerProfile] = build_fleet(config.n_servers)
        self.timeline = IncidentTimeline()
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 99]))
        self.scheduler = LiveScheduler(rng, config.incident_probability)
        self.simulator = FleetSimulator(self.fleet, self.timeline, config.step_seconds, config.seed)
        now = start or datetime.now(UTC)
        self.sim_time: datetime = now.replace(second=0, microsecond=0)
        self.latest: dict[str, MetricSample] = {}
        self.recent_logs: deque[LogRecord] = deque(maxlen=config.buffer_size)
        self.history: list[Incident] = []
        self.ticks = 0
        max_bytes = config.log_max_mb * 1024 * 1024
        self._metrics_writer = RotatingJsonlWriter(config.log_dir / "metrics.jsonl", max_bytes)
        self._logs_writer = RotatingJsonlWriter(config.log_dir / "logs.jsonl", max_bytes)

    @property
    def host_names(self) -> list[str]:
        """Names of all simulated hosts."""
        return [s.name for s in self.fleet]

    def tick(self) -> StepResult:
        """Advance simulated time by one step, possibly spawning an incident."""
        spawned = self.scheduler.maybe_spawn(self.fleet, self.timeline, self.sim_time)
        if spawned:
            self.history.append(spawned)
        result = self.simulator.step(self.sim_time)
        for sample in result.metrics:
            self.latest[sample.host] = sample
        self.recent_logs.extend(result.logs)
        self._metrics_writer.write_many([m.to_row() for m in result.metrics])
        self._logs_writer.write_many([r.to_dict() for r in result.logs])
        self.sim_time += timedelta(seconds=self.config.step_seconds)
        self.timeline.prune(self.sim_time - timedelta(hours=6))
        self.history = self.history[-500:]
        self.ticks += 1
        return result

    def inject(self, incident_type: IncidentType, host: str | None = None) -> Incident | None:
        """Force an incident to start now (used by the demo endpoint).

        Returns:
            The new incident, or None if no eligible host is free.

        Raises:
            ValueError: If ``host`` is not part of the fleet.
        """
        if host is not None and host not in self.host_names:
            raise ValueError(f"unknown host {host!r}")
        incident = self.scheduler.maybe_spawn(
            self.fleet, self.timeline, self.sim_time, forced_type=incident_type, forced_host=host
        )
        if incident:
            self.history.append(incident)
        return incident

    def active_incidents(self) -> list[Incident]:
        """Incidents active at the current simulated time."""
        return [i for i in self.timeline.incidents if i.is_active(self.sim_time)]
