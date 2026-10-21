"""Step-based fleet simulator: the single engine behind batch datasets and live mode.

One call to :meth:`FleetSimulator.step` advances every server by one time step
and returns labelled metric samples and log records. The batch CLI loops over
it; the live service calls it on a timer. Sharing one engine guarantees that
the offline training data and the live demo stream look the same.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from aiops_copilot.generator.baseline import BaselineModel, load_factor
from aiops_copilot.generator.incidents import IncidentTimeline
from aiops_copilot.generator.injectors import apply_incident, clip_values
from aiops_copilot.generator.logs import LogGenerator
from aiops_copilot.generator.models import LogRecord, MetricSample, ServerProfile

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Everything produced by one simulation step."""

    timestamp: datetime
    metrics: list[MetricSample] = field(default_factory=list)
    logs: list[LogRecord] = field(default_factory=list)


class FleetSimulator:
    """Simulates a fleet of servers with injected incidents.

    Args:
        fleet: Servers to simulate.
        timeline: Incidents that may be active (can be extended at runtime).
        step_seconds: Size of one step in seconds.
        seed: Random seed; identical seeds produce identical data.
    """

    def __init__(
        self,
        fleet: list[ServerProfile],
        timeline: IncidentTimeline,
        step_seconds: int,
        seed: int,
    ) -> None:
        if not fleet:
            raise ValueError("fleet must contain at least one server")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.fleet = fleet
        self.timeline = timeline
        self.step_seconds = step_seconds
        # Independent RNG streams so adding logs never changes the metric values.
        seeds = np.random.SeedSequence(seed).spawn(len(fleet) + 2)
        self._baselines = {
            server.name: BaselineModel(server, np.random.default_rng(s))
            for server, s in zip(fleet, seeds[: len(fleet)], strict=True)
        }
        self._inject_rng = np.random.default_rng(seeds[-2])
        self._logs = LogGenerator(np.random.default_rng(seeds[-1]))

    def step(self, ts: datetime) -> StepResult:
        """Advance the whole fleet to ``ts`` and return labelled data."""
        result = StepResult(timestamp=ts)
        load = load_factor(ts)
        for server in self.fleet:
            values, running_jobs = self._baselines[server.name].sample(ts, self.step_seconds)
            healthy_error_rate = values["error_rate_pct"]
            incident = self.timeline.active(server.name, ts)
            sample = MetricSample(timestamp=ts, host=server.name, role=server.role, values=values,
                                  benign_event=",".join(running_jobs))

            if incident is not None:
                progress = incident.progress(ts)
                sample.values = apply_incident(values, incident, progress, self._inject_rng)
                sample.is_anomaly = 1
                sample.anomaly_type = incident.incident_type.value
                sample.incident_id = incident.incident_id
                result.logs.extend(
                    self._logs.incident(server, incident, ts, progress, self.step_seconds)
                )
            sample.values = clip_values(sample.values)
            # Background logs use the *healthy* error rate so that every error line
            # caused by an incident comes from the labelled incident generator.
            result.logs.extend(
                self._logs.normal(server, ts, self.step_seconds, load,
                                  healthy_error_rate, running_jobs)
            )
            result.metrics.append(sample)
        result.logs.sort(key=lambda r: r.timestamp)
        return result
