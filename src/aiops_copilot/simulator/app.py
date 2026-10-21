"""Live fleet simulator as a small FastAPI service.

It stands in for real servers: Prometheus scrapes ``/metrics`` exactly as it
would scrape a node exporter, and later phases read logs from the JSON API or
from ``data/live/*.jsonl``.

Endpoints:
    GET  /health                       liveness + simulated clock
    GET  /metrics                      Prometheus exposition format
    GET  /api/v1/hosts                 fleet inventory
    GET  /api/v1/metrics/latest        latest sample per host (optionally one host)
    GET  /api/v1/logs                  recent log lines with filters
    GET  /api/v1/incidents             ground-truth incidents (for evaluation/demo)
    POST /api/v1/incidents/inject      start an anomaly/attack on demand (demo)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest
from pydantic import BaseModel, Field

from aiops_copilot import __version__
from aiops_copilot.core.config import Settings, get_settings
from aiops_copilot.core.logging import setup_logging
from aiops_copilot.generator.models import METRIC_NAMES, IncidentType
from aiops_copilot.simulator.engine import LiveConfig, LiveSimulation

logger = logging.getLogger(__name__)


def get_sim(request: Request) -> LiveSimulation:
    """FastAPI dependency returning the app's live simulation."""
    sim: LiveSimulation = request.app.state.sim
    return sim


SimDep = Annotated[LiveSimulation, Depends(get_sim)]


class InjectRequest(BaseModel):
    """Body of the inject endpoint."""

    incident_type: IncidentType
    host: str | None = Field(default=None, max_length=64, pattern=r"^[a-z0-9-]+$")


class Telemetry:
    """Prometheus metrics, kept in a private registry (no global state)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.gauges: dict[str, Gauge] = {
            name: Gauge(f"aiops_server_{name}", f"Simulated {name}", ["host", "role"],
                        registry=self.registry)
            for name in METRIC_NAMES
        }
        self.truth = Gauge("aiops_ground_truth_incident_active",
                           "1 while an injected incident is active (ground truth label)",
                           ["host", "incident_type"], registry=self.registry)
        self.log_lines = Counter("aiops_sim_log_lines_total", "Log lines emitted",
                                 ["host", "source"], registry=self.registry)
        self.ticks = Counter("aiops_sim_ticks_total", "Simulation ticks", registry=self.registry)

    def update(self, sim: LiveSimulation) -> None:
        """Refresh gauges from the simulation's latest state."""
        for sample in sim.latest.values():
            for name in METRIC_NAMES:
                self.gauges[name].labels(sample.host, sample.role.value).set(sample.values[name])
        self.truth.clear()
        for incident in sim.active_incidents():
            self.truth.labels(incident.host, incident.incident_type.value).set(1)


def create_app(settings: Settings | None = None, autostart: bool = True) -> FastAPI:
    """Application factory.

    Args:
        settings: Override settings (tests pass their own).
        autostart: Start the background tick loop. Tests disable it and call
            ``app.state.sim.tick()`` directly for determinism.
    """
    settings = settings or get_settings()
    config = LiveConfig(
        seed=settings.gen_seed,
        tick_seconds=settings.sim_tick_seconds,
        speedup=settings.sim_speedup,
        incident_probability=settings.sim_incident_probability,
        log_dir=settings.sim_log_dir,
        log_max_mb=settings.sim_log_max_mb,
    )
    telemetry = Telemetry()

    async def _run_loop(sim: LiveSimulation) -> None:
        while True:
            try:
                result = sim.tick()
                telemetry.ticks.inc()
                for rec in result.logs:
                    telemetry.log_lines.labels(rec.host, rec.source).inc()
                telemetry.update(sim)
            except Exception:  # keep the simulator alive; log and continue
                logger.exception("simulation tick failed")
            await asyncio.sleep(config.tick_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        if autostart:
            task = asyncio.create_task(_run_loop(app.state.sim))
            logger.info("Simulator running: tick=%ss speedup=%sx hosts=%s",
                        config.tick_seconds, config.speedup, app.state.sim.host_names)
        yield
        if task:
            task.cancel()

    app = FastAPI(title="AI Ops Copilot - Fleet Simulator", version=__version__, lifespan=lifespan)
    app.state.sim = LiveSimulation(config)
    app.state.telemetry = telemetry

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        """If SIM_ADMIN_TOKEN is set, require it on mutating endpoints."""
        expected = settings.sim_admin_token.get_secret_value() if settings.sim_admin_token else ""
        if expected and not hmac.compare_digest(x_admin_token or "", expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-Admin-Token")

    @app.get("/health")
    def health(sim: SimDep) -> dict[str, Any]:
        return {"status": "ok", "sim_time": sim.sim_time.isoformat(), "ticks": sim.ticks,
                "active_incidents": len(sim.active_incidents())}

    @app.get("/metrics")
    def prometheus_metrics(sim: SimDep) -> Response:
        telemetry.update(sim)
        return Response(generate_latest(telemetry.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/hosts")
    def hosts(sim: SimDep) -> list[dict[str, str]]:
        return [{"name": s.name, "role": s.role.value, "ip": s.ip} for s in sim.fleet]

    @app.get("/api/v1/metrics/latest")
    def latest_metrics(sim: SimDep,
                       host: Annotated[str | None, Query(max_length=64)] = None
                       ) -> list[dict[str, Any]]:
        if host is not None and host not in sim.host_names:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown host {host!r}")
        samples = [sim.latest[host]] if host and host in sim.latest else \
            ([] if host else list(sim.latest.values()))
        return [s.to_row() for s in samples]

    @app.get("/api/v1/logs")
    def logs(sim: SimDep,
             host: Annotated[str | None, Query(max_length=64)] = None,
             source: Annotated[str | None, Query(max_length=32)] = None,
             only_anomalies: bool = False,
             limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[dict[str, Any]]:
        if host is not None and host not in sim.host_names:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown host {host!r}")
        out: list[dict[str, Any]] = []
        for rec in reversed(sim.recent_logs):
            if host and rec.host != host:
                continue
            if source and rec.source != source:
                continue
            if only_anomalies and not rec.is_anomaly:
                continue
            out.append(rec.to_dict())
            if len(out) >= limit:
                break
        return out

    @app.get("/api/v1/incidents")
    def incidents(sim: SimDep, active_only: bool = False) -> list[dict[str, Any]]:
        items = sim.active_incidents() if active_only else sim.history
        return [i.to_dict() for i in items]

    @app.post("/api/v1/incidents/inject", status_code=status.HTTP_201_CREATED,
              dependencies=[Depends(require_admin)])
    def inject(body: InjectRequest, sim: SimDep) -> dict[str, Any]:
        try:
            incident = sim.inject(body.incident_type, body.host)
        except ValueError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        if incident is None:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                "no eligible host is free for this incident type right now")
        return incident.to_dict()

    return app


def main() -> None:
    """Run the simulator with uvicorn (``python -m aiops_copilot.simulator``)."""
    import uvicorn

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    uvicorn.run(create_app(settings), host=settings.sim_host, port=settings.sim_port,
                log_level=settings.log_level.lower())
