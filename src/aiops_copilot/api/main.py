"""AI Ops Copilot backend API.

Endpoints (all under ``/api/v1`` need ``X-API-Key`` when ``API_KEY`` is set):

    GET  /health                          liveness + database check
    GET  /api/v1/incidents                list (filters: status, host, category; paging)
    GET  /api/v1/incidents/{id}           detail with alerts and RCA runs
    POST /api/v1/incidents/{id}/status    acknowledge / resolve / reopen
    GET  /api/v1/stats                    counts for the dashboard
    GET  /api/v1/model-info               detector thresholds + latest evaluation results
    POST /api/v1/analyze                  run detection + RCA on a submitted window (no storage)

Run: ``python -m aiops_copilot.api`` (port 8000), docs at ``/docs``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from aiops_copilot import __version__
from aiops_copilot.api.security import RateLimiter, RateLimitMiddleware, make_api_key_checker
from aiops_copilot.core.config import Settings, get_settings
from aiops_copilot.core.logging import setup_logging
from aiops_copilot.db import repository as repo
from aiops_copilot.db.models import IncidentRow, init_db
from aiops_copilot.generator.models import METRIC_NAMES

logger = logging.getLogger(__name__)


class StatusUpdate(BaseModel):
    """Body for the status endpoint."""

    status: Literal["open", "acknowledged", "resolved"]


class MetricIn(BaseModel):
    """One metric sample submitted for analysis."""

    timestamp: datetime
    host: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    role: Literal["web", "api", "db", "worker"]
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    disk_pct: float = Field(ge=0, le=100)
    latency_p95_ms: float = Field(ge=0, le=1e6)
    error_rate_pct: float = Field(ge=0, le=100)
    requests_per_sec: float = Field(ge=0, le=1e7)
    net_in_mbps: float = Field(ge=0, le=1e6)
    net_out_mbps: float = Field(ge=0, le=1e6)
    failed_logins: float = Field(ge=0, le=1e6)
    process_count: float = Field(ge=0, le=1e6)


class LogIn(BaseModel):
    """One log line submitted for analysis."""

    timestamp: datetime
    host: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    source: str = Field(max_length=32)
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    message: str = Field(max_length=4000)


class AnalyzeRequest(BaseModel):
    """A window of telemetry to analyse (bounded to keep requests cheap)."""

    metrics: list[MetricIn] = Field(min_length=1, max_length=20_000)
    logs: list[LogIn] = Field(default_factory=list, max_length=50_000)
    run_rca: bool = True


def get_db(request: Request) -> Iterator[Session]:
    """One session per request; commit on success, roll back on error."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DB = Annotated[Session, Depends(get_db)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""
    settings = settings or get_settings()
    session_factory: sessionmaker[Session] = init_db(settings.effective_database_url)
    state: dict[str, Any] = {}

    def load_detector() -> Any:
        """Load the trained detector once; retried on later requests until training has run."""
        if "detector" not in state:
            try:
                from aiops_copilot.detection.hybrid import HybridDetector
                state["detector"] = HybridDetector.load(settings.model_dir)
            except FileNotFoundError:
                return None
        return state["detector"]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if load_detector() is None:
            logger.warning("No trained models in %s yet; /api/v1/analyze returns 503 until training has run",
                           settings.model_dir)
        yield

    app = FastAPI(title="AI Ops Copilot API", version=__version__, lifespan=lifespan)
    app.state.session_factory = session_factory
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(settings.api_rate_limit_per_minute))
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o],
                       allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])
    require_key = make_api_key_checker(settings.api_key.get_secret_value() if settings.api_key else "")
    if not settings.api_key and settings.app_env == "production":
        logger.warning("API_KEY is not set in production: the API is unauthenticated")

    auth = [Depends(require_key)]

    @app.get("/health")
    def health(session: DB) -> dict[str, Any]:
        session.execute(text("SELECT 1"))
        return {"status": "ok", "version": __version__, "models_loaded": "detector" in state}

    @app.get("/api/v1/incidents", dependencies=auth)
    def incidents(session: DB,
                  status_: Annotated[Literal["open", "acknowledged", "resolved"] | None, Query(alias="status")] = None,
                  host: Annotated[str | None, Query(max_length=64)] = None,
                  category: Annotated[Literal["operational", "security"] | None, Query()] = None,
                  limit: Annotated[int, Query(ge=1, le=200)] = 50,
                  offset: Annotated[int, Query(ge=0, le=100_000)] = 0) -> list[dict[str, Any]]:
        return [r.to_dict() for r in repo.list_incidents(session, status_, host, category, limit, offset)]

    @app.get("/api/v1/incidents/{incident_id}", dependencies=auth)
    def incident_detail(incident_id: int, session: DB) -> dict[str, Any]:
        row = session.get(IncidentRow, incident_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "incident not found")
        return {**row.to_dict(), "alerts": [a.to_dict() for a in row.alerts],
                "rca_runs": [{"engine": r.engine, "latency_ms": r.latency_ms,
                              "created_at": r.created_at.isoformat()} for r in row.rca_runs]}

    @app.post("/api/v1/incidents/{incident_id}/status", dependencies=auth)
    def update_status(incident_id: int, body: StatusUpdate, session: DB) -> dict[str, Any]:
        row = repo.set_status(session, incident_id, body.status)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "incident not found")
        return row.to_dict()

    @app.get("/api/v1/stats", dependencies=auth)
    def get_stats(session: DB) -> dict[str, Any]:
        return repo.stats(session)

    @app.get("/api/v1/model-info", dependencies=auth)
    def model_info() -> dict[str, Any]:
        detector = load_detector()
        out: dict[str, Any] = {"models_loaded": detector is not None}
        if detector is not None:
            out["detector"] = detector.meta
        for name in ("detection_results", "rca_results", "rca_results_llm"):
            path = settings.reports_dir / f"{name}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                out[name] = data.get("summary", data) if name.startswith("rca") else data
        return out

    @app.post("/api/v1/analyze", dependencies=auth)
    def analyze(body: AnalyzeRequest, request: Request) -> dict[str, Any]:
        detector = load_detector()
        if detector is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "models not trained yet")
        frame = pd.DataFrame([m.model_dump() for m in body.metrics])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values(["host", "timestamp"]).reset_index(drop=True)
        logs = [{**lg.model_dump(), "timestamp": lg.timestamp} for lg in body.logs]
        try:
            result = detector.detect(frame, logs)
        except KeyError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"unknown host/role: {exc}") from exc
        incidents = [i.to_dict() for i in result.incidents]
        if body.run_rca and result.incidents:
            from aiops_copilot.llm.evidence import FrameEvidence
            from aiops_copilot.llm.rca import make_rca_engine
            from aiops_copilot.rag.build import kb_dir
            from aiops_copilot.rag.knowledge_base import KnowledgeBase

            kb = KnowledgeBase(kb_dir(settings), settings.vector_store, settings.embedding_backend)
            if kb.ready:
                engine = make_rca_engine(settings, kb.load())
                evidence = FrameEvidence(frame, [{**lg, "timestamp": lg["timestamp"].isoformat()} for lg in logs],
                                         detector.scaler)
                for out, inc in zip(incidents, result.incidents, strict=True):
                    out["rca"] = engine.analyze(inc, evidence).model_dump()
        return {"rows": len(frame), "flagged_rows": int(result.final_flags.sum()), "incidents": incidents,
                "metrics_checked": list(METRIC_NAMES)}

    return app


def main() -> None:
    """Run with uvicorn (``python -m aiops_copilot.api``)."""
    import uvicorn

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port,
                log_level=settings.log_level.lower())
