"""Real-time pipeline worker.

Every few seconds it:

1. pulls the newest metrics and logs from the fleet (the simulator's API here;
   in production this would be Prometheus + a log shipper),
2. keeps a rolling buffer per host and runs the hybrid detector on it,
3. turns alarms on the *latest* samples into incidents (merged with any open
   incident on the same host and type),
4. for each new incident, runs root-cause analysis and routes alerts in a
   background thread so slow LLM calls never block detection.

If no trained models exist it trains them first (``--no-bootstrap`` to disable).

Usage::

    python -m aiops_copilot.pipeline.worker
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pandas as pd

from aiops_copilot.alerts.router import AlertRouter
from aiops_copilot.core.config import Settings, get_settings
from aiops_copilot.core.logging import setup_logging
from aiops_copilot.db import repository as repo
from aiops_copilot.db.models import IncidentRow, init_db
from aiops_copilot.detection.grouping import DetectedIncident
from aiops_copilot.detection.hybrid import HybridDetector
from aiops_copilot.detection.security_rules import SecurityRuleEngine
from aiops_copilot.generator.models import METRIC_NAMES
from aiops_copilot.llm.evidence import FrameEvidence
from aiops_copilot.llm.rca import RCAEngine, make_rca_engine
from aiops_copilot.rag.build import build_knowledge_base, kb_dir
from aiops_copilot.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger("aiops_copilot.worker")

BUFFER_ROWS = 240        # per host (4 hours at 1-minute resolution)
MIN_ROWS = 25            # warm-up before detecting
LOG_BUFFER = 30_000
_METRIC_COLS = ["timestamp", "host", "role", *METRIC_NAMES]


def bootstrap(settings: Settings) -> None:
    """Train detectors and build the knowledge base if they are missing."""
    if not (settings.model_dir / "detector_meta.json").exists():
        logger.info("No trained models found: running the training + evaluation pipeline (about 1-2 min)")
        from aiops_copilot.evaluation.run_detection import run as run_detection

        run_detection()
    kb = KnowledgeBase(kb_dir(settings), settings.vector_store, settings.embedding_backend)
    if not kb.ready:
        logger.info("No knowledge base found: building it (about 1 min)")
        build_knowledge_base(settings)


class Worker:
    """Stateful streaming pipeline."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(base_url=settings.simulator_url, timeout=10.0)
        self.detector = HybridDetector.load(settings.model_dir)
        self.rules = SecurityRuleEngine()
        kb = KnowledgeBase(kb_dir(settings), settings.vector_store, settings.embedding_backend)
        self.rca: RCAEngine | None = make_rca_engine(settings, kb.load()) if settings.worker_run_rca else None
        self.sessions = init_db(settings.effective_database_url)
        self.router = AlertRouter(settings)
        self.metrics: dict[str, deque[dict[str, Any]]] = {}
        self.logs: deque[dict[str, Any]] = deque(maxlen=LOG_BUFFER)
        self.last_metric_ts: dict[str, str] = {}
        self.last_log_ts = ""
        self._new_logs: list[dict[str, Any]] = []
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rca")
        self.running = True

    # ------------------------------------------------------------ ingestion
    def poll(self) -> int:
        """Fetch new samples and logs. Returns the number of new metric rows."""
        new = 0
        for row in self.client.get("/api/v1/metrics/latest").raise_for_status().json():
            host = row["host"]
            if row["timestamp"] <= self.last_metric_ts.get(host, ""):
                continue
            self.last_metric_ts[host] = row["timestamp"]
            self.metrics.setdefault(host, deque(maxlen=BUFFER_ROWS)).append({k: row[k] for k in _METRIC_COLS})
            new += 1
        fresh = [r for r in self.client.get("/api/v1/logs", params={"limit": 1000}).raise_for_status().json()
                 if r["timestamp"] > self.last_log_ts]
        fresh.sort(key=lambda r: r["timestamp"])
        if fresh:
            self.last_log_ts = fresh[-1]["timestamp"]
        # Keep only what detection/RCA may see (drop the simulator's ground-truth labels).
        self._new_logs = [{k: r[k] for k in ("timestamp", "host", "source", "level", "message")} for r in fresh]
        self.logs.extend(self._new_logs)
        return new

    def frame(self) -> pd.DataFrame:
        """Current buffers as a host-sorted DataFrame."""
        rows = [r for buf in self.metrics.values() for r in buf]
        df = pd.DataFrame(rows, columns=_METRIC_COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values(["host", "timestamp"]).reset_index(drop=True)

    # ------------------------------------------------------------ detection
    def step(self) -> list[int]:
        """One poll + detect cycle. Returns ids of newly created incidents."""
        if self.poll() == 0 or min((len(b) for b in self.metrics.values()), default=0) < MIN_ROWS:
            return []
        frame = self.frame()
        result = self.detector.detect(frame, self._new_logs, rules=self.rules)
        latest = frame.groupby("host")["timestamp"].max()
        # Only incidents still active on the newest sample (one step of slack); older
        # groups in the buffer were handled on earlier ticks.
        active = [i for i in result.incidents if i.end >= latest[i.host].to_pydatetime()]
        created: list[tuple[int, DetectedIncident]] = []
        with self.sessions() as session, session.begin():
            for inc in active:
                row, is_new = repo.upsert_incident(session, inc)
                if is_new:
                    created.append((row.id, inc))
                    logger.info("New incident #%d %s on %s (%s)", row.id, row.incident_type, row.host,
                                row.severity)
        if created:
            evidence = FrameEvidence(frame.copy(), list(self.logs), self.detector.scaler)
            for incident_id, inc in created:
                self.pool.submit(self._analyze_and_alert, incident_id, inc, evidence)
        return [i for i, _ in created]

    def _analyze_and_alert(self, incident_id: int, inc: DetectedIncident, evidence: FrameEvidence) -> None:
        try:
            with self.sessions() as session, session.begin():
                row = session.get(IncidentRow, incident_id)
                if row is None:
                    return
                if self.rca is not None:
                    result = self.rca.analyze(inc, evidence)
                    repo.save_rca(session, row, result)
                    logger.info("RCA for #%d (%s, %d ms): %s", incident_id, result.engine, result.latency_ms,
                                result.root_cause_id)
                self.router.route(session, row)
        except Exception:  # noqa: BLE001 - never kill the worker thread
            logger.exception("RCA/alerting failed for incident %d", incident_id)

    def run_forever(self) -> None:
        """Main loop with graceful shutdown on SIGINT/SIGTERM."""
        def stop(*_: Any) -> None:
            self.running = False

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        logger.info("Worker started: polling %s every %ss", self.settings.simulator_url,
                    self.settings.worker_poll_seconds)
        backoff = 1.0
        while self.running:
            started = time.monotonic()
            try:
                self.step()
                backoff = 1.0
            except httpx.HTTPError as exc:
                logger.warning("simulator not reachable (%s); retrying in %.0fs", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            except Exception:  # noqa: BLE001
                logger.exception("pipeline step failed")
            time.sleep(max(0.0, self.settings.worker_poll_seconds - (time.monotonic() - started)))
        self.pool.shutdown(wait=True)
        logger.info("Worker stopped")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-bootstrap", action="store_true", help="Fail instead of training missing models")
    args = parser.parse_args(argv)
    if not args.no_bootstrap:
        bootstrap(settings)
    Worker(settings).run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
