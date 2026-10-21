"""End-to-end: the streaming worker turns simulator data into stored, explained, alerted incidents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from sqlalchemy import select

from aiops_copilot.core.config import Settings
from aiops_copilot.db.models import AlertRow, IncidentRow
from aiops_copilot.generator.models import METRIC_NAMES
from aiops_copilot.pipeline.worker import Worker
from conftest import Trained


class FakeSimulator:
    """Replays a recorded split through the simulator's HTTP API, one minute per poll."""

    def __init__(self, frame: pd.DataFrame, logs: list[dict[str, Any]], start: pd.Timestamp) -> None:
        self.times = sorted(t for t in frame["timestamp"].unique() if t >= start)
        self.frame = frame
        self.logs = sorted(logs, key=lambda r: r["timestamp"])
        self.i = 0

    @property
    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self.times[min(self.i, len(self.times) - 1)])

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/metrics/latest":
            rows = self.frame[self.frame["timestamp"] == self.now]
            out = [{"timestamp": r["timestamp"].isoformat(), "host": r["host"], "role": r["role"],
                    **{m: float(r[m]) for m in METRIC_NAMES}} for _, r in rows.iterrows()]
            return httpx.Response(200, json=out)
        if request.url.path == "/api/v1/logs":
            cutoff = self.now + pd.Timedelta(seconds=59)
            recent = [r for r in self.logs if pd.Timestamp(r["timestamp"]) <= cutoff][-1000:]
            return httpx.Response(200, json=recent)
        return httpx.Response(404)


def test_worker_creates_explained_incident(tmp_path: Path, trained: Trained, kb_settings: Settings,
                                           knowledge_base: object) -> None:
    truth = next(i for i in trained.test_incidents if i["incident_type"] == "ssh_brute_force")
    start = pd.Timestamp(truth["start"]) - pd.Timedelta(minutes=40)
    sim = FakeSimulator(trained.test, trained.test_logs, start)
    settings = kb_settings.model_copy(update={
        "model_dir": trained.model_dir, "database_url": f"sqlite:///{(tmp_path / 'w.db').as_posix()}"})
    client = httpx.Client(base_url="http://sim", transport=httpx.MockTransport(sim.handler))
    worker = Worker(settings, client=client)

    steps = int((pd.Timestamp(truth["end"]) - start).total_seconds() // 60) + 5
    for _ in range(steps):
        worker.step()
        sim.i += 1
    worker.pool.shutdown(wait=True)

    with worker.sessions() as s:
        rows = s.scalars(select(IncidentRow).where(IncidentRow.host == truth["host"])).all()
        alerts = s.scalars(select(AlertRow)).all()
    attack = [r for r in rows if r.incident_type == "ssh_brute_force"]
    assert attack, f"no brute-force incident; got {[(r.host, r.incident_type) for r in rows]}"
    row = attack[0]
    assert row.category == "security"
    assert row.summary and row.root_cause_id.startswith("ssh_brute_force/")
    assert any(a.channel == "dashboard" and a.status == "sent" for a in alerts)
