"""Backend API: auth, validation, rate limiting and endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aiops_copilot.api.main import create_app
from aiops_copilot.core.config import Settings
from aiops_copilot.db import repository as repo
from conftest import Trained
from test_alerts_db import make_incident

KEY = {"X-API-Key": "test-key-123"}


def _settings(tmp_path: Path, **kw: Any) -> Settings:
    base: dict[str, Any] = {"database_url": f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
                            "model_dir": tmp_path / "models", "reports_dir": tmp_path / "reports",
                            "data_dir": tmp_path / "data", "api_key": "test-key-123", "llm_provider": "none"}
    base.update(kw)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as c:
        yield c  # type: ignore[misc]


def test_health_needs_no_key(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_api_key_required(client: TestClient) -> None:
    assert client.get("/api/v1/incidents").status_code == 401
    assert client.get("/api/v1/incidents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/incidents", headers=KEY).json() == []


def test_incident_lifecycle(client: TestClient) -> None:
    factory = client.app.state.session_factory  # type: ignore[attr-defined]
    with factory() as s, s.begin():
        row, _ = repo.upsert_incident(s, make_incident())
        incident_id = row.id
    listed = client.get("/api/v1/incidents", headers=KEY, params={"status": "open"}).json()
    assert [i["id"] for i in listed] == [incident_id]
    detail = client.get(f"/api/v1/incidents/{incident_id}", headers=KEY).json()
    assert detail["host"] == "web-01" and detail["alerts"] == []
    r = client.post(f"/api/v1/incidents/{incident_id}/status", json={"status": "acknowledged"}, headers=KEY)
    assert r.status_code == 200 and r.json()["status"] == "acknowledged"
    assert client.get("/api/v1/stats", headers=KEY).json()["by_status"] == {"acknowledged": 1}


@pytest.mark.parametrize("path,kwargs,code", [
    ("/api/v1/incidents/999", {}, 404),
    ("/api/v1/incidents", {"params": {"limit": 5000}}, 422),
    ("/api/v1/incidents", {"params": {"status": "deleted"}}, 422),
    ("/api/v1/incidents", {"params": {"host": "x" * 100}}, 422),
])
def test_input_validation(client: TestClient, path: str, kwargs: dict[str, Any], code: int) -> None:
    assert client.get(path, headers=KEY, **kwargs).status_code == code


def test_bad_status_rejected(client: TestClient) -> None:
    r = client.post("/api/v1/incidents/1/status", json={"status": "deleted"}, headers=KEY)
    assert r.status_code == 422


def test_analyze_without_models_is_503(client: TestClient) -> None:
    body = {"metrics": [{"timestamp": "2026-01-08T00:00:00Z", "host": "web-01", "role": "web", "cpu_pct": 30,
                         "mem_pct": 50, "disk_pct": 40, "latency_p95_ms": 100, "error_rate_pct": 0.5,
                         "requests_per_sec": 150, "net_in_mbps": 10, "net_out_mbps": 30, "failed_logins": 0,
                         "process_count": 180}]}
    assert client.post("/api/v1/analyze", json=body, headers=KEY).status_code == 503
    body["metrics"][0]["cpu_pct"] = 250  # out of range
    assert client.post("/api/v1/analyze", json=body, headers=KEY).status_code == 422
    body["metrics"][0].update(cpu_pct=30, host="web-01; DROP TABLE")
    assert client.post("/api/v1/analyze", json=body, headers=KEY).status_code == 422


def test_analyze_with_trained_models(tmp_path: Path, trained: Trained) -> None:
    app = create_app(_settings(tmp_path, model_dir=trained.model_dir))
    window = trained.test
    inc = next(i for i in trained.test_incidents if i["incident_type"] == "cpu_spike") \
        if any(i["incident_type"] == "cpu_spike" for i in trained.test_incidents) else trained.test_incidents[0]
    host_rows = window[window["host"] == inc["host"]]
    cols = ["timestamp", "host", "role", "cpu_pct", "mem_pct", "disk_pct", "latency_p95_ms", "error_rate_pct",
            "requests_per_sec", "net_in_mbps", "net_out_mbps", "failed_logins", "process_count"]
    rows = host_rows[cols].copy()
    rows["timestamp"] = rows["timestamp"].map(lambda t: t.isoformat())
    body = {"metrics": rows.to_dict("records"), "run_rca": False}
    with TestClient(app) as c:
        r = c.post("/api/v1/analyze", json=body, headers=KEY)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["rows"] == len(rows) and out["flagged_rows"] > 0 and out["incidents"]


def test_rate_limit(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, api_rate_limit_per_minute=3))
    with TestClient(app) as c:
        codes = [c.get("/api/v1/stats", headers=KEY).status_code for _ in range(5)]
        assert c.get("/health").status_code == 200  # health checks are exempt
    assert codes[:3] == [200, 200, 200] and codes[-1] == 429
