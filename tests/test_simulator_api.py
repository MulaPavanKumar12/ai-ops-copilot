"""Tests for the live simulator HTTP API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from aiops_copilot.core.config import Settings
from aiops_copilot.simulator.app import create_app


def _client(tmp_path: Path, token: str | None = None) -> TestClient:
    settings = Settings(
        _env_file=None,
        sim_log_dir=tmp_path / "live",
        sim_incident_probability=0.0,
        sim_admin_token=SecretStr(token) if token else None,
    )
    app = create_app(settings, autostart=False)
    for _ in range(3):
        app.state.sim.tick()
    return TestClient(app)


def test_health_and_hosts(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    hosts = client.get("/api/v1/hosts").json()
    assert len(hosts) == 6 and hosts[0]["name"] == "web-01"


def test_prometheus_endpoint(tmp_path: Path) -> None:
    body = _client(tmp_path).get("/metrics").text
    assert 'aiops_server_cpu_pct{host="web-01",role="web"}' in body


def test_inject_attack_shows_in_logs_metrics_and_truth(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/api/v1/incidents/inject",
                       json={"incident_type": "ssh_brute_force", "host": "web-01"})
    assert resp.status_code == 201
    incident_id = resp.json()["incident_id"]
    sim = client.app.state.sim  # type: ignore[attr-defined]
    sim.tick()
    logs = client.get("/api/v1/logs", params={"only_anomalies": True, "host": "web-01"}).json()
    assert logs and all(r["incident_id"] == incident_id for r in logs)
    latest = client.get("/api/v1/metrics/latest", params={"host": "web-01"}).json()[0]
    assert latest["failed_logins"] > 10 and latest["is_anomaly"] == 1
    assert 'aiops_ground_truth_incident_active{host="web-01",incident_type="ssh_brute_force"} 1.0' \
        in client.get("/metrics").text
    assert (tmp_path / "live" / "logs.jsonl").exists()


@pytest.mark.parametrize(
    ("params", "code"),
    [({"limit": 0}, 422), ({"limit": 5000}, 422), ({"host": "nope"}, 404)],
)
def test_log_query_validation(tmp_path: Path, params: dict, code: int) -> None:
    assert _client(tmp_path).get("/api/v1/logs", params=params).status_code == code


def test_inject_validation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/api/v1/incidents/inject", json={"incident_type": "bogus"}).status_code == 422
    assert client.post("/api/v1/incidents/inject",
                       json={"incident_type": "cpu_spike", "host": "Bad Host!"}).status_code == 422
    assert client.post("/api/v1/incidents/inject",
                       json={"incident_type": "cpu_spike", "host": "ghost-01"}).status_code == 404
    # memory_leak is not allowed on db hosts -> no eligible host
    assert client.post("/api/v1/incidents/inject",
                       json={"incident_type": "memory_leak", "host": "db-01"}).status_code == 409


def test_admin_token_required_when_configured(tmp_path: Path) -> None:
    client = _client(tmp_path, token="s3cret")
    body = {"incident_type": "cpu_spike"}
    assert client.post("/api/v1/incidents/inject", json=body).status_code == 401
    assert client.post("/api/v1/incidents/inject", json=body,
                       headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.post("/api/v1/incidents/inject", json=body,
                       headers={"X-Admin-Token": "s3cret"}).status_code == 201


def test_unwritable_log_dir_does_not_crash(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")  # a file where a directory is expected
    settings = Settings(_env_file=None, sim_log_dir=blocker / "live", sim_incident_probability=0.0)
    app = create_app(settings, autostart=False)
    app.state.sim.tick()
    assert TestClient(app).get("/api/v1/metrics/latest").status_code == 200
