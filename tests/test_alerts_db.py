"""Incident storage and alert routing (SQLite in a temp folder)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from aiops_copilot.alerts.router import AlertMessage, AlertRouter, SlackChannel
from aiops_copilot.core.config import Settings
from aiops_copilot.db import repository as repo
from aiops_copilot.db.models import IncidentRow, init_db
from aiops_copilot.detection.grouping import DetectedIncident
from aiops_copilot.generator.models import IncidentType, Severity

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def make_incident(host: str = "web-01", kind: IncidentType = IncidentType.CPU_SPIKE,
                  severity: Severity = Severity.HIGH, start: datetime = T0, minutes: int = 5) -> DetectedIncident:
    return DetectedIncident(host=host, role="web", start=start, end=start + timedelta(minutes=minutes),
                            peak_score=2.5, detectors=["isolation_forest"], suspected_type=kind,
                            severity=severity, top_metrics=[("cpu_pct", 6.1)])


@pytest.fixture()
def sessions(tmp_path: Path) -> sessionmaker[Session]:
    return init_db(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")


class RecordingChannel:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name, self.fail, self.sent = name, fail, []  # type: ignore[var-annotated]

    def configured(self) -> bool:
        return True

    def send(self, msg: AlertMessage) -> None:
        if self.fail:
            raise ConnectionError("smtp down")
        self.sent.append(msg)


def _settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, alert_cooldown_minutes=30, **kw)  # type: ignore[call-arg]


# ------------------------------------------------------------------ repository
def test_upsert_merges_alarms_on_same_host(sessions: sessionmaker[Session]) -> None:
    with sessions() as s, s.begin():
        a, created_a = repo.upsert_incident(s, make_incident())
        b, created_b = repo.upsert_incident(s, make_incident(start=T0 + timedelta(minutes=8)))
        c, created_c = repo.upsert_incident(s, make_incident(host="api-01"))
    assert created_a and not created_b and created_c
    assert a.id == b.id != c.id
    assert b.ended_at == T0 + timedelta(minutes=13)


def test_security_finding_relabels_incident(sessions: sessionmaker[Session]) -> None:
    from aiops_copilot.detection.security_rules import SecurityFinding

    with sessions() as s, s.begin():
        row, _ = repo.upsert_incident(s, make_incident())
        attack = make_incident(kind=IncidentType.SSH_BRUTE_FORCE, start=T0 + timedelta(minutes=2))
        attack.findings = [SecurityFinding("web-01", T0, "BRUTE-001", IncidentType.SSH_BRUTE_FORCE,
                                           Severity.HIGH, "20 failed logins", "Failed password ...", "203.0.113.7")]
        same, created = repo.upsert_incident(s, attack)
    assert not created and same.id == row.id
    assert same.category == "security" and same.incident_type == "ssh_brute_force"


def test_status_and_stats(sessions: sessionmaker[Session]) -> None:
    with sessions() as s, s.begin():
        row, _ = repo.upsert_incident(s, make_incident())
        repo.set_status(s, row.id, "resolved")
        with pytest.raises(ValueError):
            repo.set_status(s, row.id, "deleted")
        stats = repo.stats(s, since=T0 - timedelta(days=1))
    assert stats["by_status"] == {"resolved": 1}


# ------------------------------------------------------------------ alerts
def test_routing_by_severity_and_dedup(sessions: sessionmaker[Session]) -> None:
    channels = [RecordingChannel("slack"), RecordingChannel("email"), RecordingChannel("dashboard")]
    router = AlertRouter(_settings(), channels=channels)  # type: ignore[arg-type]
    with sessions() as s, s.begin():
        high, _ = repo.upsert_incident(s, make_incident())
        low, _ = repo.upsert_incident(s, make_incident(host="db-01", severity=Severity.LOW))
        first = router.route(s, high)
        again = router.route(s, high)          # same incident, same cooldown bucket -> suppressed
        low_alerts = router.route(s, low)
    assert sorted(a.channel for a in first) == ["dashboard", "email", "slack"]
    assert again == []
    assert [a.channel for a in low_alerts] == ["dashboard"]
    assert len(channels[0].sent) == 1


def test_failing_channel_is_recorded_and_others_still_send(sessions: sessionmaker[Session]) -> None:
    channels = [RecordingChannel("slack"), RecordingChannel("email", fail=True), RecordingChannel("dashboard")]
    router = AlertRouter(_settings(), channels=channels)  # type: ignore[arg-type]
    with sessions() as s, s.begin():
        row, _ = repo.upsert_incident(s, make_incident(severity=Severity.CRITICAL))
        alerts = {a.channel: a for a in router.route(s, row)}
    assert alerts["email"].status == "failed" and "smtp down" in (alerts["email"].error or "")
    assert alerts["slack"].status == "sent" and alerts["dashboard"].status == "sent"


def test_slack_only_accepts_real_webhooks() -> None:
    assert SlackChannel("https://hooks.slack.com/services/T/B/X").configured()
    assert not SlackChannel("http://hooks.slack.com/services/T/B/X").configured()
    assert not SlackChannel("https://evil.example.com/hooks.slack.com/").configured()
    assert not SlackChannel("").configured()


def test_alert_text_contains_fix_and_link(sessions: sessionmaker[Session]) -> None:
    with sessions() as s, s.begin():
        row, _ = repo.upsert_incident(s, make_incident())
        row.suggested_fix = "Roll back the deploy."
        msg = AlertMessage.from_incident(row, "http://localhost:8501/?incident=1")
    assert "Roll back the deploy." in msg.text() and "incident=1" in msg.text()
    assert isinstance(row, IncidentRow)
