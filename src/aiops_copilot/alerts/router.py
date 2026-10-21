"""Alert routing with severity rules and deduplication.

Routing (who gets woken up):

==========  ===========================
severity    channels
==========  ===========================
critical    Slack + email + dashboard
high        Slack + email + dashboard
medium      Slack + dashboard
low         dashboard only
==========  ===========================

Deduplication: one alert per (channel, host, incident type) per cooldown
window, enforced by a UNIQUE constraint in the database, so it also holds when
several workers run in parallel. Unconfigured channels are skipped, never
fatal; a failed send is recorded with its error for the dashboard.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from datetime import timedelta
from email.message import EmailMessage
from typing import Protocol

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aiops_copilot.core.config import Settings
from aiops_copilot.db.models import AlertRow, IncidentRow, utcnow

logger = logging.getLogger(__name__)

ROUTES: dict[str, tuple[str, ...]] = {
    "critical": ("slack", "email", "dashboard"),
    "high": ("slack", "email", "dashboard"),
    "medium": ("slack", "dashboard"),
    "low": ("dashboard",),
}
_EMOJI = {"critical": ":rotating_light:", "high": ":red_circle:", "medium": ":large_orange_circle:",
          "low": ":large_blue_circle:"}


@dataclass
class AlertMessage:
    """Channel-independent alert content."""

    title: str
    severity: str
    host: str
    category: str
    summary: str
    suggested_fix: str
    incident_id: int
    dashboard_url: str

    @classmethod
    def from_incident(cls, row: IncidentRow, dashboard_url: str) -> AlertMessage:
        """Build from a stored incident."""
        return cls(title=row.title, severity=row.severity, host=row.host, category=row.category,
                   summary=row.summary or "Root-cause analysis pending.",
                   suggested_fix=row.suggested_fix or "See runbook.", incident_id=row.id,
                   dashboard_url=dashboard_url)

    def text(self) -> str:
        """Plain-text body."""
        return (f"[{self.severity.upper()}] {self.title}\nHost: {self.host} ({self.category})\n\n"
                f"What happened: {self.summary}\n\nSuggested fix: {self.suggested_fix}\n\n"
                f"Incident #{self.incident_id}: {self.dashboard_url}")


class Channel(Protocol):
    """A notification channel."""

    name: str

    def configured(self) -> bool: ...
    def send(self, msg: AlertMessage) -> None: ...


class SlackChannel:
    """Slack incoming webhook."""

    name = "slack"

    def __init__(self, webhook_url: str, timeout: float = 10.0) -> None:
        self.webhook_url = webhook_url
        self.timeout = timeout

    def configured(self) -> bool:
        """Only real Slack webhook URLs are accepted (prevents posting data elsewhere by mistake)."""
        return self.webhook_url.startswith("https://hooks.slack.com/")

    def send(self, msg: AlertMessage) -> None:
        """POST a Block Kit message."""
        payload = {
            "text": f"{msg.severity.upper()}: {msg.title}",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"{msg.title}"[:150]}},
                {"type": "section", "text": {"type": "mrkdwn", "text":
                    f"{_EMOJI.get(msg.severity, '')} *{msg.severity.upper()}* on `{msg.host}` ({msg.category})"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*What happened*\n{msg.summary}"[:2900]}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Suggested fix*\n{msg.suggested_fix}"[:2900]}},
                {"type": "context", "elements": [{"type": "mrkdwn",
                                                  "text": f"<{msg.dashboard_url}|Open incident #{msg.incident_id}>"}]},
            ],
        }
        resp = httpx.post(self.webhook_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()


class EmailChannel:
    """SMTP with STARTTLS."""

    name = "email"

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def configured(self) -> bool:
        """Needs a host and at least one recipient."""
        return bool(self.s.smtp_host and self.s.alert_email_to)

    def send(self, msg: AlertMessage) -> None:
        """Send a plain-text email."""
        email = EmailMessage()
        email["Subject"] = f"[{msg.severity.upper()}] {msg.title}"
        email["From"] = self.s.alert_email_from
        email["To"] = self.s.alert_email_to
        email.set_content(msg.text())
        with smtplib.SMTP(self.s.smtp_host, self.s.smtp_port, timeout=15) as smtp:
            smtp.starttls()
            if self.s.smtp_user and self.s.smtp_password:
                smtp.login(self.s.smtp_user, self.s.smtp_password.get_secret_value())
            smtp.send_message(email)


class DashboardChannel:
    """Always-on channel: the incident is visible in the dashboard."""

    name = "dashboard"

    def configured(self) -> bool:
        """Always available."""
        return True

    def send(self, msg: AlertMessage) -> None:
        """Nothing to send; the database row is the notification."""


class AlertRouter:
    """Routes incidents to channels with dedup and error capture."""

    def __init__(self, settings: Settings, channels: list[Channel] | None = None,
                 dashboard_base_url: str = "http://localhost:8501") -> None:
        self.settings = settings
        self.cooldown = timedelta(minutes=settings.alert_cooldown_minutes)
        self.dashboard_base_url = dashboard_base_url.rstrip("/")
        if channels is None:
            webhook = settings.slack_webhook_url.get_secret_value() if settings.slack_webhook_url else ""
            channels = [SlackChannel(webhook), EmailChannel(settings), DashboardChannel()]
        self.channels = {c.name: c for c in channels}

    def dedup_key(self, row: IncidentRow) -> str:
        """Same host + type within one cooldown bucket = same alert."""
        seconds = max(int(self.cooldown.total_seconds()), 1)
        bucket = int(row.started_at.timestamp()) // seconds
        return f"{row.host}:{row.incident_type}:{bucket}"

    def route(self, session: Session, row: IncidentRow) -> list[AlertRow]:
        """Send the alert on every channel its severity calls for. Returns created alert rows."""
        msg = AlertMessage.from_incident(row, f"{self.dashboard_base_url}/?incident={row.id}")
        created: list[AlertRow] = []
        for name in ROUTES.get(row.severity, ("dashboard",)):
            channel = self.channels.get(name)
            if channel is None or not channel.configured():
                continue
            alert = AlertRow(incident_id=row.id, channel=name, dedup_key=self.dedup_key(row))
            try:
                with session.begin_nested():
                    session.add(alert)
                    session.flush()
            except IntegrityError:
                logger.debug("duplicate alert suppressed: %s %s", name, alert.dedup_key)
                continue
            try:
                channel.send(msg)
                alert.status, alert.sent_at = "sent", utcnow()
            except Exception as exc:  # noqa: BLE001 - one channel failing must not stop the others
                alert.status, alert.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
                logger.warning("alert via %s failed: %s", name, exc)
            created.append(alert)
        return created
