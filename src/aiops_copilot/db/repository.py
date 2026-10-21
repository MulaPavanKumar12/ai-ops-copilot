"""Data-access helpers so API and worker code never build queries by hand."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aiops_copilot.db.models import AlertRow, IncidentRow, RCARunRow, utcnow
from aiops_copilot.detection.grouping import DetectedIncident
from aiops_copilot.generator.incidents import CATALOG
from aiops_copilot.llm.rca import RCAResult

VALID_STATUSES = ("open", "acknowledged", "resolved")


def upsert_incident(session: Session, inc: DetectedIncident) -> tuple[IncidentRow, bool]:
    """Insert a new incident, or extend the open one on the same host.

    Alarms on one host that overlap or follow within 10 minutes are the same
    problem, even if the suspected type changes as more evidence arrives. When
    a security rule fires later, the incident is re-labelled as that attack,
    because a signature match is more specific than a metric guess.

    Returns:
        (row, created) where ``created`` is True for a brand-new incident.
    """
    existing = session.scalars(
        select(IncidentRow).where(
            IncidentRow.host == inc.host,
            IncidentRow.status != "resolved",
            IncidentRow.ended_at >= inc.start - timedelta(minutes=10),
        ).order_by(IncidentRow.started_at.desc())
    ).first()
    evidence = {"top_metrics": [{"metric": m, "z": round(z, 2)} for m, z in inc.top_metrics],
                "security_findings": [f.to_dict() for f in inc.findings[:10]]}
    if existing is not None:
        existing.ended_at = max(existing.ended_at or inc.end, inc.end)
        existing.peak_score = max(existing.peak_score, inc.peak_score)
        existing.detectors = sorted(set(existing.detectors) | set(inc.detectors))
        if inc.findings:
            findings = (existing.evidence.get("security_findings", []) + evidence["security_findings"])[-20:]
            existing.evidence = {**existing.evidence, "security_findings": findings}
            if existing.category != "security":
                existing.incident_type = inc.suspected_type.value
                existing.category = inc.category.value
                existing.severity = inc.severity.value
                existing.title = f"{CATALOG[inc.suspected_type].title} on {inc.host}"
        return existing, False
    row = IncidentRow(
        incident_key=inc.key, host=inc.host, role=inc.role, category=inc.category.value,
        incident_type=inc.suspected_type.value, severity=inc.severity.value, status="open",
        started_at=inc.start, ended_at=inc.end,
        title=f"{CATALOG[inc.suspected_type].title} on {inc.host}",
        peak_score=inc.peak_score, detectors=inc.detectors, evidence=evidence,
    )
    session.add(row)
    session.flush()
    return row, True


def save_rca(session: Session, row: IncidentRow, result: RCAResult) -> None:
    """Store an RCA result on the incident and keep a run record."""
    row.summary = result.summary
    row.root_cause = result.root_cause
    row.root_cause_id = result.root_cause_id
    row.suggested_fix = result.suggested_fix
    row.confidence = result.confidence
    row.evidence = {**(row.evidence or {}), "rca_evidence": result.evidence,
                    "similar_incidents": result.similar_incident_ids, "rca_engine": result.engine}
    session.add(RCARunRow(incident_id=row.id, engine=result.engine, latency_ms=result.latency_ms,
                          output=result.model_dump()))


def list_incidents(session: Session, status: str | None = None, host: str | None = None,
                   category: str | None = None, limit: int = 50, offset: int = 0) -> list[IncidentRow]:
    """Newest incidents first, with optional filters."""
    q = select(IncidentRow)
    if status:
        q = q.where(IncidentRow.status == status)
    if host:
        q = q.where(IncidentRow.host == host)
    if category:
        q = q.where(IncidentRow.category == category)
    return list(session.scalars(q.order_by(IncidentRow.started_at.desc()).offset(offset).limit(limit)))


def set_status(session: Session, incident_id: int, status: str) -> IncidentRow | None:
    """Change an incident's status."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}")
    row = session.get(IncidentRow, incident_id)
    if row is not None:
        row.status = status
    return row


def stats(session: Session, since: datetime | None = None) -> dict[str, Any]:
    """Counts for the dashboard header."""
    since = since or (utcnow() - timedelta(days=1))
    by_status = dict(session.execute(select(IncidentRow.status, func.count()).group_by(IncidentRow.status)).all())
    by_cat = dict(session.execute(select(IncidentRow.category, func.count())
                                  .where(IncidentRow.started_at >= since).group_by(IncidentRow.category)).all())
    by_sev = dict(session.execute(select(IncidentRow.severity, func.count())
                                  .where(IncidentRow.status != "resolved").group_by(IncidentRow.severity)).all())
    avg_rca = session.execute(select(func.avg(RCARunRow.latency_ms))).scalar()
    alerts_sent = session.execute(select(func.count()).select_from(AlertRow)
                                  .where(AlertRow.status == "sent")).scalar()
    return {"by_status": by_status, "by_category_24h": by_cat, "open_by_severity": by_sev,
            "avg_rca_latency_ms": round(float(avg_rca), 1) if avg_rca else None,
            "alerts_sent": int(alerts_sent or 0)}
