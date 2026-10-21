"""SQLAlchemy 2.0 models and session factory.

Works with PostgreSQL (docker-compose) and SQLite (local development and
tests) from the same code. Tables are created on start-up with
``create_all`` (idempotent); a migration tool such as Alembic would be the next
step for schema changes in production.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    """Timezone-aware now (UTC)."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Stores UTC, always returns timezone-aware datetimes (SQLite drops tzinfo otherwise)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC) if value is not None else None

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    """Declarative base."""


class IncidentRow(Base):
    """A detected incident and its analysis."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_key: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    host: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(32))
    category: Mapped[str] = mapped_column(String(16))
    incident_type: Mapped[str] = mapped_column(String(40))
    severity: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime())
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    root_cause_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    suggested_fix: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_score: Mapped[float] = mapped_column(Float, default=0.0)
    detectors: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)

    alerts: Mapped[list[AlertRow]] = relationship(back_populates="incident", cascade="all, delete-orphan")
    rca_runs: Mapped[list[RCARunRow]] = relationship(back_populates="incident", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_incidents_host_started", "host", "started_at"),)

    def to_dict(self) -> dict[str, Any]:
        """API representation."""
        return {
            "id": self.id, "incident_key": self.incident_key, "host": self.host, "role": self.role,
            "category": self.category, "incident_type": self.incident_type, "severity": self.severity,
            "status": self.status, "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None, "title": self.title,
            "summary": self.summary, "root_cause": self.root_cause, "root_cause_id": self.root_cause_id,
            "suggested_fix": self.suggested_fix, "confidence": self.confidence,
            "peak_score": self.peak_score, "detectors": self.detectors, "evidence": self.evidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AlertRow(Base):
    """One notification attempt on one channel."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    channel: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    dedup_key: Mapped[str] = mapped_column(String(200))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    incident: Mapped[IncidentRow] = relationship(back_populates="alerts")
    __table_args__ = (UniqueConstraint("channel", "dedup_key", name="uq_alert_channel_dedup"),)

    def to_dict(self) -> dict[str, Any]:
        """API representation."""
        return {"id": self.id, "incident_id": self.incident_id, "channel": self.channel,
                "status": self.status, "error": self.error,
                "sent_at": self.sent_at.isoformat() if self.sent_at else None}


class RCARunRow(Base):
    """One root-cause analysis run (kept for latency/quality tracking)."""

    __tablename__ = "rca_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    engine: Mapped[str] = mapped_column(String(80))
    latency_ms: Mapped[int] = mapped_column(Integer)
    output: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    incident: Mapped[IncidentRow] = relationship(back_populates="rca_runs")


def make_engine(url: str) -> Engine:
    """Create an engine; enables foreign keys and WAL mode for SQLite."""
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def init_db(url: str) -> sessionmaker[Any]:
    """Create tables if needed and return a session factory."""
    if url.startswith("sqlite:///"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
