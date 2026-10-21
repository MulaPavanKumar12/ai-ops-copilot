"""Data models shared by the synthetic generator, the live simulator and later phases.

These are the "contract" between components: the detectors (Phase 2/3), the RAG
pipeline (Phase 4) and the API (Phase 5) all consume these shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ServerRole(StrEnum):
    """Workload type of a server; drives its baseline behaviour."""

    WEB = "web"
    API = "api"
    DB = "db"
    WORKER = "worker"


class IncidentCategory(StrEnum):
    """Top-level split used for routing and per-category evaluation."""

    OPERATIONAL = "operational"
    SECURITY = "security"


class IncidentType(StrEnum):
    """Every anomaly / attack pattern the generator can inject."""

    # Operational
    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    DISK_FILL = "disk_fill"
    LATENCY_DEGRADATION = "latency_degradation"
    ERROR_BURST = "error_burst"
    TRAFFIC_SURGE = "traffic_surge"
    # Security
    SSH_BRUTE_FORCE = "ssh_brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_PROCESS = "suspicious_process"


class Severity(StrEnum):
    """Alert severity, aligned with common on-call conventions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


#: Order of metric columns in every dataset and API response.
METRIC_NAMES: tuple[str, ...] = (
    "cpu_pct",
    "mem_pct",
    "disk_pct",
    "latency_p95_ms",
    "error_rate_pct",
    "requests_per_sec",
    "net_in_mbps",
    "net_out_mbps",
    "failed_logins",
    "process_count",
)

#: Metrics that are percentages and must stay within [0, 100].
PERCENT_METRICS: frozenset[str] = frozenset({"cpu_pct", "mem_pct", "disk_pct", "error_rate_pct"})


@dataclass(frozen=True)
class ServerProfile:
    """Static description of one simulated server.

    Attributes:
        name: Hostname, e.g. ``web-01``.
        role: Workload type.
        ip: Private IP address of the host.
        baseline: Mean value of each metric at average daily load.
    """

    name: str
    role: ServerRole
    ip: str
    baseline: dict[str, float]


@dataclass
class Incident:
    """Ground-truth record of one injected anomaly or attack.

    The descriptive fields (``root_cause``, ``suggested_fix``) double as the seed
    for the incident-report knowledge base used by RAG in Phase 4.
    """

    incident_id: str
    incident_type: IncidentType
    category: IncidentCategory
    host: str
    start: datetime
    end: datetime
    severity: Severity
    intensity: float
    title: str
    description: str
    root_cause: str
    suggested_fix: str
    root_cause_id: str = ""
    attacker_ip: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def is_active(self, ts: datetime) -> bool:
        """Return True if ``ts`` falls inside the incident window [start, end)."""
        return self.start <= ts < self.end

    def progress(self, ts: datetime) -> float:
        """Fraction of the incident elapsed at ``ts``, clamped to [0, 1]."""
        total = (self.end - self.start).total_seconds()
        if total <= 0:
            return 1.0
        return min(1.0, max(0.0, (ts - self.start).total_seconds() / total))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        data = asdict(self)
        data["incident_type"] = self.incident_type.value
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        data["start"] = self.start.isoformat()
        data["end"] = self.end.isoformat()
        return data


@dataclass
class MetricSample:
    """One row of metrics for one host at one timestamp, with labels.

    ``benign_event`` names a scheduled job (e.g. ``nightly_backup``) running at
    this time. It is *not* an anomaly; it lets us explain false positives.
    """

    timestamp: datetime
    host: str
    role: ServerRole
    values: dict[str, float]
    is_anomaly: int = 0
    anomaly_type: str = ""
    incident_id: str = ""
    benign_event: str = ""

    def to_row(self) -> dict[str, Any]:
        """Flat dict suitable for CSV / DataFrame construction."""
        row: dict[str, Any] = {
            "timestamp": self.timestamp.isoformat(),
            "host": self.host,
            "role": self.role.value,
        }
        for name in METRIC_NAMES:
            row[name] = round(self.values[name], 3)
        row["is_anomaly"] = self.is_anomaly
        row["anomaly_type"] = self.anomaly_type
        row["incident_id"] = self.incident_id
        row["benign_event"] = self.benign_event
        return row


@dataclass
class LogRecord:
    """One structured log line, with labels.

    ``is_anomaly`` is 1 only for lines *caused* by an incident. Normal traffic
    that happens during an incident stays labelled 0, which keeps log-level
    evaluation honest.
    """

    timestamp: datetime
    host: str
    source: str
    level: str
    message: str
    is_anomaly: int = 0
    anomaly_type: str = ""
    incident_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "host": self.host,
            "source": self.source,
            "level": self.level,
            "message": self.message,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "incident_id": self.incident_id,
        }
