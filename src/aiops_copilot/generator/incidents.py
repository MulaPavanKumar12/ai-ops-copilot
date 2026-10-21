"""Incident catalogue and schedulers.

The catalogue describes *what* each anomaly/attack looks like and *why* it
happened (root cause + fix). Several root-cause variants exist per type so the
RAG system in Phase 4 has to retrieve the *right* past incident, not just the
right category.

Two schedulers decide *when* incidents happen:

* :class:`BatchScheduler` plans every incident up front for an offline dataset.
* :class:`LiveScheduler` randomly spawns incidents while the live simulator runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from aiops_copilot.generator.models import (
    Incident,
    IncidentCategory,
    IncidentType,
    ServerProfile,
    ServerRole,
    Severity,
)

logger = logging.getLogger(__name__)

_ALL_ROLES = frozenset(ServerRole)
_APP_ROLES = frozenset({ServerRole.WEB, ServerRole.API, ServerRole.WORKER})
_MIN_GAP = timedelta(minutes=30)


@dataclass(frozen=True)
class RootCause:
    """One plausible explanation for an incident, plus its remediation.

    ``key`` is a stable identifier (e.g. ``cpu_spike/regex``) used as the
    ground-truth label when scoring root-cause analysis.
    """

    key: str
    cause: str
    fix: str


@dataclass(frozen=True)
class IncidentSpec:
    """Static description of an incident type."""

    incident_type: IncidentType
    category: IncidentCategory
    severity: Severity
    duration_min: tuple[int, int]
    roles: frozenset[ServerRole]
    title: str
    description: str
    root_causes: tuple[RootCause, ...]
    night_bias: float = 0.0  # probability the incident is forced into 00:00-05:00


CATALOG: dict[IncidentType, IncidentSpec] = {
    IncidentType.CPU_SPIKE: IncidentSpec(
        IncidentType.CPU_SPIKE, IncidentCategory.OPERATIONAL, Severity.HIGH, (10, 40), _ALL_ROLES,
        "CPU saturation",
        "CPU utilisation jumped above 85% and stayed there, increasing request latency.",
        (
            RootCause("cpu_spike/regex_backtracking", "A catastrophic-backtracking regex in the request validator was hit by "
                      "long user input, pinning worker threads.",
                      "Roll back the validator change, bound input length, and replace the "
                      "regex with a linear-time pattern."),
            RootCause("cpu_spike/retry_storm", "A config push set the retry policy to unlimited retries with no "
                      "backoff, creating a hot loop against a failing dependency.",
                      "Revert the config, enforce exponential backoff with a retry cap, and "
                      "add a circuit breaker."),
            RootCause("cpu_spike/cron_misfire", "A cron entry was mistyped to run a heavy report job every minute "
                      "instead of daily.",
                      "Fix the cron expression, kill the duplicate jobs, and add a lock file "
                      "so the job cannot overlap."),
        ),
    ),
    IncidentType.MEMORY_LEAK: IncidentSpec(
        IncidentType.MEMORY_LEAK, IncidentCategory.OPERATIONAL, Severity.HIGH, (60, 180), _APP_ROLES,
        "Memory leak",
        "Memory grew steadily without being released until the OOM killer terminated the service.",
        (
            RootCause("memory_leak/unbounded_cache", "An in-process cache had no eviction policy after a library upgrade, so "
                      "entries accumulated forever.",
                      "Restart the service to recover, set a max size / TTL on the cache, and "
                      "add a memory-growth alert."),
            RootCause("memory_leak/unclosed_sessions", "HTTP client sessions were created per request and never closed, leaking "
                      "sockets and buffers.",
                      "Reuse a single pooled client, close sessions in a finally block, and "
                      "add a leak test to CI."),
        ),
    ),
    IncidentType.DISK_FILL: IncidentSpec(
        IncidentType.DISK_FILL, IncidentCategory.OPERATIONAL, Severity.CRITICAL, (45, 150), _ALL_ROLES,
        "Disk filling up",
        "Disk usage climbed rapidly toward 100%, causing write failures.",
        (
            RootCause("disk_fill/debug_logging", "Debug logging was left enabled after an investigation and log rotation "
                      "was not configured for the new log file.",
                      "Set log level back to INFO, add the file to logrotate, and purge old "
                      "logs to free space."),
            RootCause("disk_fill/wal_replication_slot", "Database WAL files accumulated because a replication slot was abandoned "
                      "and blocked WAL recycling.",
                      "Drop the stale replication slot, monitor slot lag, and alert when WAL "
                      "size exceeds a threshold."),
        ),
    ),
    IncidentType.LATENCY_DEGRADATION: IncidentSpec(
        IncidentType.LATENCY_DEGRADATION, IncidentCategory.OPERATIONAL, Severity.MEDIUM, (20, 90),
        _ALL_ROLES,
        "Latency degradation",
        "p95 latency rose several-fold while traffic stayed normal.",
        (
            RootCause("latency_degradation/missing_index", "A new query missed an index after a schema migration, causing full table "
                      "scans on a hot path.",
                      "Add the missing composite index, verify with EXPLAIN ANALYZE, and add "
                      "query-plan checks to migrations."),
            RootCause("latency_degradation/upstream_timeout", "The upstream payment provider was degraded and calls had a 30s timeout, "
                      "so worker threads blocked waiting.",
                      "Lower the upstream timeout, add a circuit breaker and fallback, and "
                      "subscribe to the provider's status page."),
        ),
    ),
    IncidentType.ERROR_BURST: IncidentSpec(
        IncidentType.ERROR_BURST, IncidentCategory.OPERATIONAL, Severity.HIGH, (5, 30), _APP_ROLES,
        "Error-rate burst",
        "HTTP 5xx error rate spiked well above its normal level.",
        (
            RootCause("error_burst/pool_exhausted", "The database connection pool was exhausted after a deploy halved the "
                      "pool size setting.",
                      "Restore the pool size, alert on pool saturation, and load-test "
                      "configuration changes before deploy."),
            RootCause("error_burst/expired_tls_cert", "An expired TLS certificate on an internal service broke service-to-"
                      "service calls.",
                      "Renew the certificate, automate renewal (e.g. cert-manager / certbot), "
                      "and alert 14 days before expiry."),
        ),
    ),
    IncidentType.TRAFFIC_SURGE: IncidentSpec(
        IncidentType.TRAFFIC_SURGE, IncidentCategory.OPERATIONAL, Severity.MEDIUM, (15, 60),
        frozenset({ServerRole.WEB, ServerRole.API}),
        "Traffic surge",
        "Requests per second jumped 3-5x, raising CPU, network and latency.",
        (
            RootCause("traffic_surge/marketing_campaign", "A marketing campaign went live without notifying the platform team, so "
                      "capacity was not scaled up in advance.",
                      "Scale out the web tier, enable autoscaling on RPS, and add launches to "
                      "the capacity-planning calendar."),
            RootCause("traffic_surge/aggressive_crawler", "An aggressive web crawler ignored robots.txt and hammered search "
                      "endpoints.",
                      "Rate-limit by client IP / user agent at the load balancer and serve "
                      "crawlers a cached response."),
        ),
    ),
    IncidentType.SSH_BRUTE_FORCE: IncidentSpec(
        IncidentType.SSH_BRUTE_FORCE, IncidentCategory.SECURITY, Severity.HIGH, (10, 45), _ALL_ROLES,
        "SSH brute-force attack",
        "Hundreds of failed SSH logins from a single external IP within minutes.",
        (
            RootCause("ssh_brute_force/exposed_ssh_password_auth", "SSH (port 22) was exposed to the internet with password authentication "
                      "enabled, and an automated botnet was guessing credentials.",
                      "Block the source IP, disable password auth (keys only), restrict SSH to "
                      "a VPN/bastion, and enable fail2ban."),
        ),
    ),
    IncidentType.DATA_EXFILTRATION: IncidentSpec(
        IncidentType.DATA_EXFILTRATION, IncidentCategory.SECURITY, Severity.CRITICAL, (15, 60),
        frozenset({ServerRole.DB, ServerRole.API, ServerRole.WORKER}),
        "Unusual outbound data transfer",
        "Outbound traffic to an unknown external IP spiked far above baseline, outside business hours.",
        (
            RootCause("data_exfiltration/compromised_service_account", "A compromised service account was used to dump the database and upload "
                      "it to an attacker-controlled storage endpoint.",
                      "Isolate the host, revoke the service-account credentials, block the "
                      "destination at the egress firewall, and start forensic review."),
        ),
        night_bias=0.7,
    ),
    IncidentType.PRIVILEGE_ESCALATION: IncidentSpec(
        IncidentType.PRIVILEGE_ESCALATION, IncidentCategory.SECURITY, Severity.CRITICAL, (5, 20),
        _ALL_ROLES,
        "Privilege escalation attempt",
        "A low-privilege account attempted sudo, modified sudoers and set SUID bits.",
        (
            RootCause("privilege_escalation/web_shell", "A web-shell uploaded via an unpatched file-upload endpoint gave the "
                      "attacker a shell as www-data, who then tried to gain root.",
                      "Isolate the host, remove the web shell, patch the upload endpoint, "
                      "audit sudoers and SUID binaries, and rotate secrets."),
        ),
    ),
    IncidentType.SUSPICIOUS_PROCESS: IncidentSpec(
        IncidentType.SUSPICIOUS_PROCESS, IncidentCategory.SECURITY, Severity.HIGH, (20, 90),
        _ALL_ROLES,
        "Suspicious process (crypto-miner)",
        "An unknown binary in /tmp consumed most of the CPU and connected to a mining pool.",
        (
            RootCause("suspicious_process/exposed_container_api", "An exposed container runtime API allowed an attacker to start a "
                      "crypto-miner from /tmp.",
                      "Kill the process, remove the binary and persistence (cron/systemd), "
                      "close the exposed API, and mount /tmp noexec."),
        ),
    ),
}

_ATTACKER_PREFIXES: tuple[str, ...] = ("203.0.113", "198.51.100", "192.0.2")  # RFC 5737 ranges


def random_attacker_ip(rng: np.random.Generator) -> str:
    """Return an IP from a documentation-only range (never a real host)."""
    prefix = _ATTACKER_PREFIXES[int(rng.integers(len(_ATTACKER_PREFIXES)))]
    return f"{prefix}.{int(rng.integers(2, 254))}"


def make_incident(
    spec: IncidentSpec,
    host: ServerProfile,
    start: datetime,
    rng: np.random.Generator,
    seq: int,
) -> Incident:
    """Instantiate an :class:`Incident` from a spec, sampling duration and root cause."""
    duration = timedelta(minutes=int(rng.integers(spec.duration_min[0], spec.duration_min[1] + 1)))
    root = spec.root_causes[int(rng.integers(len(spec.root_causes)))]
    attacker_ip = random_attacker_ip(rng) if spec.category is IncidentCategory.SECURITY else None
    extra: dict[str, object] = {}
    if spec.incident_type is IncidentType.SSH_BRUTE_FORCE:
        extra["compromised"] = bool(rng.random() < 0.4)
    return Incident(
        incident_id=f"INC-{start:%Y%m%d}-{seq:04d}",
        incident_type=spec.incident_type,
        category=spec.category,
        host=host.name,
        start=start,
        end=start + duration,
        severity=spec.severity,
        intensity=float(rng.uniform(0.6, 1.0)),
        title=f"{spec.title} on {host.name}",
        description=spec.description,
        root_cause=root.cause,
        root_cause_id=root.key,
        suggested_fix=root.fix,
        attacker_ip=attacker_ip,
        extra=extra,
    )


def _overlaps(candidate: Incident, existing: list[Incident]) -> bool:
    """True if ``candidate`` overlaps (with a safety gap) any incident on the same host."""
    for other in existing:
        if other.host != candidate.host:
            continue
        if candidate.start < other.end + _MIN_GAP and other.start < candidate.end + _MIN_GAP:
            return True
    return False


class IncidentTimeline:
    """Fast lookup of the incident active on a host at a given time."""

    def __init__(self, incidents: list[Incident] | None = None) -> None:
        self._incidents: list[Incident] = sorted(incidents or [], key=lambda i: i.start)

    @property
    def incidents(self) -> list[Incident]:
        """All incidents, sorted by start time."""
        return list(self._incidents)

    def add(self, incident: Incident) -> None:
        """Add an incident and keep the list sorted."""
        self._incidents.append(incident)
        self._incidents.sort(key=lambda i: i.start)

    def active(self, host: str, ts: datetime) -> Incident | None:
        """Return the incident active on ``host`` at ``ts``, if any."""
        for incident in self._incidents:
            if incident.start > ts:
                break
            if incident.host == host and incident.is_active(ts):
                return incident
        return None

    def prune(self, before: datetime) -> None:
        """Drop incidents that ended before ``before`` (keeps live mode bounded)."""
        self._incidents = [i for i in self._incidents if i.end >= before]


class BatchScheduler:
    """Plans all incidents for an offline dataset in advance.

    Guarantees every incident type appears at least once when the requested
    count allows it, which makes per-type evaluation possible.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def plan(
        self,
        fleet: list[ServerProfile],
        start: datetime,
        end: datetime,
        n_incidents: int,
    ) -> list[Incident]:
        """Return a non-overlapping (per host) list of incidents within [start, end)."""
        if n_incidents <= 0:
            return []
        window = end - start
        warmup = min(timedelta(hours=2), window * 0.1)  # keep the very start clean
        types = list(CATALOG)
        plan_types: list[IncidentType] = []
        while len(plan_types) < n_incidents:
            batch = list(types)
            self._rng.shuffle(batch)
            plan_types.extend(batch)
        plan_types = plan_types[:n_incidents]

        incidents: list[Incident] = []
        for seq, incident_type in enumerate(plan_types, start=1):
            spec = CATALOG[incident_type]
            eligible = [s for s in fleet if s.role in spec.roles]
            if not eligible:
                logger.warning("No eligible host for %s; skipping", incident_type.value)
                continue
            placed = False
            for _ in range(200):
                host = eligible[int(self._rng.integers(len(eligible)))]
                latest = end - timedelta(minutes=spec.duration_min[1]) - (start + warmup)
                if latest.total_seconds() <= 0:
                    break
                offset = timedelta(seconds=float(self._rng.uniform(0, latest.total_seconds())))
                ts = start + warmup + offset
                if spec.night_bias and self._rng.random() < spec.night_bias:
                    night = ts.replace(hour=int(self._rng.integers(0, 5)),
                                       minute=int(self._rng.integers(0, 60)))
                    if start + warmup <= night <= start + warmup + latest:
                        ts = night
                ts = ts.replace(second=0, microsecond=0)
                candidate = make_incident(spec, host, ts, self._rng, seq)
                if not _overlaps(candidate, incidents):
                    incidents.append(candidate)
                    placed = True
                    break
            if not placed:
                logger.warning("Could not place %s without overlap; skipping", incident_type.value)
        incidents.sort(key=lambda i: i.start)
        # Renumber chronologically so IDs read naturally in reports.
        for seq, incident in enumerate(incidents, start=1):
            incident.incident_id = f"INC-{incident.start:%Y%m%d}-{seq:04d}"
        return incidents


class LiveScheduler:
    """Randomly spawns incidents while the live simulator is running."""

    def __init__(self, rng: np.random.Generator, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be within [0, 1]")
        self._rng = rng
        self._probability = probability
        self._seq = 0

    def maybe_spawn(
        self,
        fleet: list[ServerProfile],
        timeline: IncidentTimeline,
        ts: datetime,
        forced_type: IncidentType | None = None,
        forced_host: str | None = None,
    ) -> Incident | None:
        """With the configured probability (or when forced), start a new incident at ``ts``."""
        if forced_type is None and self._rng.random() >= self._probability:
            return None
        incident_type = forced_type or list(CATALOG)[int(self._rng.integers(len(CATALOG)))]
        spec = CATALOG[incident_type]
        free = [
            s for s in fleet
            if s.role in spec.roles
            and timeline.active(s.name, ts) is None
            and (forced_host is None or s.name == forced_host)
        ]
        if not free:
            return None
        host = free[int(self._rng.integers(len(free)))]
        self._seq += 1
        incident = make_incident(spec, host, ts, self._rng, self._seq)
        timeline.add(incident)
        logger.info("Injected %s on %s (%s)", incident.incident_type.value, host.name,
                    incident.incident_id)
        return incident
