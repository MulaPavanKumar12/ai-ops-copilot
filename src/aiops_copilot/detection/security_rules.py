"""Log-based security detection.

Security incidents are often invisible in metrics (privilege escalation barely
moves CPU) but leave clear traces in logs. This engine combines:

* a **stateful rate rule** for SSH brute force (many failures from one source
  IP in a sliding window), and
* **signature rules** for exfiltration, privilege escalation and malicious
  processes, written in the spirit of Sigma rules.

Each rule is tuned to *ignore* the benign look-alikes in the data: a single
mistyped password, an admin running ``sudo systemctl``, and the nightly backup
upload to the company's own S3 bucket.

Works the same in batch (evaluation) and streaming (live worker) mode.
"""

from __future__ import annotations

import ipaddress
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from aiops_copilot.generator.models import IncidentType, Severity

_FAILED_SSH = re.compile(r"Failed password for (?:invalid user )?(\S+) from (\d+\.\d+\.\d+\.\d+)")
_ACCEPTED_SSH = re.compile(r"Accepted (?:password|publickey) for (\S+) from (\d+\.\d+\.\d+\.\d+)")
_NETFLOW = re.compile(r"-> (\d+\.\d+\.\d+\.\d+):(\d+) .*bytes_out=(\d+)")


@dataclass(frozen=True)
class SignatureRule:
    """A regex rule mapped to an incident type."""

    rule_id: str
    pattern: re.Pattern[str]
    incident_type: IncidentType
    severity: Severity
    description: str


SIGNATURES: tuple[SignatureRule, ...] = (
    SignatureRule("PRIVESC-001", re.compile(r"user NOT in sudoers"), IncidentType.PRIVILEGE_ESCALATION,
                  Severity.HIGH, "Unauthorised sudo attempt"),
    SignatureRule("PRIVESC-002", re.compile(r"name=/etc/sudoers op=write"), IncidentType.PRIVILEGE_ESCALATION,
                  Severity.CRITICAL, "sudoers file modified"),
    SignatureRule("PRIVESC-003", re.compile(r"usermod -aG sudo"), IncidentType.PRIVILEGE_ESCALATION,
                  Severity.CRITICAL, "User added to sudo group"),
    SignatureRule("PRIVESC-004", re.compile(r"chmod u\+s|find / -perm -4000"), IncidentType.PRIVILEGE_ESCALATION,
                  Severity.HIGH, "SUID enumeration or SUID bit set"),
    SignatureRule("PROC-001", re.compile(r"stratum\+tcp://|:3333 proto=tcp"), IncidentType.SUSPICIOUS_PROCESS,
                  Severity.HIGH, "Crypto-mining pool connection"),
    SignatureRule("PROC-002", re.compile(r"cmd=\"/tmp/|/dev/tcp/\d"), IncidentType.SUSPICIOUS_PROCESS,
                  Severity.CRITICAL, "Execution from /tmp or reverse shell"),
    SignatureRule("EXFIL-001", re.compile(r"pg_dump .*\| *rclone"), IncidentType.DATA_EXFILTRATION,
                  Severity.CRITICAL, "Database dump piped to an external uploader"),
)


@dataclass
class SecurityFinding:
    """One rule hit, with the evidence that triggered it."""

    host: str
    timestamp: datetime
    rule_id: str
    incident_type: IncidentType
    severity: Severity
    description: str
    evidence: str
    source_ip: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "host": self.host, "timestamp": self.timestamp.isoformat(), "rule_id": self.rule_id,
            "incident_type": self.incident_type.value, "severity": self.severity.value,
            "description": self.description, "evidence": self.evidence, "source_ip": self.source_ip,
        }


_INTERNAL_NETS = tuple(ipaddress.ip_network(n) for n in
                       ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"))


def _is_external(ip: str) -> bool:
    """True for addresses outside our own (RFC 1918 / loopback) networks.

    Note: Python's ``is_private`` also treats the RFC 5737 documentation ranges
    as private, and the simulator uses those ranges for attackers, so the
    internal networks are listed explicitly instead.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not any(addr in net for net in _INTERNAL_NETS)


@dataclass
class SecurityRuleEngine:
    """Streaming rule engine. Feed log records in time order via :meth:`process`.

    Attributes:
        brute_force_threshold: Failed logins from one IP within the window.
        brute_force_window: Sliding window length.
        exfil_bytes_threshold: Outbound bytes to one external IP in one flow record.
    """

    brute_force_threshold: int = 15
    brute_force_window: timedelta = timedelta(minutes=5)
    exfil_bytes_threshold: int = 100_000_000
    _failures: dict[tuple[str, str], deque[datetime]] = field(
        default_factory=lambda: defaultdict(deque), repr=False)
    _suspicious_ips: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), repr=False)

    def process(self, record: dict[str, Any]) -> list[SecurityFinding]:
        """Evaluate one log record (dict with timestamp, host, source, message)."""
        message = str(record.get("message", ""))
        host = str(record.get("host", ""))
        ts = record["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        findings: list[SecurityFinding] = []

        failed = _FAILED_SSH.search(message)
        if failed:
            ip = failed.group(2)
            q = self._failures[(host, ip)]
            q.append(ts)
            while q and ts - q[0] > self.brute_force_window:
                q.popleft()
            if len(q) >= self.brute_force_threshold:
                self._suspicious_ips[host].add(ip)
                findings.append(SecurityFinding(
                    host, ts, "BRUTE-001", IncidentType.SSH_BRUTE_FORCE, Severity.HIGH,
                    f"{len(q)} failed SSH logins from {ip} in {int(self.brute_force_window.total_seconds()/60)} min",
                    message, ip))

        accepted = _ACCEPTED_SSH.search(message)
        if accepted and accepted.group(2) in self._suspicious_ips[host]:
            findings.append(SecurityFinding(
                host, ts, "BRUTE-002", IncidentType.SSH_BRUTE_FORCE, Severity.CRITICAL,
                f"Successful login from brute-force source {accepted.group(2)} (likely compromise)",
                message, accepted.group(2)))

        flow = _NETFLOW.search(message)
        if flow and _is_external(flow.group(1)) and int(flow.group(3)) >= self.exfil_bytes_threshold:
            findings.append(SecurityFinding(
                host, ts, "EXFIL-002", IncidentType.DATA_EXFILTRATION, Severity.CRITICAL,
                f"{int(flow.group(3)) / 1e6:.0f} MB sent to external {flow.group(1)}:{flow.group(2)}",
                message, flow.group(1)))

        for rule in SIGNATURES:
            if rule.pattern.search(message):
                ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", message)
                ip = ip_match.group(1) if ip_match and _is_external(ip_match.group(1)) else None
                findings.append(SecurityFinding(host, ts, rule.rule_id, rule.incident_type, rule.severity,
                                                rule.description, message, ip))
        return findings

    def process_many(self, records: Iterable[dict[str, Any]]) -> list[SecurityFinding]:
        """Process records in order and return all findings."""
        out: list[SecurityFinding] = []
        for record in records:
            out.extend(self.process(record))
        return out
