"""Definition of the simulated server fleet.

Each role has a distinct "normal" so detectors must learn per-host baselines
instead of one global threshold (a common real-world failure mode).
"""

from __future__ import annotations

from aiops_copilot.generator.models import ServerProfile, ServerRole

#: Mean metric values at average load, per role.
ROLE_BASELINES: dict[ServerRole, dict[str, float]] = {
    ServerRole.WEB: {
        "cpu_pct": 35.0, "mem_pct": 55.0, "disk_pct": 40.0, "latency_p95_ms": 120.0,
        "error_rate_pct": 0.5, "requests_per_sec": 250.0, "net_in_mbps": 20.0,
        "net_out_mbps": 45.0, "failed_logins": 0.05, "process_count": 180.0,
    },
    ServerRole.API: {
        "cpu_pct": 42.0, "mem_pct": 60.0, "disk_pct": 35.0, "latency_p95_ms": 180.0,
        "error_rate_pct": 0.8, "requests_per_sec": 180.0, "net_in_mbps": 15.0,
        "net_out_mbps": 25.0, "failed_logins": 0.05, "process_count": 210.0,
    },
    ServerRole.DB: {
        "cpu_pct": 30.0, "mem_pct": 72.0, "disk_pct": 62.0, "latency_p95_ms": 15.0,
        "error_rate_pct": 0.1, "requests_per_sec": 400.0, "net_in_mbps": 30.0,
        "net_out_mbps": 35.0, "failed_logins": 0.02, "process_count": 140.0,
    },
    ServerRole.WORKER: {
        "cpu_pct": 50.0, "mem_pct": 48.0, "disk_pct": 30.0, "latency_p95_ms": 900.0,
        "error_rate_pct": 1.0, "requests_per_sec": 20.0, "net_in_mbps": 8.0,
        "net_out_mbps": 10.0, "failed_logins": 0.02, "process_count": 160.0,
    },
}

_DEFAULT_LAYOUT: tuple[tuple[str, ServerRole], ...] = (
    ("web-01", ServerRole.WEB),
    ("web-02", ServerRole.WEB),
    ("api-01", ServerRole.API),
    ("api-02", ServerRole.API),
    ("db-01", ServerRole.DB),
    ("worker-01", ServerRole.WORKER),
)


def default_fleet() -> list[ServerProfile]:
    """Return the standard six-server fleet used across the project."""
    fleet: list[ServerProfile] = []
    for index, (name, role) in enumerate(_DEFAULT_LAYOUT, start=10):
        fleet.append(
            ServerProfile(
                name=name,
                role=role,
                ip=f"10.0.1.{index}",
                baseline=dict(ROLE_BASELINES[role]),
            )
        )
    return fleet


def build_fleet(size: int) -> list[ServerProfile]:
    """Build a fleet of ``size`` servers by cycling through the default roles.

    Args:
        size: Number of servers, 1..60.

    Raises:
        ValueError: If ``size`` is out of range.
    """
    if not 1 <= size <= 60:
        raise ValueError(f"fleet size must be between 1 and 60, got {size}")
    if size == len(_DEFAULT_LAYOUT):
        return default_fleet()
    counters: dict[ServerRole, int] = {}
    fleet: list[ServerProfile] = []
    for i in range(size):
        role = _DEFAULT_LAYOUT[i % len(_DEFAULT_LAYOUT)][1]
        counters[role] = counters.get(role, 0) + 1
        fleet.append(
            ServerProfile(
                name=f"{role.value}-{counters[role]:02d}",
                role=role,
                ip=f"10.0.1.{10 + i}",
                baseline=dict(ROLE_BASELINES[role]),
            )
        )
    return fleet
