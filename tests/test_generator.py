"""Tests for the synthetic data generator: realism, labels and reproducibility."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiops_copilot.generator.baseline import BaselineModel, load_factor
from aiops_copilot.generator.dataset import (
    METRIC_COLUMNS,
    DatasetConfig,
    DatasetResult,
    generate_dataset,
    resolve_run_dir,
)
from aiops_copilot.generator.fleet import build_fleet, default_fleet
from aiops_copilot.generator.incidents import CATALOG, BatchScheduler, IncidentTimeline
from aiops_copilot.generator.injectors import INJECTORS, clip_values
from aiops_copilot.generator.models import METRIC_NAMES, IncidentCategory, IncidentType

UTC = UTC


# ----------------------------------------------------------------- fleet
def test_default_fleet_has_unique_hosts_and_all_roles() -> None:
    fleet = default_fleet()
    assert len({s.name for s in fleet}) == len(fleet) == 6
    assert {s.role.value for s in fleet} == {"web", "api", "db", "worker"}


@pytest.mark.parametrize("size", [0, 61])
def test_build_fleet_rejects_bad_sizes(size: int) -> None:
    with pytest.raises(ValueError):
        build_fleet(size)


# -------------------------------------------------------------- baseline
def test_load_factor_peaks_in_afternoon() -> None:
    day = datetime(2026, 1, 6, tzinfo=UTC)  # a Tuesday
    assert load_factor(day.replace(hour=14)) > load_factor(day.replace(hour=2))


def test_baseline_values_stay_physical() -> None:
    server = default_fleet()[0]
    model = BaselineModel(server, np.random.default_rng(0))
    ts = datetime(2026, 1, 5, tzinfo=UTC)
    for step in range(2000):
        values, _ = model.sample(ts + timedelta(minutes=step), 60)
        values = clip_values(values)
        assert set(values) == set(METRIC_NAMES)
        assert 0 <= values["cpu_pct"] <= 100
        assert values["requests_per_sec"] >= 0


# ------------------------------------------------------------- incidents
def test_every_incident_type_has_spec_and_injector() -> None:
    assert set(CATALOG) == set(IncidentType) == set(INJECTORS)


def test_batch_scheduler_no_overlap_per_host() -> None:
    fleet = default_fleet()
    start = datetime(2026, 1, 5, tzinfo=UTC)
    incidents = BatchScheduler(np.random.default_rng(1)).plan(
        fleet, start, start + timedelta(days=3), 40
    )
    assert len(incidents) >= 35
    by_host: dict[str, list] = {}
    for inc in incidents:
        by_host.setdefault(inc.host, []).append(inc)
    for items in by_host.values():
        items.sort(key=lambda i: i.start)
        for a, b in zip(items, items[1:], strict=False):
            assert a.end <= b.start, f"{a.incident_id} overlaps {b.incident_id}"


def test_incidents_respect_role_constraints() -> None:
    fleet = {s.name: s for s in default_fleet()}
    start = datetime(2026, 1, 5, tzinfo=UTC)
    incidents = BatchScheduler(np.random.default_rng(3)).plan(
        list(fleet.values()), start, start + timedelta(days=2), 30
    )
    for inc in incidents:
        assert fleet[inc.host].role in CATALOG[inc.incident_type].roles


def test_timeline_lookup() -> None:
    fleet = default_fleet()
    start = datetime(2026, 1, 5, tzinfo=UTC)
    incidents = BatchScheduler(np.random.default_rng(5)).plan(fleet, start, start + timedelta(days=1), 5)
    timeline = IncidentTimeline(incidents)
    inc = incidents[0]
    assert timeline.active(inc.host, inc.start) is inc
    assert timeline.active(inc.host, inc.end) is not inc


# --------------------------------------------------------------- dataset
def test_dataset_files_and_columns(small_dataset: DatasetResult) -> None:
    for path in (small_dataset.metrics_path, small_dataset.logs_path,
                 small_dataset.incidents_path, small_dataset.manifest_path):
        assert path.exists() and path.stat().st_size > 0
    with small_dataset.metrics_path.open() as fh:
        assert next(csv.reader(fh)) == METRIC_COLUMNS


def test_all_incident_types_present(small_dataset: DatasetResult) -> None:
    incidents = json.loads(small_dataset.incidents_path.read_text())
    assert {i["incident_type"] for i in incidents} == {t.value for t in IncidentType}
    assert {i["category"] for i in incidents} == {c.value for c in IncidentCategory}
    for inc in incidents:
        assert inc["root_cause"] and inc["suggested_fix"]


def test_metric_labels_match_incident_windows(small_dataset: DatasetResult) -> None:
    df = pd.read_csv(small_dataset.metrics_path, parse_dates=["timestamp"])
    incidents = json.loads(small_dataset.incidents_path.read_text())
    expected = pd.Series(0, index=df.index)
    for inc in incidents:
        mask = (df.host == inc["host"]) & (df.timestamp >= pd.Timestamp(inc["start"])) & \
               (df.timestamp < pd.Timestamp(inc["end"]))
        expected[mask] = 1
        assert (df.loc[mask, "incident_id"] == inc["incident_id"]).all()
    assert (expected == df.is_anomaly).all()
    assert 0.02 < df.is_anomaly.mean() < 0.25  # realistic class imbalance


def test_percent_metrics_are_bounded(small_dataset: DatasetResult) -> None:
    df = pd.read_csv(small_dataset.metrics_path)
    for col in ("cpu_pct", "mem_pct", "disk_pct", "error_rate_pct"):
        assert df[col].between(0, 100).all(), col
    assert (df[list(METRIC_NAMES)] >= 0).all().all()


def test_anomalies_are_visible_in_metrics(small_dataset: DatasetResult) -> None:
    """Injected incidents must shift their signature metric versus normal rows."""
    df = pd.read_csv(small_dataset.metrics_path)
    normal = df[df.is_anomaly == 0]
    checks = {
        "cpu_spike": "cpu_pct",
        "suspicious_process": "cpu_pct",
        "ssh_brute_force": "failed_logins",
        "data_exfiltration": "net_out_mbps",
        "error_burst": "error_rate_pct",
        "traffic_surge": "requests_per_sec",
    }
    for kind, metric in checks.items():
        rows = df[df.anomaly_type == kind]
        hosts = rows.host.unique()
        base = normal[normal.host.isin(hosts)][metric].mean()
        assert rows[metric].mean() > 1.5 * base, f"{kind} not visible in {metric}"


def test_log_labels_and_attack_signatures(small_dataset: DatasetResult) -> None:
    records = [json.loads(line) for line in small_dataset.logs_path.open()]
    anomalous = [r for r in records if r["is_anomaly"]]
    normal = [r for r in records if not r["is_anomaly"]]
    assert anomalous and normal
    assert all(r["incident_id"] and r["anomaly_type"] for r in anomalous)
    assert all(not r["incident_id"] for r in normal)

    brute = [r for r in anomalous if r["anomaly_type"] == "ssh_brute_force"]
    assert len(brute) > 100
    assert len({r["message"].split(" from ")[1].split()[0] for r in brute
                if " from " in r["message"]}) == 1, "brute force should come from one IP"
    miner = [r for r in anomalous if r["anomaly_type"] == "suspicious_process"]
    assert any("stratum+tcp" in r["message"] or ":3333" in r["message"] for r in miner)
    privesc = [r for r in anomalous if r["anomaly_type"] == "privilege_escalation"]
    assert any("sudoers" in r["message"] for r in privesc)


def test_hard_negative_logs_are_labelled_normal(tmp_path: Path) -> None:
    """Benign look-alikes (single failed login, admin sudo) must stay label 0."""
    result = generate_dataset(DatasetConfig(seed=11, duration_hours=48, incidents_per_day=0,
                                            output_dir=tmp_path, run_name="clean"))
    records = [json.loads(line) for line in result.logs_path.open()]
    assert all(r["is_anomaly"] == 0 for r in records)
    assert any("Failed password for ops" in r["message"] for r in records)
    assert any("COMMAND=/bin/systemctl" in r["message"] for r in records)


def test_generation_is_deterministic(tmp_path: Path) -> None:
    kwargs = dict(seed=123, duration_hours=3, incidents_per_day=40, output_dir=tmp_path)
    a = generate_dataset(DatasetConfig(run_name="a", **kwargs))
    b = generate_dataset(DatasetConfig(run_name="b", **kwargs))
    assert a.metrics_path.read_bytes() == b.metrics_path.read_bytes()
    assert a.logs_path.read_bytes() == b.logs_path.read_bytes()
    c = generate_dataset(DatasetConfig(run_name="c", **{**kwargs, "seed": 124}))
    assert a.metrics_path.read_bytes() != c.metrics_path.read_bytes()


@pytest.mark.parametrize(
    "kwargs",
    [{"duration_hours": 0}, {"step_seconds": 5}, {"n_servers": 0},
     {"run_name": "../escape"}, {"run_name": "a/b"}, {"seed": -1}],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DatasetConfig(**kwargs)


def test_resolve_run_dir_latest(small_dataset: DatasetResult) -> None:
    out = small_dataset.run_dir.parent
    assert resolve_run_dir(out) == small_dataset.run_dir
    with pytest.raises(ValueError):
        resolve_run_dir(out, "../etc")
