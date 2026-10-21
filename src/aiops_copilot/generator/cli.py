"""Command-line interface for the synthetic data generator.

Examples::

    python -m aiops_copilot.generator.cli generate
    python -m aiops_copilot.generator.cli generate --hours 168 --seed 7 --run-name week1
    python -m aiops_copilot.generator.cli summary --examples 2

Defaults come from environment variables / ``.env`` (see ``.env.example``),
and every flag overrides its setting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from aiops_copilot.core.config import get_settings
from aiops_copilot.core.logging import setup_logging
from aiops_copilot.generator.dataset import DatasetConfig, generate_dataset, resolve_run_dir

logger = logging.getLogger("aiops_copilot.generator")


def _parse_start(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are treated as UTC."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value!r}") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (defaults pulled from settings)."""
    s = get_settings()
    parser = argparse.ArgumentParser(prog="aiops-generate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a labelled metrics + logs dataset")
    gen.add_argument("--seed", type=int, default=s.gen_seed)
    gen.add_argument("--hours", type=float, default=s.gen_duration_hours,
                     help="Length of the simulated window")
    gen.add_argument("--step-seconds", type=int, default=s.gen_step_seconds)
    gen.add_argument("--servers", type=int, default=6, help="Fleet size (1-60)")
    gen.add_argument("--incidents-per-day", type=float, default=s.gen_incidents_per_day)
    gen.add_argument("--start", type=_parse_start, default=s.gen_start,
                     help="Window start, ISO-8601 (UTC if no offset)")
    gen.add_argument("--output-dir", type=Path, default=s.gen_output_dir)
    gen.add_argument("--run-name", type=str, default=None)

    summ = sub.add_parser("summary", help="Print statistics of a generated dataset")
    summ.add_argument("--output-dir", type=Path, default=s.gen_output_dir)
    summ.add_argument("--run-name", type=str, default=None, help="Defaults to the latest run")
    summ.add_argument("--examples", type=int, default=1,
                      help="Anomalous log lines to show per incident type (0-10)")
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    config = DatasetConfig(
        seed=args.seed,
        start=args.start,
        duration_hours=args.hours,
        step_seconds=args.step_seconds,
        n_servers=args.servers,
        incidents_per_day=args.incidents_per_day,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    result = generate_dataset(config)
    print(f"\nDataset written to: {result.run_dir}")
    print(json.dumps({k: v for k, v in result.summary.items()
                      if k not in {"logs_by_source", "anomaly_rows_by_type"}}, indent=2))
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    if not 0 <= args.examples <= 10:
        raise ValueError("--examples must be between 0 and 10")
    run_dir = resolve_run_dir(args.output_dir, args.run_name)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    incidents = json.loads((run_dir / "incidents.json").read_text(encoding="utf-8"))
    summary = manifest["summary"]

    print(f"Run: {run_dir}")
    print(f"Window: {manifest['created_with']['start']} -> {manifest['created_with']['end']}")
    print(f"Hosts: {', '.join(h['name'] for h in manifest['hosts'])}")
    print(f"Metric rows: {summary['metric_rows']:,} "
          f"({summary['metric_anomaly_ratio']:.1%} anomalous)")
    print(f"Log lines:   {summary['log_rows']:,} ({summary['log_anomaly_ratio']:.1%} anomalous)\n")

    header = f"{'incident_id':<18} {'type':<22} {'category':<12} {'host':<10} {'start (UTC)':<17} {'min':>4}"
    print(header)
    print("-" * len(header))
    for inc in incidents:
        start = datetime.fromisoformat(inc["start"])
        minutes = int((datetime.fromisoformat(inc["end"]) - start).total_seconds() // 60)
        print(f"{inc['incident_id']:<18} {inc['incident_type']:<22} {inc['category']:<12} "
              f"{inc['host']:<10} {start:%Y-%m-%d %H:%M}  {minutes:>4}")

    if args.examples:
        print("\nExample anomalous log lines:")
        shown: dict[str, int] = defaultdict(int)
        with (run_dir / "logs.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                kind = rec["anomaly_type"]
                if rec["is_anomaly"] and shown[kind] < args.examples:
                    shown[kind] += 1
                    print(f"  [{kind}] {rec['host']} {rec['source']}: {rec['message'][:120]}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            return _cmd_generate(args)
        return _cmd_summary(args)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
