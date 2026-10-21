"""Evaluate root-cause analysis end to end on the test split.

For every incident the *detectors* found on the test data (no ground truth is
used to find them), the RCA engine explains it. We then compare its answer
with the ground truth of the real incident it overlaps:

* **Root-cause accuracy**: predicted ``root_cause_id`` equals the true one
  (e.g. ``error_burst/expired_tls_cert``, not just "error burst").
* **Type accuracy**: the incident family is right.
* **Latency**: wall-clock time per summary, p50 and p95.

Usage::

    python -m aiops_copilot.evaluation.run_rca                    # offline retrieval engine
    python -m aiops_copilot.evaluation.run_rca --engine llm --limit 20   # needs LLM_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from aiops_copilot.core.config import get_settings
from aiops_copilot.core.logging import setup_logging
from aiops_copilot.detection.hybrid import HybridDetector
from aiops_copilot.evaluation.run_detection import SPLITS_FULL, SPLITS_QUICK, ensure_split, load_split
from aiops_copilot.llm.evidence import FrameEvidence
from aiops_copilot.llm.rca import ClaudeRCA, RCAEngine, RetrievalRCA
from aiops_copilot.rag.build import build_knowledge_base, kb_dir
from aiops_copilot.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger("aiops_copilot.evaluation")


def run(engine_name: str = "retrieval", limit: int | None = None, quick: bool = False) -> dict[str, Any]:
    """Run the RCA evaluation and write ``reports/rca_results.{json,md}``."""
    settings = get_settings()
    data_dir = settings.data_dir / ("eval_quick" if quick else "eval")
    splits = SPLITS_QUICK if quick else SPLITS_FULL
    test_split = next(s for s in splits if s.name == "test")
    test, logs, truth = load_split(ensure_split(test_split, data_dir, 14.0))
    truth_by_id = {t["incident_id"]: t for t in truth}

    detector = HybridDetector.load(settings.model_dir)
    detected = detector.detect(test, logs).incidents

    kb = KnowledgeBase(kb_dir(settings), settings.vector_store, settings.embedding_backend)
    if not kb.ready:
        kb = build_knowledge_base(settings)
    kb.load()

    engine: RCAEngine
    if engine_name == "llm":
        if not settings.llm_enabled:
            raise ValueError("--engine llm needs LLM_API_KEY (or ANTHROPIC_API_KEY) in .env")
        engine = ClaudeRCA(settings, kb)
    else:
        engine = RetrievalRCA(kb, settings.rag_top_k)

    evidence = FrameEvidence(test, logs, detector.scaler)
    inc_ids = test.set_index(["host", "timestamp"])["incident_id"]
    rows: list[dict[str, Any]] = []
    for inc in detected:
        window = inc_ids.loc[inc.host].loc[inc.start: inc.end]
        ids = window[window != ""]
        if ids.empty:
            continue  # false alarm: nothing to score
        true = truth_by_id[ids.mode().iloc[0]]
        result = engine.analyze(inc, evidence)
        rows.append({
            "incident_id": true["incident_id"], "host": inc.host,
            "true_type": true["incident_type"], "true_root_cause_id": true.get("root_cause_id", ""),
            "predicted_root_cause_id": result.root_cause_id, "confidence": result.confidence,
            "root_cause_correct": result.root_cause_id == true.get("root_cause_id"),
            "type_correct": result.root_cause_id.split("/")[0] == true["incident_type"],
            "latency_ms": result.latency_ms, "engine": result.engine, "summary": result.summary,
        })
        if limit and len(rows) >= limit:
            break

    lat = np.array([r["latency_ms"] for r in rows], dtype=float)
    summary = {
        "engine": engine.name, "incidents_scored": len(rows),
        "root_cause_accuracy": round(float(np.mean([r["root_cause_correct"] for r in rows])), 3) if rows else None,
        "type_accuracy": round(float(np.mean([r["type_correct"] for r in rows])), 3) if rows else None,
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1) if rows else None,
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1) if rows else None,
        "fallbacks_used": sum("fallback" in r["engine"] for r in rows),
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if engine_name == "retrieval" else f"_{engine_name}"
    (settings.reports_dir / f"rca_results{suffix}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    md = [
        f"# Root-cause analysis results ({engine.name})", "",
        "| Metric | Value |", "|---|---|",
        f"| Incidents explained | {summary['incidents_scored']} |",
        f"| Root-cause accuracy (exact cause) | {summary['root_cause_accuracy']:.1%} |",
        f"| Incident-type accuracy | {summary['type_accuracy']:.1%} |",
        f"| Latency p50 | {summary['latency_p50_ms']:.0f} ms |",
        f"| Latency p95 | {summary['latency_p95_ms']:.0f} ms |",
        "", "## Example explanations", "",
    ]
    for r in [r for r in rows if r["root_cause_correct"]][:4]:
        md.append(f"- **{r['incident_id']}** ({r['true_type']}, predicted `{r['predicted_root_cause_id']}`): "
                  f"{r['summary']}")
    misses = [r for r in rows if not r["root_cause_correct"]]
    md += ["", f"## Misses ({len(misses)})", ""]
    md += [f"- **{r['incident_id']}** on {r['host']}: true `{r['true_root_cause_id']}`, predicted "
           f"`{r['predicted_root_cause_id']}` (confidence {r['confidence']:.2f}). {r['summary']}" for r in misses]
    if not misses:
        md.append("None.")
    (settings.reports_dir / f"rca_results{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["retrieval", "llm"], default="retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Max incidents to explain (cost control)")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= 500:
        logger.error("--limit must be between 1 and 500")
        return 2
    try:
        summary = run(args.engine, args.limit, args.quick)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
