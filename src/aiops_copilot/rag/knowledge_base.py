"""Incident knowledge base: past incident reports + runbooks, searchable by similarity.

Where the reports come from: in a real company they are post-mortems written
by engineers. Here they are generated from a separate 30-day "history" run of
the simulator: for every past incident we record its symptoms (which metrics
moved and by how much), the characteristic log lines, the confirmed root cause
and the fix. Runbooks (``knowledge_base/runbooks/*.md``) are indexed too.

Retrieval uses the same text shape for queries and documents: symptoms plus
log templates. Log lines are turned into *templates* (numbers and IPs replaced
by placeholders) so "QueuePool limit of size 5" matches across incidents.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aiops_copilot.generator.models import METRIC_NAMES
from aiops_copilot.rag.store import Embedder, Hit, VectorStore, load_embedder, make_embedder, make_store

logger = logging.getLogger(__name__)

_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_HEX = re.compile(r"\b0x[0-9a-f]+\b", re.I)

METRIC_PHRASES: dict[str, str] = {
    "cpu_pct": "CPU utilisation", "mem_pct": "memory usage", "disk_pct": "disk usage",
    "latency_p95_ms": "p95 latency", "error_rate_pct": "HTTP error rate",
    "requests_per_sec": "request rate", "net_in_mbps": "inbound traffic",
    "net_out_mbps": "outbound traffic", "failed_logins": "failed SSH logins",
    "process_count": "process count",
}


def log_template(message: str) -> str:
    """Replace IPs, hex and numbers with placeholders."""
    msg = _IP.sub("<ip>", message)
    msg = _HEX.sub("<hex>", msg)
    return _NUM.sub("<n>", msg)


def describe_deviations(deviations: list[tuple[str, float]]) -> str:
    """Plain-English symptoms from (metric, z-score) pairs."""
    parts = []
    for metric, z in deviations:
        if abs(z) < 2.5:
            continue
        direction = "far above" if z > 0 else "far below"
        parts.append(f"{METRIC_PHRASES.get(metric, metric)} {direction} normal")
    return "; ".join(parts) if parts else "no strong metric deviation"


def top_log_templates(messages: list[str], k: int = 6) -> list[str]:
    """Most frequent templates among a list of log messages."""
    return [t for t, _ in Counter(log_template(m) for m in messages).most_common(k)]


@dataclass
class KnowledgeDoc:
    """One indexed document."""

    doc_id: str
    doc_type: str  # "incident_report" or "runbook"
    title: str
    text: str
    incident_type: str = ""
    category: str = ""
    root_cause_id: str = ""
    root_cause: str = ""
    suggested_fix: str = ""
    host_role: str = ""

    def metadata(self) -> dict[str, Any]:
        """Flat metadata for the vector store (no None values)."""
        return {k: v for k, v in asdict(self).items() if k != "text"}


def reports_from_history(run_dir: Path) -> list[KnowledgeDoc]:
    """Turn every incident of a generated dataset into a post-mortem document."""
    metrics = pd.read_csv(run_dir / "metrics.csv", parse_dates=["timestamp"])
    incidents = json.loads((run_dir / "incidents.json").read_text(encoding="utf-8"))
    by_incident: dict[str, list[str]] = {}
    with (run_dir / "logs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["is_anomaly"] and rec["level"] in ("WARNING", "ERROR", "CRITICAL", "INFO"):
                by_incident.setdefault(rec["incident_id"], []).append(rec["message"])

    normal = metrics[metrics["is_anomaly"] == 0]
    base = normal.groupby("host")[list(METRIC_NAMES)].agg(["median", "std"])
    roles = metrics.groupby("host")["role"].first()
    docs: list[KnowledgeDoc] = []
    for inc in incidents:
        rows = metrics[metrics["incident_id"] == inc["incident_id"]]
        if rows.empty:
            continue
        host = inc["host"]
        devs = []
        for m in METRIC_NAMES:
            med, std = float(base.loc[host, (m, "median")]), float(base.loc[host, (m, "std")]) or 1.0
            peak = float(rows[m].max()) if rows[m].mean() >= med else float(rows[m].min())
            devs.append((m, (peak - med) / max(std, 1e-6)))
        devs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        symptoms = describe_deviations(devs[:4])
        templates = top_log_templates(by_incident.get(inc["incident_id"], []))
        text = (f"{inc['title'].split(' on ')[0]} on a {roles[host]} server. Symptoms: {symptoms}. "
                f"Log evidence: {' | '.join(templates)}. Root cause: {inc['root_cause']} "
                f"Resolution: {inc['suggested_fix']}")
        docs.append(KnowledgeDoc(
            doc_id=f"report-{inc['incident_id']}", doc_type="incident_report",
            title=f"Post-mortem {inc['incident_id']}: {inc['title']}", text=text,
            incident_type=inc["incident_type"], category=inc["category"],
            root_cause_id=inc.get("root_cause_id", ""), root_cause=inc["root_cause"],
            suggested_fix=inc["suggested_fix"], host_role=str(roles[host])))
    return docs


def runbooks_from_dir(directory: Path) -> list[KnowledgeDoc]:
    """Load Markdown runbooks; the incident type is taken from the file name."""
    docs = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), path.stem)
        docs.append(KnowledgeDoc(doc_id=f"runbook-{path.stem}", doc_type="runbook", title=title,
                                 text=text, incident_type=path.stem))
    return docs


class KnowledgeBase:
    """Build, persist and search the knowledge base."""

    def __init__(self, directory: Path, store_kind: str = "chroma", embedding_backend: str = "lsa") -> None:
        self.directory = directory
        self.store_kind = store_kind
        self.embedding_backend = embedding_backend
        self.embedder: Embedder | None = None
        self.store: VectorStore | None = None

    @property
    def ready(self) -> bool:
        """True if an index exists on disk."""
        return (self.directory / "embedder.joblib").exists() and (self.directory / "docs.jsonl").exists()

    def build(self, docs: list[KnowledgeDoc]) -> int:
        """(Re)build the index from documents. Returns the number indexed."""
        if not docs:
            raise ValueError("no documents to index")
        self.directory.mkdir(parents=True, exist_ok=True)
        texts = [d.text for d in docs]
        self.embedder = make_embedder(self.embedding_backend).fit(texts)
        vectors = self.embedder.embed(texts)
        self.store = make_store(self.store_kind, self.directory)
        self.store.reset()
        self.store.add([d.doc_id for d in docs], vectors, [d.metadata() for d in docs], texts)
        if hasattr(self.store, "persist"):
            self.store.persist(self.directory / "memory_store.joblib")
        self.embedder.save(self.directory / "embedder.joblib")
        with (self.directory / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(asdict(d)) + "\n")
        logger.info("Indexed %d documents into %s (%s store)", len(docs), self.directory,
                    type(self.store).__name__)
        return len(docs)

    def load(self) -> KnowledgeBase:
        """Open an existing index."""
        if not self.ready:
            raise FileNotFoundError(f"no knowledge base in {self.directory}; run `python -m "
                                    "aiops_copilot.rag.build` first")
        self.embedder = load_embedder(self.directory / "embedder.joblib")
        self.store = make_store(self.store_kind, self.directory)
        return self

    def search(self, query: str, k: int = 5, doc_type: str | None = None) -> list[Hit]:
        """Top-k documents most similar to ``query``."""
        if self.embedder is None or self.store is None:
            self.load()
        assert self.embedder is not None and self.store is not None
        vec: np.ndarray = self.embedder.embed([query])[0]
        return self.store.query(vec, k, where={"doc_type": doc_type} if doc_type else None)
