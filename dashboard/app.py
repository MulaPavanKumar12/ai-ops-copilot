"""AI Ops Copilot dashboard (Streamlit).

Run:  streamlit run dashboard/app.py
Reads the backend API (incidents, stats, model results) and the fleet
simulator (live metrics, attack injection for demos). URLs and keys come from
environment variables / .env (API_URL, SIMULATOR_URL, API_KEY, SIM_ADMIN_TOKEN).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aiops_copilot.core.config import get_settings  # noqa: E402
from aiops_copilot.generator.models import METRIC_NAMES, IncidentType  # noqa: E402

settings = get_settings()
API_URL = os.environ.get("API_URL", settings.api_url).rstrip("/")
SIM_URL = os.environ.get("SIMULATOR_URL", settings.simulator_url).rstrip("/")
API_HEADERS = {"X-API-Key": settings.api_key.get_secret_value()} if settings.api_key else {}
SIM_HEADERS = {"X-Admin-Token": settings.sim_admin_token.get_secret_value()} if settings.sim_admin_token else {}
SEV_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
METRIC_LABELS = {
    "cpu_pct": "CPU %", "mem_pct": "Memory %", "disk_pct": "Disk %", "latency_p95_ms": "p95 latency (ms)",
    "error_rate_pct": "Error rate %", "requests_per_sec": "Requests/s", "net_in_mbps": "Inbound Mbps",
    "net_out_mbps": "Outbound Mbps", "failed_logins": "Failed logins/min", "process_count": "Processes",
}

st.set_page_config(page_title="AI Ops Copilot", page_icon="🛰️", layout="wide")


def api_get(path: str, **params: Any) -> Any:
    """GET from the backend; returns None (and shows a warning) on failure."""
    try:
        r = httpx.get(f"{API_URL}{path}", params=params, headers=API_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.warning(f"API not reachable at {API_URL}{path}: {exc}")
        return None


def sim_get(path: str, **params: Any) -> Any:
    """GET from the simulator; None on failure."""
    try:
        r = httpx.get(f"{SIM_URL}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


st.title("🛰️ AI Ops Copilot")
st.caption("ML anomaly detection, security detection and LLM root-cause analysis for a server fleet.")

tab_live, tab_incidents, tab_models, tab_demo = st.tabs(
    ["Live fleet", "Incidents", "Model performance", "Demo: inject an incident"])


# ----------------------------------------------------------------------------- live
@st.fragment(run_every=5)
def live_view() -> None:
    """Auto-refreshing KPIs and charts."""
    stats = api_get("/api/v1/stats") or {}
    by_status = stats.get("by_status", {})
    sev = stats.get("open_by_severity", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open incidents", by_status.get("open", 0))
    c2.metric("Critical / high open", sev.get("critical", 0) + sev.get("high", 0))
    c3.metric("Security incidents (24h)", stats.get("by_category_24h", {}).get("security", 0))
    c4.metric("Avg RCA latency", f"{stats['avg_rca_latency_ms']:.0f} ms" if stats.get("avg_rca_latency_ms") else "-")
    c5.metric("Alerts sent", stats.get("alerts_sent", 0))

    latest = sim_get("/api/v1/metrics/latest")
    if latest is None:
        st.info(f"Simulator not reachable at {SIM_URL}. Start it with `python -m aiops_copilot.simulator`.")
        return
    hist: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    seen = {(r["host"], r["timestamp"]) for r in hist[-50:]}
    hist.extend(r for r in latest if (r["host"], r["timestamp"]) not in seen)
    st.session_state["history"] = hist[-6 * 180:]
    df = pd.DataFrame(st.session_state["history"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    metric = st.session_state.get("live_metric", "cpu_pct")
    chart = df.pivot_table(index="timestamp", columns="host", values=metric)
    st.subheader(f"{METRIC_LABELS[metric]} by host")
    st.line_chart(chart, height=320)

    open_inc = api_get("/api/v1/incidents", status="open", limit=10) or []
    if open_inc:
        st.subheader("Open incidents")
        for inc in open_inc:
            st.markdown(f"{SEV_ICON.get(inc['severity'], '')} **{inc['title']}** · {inc['category']} · "
                        f"since {inc['started_at'][11:16]} UTC  \n{inc.get('summary') or '_analysing..._'}")


with tab_live:
    st.selectbox("Metric", list(METRIC_NAMES), key="live_metric", format_func=lambda m: METRIC_LABELS[m])
    live_view()


# ------------------------------------------------------------------------ incidents
with tab_incidents:
    f1, f2, f3 = st.columns(3)
    status_f = f1.selectbox("Status", ["all", "open", "acknowledged", "resolved"])
    cat_f = f2.selectbox("Category", ["all", "operational", "security"])
    limit = f3.slider("Show", 10, 200, 50)
    params: dict[str, Any] = {"limit": limit}
    if status_f != "all":
        params["status"] = status_f
    if cat_f != "all":
        params["category"] = cat_f
    rows = api_get("/api/v1/incidents", **params) or []
    if not rows:
        st.info("No incidents yet. Inject one from the Demo tab, or wait for the simulator to create one.")
    else:
        table = pd.DataFrame(rows)[["id", "severity", "title", "category", "status", "started_at",
                                    "root_cause_id", "confidence"]]
        st.dataframe(table, hide_index=True, use_container_width=True)
        default_id = int(st.query_params.get("incident", rows[0]["id"]))
        ids = [r["id"] for r in rows]
        chosen = st.selectbox("Incident details", ids, index=ids.index(default_id) if default_id in ids else 0,
                              format_func=lambda i: next(f"#{r['id']} {r['title']}" for r in rows if r["id"] == i))
        detail = api_get(f"/api/v1/incidents/{chosen}")
        if detail:
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"### {SEV_ICON.get(detail['severity'], '')} {detail['title']}")
                st.markdown(f"**What happened:** {detail.get('summary') or '_RCA pending_'}")
                st.markdown(f"**Likely root cause:** {detail.get('root_cause') or '-'}")
                st.markdown(f"**Suggested fix:** {detail.get('suggested_fix') or '-'}")
                ev = detail.get("evidence") or {}
                if ev.get("rca_evidence"):
                    st.markdown("**Evidence**")
                    for e in ev["rca_evidence"]:
                        st.code(e, language=None)
                if ev.get("security_findings"):
                    st.markdown("**Security rule hits**")
                    st.dataframe(pd.DataFrame(ev["security_findings"])[["timestamp", "rule_id", "description",
                                                                         "source_ip"]], hide_index=True)
            with right:
                st.metric("Confidence", f"{(detail.get('confidence') or 0):.0%}")
                st.write(f"Detectors: {', '.join(detail.get('detectors') or [])}")
                st.write(f"RCA engine: {ev.get('rca_engine', '-')}")
                st.write("Alerts:", ", ".join(f"{a['channel']} ({a['status']})" for a in detail["alerts"]) or "none")
                b1, b2 = st.columns(2)
                for label, new_status, col in (("Acknowledge", "acknowledged", b1), ("Resolve", "resolved", b2)):
                    if col.button(label, key=f"{label}-{chosen}"):
                        httpx.post(f"{API_URL}/api/v1/incidents/{chosen}/status", json={"status": new_status},
                                   headers=API_HEADERS, timeout=10)
                        st.rerun()


# ---------------------------------------------------------------------------- models
with tab_models:
    info = api_get("/api/v1/model-info") or {}
    det = info.get("detection_results")
    if not det:
        st.info("No evaluation results yet. Run `python -m aiops_copilot.evaluation.run_detection`.")
    else:
        m = det["meta"]
        st.markdown(f"Test split: **{m['test_rows']:,}** metric rows, **{m['test_incidents']}** injected incidents "
                    f"over {m['test_days']:.0f} days. Thresholds were chosen on a separate validation split.")
        table = pd.DataFrame([{
            "Detector": d["detector"], "Precision": d["point"]["precision"], "Recall": d["point"]["recall"],
            "F1": d["point"]["f1"], "Incidents detected": f"{d['events']['detected']}/{d['events']['incidents']}",
            "Median time-to-detect (min)": d["events"]["median_time_to_detect_min"],
            "False alarms/day": d["events"]["false_alarms_per_day"],
        } for d in det["detectors"]])
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.bar_chart(table.set_index("Detector")[["Precision", "Recall", "F1"]], stack=False, height=300)
    for key, label in (("rca_results", "Root-cause analysis (offline retrieval)"),
                       ("rca_results_llm", "Root-cause analysis (LLM agent)")):
        rca = info.get(key)
        if rca:
            st.subheader(label)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Exact root cause correct", f"{rca['root_cause_accuracy']:.0%}")
            c2.metric("Incident type correct", f"{rca['type_accuracy']:.0%}")
            c3.metric("Latency p50", f"{rca['latency_p50_ms']:.0f} ms")
            c4.metric("Latency p95", f"{rca['latency_p95_ms']:.0f} ms")


# ------------------------------------------------------------------------------ demo
with tab_demo:
    st.markdown("Start an anomaly or attack on the simulated fleet and watch the pipeline detect, explain "
                "and alert on it (usually within 10-30 seconds).")
    hosts = [h["name"] for h in (sim_get("/api/v1/hosts") or [])]
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Incident type", [t.value for t in IncidentType])
    host = c2.selectbox("Host", ["(any eligible)"] + hosts)
    if st.button("Inject incident", type="primary"):
        body: dict[str, Any] = {"incident_type": kind}
        if host != "(any eligible)":
            body["host"] = host
        try:
            r = httpx.post(f"{SIM_URL}/api/v1/incidents/inject", json=body, headers=SIM_HEADERS, timeout=10)
            if r.status_code == 201:
                st.success(f"Injected {kind} on {r.json()['host']} ({r.json()['incident_id']}).")
            else:
                st.error(f"Simulator answered {r.status_code}: {r.json().get('detail')}")
        except httpx.HTTPError as exc:
            st.error(f"Simulator not reachable: {exc}")
    truth = sim_get("/api/v1/incidents", active_only=True) or []
    st.subheader("Ground truth: incidents active in the simulator")
    st.caption("Shown for the demo only; the detectors never see these labels.")
    if truth:
        st.dataframe(pd.DataFrame(truth)[["incident_id", "incident_type", "host", "start", "end", "root_cause_id"]],
                     hide_index=True)
    else:
        st.write("None right now.")
