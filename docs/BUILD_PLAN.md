# Build plan (8 phases)

All 8 phases are complete. Each phase ended with passing tests, a "what to test / expected output" section, and a working
`docker compose up`.

| # | Phase | Delivers | Status |
|---|---|---|---|
| 1 | **Foundation + synthetic data** | Project skeleton, settings/logging, `.env.example`, docker-compose (simulator, Prometheus, Grafana, Postgres), batch generator with labelled anomalies and attacks, live simulator API, 38 tests | **Done** |
| 2 | **Features + Isolation Forest + evaluation harness** | Feature pipeline (rolling windows, per-host normalisation), Isolation Forest detector, z-score baseline, evaluation module (point-wise and event-level precision/recall/F1, time-to-detect, per-type breakdown), first results table | **Done** |
| 3 | **LSTM autoencoder + security detection** | PyTorch LSTM AE trained on normal windows with a threshold set on validation data, log-based security detectors (brute-force rate rules, exfiltration volume, priv-esc and miner signatures), score fusion, comparison table IF vs LSTM vs hybrid | **Done** |
| 4 | **RAG + LLM root-cause analysis** | Incident-report corpus built from history, ChromaDB index, LLM agent with function calling (`get_metrics_window`, `get_logs`, `search_similar_incidents`), JSON-schema-validated output, RCA accuracy and p50/p95 latency evaluation, offline fallback when no API key is set | **Done** |
| 5 | **Backend API + persistence + alerting** | FastAPI service (incidents, detections, RCA endpoints), SQLAlchemy models on the Phase 1 schema, Slack webhook and SMTP alerts with dedup and severity routing, input validation, API-key auth, rate limiting | **Done** |
| 6 | **Dashboard + real-time pipeline** | Streamlit dashboard (fleet health, live charts, open incidents, RCA panel, "inject attack" demo button), background worker that streams from the simulator through detection, RCA and alerting | **Done** |
| 7 | **Hardening + CI** | GitHub Actions (ruff, mypy, pytest, docker build), coverage gate, health checks, structured logging everywhere, load test of the pipeline, security review (secrets scan, dependency audit) | **Done** |
| 8 | **Portfolio packaging** | Final README with results table and screenshots, 3-minute demo script, 5 resume bullets with measured numbers, 10 interview questions with answers | **Done** |

## Why this order

* **Data before models.** Labels and realistic noise are what make every later number trustworthy.
* **Baseline before deep learning.** The Isolation Forest result is the bar the LSTM must beat. An interviewer will ask "did the LSTM actually help?"
* **Detection before LLM.** The LLM explains incidents; it should not be the detector. This keeps cost and latency down and the system testable.
* **API and alerting after the core works**, so they wrap proven components.
