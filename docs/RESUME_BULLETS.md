# Resume bullets

**AI Ops Copilot** · Python, PyTorch, scikit-learn, FastAPI, ChromaDB, Claude API, PostgreSQL, Docker · [github.com/&lt;you&gt;/ai-ops-copilot]

- Built an AI monitoring assistant that detects operational and security anomalies across a 6-server fleet, fusing a z-score baseline, Isolation Forest, a PyTorch LSTM autoencoder and log-based security rules; caught **100% of 42 incidents** in a held-out 3-day test with a **1-minute median time to detect** and **5.7 false alarms/day**.
- Designed a leakage-free evaluation (separate train/validation/test seeds, thresholds tuned on validation only) and a synthetic telemetry generator with **10 incident types, 17 root causes and hard negatives** (backups, deploys, mistyped passwords), reporting point-wise P/R/F1 (**hybrid F1 0.88, recall 0.88**) plus event recall and time to detect.
- Implemented RAG-based root-cause analysis over **190 post-mortems and runbooks** in ChromaDB, with a Claude tool-calling agent (metrics, logs, log-pattern and knowledge-base tools, JSON-schema output) and an offline retrieval engine that identifies the **exact root cause for 97.7% of incidents in 0.27 s (p95)**.
- Shipped the system as production-style services: FastAPI backend with API-key auth, rate limiting and validated inputs; real-time streaming worker; Slack/email alerts with severity routing and deduplication; Streamlit dashboard; PostgreSQL; **7-service docker-compose** with health checks.
- Engineered for reliability: automatic fallback when the LLM is unavailable, refuses or returns invalid output (every failure path covered by tests), **86 automated tests** including an end-to-end pipeline test, strict mypy and CI on GitHub Actions; scoring cost of **0.1 ms per metric row**.

---

**Shorter versions (one line each)**

- Built an ML + LLM monitoring copilot (Isolation Forest, LSTM autoencoder, security rules, RAG with Claude) detecting 100% of test incidents in a median 1 minute.
- Delivered root-cause analysis with 97.7% exact-cause accuracy at 0.27 s p95 using ChromaDB retrieval, with a tool-calling Claude agent and graceful fallback.

**Tip:** if you run the LLM evaluation (`python -m aiops_copilot.evaluation.run_rca --engine llm --limit 20`)
add its accuracy and median latency to bullet 3. All numbers above come from `reports/` and are
reproducible with the fixed seeds.
