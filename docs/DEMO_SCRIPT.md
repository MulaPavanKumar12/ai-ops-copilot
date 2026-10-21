# 3-minute demo script

**Before you start (5 minutes earlier):** run `scripts/run_local.ps1` (or `docker compose up -d`), open
http://localhost:8501 and let the live chart fill for a few minutes. Have three things ready in browser
tabs: the dashboard, `reports/detection_results.md` on GitHub, and http://localhost:8000/docs.
Optional: set `LLM_API_KEY` so the incident shows `llm:claude-opus-5-5` as the RCA engine.

---

### 0:00 – 0:20 · The problem

> "On-call engineers get flooded with alerts, and when something real breaks they spend the first half
> hour figuring out what and why. I built AI Ops Copilot: it watches a server fleet, detects
> operational and security anomalies with ML, and explains the likely root cause with a fix, grounded
> in the team's past incidents."

*Show:* the **Live fleet** tab: KPI tiles and the CPU chart for six hosts.

### 0:20 – 0:50 · Architecture in one breath

> "Metrics and logs stream from the fleet into a worker. Every metric is compared with that host's
> normal level for that hour of day. Four detectors score it: a z-score baseline, an Isolation Forest,
> an LSTM autoencoder in PyTorch, and security rules on the logs. The scores are fused, grouped into
> incidents, and handed to a Claude agent that investigates with tools and a ChromaDB knowledge base of
> 180 past post-mortems. Results go to PostgreSQL, Slack and email, and this dashboard."

*Show:* the Mermaid diagram in the README (or `docs/architecture.png`).

### 0:50 – 1:50 · Live attack

*Go to* **Demo: inject an incident** → choose `ssh_brute_force` → **Inject incident**.

> "This simulates a botnet guessing SSH passwords on one of the servers. The detectors don't see
> this ground-truth table, it's just for the demo."

*Switch to* **Incidents** (refresh after ~20-30 s).

> "Here it is: a high-severity security incident. The rule engine saw more than 15 failed logins from
> one external IP in five minutes, and the metric models saw failed logins far above normal. The RCA
> says what happened, that SSH password authentication was exposed to the internet, and the fix: block
> the IP, keys only, put SSH behind a VPN, enable fail2ban. It found five similar past incidents, and
> took under 100 milliseconds offline, or a few seconds with the LLM agent."

*Point at:* evidence lines, security rule hits (BRUTE-001), confidence, the alerts sent.
*Click* **Acknowledge**.

### 1:50 – 2:35 · Does it actually work? (numbers)

*Show:* the **Model performance** tab.

> "I evaluated on three days of unseen data with 42 incidents, with thresholds chosen on a separate
> validation set. The hybrid catches all 42, median one minute after they start, with about six false
> alarms a day across six hosts. An honest finding: the simple z-score baseline has the best F1, 0.90,
> because the seasonal scaling does most of the work. The LSTM alone is weaker, but in the hybrid it
> gets recall to 100% and halves time to detect. Root-cause analysis picks the exact cause out of 17
> for 97.7% of incidents."

### 2:35 – 3:00 · Engineering and wrap-up

> "It degrades gracefully: with no API key, or if the LLM fails or returns invalid JSON, it falls back
> to retrieval so alerts never stop. The API has key auth, rate limiting and validated inputs, no
> secrets in code, 86 tests including an end-to-end pipeline test, CI, and one `docker compose up` for
> the whole stack. Next I'd connect it to real Prometheus and Loki data and add a feedback loop so
> resolved incidents become new knowledge."

*Show:* http://localhost:8000/docs for two seconds, then stop.

---

**If something goes wrong live:** the incident takes longer than 30 s → say "detection waits for a few
minutes of evidence to avoid false alarms" and show an existing incident in the table. Dashboard says
"API not reachable" → the API window was closed; restart it with `python -m aiops_copilot.api`.
