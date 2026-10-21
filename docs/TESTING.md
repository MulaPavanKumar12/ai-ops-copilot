# What to test and the expected output

Run these from the project folder with the virtual environment active
(Windows: `.\.venv\Scripts\Activate.ps1`, macOS/Linux: `source .venv/bin/activate`).
`scripts/setup.ps1` / `scripts/setup.sh` run steps 1-4 for you.

Numbers below come from the fixed random seeds, so you should see the same or very close values
(PyTorch can differ in the last decimal between machines).

## 1. Unit, integration and end-to-end tests

```bash
python -m pytest
```

Expected: `86 passed` in about 1-2 minutes. What they cover:

| File | What it proves |
|---|---|
| `test_generator.py`, `test_simulator_api.py`, `test_config.py` | Data generator is reproducible and labelled correctly; simulator API validates input; settings load from `.env` |
| `test_features.py` | Normal traffic scales to z ≈ 0; anomalies deviate far more; scaler save/load is exact |
| `test_detection.py` | Hybrid detector catches ≥80% of incidents on unseen data; save/load gives identical scores; LSTM scores anomalies higher; brute-force rule needs 15 failures in 5 min; one mistyped password is not an attack; exfiltration only to external IPs |
| `test_metrics.py` | Precision/recall/F1 and time-to-detect math on hand-made cases |
| `test_rag_rca.py` | Knowledge-base search finds the right incident type; retrieval RCA gets ≥70% of types right; the Claude agent loop with a fake client (tool calls, tool errors, structured answer) and every fallback path (API error, refusal, bad JSON, schema violation, endless tool loop) |
| `test_alerts_db.py` | Incident merging, re-labelling on a security hit, severity routing, dedup, a failing channel does not block others, Slack URL allow-list |
| `test_api.py` | API key required, input validation (422), 404, rate limiting (429), `/analyze` with trained models |
| `test_worker.py` | End to end: replayed simulator data → worker → stored brute-force incident with root cause and a sent alert |

Also: `ruff check src tests dashboard` → `All checks passed!` and `mypy` → `Success: no issues found`.

## 2. Train and evaluate the detectors

```bash
python -m aiops_copilot.evaluation.run_detection
```

Expected (about 1 minute): it generates three datasets in `data/eval/` (train 7 days, validation 2 days,
test 3 days), trains, then prints and writes `reports/detection_results.md`:

```
| Z-score baseline | 0.920 | 0.878 | 0.899 | 40/42 (95%) | 2.0 min | 3.7 |
| Isolation Forest | 0.770 | 0.727 | 0.748 | 42/42 (100%) | 2.5 min | 32.7 |
| LSTM autoencoder | 0.845 | 0.818 | 0.831 | 41/42 (98%) | 2.0 min | 12.7 |
| Security log rules | 1.000 | 0.165 | 0.283 | 17/42 (40%) | 0.0 min | 0.0 |
| Hybrid (z-score + IF + LSTM + rules) | 0.889 | 0.880 | 0.884 | 42/42 (100%) | 1.0 min | 5.7 |
```

Model files appear in `models/` (`scaler.json`, `isolation_forest.joblib`, `lstm_autoencoder.pt`,
`detector_meta.json`). Faster smoke test: add `--quick --lstm-epochs 2`.

## 3. Build the knowledge base

```bash
python -m aiops_copilot.rag.build
```

Expected (about 1 minute): `Indexed 190 documents` (180 post-mortems + 10 runbooks) and an example
search for a brute-force query whose top hits are `ssh_brute_force` reports.

## 4. Evaluate root-cause analysis

```bash
python -m aiops_copilot.evaluation.run_rca                       # offline retrieval engine
python -m aiops_copilot.evaluation.run_rca --engine llm --limit 20   # Claude agent (needs LLM_API_KEY)
```

Expected for retrieval: `root_cause_accuracy: 0.977`, `type_accuracy: 0.977`, p50 about 0.2 s, written to
`reports/rca_results.md` with example explanations and the one miss. With the LLM engine the report is
`reports/rca_results_llm.md`; each incident takes several seconds (several tool calls), and
`fallbacks_used` tells you how many fell back to retrieval.

## 5. Live system

Start the four services (`scripts/run_local.ps1`, `scripts/run_local.sh`, the VS Code compound
"Live system", or `docker compose up -d --build`).

| Check | Expected |
|---|---|
| http://localhost:8000/health | `{"status":"ok","version":"1.0.0","models_loaded":true}` |
| http://localhost:8000/docs | Swagger UI with incidents, stats, model-info, analyze |
| http://localhost:8501 | Dashboard; the CPU chart fills in over the first minute |
| Dashboard → Demo → inject `ssh_brute_force` | Within ~30 s: Incidents tab shows "SSH brute-force attack on <host>", category security, root cause `ssh_brute_force/exposed_ssh_password_auth`, fix "Block the source IP, disable password auth...", security rule hits BRUTE-001 |
| Inject `data_exfiltration` | Critical incident, root cause `data_exfiltration/...`, rule EXFIL-002 with the external IP |
| Inject `memory_leak` | Appears after a few minutes (it is a slow ramp), root cause `memory_leak/...` |
| Worker console | `New incident #N ...` then `RCA for #N (retrieval, 80 ms): ...` (or `llm:claude-opus-5-5` with a key) |
| Acknowledge / Resolve buttons | Status changes in the table |

Note: the worker needs 25 samples per host before it starts detecting (about 2 minutes with the default
5-second simulator tick).

With `API_KEY` set in `.env`, calls without the header fail:

```bash
curl -i http://localhost:8000/api/v1/incidents                      # 401
curl -H "X-API-Key: <your key>" http://localhost:8000/api/v1/stats  # 200 with counts
```

## 6. Alerts

Set `SLACK_WEBHOOK_URL` (a real `https://hooks.slack.com/...` URL) and restart the worker. Inject a
`data_exfiltration` attack: a Slack message arrives with the title, severity, what happened, the
suggested fix and a link to the incident. Injecting the same attack on the same host again within
`ALERT_COOLDOWN_MINUTES` does not send a second message. Any URL that is not a Slack webhook is ignored.

## 7. Docker

```bash
docker compose up -d --build
docker compose ps          # all services "Up", postgres/simulator/api "healthy"
docker compose logs -f worker
```

On the first start the worker trains the models and builds the knowledge base (about 2-3 minutes:
`Models saved to /app/models`, `Indexed 190 documents`, `Worker started`). Incidents are stored in
PostgreSQL. Stop with `docker compose down` (add `-v` to delete the data volumes).
