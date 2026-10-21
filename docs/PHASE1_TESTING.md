# Phase 1: what to test and what you should see

All of these were run while building the phase. Your numbers should match exactly for the
batch dataset (same seed, same output), and roughly for the live simulator (it is random).

## 0. Setup (once)

**Windows (PowerShell, in VS Code terminal):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Setup creates `.venv`, installs dependencies, copies `.env.example` to `.env`, runs the tests and
generates the first dataset. Then in VS Code press `Ctrl+Shift+P`, choose **Python: Select Interpreter**,
and pick `.venv`.

## 1. Unit tests

```bash
python -m pytest -v
```
**Expected:** `38 passed`. The tests check that:
* labels exactly match the incident windows, and every incident type and both categories appear;
* injected anomalies visibly move their signature metric (CPU, failed logins, outbound traffic, ...);
* attack logs contain the right signatures (one attacker IP for brute force, `stratum+tcp` for the miner, sudoers writes for privilege escalation);
* benign look-alikes (a single mistyped password, an admin running sudo) stay labelled normal;
* the same seed gives byte-identical files, and a different seed gives different files;
* bad inputs are rejected (path traversal in `--run-name`, out-of-range settings, unknown hosts, `limit=5000`);
* the inject endpoint requires `X-Admin-Token` when one is configured.

## 2. Batch dataset

```bash
python -m aiops_copilot.generator generate
python -m aiops_copilot.generator summary --examples 1
```
**Expected (seed 42, 24 h, 6 hosts):**
```
Done in ~2s: 8640 metric rows (5.6% anomalous), 33785 log lines (9.4% anomalous)
...
INC-20260105-0001  data_exfiltration      security     api-02     2026-01-05 04:03    33
INC-20260105-0002  error_burst            operational  api-02     2026-01-05 07:16    19
INC-20260105-0003  ssh_brute_force        security     worker-01  2026-01-05 08:48    20
... 14 incidents in total, all 10 types ...
Example anomalous log lines:
  [ssh_brute_force] worker-01 sshd: Failed password for invalid user postgres from 198.51.100.60 port ... ssh2
  [suspicious_process] db-01 auditd: type=EXECVE uid=www-data cmd="/tmp/.cache/kworkerds --url stratum+tcp://..."
```
Files appear in `data/synthetic/seed42_24h_6hosts/`: `metrics.csv`, `logs.jsonl`, `incidents.json`, `manifest.json`.

Try a bigger run for Phase 2 training: `python -m aiops_copilot.generator generate --hours 168 --run-name week1`
(about 12 s, 60,480 metric rows, 245k log lines).

**Quick look in Python** (optional):
```python
import pandas as pd
df = pd.read_csv("data/synthetic/seed42_24h_6hosts/metrics.csv")
df.groupby("anomaly_type")[["cpu_pct", "net_out_mbps", "failed_logins"]].mean()
```
You should see `cpu_spike` and `suspicious_process` with CPU around 90, `data_exfiltration` with high
`net_out_mbps`, and `ssh_brute_force` with around 100 failed logins per minute, while normal rows stay
near CPU 35-50 and 0 failed logins.

## 3. Live simulator (no Docker)

```bash
python -m aiops_copilot.simulator
```
(or VS Code: Run and Debug, then **3. Run live fleet simulator**). Then in a second terminal:

| Check | Command | Expected |
|---|---|---|
| Health | `curl http://localhost:8001/health` | `{"status":"ok","sim_time":"...","ticks":N,...}` |
| API docs | open http://localhost:8001/docs | Swagger UI listing 7 endpoints |
| Prometheus format | `curl http://localhost:8001/metrics` | lines like `aiops_server_cpu_pct{host="web-01",role="web"} 41.2` |
| Inject an attack | `curl -X POST http://localhost:8001/api/v1/incidents/inject -H "Content-Type: application/json" -d "{\"incident_type\":\"ssh_brute_force\",\"host\":\"web-01\"}"` | `201` with the incident JSON (id, attacker_ip, root_cause, ...) |
| Attack logs | after ~5 s: `curl "http://localhost:8001/api/v1/logs?only_anomalies=true&limit=3"` | `Failed password for invalid user ... from 198.51.100.x` lines with `is_anomaly: 1` |
| Attack metrics | `curl "http://localhost:8001/api/v1/metrics/latest?host=web-01"` | `failed_logins` around 50-150 and `is_anomaly: 1` |
| Validation | `curl "http://localhost:8001/api/v1/logs?limit=5000"` | `422` error |

PowerShell users: use `curl.exe` instead of `curl`, or open the URLs in a browser. The easiest way to send the
POST is the **Try it out** button in http://localhost:8001/docs.

## 4. Full stack with Docker

```bash
docker compose up -d --build
docker compose ps
```
**Expected:** `postgres (healthy)`, `simulator (healthy)`, `prometheus`, `grafana` all `Up`.

| Check | Where | Expected |
|---|---|---|
| Prometheus target | http://localhost:9090/targets | `fleet-simulator` is **UP** |
| Prometheus query | http://localhost:9090, query `aiops_server_cpu_pct` | 6 series, one per host |
| Grafana | http://localhost:3000 (login from `.env`) | Dashboard **AI Ops Copilot / Fleet Overview** with 8 live charts |
| See an attack | inject `data_exfiltration` via http://localhost:8001/docs | Within ~10 s the "Outbound traffic" chart jumps by 90-300 Mbps on one host, and the ground-truth table shows the incident |
| Postgres schema | `docker compose exec postgres psql -U aiops -d aiops -c "\dt"` | tables `alerts`, `detections`, `incidents`, `rca_runs` |
| Batch generator in Docker | `docker compose --profile tools run --rm generator` | same `Done in ...` line; files in `./data/synthetic` |

Stop everything with `docker compose down` (add `-v` to also delete the database and simulator volumes).

## Troubleshooting

* **`POSTGRES_PASSWORD` / `GRAFANA_ADMIN_PASSWORD` error on `docker compose up`**: you have no `.env`. Run `copy .env.example .env` (Windows) or `cp .env.example .env`.
* **Port already in use**: change `SIM_PORT` / `POSTGRES_PORT` in `.env`, or stop the local service using it.
* **`ModuleNotFoundError: aiops_copilot`**: activate `.venv` and run `pip install -e . --no-deps`, or use the VS Code launch configs (they set `PYTHONPATH`).
* **Linux: generator container cannot write `./data`**: set `HOST_UID`/`HOST_GID` in `.env` to the output of `id -u` / `id -g`.
