#!/usr/bin/env bash
# Start simulator, API, worker and dashboard in the background (macOS / Linux, no Docker needed).
# Logs go to logs/*.log. Stop everything with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs
pids=()
python -m aiops_copilot.simulator        > logs/simulator.log 2>&1 & pids+=($!); sleep 3
python -m aiops_copilot.api              > logs/api.log 2>&1 & pids+=($!); sleep 3
python -m aiops_copilot.pipeline.worker  > logs/worker.log 2>&1 & pids+=($!)
streamlit run dashboard/app.py --server.headless true --browser.gatherUsageStats false \
                                         > logs/dashboard.log 2>&1 & pids+=($!)
trap 'kill "${pids[@]}" 2>/dev/null; echo stopped' INT TERM EXIT
echo "Running. Dashboard: http://localhost:8501   API docs: http://localhost:8000/docs   (Ctrl+C to stop)"
wait
