#!/usr/bin/env bash
# One-time local setup for macOS / Linux:
#   venv -> dependencies -> .env -> tests -> train + evaluate models -> knowledge base -> RCA evaluation
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3.11}
command -v "$PY" >/dev/null 2>&1 || PY=python3
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
# On Linux without a GPU the CPU build of PyTorch is ~10x smaller.
if [ "$(uname -s)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu || true
fi
pip install -r requirements-dev.txt
pip install -e . --no-deps

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example. Add LLM_API_KEY to enable the Claude RCA agent."; }

echo "==> Running tests";                       python -m pytest
echo "==> Training + evaluating detectors";     python -m aiops_copilot.evaluation.run_detection
echo "==> Building the RAG knowledge base";     python -m aiops_copilot.rag.build
echo "==> Evaluating root-cause analysis";      python -m aiops_copilot.evaluation.run_rca
echo
echo "Setup complete. Results: reports/detection_results.md and reports/rca_results.md"
echo "Start the live system:  bash scripts/run_local.sh   (then open http://localhost:8501)"
