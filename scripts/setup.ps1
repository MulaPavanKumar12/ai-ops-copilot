# One-time local setup for Windows (PowerShell):
#   venv -> dependencies -> .env -> tests -> train + evaluate models -> knowledge base -> RCA evaluation
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
# Takes about 5-10 minutes (mostly downloading PyTorch the first time).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Prefer the Python launcher with 3.11; fall back to "python".
$py = "python"
try { & py -3.11 --version *> $null; if ($LASTEXITCODE -eq 0) { $py = "py -3.11" } } catch { }
Invoke-Expression "$py -c `"import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'`""

if (-not (Test-Path ".venv")) { Invoke-Expression "$py -m venv .venv" }
$venvPy = ".\.venv\Scripts\python.exe"

function Step($title, [scriptblock]$action) {
    Write-Host ""
    Write-Host "==> $title" -ForegroundColor Cyan
    & $action
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $title" }
}

Step "Installing dependencies" {
    & $venvPy -m pip install --upgrade pip | Out-Null
    & $venvPy -m pip install -r requirements-dev.txt
    & $venvPy -m pip install -e . --no-deps
}

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example. Add LLM_API_KEY there to enable the Claude RCA agent."
}

Step "Running tests"                              { & $venvPy -m pytest }
Step "Training + evaluating detectors (~1 min)"   { & $venvPy -m aiops_copilot.evaluation.run_detection }
Step "Building the RAG knowledge base (~1 min)"   { & $venvPy -m aiops_copilot.rag.build }
Step "Evaluating root-cause analysis"             { & $venvPy -m aiops_copilot.evaluation.run_rca }

Write-Host ""
Write-Host "Setup complete. Results are in reports\detection_results.md and reports\rca_results.md" -ForegroundColor Green
Write-Host "Start the live system (4 windows):  powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1"
