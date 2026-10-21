# Start the four live services in separate PowerShell windows (Windows, no Docker needed):
#   simulator :8001 -> worker (detection + RCA + alerts) -> API :8000 -> dashboard :8501
# Run setup.ps1 once first. Close the windows (or press Ctrl+C in each) to stop.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run scripts/setup.ps1 first" }

$services = @(
    @{ Name = "simulator"; Args = "-m aiops_copilot.simulator" },
    @{ Name = "api";       Args = "-m aiops_copilot.api" },
    @{ Name = "worker";    Args = "-m aiops_copilot.pipeline.worker" },
    @{ Name = "dashboard"; Args = "-m streamlit run dashboard/app.py --browser.gatherUsageStats=false" }
)
foreach ($s in $services) {
    Start-Process powershell -WorkingDirectory $root -ArgumentList @(
        "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='aiops $($s.Name)'; & '$py' $($s.Args)")
    Start-Sleep -Seconds 3
}
Write-Host "Started. Dashboard: http://localhost:8501   API docs: http://localhost:8000/docs"
