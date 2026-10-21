# Runbook: CPU saturation (cpu_spike)

**Category:** operational  
**Default severity:** high

## What it looks like

CPU utilisation jumped above 85% and stayed there, increasing request latency.

## Triage steps

1. Run `top -o %CPU` / `pidstat 1` to find the hot process.
2. Check recent deploys and config pushes in the last hour.
3. Look for retry loops or regex timeouts in application logs.
4. Check `crontab -l` for jobs running more often than intended.

## Known root causes and fixes

### `cpu_spike/regex_backtracking`

- **Cause:** A catastrophic-backtracking regex in the request validator was hit by long user input, pinning worker threads.
- **Fix:** Roll back the validator change, bound input length, and replace the regex with a linear-time pattern.

### `cpu_spike/retry_storm`

- **Cause:** A config push set the retry policy to unlimited retries with no backoff, creating a hot loop against a failing dependency.
- **Fix:** Revert the config, enforce exponential backoff with a retry cap, and add a circuit breaker.

### `cpu_spike/cron_misfire`

- **Cause:** A cron entry was mistyped to run a heavy report job every minute instead of daily.
- **Fix:** Fix the cron expression, kill the duplicate jobs, and add a lock file so the job cannot overlap.
