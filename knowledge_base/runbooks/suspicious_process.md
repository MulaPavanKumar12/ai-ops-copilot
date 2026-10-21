# Runbook: Suspicious process (crypto-miner) (suspicious_process)

**Category:** security  
**Default severity:** high

## What it looks like

An unknown binary in /tmp consumed most of the CPU and connected to a mining pool.

## Triage steps

1. Find processes running from /tmp or with high CPU and unknown names.
2. Check outbound connections to mining pools (ports 3333/4444/5555) or reverse shells.
3. Remove persistence: cron entries, systemd units, authorized_keys.
4. Close the entry point (exposed Docker/K8s API), mount /tmp noexec.

## Known root causes and fixes

### `suspicious_process/exposed_container_api`

- **Cause:** An exposed container runtime API allowed an attacker to start a crypto-miner from /tmp.
- **Fix:** Kill the process, remove the binary and persistence (cron/systemd), close the exposed API, and mount /tmp noexec.
