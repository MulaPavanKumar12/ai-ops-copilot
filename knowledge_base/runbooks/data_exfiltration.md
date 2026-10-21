# Runbook: Unusual outbound data transfer (data_exfiltration)

**Category:** security  
**Default severity:** critical

## What it looks like

Outbound traffic to an unknown external IP spiked far above baseline, outside business hours.

## Triage steps

1. Identify the process and destination of the outbound flows (netflow / `ss -tp`).
2. Compare with the scheduled backup window and destination before escalating.
3. Isolate the host from the network if the destination is unknown.
4. Revoke the credentials used and preserve evidence for forensics.

## Known root causes and fixes

### `data_exfiltration/compromised_service_account`

- **Cause:** A compromised service account was used to dump the database and upload it to an attacker-controlled storage endpoint.
- **Fix:** Isolate the host, revoke the service-account credentials, block the destination at the egress firewall, and start forensic review.
