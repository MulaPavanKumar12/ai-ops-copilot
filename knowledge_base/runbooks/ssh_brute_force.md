# Runbook: SSH brute-force attack (ssh_brute_force)

**Category:** security  
**Default severity:** high

## What it looks like

Hundreds of failed SSH logins from a single external IP within minutes.

## Triage steps

1. Count failed logins per source IP in `/var/log/auth.log`.
2. Check whether any login from that IP succeeded (`Accepted password`).
3. Block the IP at the firewall and enable fail2ban.
4. Disable password authentication; allow SSH only through a bastion/VPN.
5. If a login succeeded: treat the host as compromised and start incident response.

## Known root causes and fixes

### `ssh_brute_force/exposed_ssh_password_auth`

- **Cause:** SSH (port 22) was exposed to the internet with password authentication enabled, and an automated botnet was guessing credentials.
- **Fix:** Block the source IP, disable password auth (keys only), restrict SSH to a VPN/bastion, and enable fail2ban.
