# Runbook: Privilege escalation attempt (privilege_escalation)

**Category:** security  
**Default severity:** critical

## What it looks like

A low-privilege account attempted sudo, modified sudoers and set SUID bits.

## Triage steps

1. Review `auth.log` and auditd for sudo failures, sudoers writes and new group members.
2. List SUID binaries changed recently (`find / -perm -4000 -newer /etc/hostname`).
3. Look for web shells in upload directories.
4. Isolate the host, rotate secrets, and rebuild from a clean image.

## Known root causes and fixes

### `privilege_escalation/web_shell`

- **Cause:** A web-shell uploaded via an unpatched file-upload endpoint gave the attacker a shell as www-data, who then tried to gain root.
- **Fix:** Isolate the host, remove the web shell, patch the upload endpoint, audit sudoers and SUID binaries, and rotate secrets.
