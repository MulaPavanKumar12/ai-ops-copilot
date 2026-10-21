# Runbook: Disk filling up (disk_fill)

**Category:** operational  
**Default severity:** critical

## What it looks like

Disk usage climbed rapidly toward 100%, causing write failures.

## Triage steps

1. `df -h` and `du -xh / | sort -h | tail` to find the largest directories.
2. Check log levels and logrotate config for new log files.
3. On PostgreSQL check `pg_wal` size and inactive replication slots.
4. Free space safely: compress or delete rotated logs first.

## Known root causes and fixes

### `disk_fill/debug_logging`

- **Cause:** Debug logging was left enabled after an investigation and log rotation was not configured for the new log file.
- **Fix:** Set log level back to INFO, add the file to logrotate, and purge old logs to free space.

### `disk_fill/wal_replication_slot`

- **Cause:** Database WAL files accumulated because a replication slot was abandoned and blocked WAL recycling.
- **Fix:** Drop the stale replication slot, monitor slot lag, and alert when WAL size exceeds a threshold.
