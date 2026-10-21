# Root-cause analysis results (retrieval)

| Metric | Value |
|---|---|
| Incidents explained | 43 |
| Root-cause accuracy (exact cause) | 97.7% |
| Incident-type accuracy | 97.7% |
| Latency p50 | 206 ms |
| Latency p95 | 271 ms |

## Example explanations

- **INC-20260114-0001** (memory_leak, predicted `memory_leak/unbounded_cache`): worker-01 (worker) has shown memory usage far above normal; p95 latency far above normal since 10:19 UTC. New log patterns include: "Out of memory: Killed process <n> (python3) total-vm:8123456kB, anon-rss:7340032kB". This matches 5 of 5 similar past incidents (memory leak).
- **INC-20260114-0002** (memory_leak, predicted `memory_leak/unbounded_cache`): web-01 (web) has shown memory usage far above normal; HTTP error rate far above normal; p95 latency far above normal since 12:53 UTC. New log patterns include: "Out of memory: Killed process <n> (python3) total-vm:8123456kB, anon-rss:7340032kB". This matches 5 of 5 similar past incidents (memory leak).
- **INC-20260114-0003** (suspicious_process, predicted `suspicious_process/exposed_container_api`): api-01 (api) has shown CPU utilisation far above normal since 13:55 UTC. New log patterns include: "ESTABLISHED <ip>:<n> -> <ip>:<n> proto=tcp process=kworkerds". This matches 5 of 5 similar past incidents (suspicious process).
- **INC-20260114-0004** (disk_fill, predicted `disk_fill/wal_replication_slot`): web-01 (web) has shown disk usage far above normal; HTTP error rate far above normal since 15:17 UTC. New log patterns include: "filesystem /var usage <n>% (+6GB/h)". This matches 5 of 5 similar past incidents (disk fill).

## Misses (1)

- **INC-20260114-0001** on worker-01: true `memory_leak/unbounded_cache`, predicted `ssh_brute_force/exposed_ssh_password_auth` (confidence 1.00). worker-01 (worker) has shown memory usage far above normal since 10:05 UTC. New log patterns include: "Accepted publickey for deploy from <ip> port <n> ssh2". This matches 5 of 5 similar past incidents (ssh brute force).
