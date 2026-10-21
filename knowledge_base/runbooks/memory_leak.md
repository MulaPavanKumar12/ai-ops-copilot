# Runbook: Memory leak (memory_leak)

**Category:** operational  
**Default severity:** high

## What it looks like

Memory grew steadily without being released until the OOM killer terminated the service.

## Triage steps

1. Plot memory over the last 6 hours: a straight ramp means a leak, a step means a load change.
2. Take a heap profile (py-spy dump / tracemalloc) before restarting.
3. Check cache sizes and open sockets (`ss -s`).
4. Restart one instance to buy time; keep one for analysis.

## Known root causes and fixes

### `memory_leak/unbounded_cache`

- **Cause:** An in-process cache had no eviction policy after a library upgrade, so entries accumulated forever.
- **Fix:** Restart the service to recover, set a max size / TTL on the cache, and add a memory-growth alert.

### `memory_leak/unclosed_sessions`

- **Cause:** HTTP client sessions were created per request and never closed, leaking sockets and buffers.
- **Fix:** Reuse a single pooled client, close sessions in a finally block, and add a leak test to CI.
