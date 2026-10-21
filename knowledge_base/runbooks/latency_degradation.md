# Runbook: Latency degradation (latency_degradation)

**Category:** operational  
**Default severity:** medium

## What it looks like

p95 latency rose several-fold while traffic stayed normal.

## Triage steps

1. Compare p95 latency with request rate: latency up with flat traffic points to a dependency.
2. Check slow-query logs and `EXPLAIN ANALYZE` for new queries.
3. Check upstream provider status pages and timeout settings.
4. Look for thread-pool exhaustion in application metrics.

## Known root causes and fixes

### `latency_degradation/missing_index`

- **Cause:** A new query missed an index after a schema migration, causing full table scans on a hot path.
- **Fix:** Add the missing composite index, verify with EXPLAIN ANALYZE, and add query-plan checks to migrations.

### `latency_degradation/upstream_timeout`

- **Cause:** The upstream payment provider was degraded and calls had a 30s timeout, so worker threads blocked waiting.
- **Fix:** Lower the upstream timeout, add a circuit breaker and fallback, and subscribe to the provider's status page.
