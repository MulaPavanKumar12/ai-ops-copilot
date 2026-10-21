# Runbook: Traffic surge (traffic_surge)

**Category:** operational  
**Default severity:** medium

## What it looks like

Requests per second jumped 3-5x, raising CPU, network and latency.

## Triage steps

1. Break traffic down by client IP, user agent and URL.
2. Distinguish real users (campaign referrers) from bots (one user agent, robots.txt ignored).
3. Scale out or enable autoscaling for real demand.
4. Rate-limit abusive clients at the load balancer.

## Known root causes and fixes

### `traffic_surge/marketing_campaign`

- **Cause:** A marketing campaign went live without notifying the platform team, so capacity was not scaled up in advance.
- **Fix:** Scale out the web tier, enable autoscaling on RPS, and add launches to the capacity-planning calendar.

### `traffic_surge/aggressive_crawler`

- **Cause:** An aggressive web crawler ignored robots.txt and hammered search endpoints.
- **Fix:** Rate-limit by client IP / user agent at the load balancer and serve crawlers a cached response.
