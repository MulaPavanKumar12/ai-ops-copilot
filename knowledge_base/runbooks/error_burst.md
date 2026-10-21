# Runbook: Error-rate burst (error_burst)

**Category:** operational  
**Default severity:** high

## What it looks like

HTTP 5xx error rate spiked well above its normal level.

## Triage steps

1. Group 5xx errors by exception type in the last 15 minutes.
2. Check DB connection-pool saturation and recent config changes.
3. Check TLS certificate expiry for internal services (`openssl s_client`).
4. Roll back the most recent deploy if errors started right after it.

## Known root causes and fixes

### `error_burst/pool_exhausted`

- **Cause:** The database connection pool was exhausted after a deploy halved the pool size setting.
- **Fix:** Restore the pool size, alert on pool saturation, and load-test configuration changes before deploy.

### `error_burst/expired_tls_cert`

- **Cause:** An expired TLS certificate on an internal service broke service-to-service calls.
- **Fix:** Renew the certificate, automate renewal (e.g. cert-manager / certbot), and alert 14 days before expiry.
