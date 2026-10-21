"""API protection: API-key auth and per-client rate limiting."""

from __future__ import annotations

import hmac
import threading
import time
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


class RateLimiter:
    """Token bucket per client key (in-process; use Redis when running several replicas)."""

    def __init__(self, per_minute: int) -> None:
        self.rate = per_minute / 60.0
        self.capacity = float(per_minute)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Consume one token for ``key`` if available."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            if len(self._buckets) > 10_000:  # bound memory
                self._buckets.pop(next(iter(self._buckets)))
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Return 429 when a client exceeds its budget (health checks are exempt)."""

    def __init__(self, app: object, limiter: RateLimiter) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path != "/health":
            client = request.client.host if request.client else "unknown"
            if not self.limiter.allow(client):
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                                    headers={"Retry-After": "10"})
        return await call_next(request)


def make_api_key_checker(expected: str) -> Callable[[Request], None]:
    """Dependency that enforces ``X-API-Key`` when an API key is configured."""

    def check(request: Request) -> None:
        if not expected:
            return
        supplied = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")

    return check
