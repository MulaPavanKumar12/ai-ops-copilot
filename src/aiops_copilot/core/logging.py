"""Logging setup shared by every entry point (CLI, simulator, API).

Supports human-readable text logs for local work and single-line JSON logs
for containers, where a log shipper can parse them.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via ``extra={...}`` is included as structured fields.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure the root logger once.

    Args:
        level: Minimum log level name, e.g. ``"INFO"``.
        fmt: ``"text"`` for humans or ``"json"`` for machines.
    """
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Keep noisy third-party loggers quiet unless we are debugging.
    for noisy in ("uvicorn.access", "httpx"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))
