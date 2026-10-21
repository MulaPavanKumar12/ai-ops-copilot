"""Central application settings.

All configuration is read from environment variables (optionally loaded from a
``.env`` file). Nothing secret is ever hard-coded: secrets are typed as
:class:`pydantic.SecretStr` so they are masked in logs and ``repr`` output.

Usage::

    from aiops_copilot.core.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed, validated settings for every component of AI Ops Copilot."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # ------------------------------------------------------------------ app
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    # ------------------------------------------------------------- postgres
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "aiops"
    postgres_user: str = "aiops"
    postgres_password: SecretStr = SecretStr("change-me")

    # ------------------------------------------------- synthetic generator
    gen_seed: int = Field(default=42, ge=0)
    gen_duration_hours: float = Field(default=24.0, gt=0, le=24 * 30)
    gen_step_seconds: int = Field(default=60, ge=10, le=3600)
    gen_incidents_per_day: float = Field(default=14.0, ge=0, le=200)
    gen_start: datetime = datetime(2026, 1, 5, tzinfo=UTC)
    gen_output_dir: Path = PROJECT_ROOT / "data" / "synthetic"

    # ------------------------------------------------------ live simulator
    sim_host: str = "0.0.0.0"
    sim_port: int = Field(default=8001, ge=1, le=65535)
    sim_tick_seconds: float = Field(default=5.0, ge=0.5, le=300)
    sim_speedup: float = Field(default=12.0, ge=1.0, le=720.0)
    sim_incident_probability: float = Field(default=0.02, ge=0.0, le=1.0)
    sim_log_dir: Path = PROJECT_ROOT / "data" / "live"
    sim_log_max_mb: int = Field(default=50, ge=1, le=2048)
    sim_admin_token: SecretStr | None = None

    # ------------------------------------------------------ storage paths
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    # Empty = local SQLite file in data/. docker-compose sets a PostgreSQL URL.
    database_url: str = ""

    # ------------------------------------------------------------- RAG
    embedding_backend: Literal["lsa", "sentence-transformers"] = "lsa"
    vector_store: Literal["chroma", "memory"] = "chroma"
    rag_top_k: int = Field(default=5, ge=1, le=20)

    # ------------------------------------------------------------- LLM
    llm_provider: Literal["anthropic", "none"] = "anthropic"
    llm_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high"] = "medium"
    llm_timeout_seconds: float = Field(default=90.0, ge=5, le=600)
    llm_max_tool_rounds: int = Field(default=6, ge=1, le=15)
    llm_server_fallbacks: bool = True

    # ------------------------------------------------------------- API
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: SecretStr | None = None
    api_rate_limit_per_minute: int = Field(default=240, ge=1, le=100_000)
    cors_origins: str = "http://localhost:8501"

    # ------------------------------------------------------------- alerting
    slack_webhook_url: SecretStr | None = None
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = ""
    smtp_password: SecretStr | None = None
    alert_email_from: str = "aiops-copilot@localhost"
    alert_email_to: str = ""
    alert_cooldown_minutes: int = Field(default=30, ge=0, le=24 * 60)

    # ------------------------------------------------------------- worker / dashboard
    simulator_url: str = "http://localhost:8001"
    api_url: str = "http://localhost:8000"
    worker_poll_seconds: float = Field(default=2.0, ge=0.5, le=300)
    worker_run_rca: bool = True

    @field_validator("gen_start")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        """Treat naive timestamps as UTC so every component agrees on time."""
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @property
    def effective_database_url(self) -> str:
        """DATABASE_URL if set, else a SQLite file under ``data_dir``."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'aiops.db').as_posix()}"

    @property
    def postgres_url(self) -> str:
        """PostgreSQL DSN built from the POSTGRES_* settings (used by docker-compose)."""
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def llm_key(self) -> str:
        """The LLM API key (LLM_API_KEY, or ANTHROPIC_API_KEY), or "" if none is set."""
        for secret in (self.llm_api_key, self.anthropic_api_key):
            if secret is not None and secret.get_secret_value().strip():
                return secret.get_secret_value().strip()
        return ""

    @property
    def llm_enabled(self) -> bool:
        """True when an LLM provider and key are configured."""
        return self.llm_provider != "none" and bool(self.llm_key())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
