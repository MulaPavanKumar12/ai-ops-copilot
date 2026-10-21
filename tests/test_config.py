"""Tests for settings: secrets stay masked, invalid values are rejected."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from aiops_copilot.core.config import Settings


def test_secrets_are_masked() -> None:
    settings = Settings(_env_file=None, postgres_password=SecretStr("hunter2"))
    assert "hunter2" not in repr(settings)
    assert "hunter2" in settings.postgres_url
    assert settings.effective_database_url.startswith("sqlite:///")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_SEED", "99")
    monkeypatch.setenv("SIM_TICK_SECONDS", "2.5")
    settings = Settings(_env_file=None)
    assert settings.gen_seed == 99 and settings.sim_tick_seconds == 2.5


@pytest.mark.parametrize(("key", "value"), [("POSTGRES_PORT", "70000"), ("GEN_SEED", "-1"),
                                             ("SIM_INCIDENT_PROBABILITY", "1.5")])
def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
