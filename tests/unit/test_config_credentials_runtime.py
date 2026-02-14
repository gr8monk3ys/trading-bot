import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _reload_config_module():
    config = importlib.import_module("config")
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _reset_config_after_test():
    yield
    _reload_config_module()


def test_get_alpaca_creds_supports_alias_env_vars(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "alias_key")
    monkeypatch.setenv("API_SECRET", "alias_secret")
    monkeypatch.setenv("PAPER", "false")

    config = _reload_config_module()
    creds = config.get_alpaca_creds(refresh=True)

    assert creds["API_KEY"] == "alias_key"
    assert creds["API_SECRET"] == "alias_secret"
    assert creds["PAPER"] is False


def test_require_alpaca_credentials_raises_when_missing(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_SECRET", raising=False)

    config = _reload_config_module()

    with pytest.raises(ValueError, match="Alpaca API credentials not found"):
        config.require_alpaca_credentials("unit-test mode")


def test_main_help_without_credentials_has_no_startup_warning():
    env = os.environ.copy()
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "API_KEY", "API_SECRET"):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "must be set in environment variables" not in result.stderr
    assert "Trading will fail until valid API credentials are provided" not in result.stderr
