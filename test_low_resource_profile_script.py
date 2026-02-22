from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_low_resource_profile.py"
    spec = importlib.util.spec_from_file_location("run_low_resource_profile", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_command_uses_small_defaults() -> None:
    module = _load_module()
    args = module._parse_args([])

    command = module._build_command(args)

    assert "live_trader.py" in command[1]
    assert "--strategy" in command
    assert "momentum" in command
    assert "--symbols" in command
    assert "--risk-profile" in command
    risk_profile_index = command.index("--risk-profile")
    assert command[risk_profile_index + 1] == "conservative"
    symbol_index = command.index("--symbols")
    assert command[symbol_index + 1 : symbol_index + 3] == ["BTC/USD", "ETH/USD"]


def test_build_command_stock_profile_defaults() -> None:
    module = _load_module()
    args = module._parse_args(["--asset-class", "stock"])

    command = module._build_command(args)

    symbol_index = command.index("--symbols")
    assert command[symbol_index + 1 : symbol_index + 3] == ["AAPL", "MSFT"]


def test_build_command_includes_explicit_risk_overrides() -> None:
    module = _load_module()
    args = module._parse_args(
        [
            "--risk-profile",
            "aggressive",
            "--position-size",
            "0.12",
            "--max-position-size",
            "0.18",
            "--max-daily-loss",
            "0.04",
            "--max-intraday-drawdown",
            "0.07",
        ]
    )

    command = module._build_command(args)

    assert "--risk-profile" in command
    assert command[command.index("--risk-profile") + 1] == "aggressive"
    assert "--position-size" in command
    assert command[command.index("--position-size") + 1] == "0.12"
    assert "--max-position-size" in command
    assert command[command.index("--max-position-size") + 1] == "0.18"
    assert "--max-daily-loss" in command
    assert command[command.index("--max-daily-loss") + 1] == "0.04"
    assert "--max-intraday-drawdown" in command
    assert command[command.index("--max-intraday-drawdown") + 1] == "0.07"


def test_build_environment_sets_safe_defaults(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.delenv("PAPER", raising=False)
    monkeypatch.delenv("SLO_PAGING_ENABLED", raising=False)

    env = module._build_environment()

    assert env["PAPER"] == "true"
    assert env["SLO_PAGING_ENABLED"] == "false"
    assert env["PAPER_LIVE_SHADOW_DRIFT_WARNING"] == "0.10"
    assert env["PAPER_LIVE_SHADOW_DRIFT_MAX"] == "0.12"
