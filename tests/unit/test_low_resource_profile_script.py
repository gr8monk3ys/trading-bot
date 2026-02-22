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
    assert command[symbol_index + 1 : symbol_index + 11] == [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "LTC/USD",
        "DOGE/USD",
        "AVAX/USD",
        "LINK/USD",
        "BCH/USD",
        "DOT/USD",
        "UNI/USD",
    ]


def test_build_command_stock_profile_defaults() -> None:
    module = _load_module()
    args = module._parse_args(["--asset-class", "stock"])

    command = module._build_command(args)

    symbol_index = command.index("--symbols")
    assert command[symbol_index + 1 : symbol_index + 3] == ["AAPL", "MSFT"]


def test_build_command_honors_crypto_symbol_env_override(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("LOW_RESOURCE_CRYPTO_SYMBOLS", "BTC/USD,ETH/USD,LINK/USD")
    args = module._parse_args([])

    command = module._build_command(args)

    symbol_index = command.index("--symbols")
    assert command[symbol_index + 1 : symbol_index + 4] == ["BTC/USD", "ETH/USD", "LINK/USD"]


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


def test_build_environment_sets_aggressive_crypto_risk_overrides(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.delenv("MAX_PORTFOLIO_RISK", raising=False)
    monkeypatch.delenv("MAX_POSITION_RISK", raising=False)
    args = module._parse_args(["--asset-class", "crypto", "--risk-profile", "aggressive"])

    env = module._build_environment(args)

    assert env["MAX_PORTFOLIO_RISK"] == "0.30"
    assert env["MAX_POSITION_RISK"] == "0.15"


def test_build_command_aggressive_crypto_applies_higher_action_tuning() -> None:
    module = _load_module()
    args = module._parse_args(["--asset-class", "crypto", "--risk-profile", "aggressive"])

    command = module._build_command(args)

    assert "--crypto-buy-score-threshold" in command
    assert command[command.index("--crypto-buy-score-threshold") + 1] == "0.75"
    assert "--crypto-dip-rsi-max" in command
    assert command[command.index("--crypto-dip-rsi-max") + 1] == "45.0"
    assert "--crypto-dip-min-macd-hist-delta" in command
    assert command[command.index("--crypto-dip-min-macd-hist-delta") + 1] == "0.005"
    assert "--crypto-dip-min-rebound-pct" in command
    assert command[command.index("--crypto-dip-min-rebound-pct") + 1] == "0.0005"
