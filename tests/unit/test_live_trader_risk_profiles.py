from __future__ import annotations

from types import SimpleNamespace

import pytest

from live_trader import (
    LiveTrader,
    _determine_position_scale,
    _resolve_runtime_parameters,
    _validate_runtime_risk_parameters,
)


def _args(**overrides):
    defaults = {
        "risk_profile": "custom",
        "position_size": None,
        "max_position_size": None,
        "stop_loss": None,
        "take_profit": None,
        "max_daily_loss": None,
        "max_intraday_drawdown": None,
        "drawdown_soft_limit": None,
        "drawdown_soft_scale": None,
        "drawdown_medium_limit": None,
        "drawdown_medium_scale": None,
        "kill_switch_cooldown_minutes": None,
        "crypto_buy_score_threshold": None,
        "crypto_dip_rsi_max": None,
        "crypto_dip_min_macd_hist_delta": None,
        "crypto_dip_min_rebound_pct": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_runtime_parameters_aggressive_profile_defaults() -> None:
    parameters = _resolve_runtime_parameters(_args(risk_profile="aggressive"))

    assert parameters["position_size"] == pytest.approx(0.10)
    assert parameters["max_position_size"] == pytest.approx(0.15)
    assert parameters["max_daily_loss"] == pytest.approx(0.04)
    assert parameters["drawdown_hard_limit_pct"] == pytest.approx(0.07)


def test_resolve_runtime_parameters_allows_cli_override_over_profile() -> None:
    parameters = _resolve_runtime_parameters(
        _args(
            risk_profile="aggressive",
            position_size=0.12,
            max_position_size=0.2,
            max_daily_loss=0.05,
        )
    )

    assert parameters["position_size"] == pytest.approx(0.12)
    assert parameters["max_position_size"] == pytest.approx(0.2)
    assert parameters["max_daily_loss"] == pytest.approx(0.05)


def test_resolve_runtime_parameters_accepts_crypto_momentum_overrides() -> None:
    parameters = _resolve_runtime_parameters(
        _args(
            risk_profile="aggressive",
            crypto_buy_score_threshold=0.7,
            crypto_dip_rsi_max=48.0,
            crypto_dip_min_macd_hist_delta=0.002,
            crypto_dip_min_rebound_pct=0.0003,
        )
    )

    assert parameters["crypto_long_only_buy_score_threshold"] == pytest.approx(0.7)
    assert parameters["crypto_long_only_dip_rsi_max"] == pytest.approx(48.0)
    assert parameters["crypto_long_only_dip_min_macd_hist_delta"] == pytest.approx(0.002)
    assert parameters["crypto_long_only_dip_min_rebound_pct"] == pytest.approx(0.0003)


def test_validate_runtime_risk_parameters_rejects_invalid_drawdown_ladder() -> None:
    with pytest.raises(ValueError):
        _validate_runtime_risk_parameters(
            {
                "position_size": 0.1,
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "max_daily_loss": 0.03,
                "drawdown_soft_limit_pct": 0.05,
                "drawdown_medium_limit_pct": 0.04,
                "drawdown_hard_limit_pct": 0.07,
                "drawdown_soft_scale": 0.75,
                "drawdown_medium_scale": 0.5,
                "kill_switch_cooldown_minutes": 60,
            }
        )


def test_validate_runtime_risk_parameters_rejects_invalid_crypto_rsi() -> None:
    with pytest.raises(ValueError):
        _validate_runtime_risk_parameters(
            {
                "position_size": 0.1,
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "max_daily_loss": 0.03,
                "drawdown_soft_limit_pct": 0.03,
                "drawdown_medium_limit_pct": 0.05,
                "drawdown_hard_limit_pct": 0.07,
                "drawdown_soft_scale": 0.75,
                "drawdown_medium_scale": 0.5,
                "kill_switch_cooldown_minutes": 60,
                "crypto_long_only_dip_rsi_max": 120.0,
            }
        )


def test_determine_position_scale_uses_ladder() -> None:
    parameters = {
        "drawdown_soft_limit_pct": 0.03,
        "drawdown_soft_scale": 0.8,
        "drawdown_medium_limit_pct": 0.05,
        "drawdown_medium_scale": 0.6,
        "drawdown_hard_limit_pct": 0.07,
    }

    assert _determine_position_scale(0.01, parameters) == pytest.approx(1.0)
    assert _determine_position_scale(0.035, parameters) == pytest.approx(0.8)
    assert _determine_position_scale(0.055, parameters) == pytest.approx(0.6)
    assert _determine_position_scale(0.08, parameters) == pytest.approx(0.0)


class _DummyStrategy:
    def __init__(self) -> None:
        self.parameters = {"position_size": 0.1, "max_position_size": 0.2, "short_position_size": 0.08}
        self.position_size = 0.1
        self.max_position_size = 0.2
        self.short_position_size = 0.08


class _KillSwitchRecorder:
    def __init__(self) -> None:
        self.calls = []

    def activate_kill_switch(self, **kwargs):
        self.calls.append(kwargs)


def test_live_trader_scales_strategy_size_on_derisk() -> None:
    trader = LiveTrader(strategy_name="momentum", symbols=["BTC/USD"], parameters={})
    trader.strategy = _DummyStrategy()
    trader._capture_strategy_risk_baselines()

    trader._apply_strategy_position_scale(0.5, reason="test")

    assert trader.strategy.position_size == pytest.approx(0.05)
    assert trader.strategy.short_position_size == pytest.approx(0.04)
    assert trader.strategy.max_position_size == pytest.approx(0.1)


def test_live_trader_halts_on_hard_drawdown() -> None:
    trader = LiveTrader(
        strategy_name="momentum",
        symbols=["BTC/USD"],
        parameters={
            "drawdown_soft_limit_pct": 0.03,
            "drawdown_soft_scale": 0.8,
            "drawdown_medium_limit_pct": 0.05,
            "drawdown_medium_scale": 0.6,
            "drawdown_hard_limit_pct": 0.07,
            "position_size": 0.1,
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "max_daily_loss": 0.03,
        },
    )
    trader.order_gateway = _KillSwitchRecorder()
    trader.running = True
    trader._peak_equity = 100_000

    trader._apply_drawdown_controls(92_000)

    assert trader.running is False
    assert trader.shutdown_event.is_set()
    assert trader.order_gateway.calls
