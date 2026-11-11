"""BacktestBroker(fixed_cost_bps=...): a flat, stated cost per market fill (ADR 0013)."""

import pytest

from brokers.backtest import BacktestBroker
from engine.backtest.runner import _mean_fill_cost_bps


def test_fixed_cost_moves_price_against_the_trader():
    b = BacktestBroker(initial_balance=1_000, fixed_cost_bps=50)
    assert b._calculate_slippage("SPY", 10, "buy", 100.0, "market") == pytest.approx(100.5)
    assert b._calculate_slippage("SPY", 10, "sell", 100.0, "market") == pytest.approx(99.5)


def test_zero_fixed_cost_is_free():
    b = BacktestBroker(initial_balance=1_000, fixed_cost_bps=0)
    assert b._calculate_slippage("SPY", 10, "buy", 100.0, "market") == pytest.approx(100.0)


def test_default_is_the_dynamic_model():
    assert BacktestBroker(initial_balance=1_000).fixed_cost_bps is None


def test_mean_fill_cost_is_notional_weighted_and_skips_unfilled():
    orders = [
        {"filled_qty": 10, "price": 100.0, "slippage_bps": 50.0},  # $1,000
        {"filled_qty": 30, "price": 100.0, "slippage_bps": 10.0},  # $3,000
        {"filled_qty": 0, "price": 100.0, "slippage_bps": 0.0},  # rejected
    ]
    assert _mean_fill_cost_bps(orders) == pytest.approx(20.0)
    assert _mean_fill_cost_bps([]) == 0.0
