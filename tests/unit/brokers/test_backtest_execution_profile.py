#!/usr/bin/env python3
"""
Unit tests for BacktestBroker execution realism profiles.
"""

from datetime import datetime

import pandas as pd

from brokers.backtest_broker import BacktestBroker, ExecutionProfile


def _price_data() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.Series(range(100, 130), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": prices - 0.2,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def test_order_includes_latency_and_profile_metadata():
    broker = BacktestBroker(execution_profile="realistic", random_seed=42)
    data = _price_data()
    broker.set_price_data("AAPL", data)
    broker._current_date = data.index[-1]

    order = broker.place_order("AAPL", 10, "buy")

    assert "latency_ms" in order
    assert "execution_profile" in order
    assert order["execution_profile"] == "realistic"
    assert order["status"] in {"filled", "partially_filled"}


def test_stressed_profile_has_higher_slippage_than_idealistic():
    data = _price_data()
    check_date = data.index[-1]

    ideal = BacktestBroker(random_seed=7)
    stressed = BacktestBroker(random_seed=7)
    ideal.set_execution_profile(
        ExecutionProfile(
            name="idealistic_custom",
            slippage_multiplier=0.7,
            partial_fill_multiplier=1.0,
            min_latency_ms=1,
            max_latency_ms=1,
            reject_probability=0.0,
        )
    )
    stressed.set_execution_profile(
        ExecutionProfile(
            name="stressed_custom",
            slippage_multiplier=1.6,
            partial_fill_multiplier=1.0,
            min_latency_ms=1,
            max_latency_ms=1,
            reject_probability=0.0,
        )
    )

    ideal.set_price_data("AAPL", data)
    stressed.set_price_data("AAPL", data)
    ideal._current_date = check_date
    stressed._current_date = check_date

    ideal_order = ideal.place_order("AAPL", 100, "buy")
    stressed_order = stressed.place_order("AAPL", 100, "buy")

    assert stressed_order["slippage_bps"] > ideal_order["slippage_bps"]


def test_reject_probability_can_simulate_liquidity_reject():
    broker = BacktestBroker(random_seed=123)
    broker.set_execution_profile(
        ExecutionProfile(
            name="always_reject",
            slippage_multiplier=1.0,
            partial_fill_multiplier=1.0,
            min_latency_ms=5,
            max_latency_ms=5,
            reject_probability=1.0,
        )
    )
    data = _price_data()
    broker.set_price_data("AAPL", data)
    broker._current_date = datetime(2024, 2, 9)

    order = broker.place_order("AAPL", 10, "buy")

    assert order["status"] == "rejected"
    assert order["filled_qty"] == 0
    assert broker.get_positions() == []
    assert broker.get_trades() == []
