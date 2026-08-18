"""Tests for BacktestBroker.get_gross_exposure.

The 2020-2024 baselines were run with cash-based sizing that left the
strategy ~75% idle, and nothing measured it: no exposure series existed
anywhere in the engine. Gross exposure (sum of |position| notional divided
by portfolio value) is the denominator every performance comparison against
buy-and-hold needs.
"""

import numpy as np
import pandas as pd
import pytest

from brokers.backtest_broker import BacktestBroker


@pytest.fixture
def broker_with_prices():
    broker = BacktestBroker(initial_balance=100_000)
    dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
    frame = pd.DataFrame(
        {
            "open": np.full(10, 100.0),
            "high": np.full(10, 101.0),
            "low": np.full(10, 99.0),
            "close": np.full(10, 100.0),
            "volume": np.full(10, 1_000_000),
        },
        index=dates,
    )
    broker.set_price_data("SPY", frame)
    broker._current_date = dates[5]
    return broker, dates[5]


def test_flat_book_has_zero_exposure(broker_with_prices):
    broker, date = broker_with_prices

    assert broker.get_gross_exposure(date) == 0.0


def test_long_position_exposure_is_notional_over_portfolio_value(broker_with_prices):
    broker, date = broker_with_prices
    broker.balance = 90_000
    broker.positions["SPY"] = {"symbol": "SPY", "quantity": 100, "avg_price": 100.0}

    # Notional 100 * $100 = $10k; portfolio value = 90k cash + 10k = 100k.
    assert broker.get_gross_exposure(date) == pytest.approx(0.10)


def test_short_position_counts_absolute_notional(broker_with_prices):
    broker, date = broker_with_prices
    broker.balance = 110_000
    broker.positions["SPY"] = {"symbol": "SPY", "quantity": -100, "avg_price": 100.0}

    # |−100| * $100 = $10k gross; portfolio value = 110k − 10k = 100k.
    assert broker.get_gross_exposure(date) == pytest.approx(0.10)
