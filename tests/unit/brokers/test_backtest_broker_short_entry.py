"""Regression tests: a sell with no existing position must open a short.

place_order's sell path credited cash and then only *updated* an existing
position — a naked short booked no liability at all. Short signals were
literally free money: the 2020-2024 reruns credited ~$83k of phantom cash
from two IWM shorts that never had to be bought back.
"""

import numpy as np
import pandas as pd
import pytest

from brokers.backtest_broker import BacktestBroker


@pytest.fixture
def broker():
    b = BacktestBroker(initial_balance=100_000, enable_partial_fills=False)
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
    b.set_price_data("SPY", frame)
    b._current_date = dates[5]
    return b


def test_naked_sell_opens_short_position(broker):
    broker.place_order("SPY", 100, "sell", order_type="market")

    assert "SPY" in broker.positions
    assert broker.positions["SPY"]["quantity"] == -100


def test_naked_sell_does_not_create_free_equity(broker):
    start_value = broker.get_portfolio_value()

    broker.place_order("SPY", 100, "sell", order_type="market")

    # Cash rises by proceeds but the short liability offsets it: portfolio
    # value must not jump by the sale proceeds (only execution costs move it).
    end_value = broker.get_portfolio_value()
    assert end_value == pytest.approx(start_value, rel=0.01)
    assert end_value < start_value + 5_000


def test_short_then_cover_returns_to_flat(broker):
    broker.place_order("SPY", 100, "sell", order_type="market")
    broker.place_order("SPY", 100, "buy", order_type="market")

    assert "SPY" not in broker.positions
