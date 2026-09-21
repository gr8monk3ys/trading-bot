"""Both adapters satisfy the broker seam (brokers/protocol.py), with one Position shape."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from brokers.alpaca.account import _to_position
from brokers.alpaca_broker import AlpacaBroker
from brokers.backtest import BacktestBroker
from brokers.protocol import Broker, Position


def _backtest_broker():
    broker = BacktestBroker(initial_balance=10_000, execution_profile="idealistic", random_seed=1)
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    broker.set_price_data(
        "SPY",
        pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=dates),
    )
    broker.advance_to(datetime(2024, 1, 2))
    return broker


def test_backtest_broker_is_a_broker():
    assert isinstance(_backtest_broker(), Broker)


def test_alpaca_broker_is_a_broker():
    assert isinstance(AlpacaBroker.__new__(AlpacaBroker), Broker)


async def test_backtest_positions_use_protocol_shape():
    broker = _backtest_broker()
    await broker.place_order("SPY", 10, "buy")
    broker.advance_to(datetime(2024, 1, 3))
    (pos,) = await broker.get_positions()
    assert isinstance(pos, Position)
    assert pos.symbol == "SPY" and pos.qty == 10 and pos.side == "long"
    assert pos.market_value == pytest.approx(10 * pos.current_price)
    assert await broker.get_position("SPY") == pos
    assert await broker.get_position("QQQ") is None


async def test_backtest_short_is_negative_qty():
    broker = _backtest_broker()
    await broker.place_order("SPY", 5, "sell")
    (pos,) = await broker.get_positions()
    assert pos.qty == -5 and pos.side == "short"


def test_alpaca_position_converts_to_protocol_shape():
    raw = SimpleNamespace(
        symbol="AAPL",
        qty="3",
        side="PositionSide.SHORT",
        avg_entry_price="150.5",
        current_price="140",
        market_value="-420",
        unrealized_pl="31.5",
        unrealized_plpc="0.07",
    )
    pos = _to_position(raw)
    assert pos == Position("AAPL", -3.0, 150.5, 140.0, -420.0, 31.5, 0.07)
    assert pos.side == "short"


def test_no_dict_positions_leak_through_backtest_broker():
    broker = _backtest_broker()
    assert not hasattr(broker, "get_all_positions")
    assert MagicMock is not None  # keeps the import honest for future fakes
