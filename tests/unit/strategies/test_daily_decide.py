"""Strategies decide, the session submits: the daily execution semantics the
2020-2024 baseline runs on, pinned through the decider interface.

Enter when flat; an opposite signal while holding closes the position (no
stop-and-reverse); integer shares; equity-based sizing by default with the
buy quantity capped by cash.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from brokers.protocol import Position
from engine.session import PortfolioView
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.momentum_strategy_backtest import MomentumStrategyBacktest
from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

WHEN = datetime(2024, 3, 1)


def _view(*, cash=100_000.0, equity=100_000.0, price=100.0, positions=()):
    return PortfolioView(
        equity=equity,
        cash=cash,
        positions={p.symbol: p for p in positions},
        _ask=AsyncMock(return_value=price),
    )


def _momentum(signal, **params):
    strategy = MomentumStrategyBacktest(broker=AsyncMock(), parameters=params)
    strategy.signals = {"SPY": signal}
    return strategy


def _long(qty=100):
    return Position("SPY", float(qty), 90.0)


def _short(qty=100):
    return Position("SPY", -float(qty), 110.0)


# --- sizing ---------------------------------------------------------------


async def test_default_sizing_uses_equity_not_cash():
    (intent,) = await _momentum("buy").decide("SPY", WHEN, _view(cash=50_000, equity=100_000))
    assert intent.side == "buy" and not intent.is_exit
    assert intent.qty == 100  # 10% of equity / 100


async def test_position_size_pct_scales_equity_sizing():
    (intent,) = await _momentum("buy", position_size_pct=0.25).decide("SPY", WHEN, _view())
    assert intent.qty == 250


async def test_buy_quantity_is_capped_by_available_cash():
    (intent,) = await _momentum("buy", position_size_pct=0.5).decide(
        "SPY", WHEN, _view(cash=1_000, equity=100_000)
    )
    assert intent.qty == 10


async def test_legacy_cash_sizing_basis_still_available():
    (intent,) = await _momentum("buy", sizing_basis="cash").decide(
        "SPY", WHEN, _view(cash=50_000, equity=100_000)
    )
    assert intent.qty == 50


async def test_too_small_for_one_share_means_no_intent():
    assert await _momentum("buy").decide("SPY", WHEN, _view(cash=5, equity=5)) == []


# --- entries and opposite-signal exits ---------------------------------------


@pytest.mark.parametrize(
    "signal, positions, expected",
    [
        ("buy", (), ("buy", False, 100)),
        ("short", (), ("sell", False, 100)),
        ("short", (_long(100),), ("sell", True, 100)),
        ("buy", (_short(40),), ("buy", True, 40)),
        ("sell", (_long(70),), ("sell", True, 70)),
    ],
)
async def test_entries_and_exits(signal, positions, expected):
    (intent,) = await _momentum(signal).decide("SPY", WHEN, _view(positions=positions))
    assert (intent.side, intent.is_exit, intent.qty) == expected
    assert intent.symbol == "SPY"


@pytest.mark.parametrize(
    "signal, positions",
    [("buy", (_long(),)), ("short", (_short(),)), ("neutral", ()), ("hold", (_long(),))],
)
async def test_same_direction_or_no_signal_does_nothing(signal, positions):
    assert await _momentum(signal).decide("SPY", WHEN, _view(positions=positions)) == []


async def test_dict_signals_are_read_by_action():
    (intent,) = await _momentum({"action": "buy"}).decide("SPY", WHEN, _view())
    assert intent.side == "buy"


# --- the other deciders ------------------------------------------------------


async def test_plain_momentum_shares_the_daily_semantics():
    strategy = MomentumStrategy(broker=AsyncMock(), parameters={})
    strategy.signals = {"SPY": "buy"}
    (intent,) = await strategy.decide("SPY", WHEN, _view())
    assert intent.qty == 100 and intent.reason == "momentum_backtest_entry"


async def test_mean_reversion_trades_its_signal():
    strategy = MeanReversionStrategy(broker=AsyncMock(), parameters={})
    strategy.signals = {"SPY": "short"}
    (intent,) = await strategy.decide("SPY", WHEN, _view(positions=(_long(30),)))
    assert intent.is_exit and intent.side == "sell" and intent.qty == 30


async def test_simple_ma_buys_a_fifth_of_cash_and_sells_the_position():
    strategy = SimpleMACrossoverStrategy(broker=AsyncMock())
    strategy.signals = {"SPY": "buy"}
    (buy,) = await strategy.decide("SPY", WHEN, _view(cash=10_000, price=50.0))
    assert buy.qty == 40 and not buy.is_exit
    strategy.signals = {"SPY": "sell"}
    (sell,) = await strategy.decide("SPY", WHEN, _view(positions=(_long(40),)))
    assert sell.qty == 40 and sell.is_exit
    strategy.signals = {"SPY": "sell"}
    assert await strategy.decide("SPY", WHEN, _view()) == []
