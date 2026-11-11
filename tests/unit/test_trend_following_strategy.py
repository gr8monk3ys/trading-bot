"""TrendFollowingStrategy: monthly signal, exits before entries, slot sizing, no margin."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.session import PortfolioView
from strategies.trend_following_strategy import TrendFollowingStrategy

SYMS = ["AAA", "BBB"]


def _hist(start: float, end: float, n: int = 300) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"close": np.linspace(start, end, n)}, index=idx)


def _view(cash=100_000.0, equity=100_000.0, positions=None, price=100.0):
    async def ask(_symbol):
        return price

    return PortfolioView(equity=equity, cash=cash, positions=positions or {}, _ask=ask)


def _strategy():
    return TrendFollowingStrategy(parameters={"symbols": SYMS})


async def test_signal_is_sign_of_12_month_return():
    s = _strategy()
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(50, 100), "BBB": _hist(100, 50)})
    assert s.signals == {"AAA": True, "BBB": False}


async def test_short_history_is_no_signal_and_retries():
    s = _strategy()
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(50, 100, n=100)})
    assert s.signals["AAA"] is None
    assert s._rebalance_today is False
    assert await s.decide("AAA", datetime(2024, 3, 1), _view()) == []


async def test_rebalances_once_per_month():
    s = _strategy()
    h = {"AAA": _hist(50, 100), "BBB": _hist(50, 100)}
    await s.prepare(datetime(2024, 3, 1), h)
    assert s._rebalance_today
    await s.prepare(datetime(2024, 3, 4), h)
    assert not s._rebalance_today
    await s.prepare(datetime(2024, 4, 1), h)
    assert s._rebalance_today


async def test_buy_sizes_one_slot_and_never_uses_margin():
    s = _strategy()
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(50, 100), "BBB": _hist(50, 100)})
    (intent,) = await s.decide("AAA", datetime(2024, 3, 1), _view())
    assert (intent.side, intent.qty) == ("buy", 500)  # 100k / 2 slots / $100
    s2 = _strategy()
    await s2.prepare(datetime(2024, 3, 1), {"AAA": _hist(50, 100), "BBB": _hist(50, 100)})
    (intent,) = await s2.decide("AAA", datetime(2024, 3, 1), _view(cash=10_000))
    assert intent.qty == int(10_000 * 0.995 / 100)


async def test_exits_for_all_symbols_go_out_first():
    s = _strategy()
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(50, 100), "BBB": _hist(100, 50)})
    held = {"BBB": SimpleNamespace(symbol="BBB", qty=10)}
    intents = await s.decide("AAA", datetime(2024, 3, 1), _view(positions=held))
    assert [(i.symbol, i.side, i.qty, i.is_exit) for i in intents] == [("BBB", "sell", 10, True)]
    # next call (portfolio refreshed by the session) buys AAA, no second exit
    intents = await s.decide("AAA", datetime(2024, 3, 1), _view())
    assert [(i.symbol, i.side) for i in intents] == [("AAA", "buy")]


async def test_holds_existing_position_without_resizing():
    s = _strategy()
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(50, 100), "BBB": _hist(50, 100)})
    held = {"AAA": SimpleNamespace(symbol="AAA", qty=3)}
    assert await s.decide("AAA", datetime(2024, 3, 1), _view(positions=held)) == []


def test_unknown_parameter_raises():
    with pytest.raises(ValueError):
        TrendFollowingStrategy(parameters={"lookback": 100})
