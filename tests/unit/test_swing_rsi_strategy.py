"""SwingRSIStrategy: buy RSI(2) dips above the 200-day average, sell above the 5-day average."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.session import PortfolioView
from strategies.swing_rsi_strategy import SwingRSIStrategy, wilder_rsi

SYMS = ["AAA", "BBB"]
WHEN = datetime(2024, 3, 1)


def _hist(tail: list[float], n: int = 220, start: float = 50.0, end: float = 100.0):
    """A steady uptrend from ``start`` to ``end``, then the given closes."""
    closes = list(np.linspace(start, end, n)) + tail
    idx = pd.bdate_range("2023-01-02", periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _view(cash=100_000.0, equity=100_000.0, positions=None, price=100.0):
    async def ask(_symbol):
        return price

    return PortfolioView(equity=equity, cash=cash, positions=positions or {}, _ask=ask)


def _strategy(**extra):
    return SwingRSIStrategy(parameters={"symbols": SYMS, **extra})


def test_wilder_rsi_extremes():
    assert wilder_rsi(pd.Series([1.0, 2.0, 3.0, 4.0]), 2) == 100.0
    assert wilder_rsi(pd.Series([4.0, 3.0, 2.0, 1.0]), 2) == pytest.approx(0.0)
    assert 0 < wilder_rsi(pd.Series([10.0, 11.0, 10.5, 10.2]), 2) < 50


async def test_sharp_dip_in_uptrend_enters_with_a_slot():
    s = _strategy()
    await s.prepare(WHEN, {"AAA": _hist([97.0, 94.0, 91.0])})
    assert s.signals["AAA"] == "enter"
    (intent,) = await s.decide("AAA", WHEN, _view())
    assert (intent.side, intent.qty) == ("buy", int(100_000 / 2 / 100))


async def test_dip_below_the_200_day_average_does_not_enter():
    s = _strategy()
    # Downtrend: the dip is below the long average, so no entry.
    await s.prepare(WHEN, {"AAA": _hist([60.0, 57.0, 54.0], start=150.0, end=63.0)})
    assert s.signals["AAA"] is None
    assert await s.decide("AAA", WHEN, _view()) == []


async def test_close_above_short_average_exits_whole_position():
    s = _strategy()
    await s.prepare(WHEN, {"AAA": _hist([97.0, 94.0, 99.0])})
    assert s.signals["AAA"] == "exit"
    held = {"AAA": SimpleNamespace(symbol="AAA", qty=300)}
    (intent,) = await s.decide("AAA", WHEN, _view(positions=held))
    assert (intent.side, intent.qty, intent.is_exit) == ("sell", 300, True)


async def test_no_second_entry_while_held_and_no_margin():
    s = _strategy()
    await s.prepare(WHEN, {"AAA": _hist([97.0, 94.0, 91.0])})
    held = {"AAA": SimpleNamespace(symbol="AAA", qty=100)}
    assert await s.decide("AAA", WHEN, _view(positions=held)) == []
    (intent,) = await s.decide("AAA", WHEN, _view(cash=2_000))
    assert intent.qty == int(2_000 * 0.995 / 100)


async def test_short_history_gives_no_signal():
    s = _strategy()
    await s.prepare(WHEN, {"AAA": _hist([], n=50)})
    assert s.signals["AAA"] is None


def test_unknown_parameter_raises():
    with pytest.raises(ValueError):
        SwingRSIStrategy(parameters={"rsi_len": 3})
