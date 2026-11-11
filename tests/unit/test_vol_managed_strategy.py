"""VolManagedStrategy: weight = min(1, target_vol / realised vol), banded resizing, no margin."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.session import PortfolioView
from strategies.vol_managed_strategy import VolManagedStrategy

SYMS = ["AAA", "BBB"]


def _hist(daily_vol: float, n: int = 60) -> pd.DataFrame:
    """Prices whose daily returns alternate +/-daily_vol (std == daily_vol, exactly)."""
    r = np.where(np.arange(n) % 2 == 0, daily_vol, -daily_vol)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"close": 100 * np.cumprod(1 + r)}, index=idx)


def _view(cash=100_000.0, equity=100_000.0, positions=None, price=100.0):
    async def ask(_symbol):
        return price

    return PortfolioView(equity=equity, cash=cash, positions=positions or {}, _ask=ask)


def _strategy(**extra):
    return VolManagedStrategy(parameters={"symbols": SYMS, **extra})


async def test_weight_is_target_over_realised_vol_capped_at_one():
    s = _strategy()
    calm = 0.10 / np.sqrt(252)  # 10% annualised -> capped at 1
    wild = 0.30 / np.sqrt(252)  # 30% annualised -> 0.15 / 0.30
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(calm), "BBB": _hist(wild)})
    assert s.weights["AAA"] == 1.0
    assert s.weights["BBB"] == pytest.approx(0.5, rel=0.05)


async def test_short_history_is_no_weight_and_retries():
    s = _strategy()
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(0.01, n=10)})
    assert s.weights["AAA"] is None
    assert s._rebalance_today is False
    assert await s.decide("AAA", datetime(2024, 3, 1), _view()) == []


async def test_rebalances_once_per_month():
    s = _strategy()
    h = {"AAA": _hist(0.01), "BBB": _hist(0.01)}
    await s.prepare(datetime(2024, 3, 1), h)
    assert s._rebalance_today
    await s.prepare(datetime(2024, 3, 4), h)
    assert not s._rebalance_today
    await s.prepare(datetime(2024, 4, 1), h)
    assert s._rebalance_today


async def test_buy_sizes_slot_times_weight_and_never_uses_margin():
    s = _strategy()
    wild = 0.30 / np.sqrt(252)
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(wild), "BBB": _hist(wild)})
    (intent,) = await s.decide("AAA", datetime(2024, 3, 1), _view())
    expected = int(100_000 / 2 * s.weights["AAA"] / 100)
    assert (intent.side, intent.qty) == ("buy", expected)
    s2 = _strategy()
    await s2.prepare(datetime(2024, 3, 1), {"AAA": _hist(wild), "BBB": _hist(wild)})
    (intent,) = await s2.decide("AAA", datetime(2024, 3, 1), _view(cash=5_000))
    assert intent.qty == int(5_000 * 0.995 / 100)


async def test_reductions_for_all_symbols_go_out_first():
    s = _strategy()
    calm = 0.10 / np.sqrt(252)
    wild = 0.60 / np.sqrt(252)  # weight 0.25 -> target 125 shares of a 500-share slot
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(calm), "BBB": _hist(wild)})
    held = {"BBB": SimpleNamespace(symbol="BBB", qty=500)}
    intents = await s.decide("AAA", datetime(2024, 3, 1), _view(positions=held))
    assert len(intents) == 1
    (cut,) = intents
    assert (cut.symbol, cut.side, cut.is_exit) == ("BBB", "sell", True)
    assert cut.qty == 500 - int(100_000 / 2 * s.weights["BBB"] / 100)
    # next call (portfolio refreshed by the session) buys AAA, no second reduction
    intents = await s.decide("AAA", datetime(2024, 3, 1), _view())
    assert [(i.symbol, i.side) for i in intents] == [("AAA", "buy")]


async def test_small_target_moves_inside_the_band_do_not_trade():
    s = _strategy()
    calm = 0.10 / np.sqrt(252)  # weight 1 -> target 500 shares
    await s.prepare(datetime(2024, 3, 1), {"AAA": _hist(calm), "BBB": _hist(calm)})
    held = {"AAA": SimpleNamespace(symbol="AAA", qty=450)}  # 50 short < 25% band (125)
    assert await s.decide("AAA", datetime(2024, 3, 1), _view(positions=held)) == []


def test_unknown_parameter_raises():
    with pytest.raises(ValueError):
        VolManagedStrategy(parameters={"vol_window": 10})
