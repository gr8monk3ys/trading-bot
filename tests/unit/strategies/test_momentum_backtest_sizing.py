"""Tests for MomentumStrategyBacktest position sizing basis.

Legacy sizing was hardcoded `cash * 0.10`: each entry consumed 10% of
*remaining* cash, so a portfolio that bought its 4-symbol universe peaked
near 25-30% gross exposure and sat mostly idle. Equity-based sizing
(`equity * position_size_pct`) makes target exposure an explicit, sweepable
input instead of an accident of entry order.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from strategies.momentum_strategy_backtest import MomentumStrategyBacktest


def _strategy(parameters, *, cash, equity, price=100.0):
    broker = AsyncMock()
    broker.get_all_positions = AsyncMock(return_value=[])
    broker.get_account = AsyncMock(return_value=SimpleNamespace(cash=cash, equity=equity))
    broker.get_latest_quote = AsyncMock(return_value=SimpleNamespace(ask_price=price))
    strategy = MomentumStrategyBacktest(broker=broker, parameters=parameters)
    strategy._place_backtest_order = AsyncMock()
    return strategy


async def test_default_sizing_uses_equity_not_cash():
    strategy = _strategy({}, cash=10_000, equity=100_000)

    await strategy.execute_trade("SPY", "buy")

    # 10% of $100k equity = $10k -> 100 shares (not 10% of $10k cash = 10).
    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "buy", is_exit=False)


async def test_position_size_pct_parameter_scales_equity_sizing():
    strategy = _strategy({"position_size_pct": 0.25}, cash=100_000, equity=100_000)

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 250, "buy", is_exit=False)


async def test_buy_quantity_is_capped_by_available_cash():
    strategy = _strategy({"position_size_pct": 0.25}, cash=5_000, equity=100_000)

    await strategy.execute_trade("SPY", "buy")

    # Wants $25k but only $5k cash remains -> 50 shares.
    strategy._place_backtest_order.assert_awaited_once_with("SPY", 50, "buy", is_exit=False)


async def test_legacy_cash_sizing_basis_still_available():
    strategy = _strategy({"sizing_basis": "cash"}, cash=10_000, equity=100_000)

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 10, "buy", is_exit=False)
