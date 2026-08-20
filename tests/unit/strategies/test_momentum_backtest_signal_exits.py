"""Tests for MomentumStrategyBacktest opposite-signal exits.

The momentum signal generator emits only "buy" / "short" / "neutral" — never
"sell" — and execute_trade's decision table only opened positions when flat.
Net effect measured in the 2020-2024 ETF baseline: once entered, a symbol
could never be exited or re-entered, so every run bottomed out at 4 entries +
4 forced end-of-backtest liquidations ("8 trades in 5 years"), and the
published verdict was rendered on enter-once-and-hold, not on the strategy.

These tests pin the fix: an opposite signal while holding closes the
position (long + "short" -> exit sell; short + "buy" -> exit cover), and
entries while flat keep working exactly as before.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from strategies.momentum_strategy_backtest import MomentumStrategyBacktest


def _strategy(*, position=None, cash=100_000, equity=100_000, price=100.0):
    broker = AsyncMock()
    positions = [position] if position is not None else []
    broker.get_all_positions = AsyncMock(return_value=positions)
    broker.get_account = AsyncMock(return_value=SimpleNamespace(cash=cash, equity=equity))
    broker.get_latest_quote = AsyncMock(return_value=SimpleNamespace(ask_price=price))
    strategy = MomentumStrategyBacktest(broker=broker, parameters={})
    strategy._place_backtest_order = AsyncMock()
    return strategy


def _long(symbol="SPY", qty=100):
    return SimpleNamespace(symbol=symbol, quantity=qty, qty=str(qty), entry_price=90.0)


def _short(symbol="SPY", qty=-100):
    return SimpleNamespace(symbol=symbol, quantity=qty, qty=str(qty), entry_price=110.0)


async def test_short_signal_while_long_exits_the_long():
    strategy = _strategy(position=_long(qty=100))

    await strategy.execute_trade("SPY", "short")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "sell", is_exit=True)


async def test_buy_signal_while_short_covers_the_short():
    strategy = _strategy(position=_short(qty=-100))

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "buy", is_exit=True)


async def test_buy_signal_while_flat_still_opens_long():
    strategy = _strategy(position=None)

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "buy", is_exit=False)


async def test_short_signal_while_flat_still_opens_short():
    strategy = _strategy(position=None)

    await strategy.execute_trade("SPY", "short")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "sell", is_exit=False)


async def test_buy_signal_while_long_does_nothing():
    strategy = _strategy(position=_long(qty=100))

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_not_awaited()


async def test_short_signal_while_short_does_nothing():
    strategy = _strategy(position=_short(qty=-100))

    await strategy.execute_trade("SPY", "short")

    strategy._place_backtest_order.assert_not_awaited()


async def test_dict_position_shape_also_exits():
    strategy = _strategy(position={"symbol": "SPY", "quantity": 40, "entry_price": 90.0})

    await strategy.execute_trade("SPY", "short")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 40, "sell", is_exit=True)
