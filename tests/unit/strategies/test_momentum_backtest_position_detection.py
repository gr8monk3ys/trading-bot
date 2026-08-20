"""Regression tests for MomentumStrategyBacktest position detection.

The backtest broker returns position *objects* (MockPosition), not dicts.
A precedence bug in execute_trade's position scan —

    getattr(pos, "symbol", None) or pos.get("symbol")
    if isinstance(pos, dict)
    else None

parses as ``(...) if isinstance(pos, dict) else None`` — made pos_symbol
always None for object positions, so the strategy could never see its own
book: buys re-entered held symbols, shorts fired against existing longs,
and the signal-driven sell branch was unreachable. These tests pin the
corrected behavior with object positions, dict positions, and an empty book.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from strategies.momentum_strategy_backtest import MomentumStrategyBacktest


def _strategy_with_positions(positions):
    broker = AsyncMock()
    broker.get_all_positions = AsyncMock(return_value=positions)
    broker.get_account = AsyncMock(return_value=SimpleNamespace(cash=100_000.0))
    broker.get_latest_quote = AsyncMock(return_value=SimpleNamespace(ask_price=100.0))
    strategy = MomentumStrategyBacktest(broker=broker, parameters={})
    strategy._place_backtest_order = AsyncMock()
    return strategy


async def test_buy_signal_does_not_reenter_symbol_held_as_object_position():
    strategy = _strategy_with_positions([SimpleNamespace(symbol="SPY", qty=100, quantity=100)])

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_not_awaited()


async def test_sell_signal_closes_symbol_held_as_object_position():
    strategy = _strategy_with_positions([SimpleNamespace(symbol="SPY", qty=100, quantity=100)])

    await strategy.execute_trade("SPY", "sell")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "sell", is_exit=True)


async def test_short_signal_against_existing_long_exits_instead_of_shorting():
    # The original intent of this test stands — a bearish signal must never
    # stack a naked short on top of a held long. Since the 2026-08 exit fix
    # the correct response is to close the long (opposite-signal exit).
    strategy = _strategy_with_positions([SimpleNamespace(symbol="SPY", qty=100, quantity=100)])

    await strategy.execute_trade("SPY", "short")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "sell", is_exit=True)


async def test_buy_signal_enters_when_book_is_empty():
    strategy = _strategy_with_positions([])

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_awaited_once_with("SPY", 100, "buy", is_exit=False)


async def test_dict_positions_still_detected():
    strategy = _strategy_with_positions([{"symbol": "SPY", "qty": 100, "quantity": 100}])

    await strategy.execute_trade("SPY", "buy")

    strategy._place_backtest_order.assert_not_awaited()
