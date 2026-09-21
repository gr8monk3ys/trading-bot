"""Regression tests for strategy gateway routing and constructor compatibility."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engine.order_submission import OrderIntent


class _FakeOrderBuilder:
    """Minimal order builder stub used to avoid external dependency details in tests."""

    def __init__(self, symbol: str, side: str, qty: int):
        self.symbol = symbol
        self.side = side
        self.qty = qty

    def market(self):
        return self

    def day(self):
        return self

    def build(self):
        return SimpleNamespace(
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            type="market",
        )


class _ConcreteBaseStrategy:
    """Factory mixin for lightweight BaseStrategy concrete classes in tests."""

    @staticmethod
    def build():
        from strategies.base_strategy import BaseStrategy

        class _TestStrategy(BaseStrategy):
            async def analyze_symbol(self, symbol):
                return {"action": "hold"}

            async def execute_trade(self, symbol, signal):
                return None

        return _TestStrategy


@pytest.mark.asyncio
async def test_momentum_backtest_entry_order_uses_strategy_gateway_helpers():
    """Backtest entry path should not call broker.submit_order_advanced directly."""
    from strategies import momentum_strategy_backtest as module

    broker = AsyncMock()
    broker.submit_order_advanced = AsyncMock()
    strategy = module.MomentumStrategyBacktest(broker=broker, parameters={})
    strategy.submit_entry_order = AsyncMock(return_value=SimpleNamespace(success=True))
    strategy.submit_exit_order = AsyncMock()

    await strategy._place_backtest_order("AAPL", 10, "buy", is_exit=False)

    strategy.submit_entry_order.assert_awaited_once()
    strategy.submit_exit_order.assert_not_awaited()
    broker.submit_order_advanced.assert_not_awaited()

    intent = strategy.submit_entry_order.await_args.args[0]
    assert intent.reason == "momentum_backtest_entry"
    assert intent.symbol == "AAPL" and intent.side == "buy" and intent.qty == 10


@pytest.mark.asyncio
async def test_momentum_backtest_exit_order_uses_submit_exit_order():
    """Backtest exit path should route through submit_exit_order for safety checks."""
    from strategies import momentum_strategy_backtest as module

    strategy = module.MomentumStrategyBacktest(broker=AsyncMock(), parameters={})
    strategy.submit_entry_order = AsyncMock()
    strategy.submit_exit_order = AsyncMock(return_value=SimpleNamespace(success=True))

    await strategy._place_backtest_order("MSFT", 7, "sell", is_exit=True)

    strategy.submit_exit_order.assert_awaited_once_with(
        symbol="MSFT",
        qty=7,
        side="sell",
        reason="backtest_exit",
    )
    strategy.submit_entry_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_strategy_entry_order_blocks_when_gateway_missing():
    """Entry orders must fail closed if no gateway is configured."""
    TestStrategy = _ConcreteBaseStrategy.build()

    broker = AsyncMock()
    broker.submit_order_advanced = AsyncMock(return_value=SimpleNamespace(id="direct-order"))

    strategy = TestStrategy(broker=broker, parameters={})
    result = await strategy.submit_entry_order(
        OrderIntent(symbol="AAPL", side="buy", qty=1, reason="test-entry")
    )

    assert result is None
    broker.submit_order_advanced.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_strategy_exit_order_blocks_when_gateway_missing():
    """Exit orders must fail closed if no gateway is configured."""
    TestStrategy = _ConcreteBaseStrategy.build()

    broker = AsyncMock()
    broker.get_positions.return_value = [SimpleNamespace(symbol="AAPL", qty="5")]
    broker.submit_order_advanced = AsyncMock(return_value=SimpleNamespace(id="direct-order"))

    strategy = TestStrategy(broker=broker, parameters={})

    result = await strategy.submit_exit_order(symbol="AAPL", qty=3, side="sell", reason="test-exit")

    assert result is None
    broker.submit_order_advanced.assert_not_awaited()
