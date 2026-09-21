"""Regression tests for strategy gateway routing and constructor compatibility."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engine.order_submission import OrderIntent


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
