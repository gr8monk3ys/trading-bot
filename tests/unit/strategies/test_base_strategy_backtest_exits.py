"""BaseStrategy.submit_exit_order must work against BacktestBroker.

BacktestBroker.get_positions() is synchronous (returning raw dicts) while
BaseStrategy.submit_exit_order awaits broker.get_positions() and reads
.symbol/.qty off the results. Under the backtest broker that raised
TypeError inside the method's try/except, was swallowed as a generic
failure, and every signal-driven exit in every backtest silently no-oped
(tests/unit/engine/test_backtest_end_liquidation.py documents the symptom).
These tests pin the fix: exits must route to the gateway under both broker
APIs.
"""

from unittest.mock import AsyncMock, Mock

from brokers.backtest import BacktestBroker
from engine.order_submission import OrderOutcome, OrderStatus
from strategies.base_strategy import BaseStrategy


class _ExitProbeStrategy(BaseStrategy):
    NAME = "exit_probe"

    async def analyze_symbol(self, symbol):
        return None

    async def execute_trade(self, symbol, signal):
        return None


def _strategy_with_position():
    broker = BacktestBroker(initial_balance=100_000)
    broker.positions["SPY"] = {"symbol": "SPY", "quantity": 10, "entry_price": 100.0}
    gateway = Mock()
    gateway.submit = AsyncMock(
        return_value=OrderOutcome(OrderStatus.FILLED, "SPY", "sell", 10, 10, "bt-1")
    )
    strategy = _ExitProbeStrategy(broker=broker, order_submission=gateway)
    return strategy, gateway


async def test_submit_exit_order_reaches_gateway_under_backtest_broker():
    strategy, gateway = _strategy_with_position()

    result = await strategy.submit_exit_order(symbol="SPY", qty=10, reason="signal_exit")

    assert result is not None and result.ok
    gateway.submit.assert_awaited_once()
    assert gateway.submit.call_args.args[0].symbol == "SPY"


async def test_submit_exit_order_still_rejects_unheld_symbol():
    strategy, gateway = _strategy_with_position()

    result = await strategy.submit_exit_order(symbol="QQQ", qty=5, reason="signal_exit")

    assert result is None
    gateway.submit.assert_not_awaited()
