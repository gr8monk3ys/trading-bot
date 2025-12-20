from datetime import datetime

import pytest

from engine.backtest_engine import BacktestEngine


class _AsyncStrategy:
    def __init__(self):
        self.called = False
        self.current_date = None

    async def on_trading_iteration(self):
        self.called = True


@pytest.mark.asyncio
async def test_run_strategy_iteration_awaits_async_method():
    engine = BacktestEngine()
    strategy = _AsyncStrategy()

    await engine._run_strategy_iteration(strategy, datetime(2024, 1, 2))

    assert strategy.called is True
