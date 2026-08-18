"""Tests for daily gross-exposure tracking in BacktestEngine.run_backtest.

The 2020-2024 baselines compared strategy returns at ~25% average gross
exposure against SPY buy-and-hold at 100% without measuring either side.
The engine must report the exposure series so every future result can be
read per unit of capital actually deployed.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from engine.backtest_engine import BacktestEngine
from tests.unit.engine.test_backtest_end_liquidation import (
    _bars_by_symbol,
    _BuyAndHoldStrategy,
)


async def test_run_backtest_reports_daily_exposure_series():
    symbols = ["AAA", "BBB"]
    start = datetime(2024, 1, 2)
    n_days = 10
    get_bars, _ = _bars_by_symbol(symbols, start, n=n_days)

    data_broker = MagicMock()
    data_broker.get_bars = AsyncMock(side_effect=get_bars)

    engine = BacktestEngine(broker=data_broker)
    result = await engine.run_backtest(
        strategy_class=_BuyAndHoldStrategy,
        symbols=symbols,
        start_date=start,
        end_date=start + pd.Timedelta(days=n_days + 1),
        initial_capital=100_000,
    )

    exposure = result["exposure_curve"]
    # One sample per recorded trading day (equity_curve also has the
    # initial-capital seed point at index 0).
    assert len(exposure) == len(result["equity_curve"]) - 1
    # Buy-and-hold of 10 shares x 2 symbols at ~$100 is a small but non-zero
    # fraction of $100k, held every day after entry.
    assert all(0.0 <= x <= 1.0 for x in exposure)
    assert max(exposure) > 0.0

    avg = result["avg_gross_exposure"]
    assert 0.0 < avg <= 1.0
    assert abs(avg - sum(exposure) / len(exposure)) < 1e-9
    assert result["peak_gross_exposure"] == max(exposure)
