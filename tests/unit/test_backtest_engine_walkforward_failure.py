from datetime import datetime

import pytest

from engine.backtest_engine import BacktestEngine


@pytest.mark.asyncio
async def test_run_walk_forward_backtest_skips_invalid_folds(monkeypatch):
    engine = BacktestEngine()

    async def _fake_run_backtest(*_args, **_kwargs):
        return {"equity_curve": [100000, 100000], "total_trades": 0}

    monkeypatch.setattr(engine, "run_backtest", _fake_run_backtest)

    # Short range -> folds are skipped
    result = await engine.run_walk_forward_backtest(
        strategy_class=object,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 2, 1),
        n_folds=3,
        train_pct=0.9,
        embargo_days=5,
    )

    assert result["fold_results"] == []
    assert result["overfit_detected"] is True
