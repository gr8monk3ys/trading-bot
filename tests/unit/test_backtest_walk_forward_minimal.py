import pytest

from datetime import datetime

from engine.backtest_engine import BacktestEngine


@pytest.mark.asyncio
async def test_run_walk_forward_backtest_no_folds(monkeypatch):
    engine = BacktestEngine()

    async def _fake_run_backtest(*_args, **_kwargs):
        return {"equity_curve": [100000, 100000], "total_trades": 0}

    monkeypatch.setattr(engine, "run_backtest", _fake_run_backtest)

    result = await engine.run_walk_forward_backtest(
        strategy_class=object,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
        n_folds=3,
        train_pct=0.7,
        embargo_days=5,
    )

    assert result["fold_results"] == []
    assert bool(result["overfit_detected"]) is True


@pytest.mark.asyncio
async def test_run_walk_forward_backtest_single_fold(monkeypatch):
    engine = BacktestEngine()

    async def _fake_run_backtest(*_args, **_kwargs):
        return {
            "equity_curve": [100000, 105000],
            "total_trades": 5,
        }

    monkeypatch.setattr(engine, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(engine, "_calculate_sharpe_from_equity", lambda _eq: 1.0)

    result = await engine.run_walk_forward_backtest(
        strategy_class=object,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 4, 30),
        n_folds=1,
        train_pct=0.6,
        embargo_days=2,
    )

    assert result["fold_results"]
    assert result["n_folds"] == 1
    assert result["is_sharpe"] >= 0
    assert result["overfit_detected"] == False


@pytest.mark.asyncio
async def test_run_walk_forward_backtest_overfit_detected(monkeypatch):
    engine = BacktestEngine()

    async def _fake_run_backtest(*_args, **_kwargs):
        return {
            "equity_curve": [100000, 101000],
            "total_trades": 5,
        }

    # First call (IS) -> high sharpe, second call (OOS) -> low sharpe
    calls = {"count": 0}

    def _fake_sharpe(_eq):
        calls["count"] += 1
        return 2.0 if calls["count"] == 1 else 0.5

    monkeypatch.setattr(engine, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(engine, "_calculate_sharpe_from_equity", _fake_sharpe)

    result = await engine.run_walk_forward_backtest(
        strategy_class=object,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 4, 30),
        n_folds=1,
        train_pct=0.6,
        embargo_days=2,
    )

    assert bool(result["overfit_detected"]) is True
