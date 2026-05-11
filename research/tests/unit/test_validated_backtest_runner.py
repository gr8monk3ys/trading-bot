import pandas as pd
import pytest

from engine.validated_backtest import ValidatedBacktestRunner


@pytest.mark.asyncio
async def test_run_validated_backtest_without_walk_forward(monkeypatch):
    runner = ValidatedBacktestRunner(broker=None, walk_forward_enabled=False)

    async def _fake_run_backtest(*_args, **_kwargs):
        return {
            "total_return": 0.1,
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.05,
            "num_trades": 25,
            "win_rate": 0.6,
            "equity_curve": [100, 110],
            "daily_returns": pd.Series([0.1, 0.0]),
        }

    monkeypatch.setattr(runner, "_run_backtest", _fake_run_backtest)
    monkeypatch.setattr(runner, "_calculate_significance", lambda _r: (False, None))
    # Patch helper to avoid numpy/scipy work in tests
    monkeypatch.setattr(runner, "_calculate_permutation_tests", lambda _r: {"error": "n/a"})

    result = await runner.run_validated_backtest(
        strategy_class=object,
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-02-01",
    )

    assert result.walk_forward_validated is False
    assert result.total_return > 0
    assert result.validation_gates
