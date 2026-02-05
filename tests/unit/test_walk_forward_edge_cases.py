import asyncio
from datetime import datetime, timedelta

import pytest

from engine.walk_forward import WalkForwardValidator


def test_create_time_splits_raises_on_short_range():
    validator = WalkForwardValidator(min_train_days=30, n_splits=3)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 2, 1)

    with pytest.raises(ValueError):
        validator.create_time_splits(start, end)


def test_overfit_ratio_edge_cases_in_validate():
    async def _backtest_fn(_symbols, _start, _end, **_kwargs):
        return {"total_return": 0.1, "sharpe_ratio": 1.0, "num_trades": 10}

    validator = WalkForwardValidator(n_splits=2, min_train_days=10, gap_days=1)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 1)
    splits = validator.create_time_splits(start, end)

    async def run():
        # Force OOS negative on second call by wrapping
        calls = {"count": 0}

        async def backtest_fn(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] % 2 == 0:
                return {"total_return": -0.05, "sharpe_ratio": -0.2, "num_trades": 5}
            return {"total_return": 0.1, "sharpe_ratio": 1.0, "num_trades": 10}

        result = await validator.validate(
            backtest_fn,
            symbols=["AAPL"],
            start_date_str=splits[0][0].strftime("%Y-%m-%d"),
            end_date_str=splits[-1][3].strftime("%Y-%m-%d"),
        )
        return result

    result = asyncio.run(run())

    assert "avg_overfit_ratio" in result
    assert result["n_folds"] >= 1
