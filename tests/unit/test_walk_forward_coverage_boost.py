from __future__ import annotations

import sys
import types
from datetime import datetime

import numpy as np
import pytest

from engine.walk_forward import (
    WalkForwardValidator,
    _coerce_float,
    _coerce_int,
    calculate_sharpe_confidence_interval,
    check_degradation_significance,
    run_walk_forward_validation,
)


def test_coerce_helpers_cover_edge_types():
    assert _coerce_float(None, 1.0) == 1.0
    assert _coerce_float(True, 0.0) == 1.0
    assert _coerce_float("bad", 2.0) == 2.0
    assert _coerce_float(object(), 3.0) == 3.0

    assert _coerce_int(None, 1) == 1
    assert _coerce_int(True, 0) == 1
    assert _coerce_int(3.9, 0) == 3
    assert _coerce_int("bad", 4) == 4
    assert _coerce_int(object(), 5) == 5


def test_check_degradation_significance_wilcoxon_value_error(monkeypatch):
    monkeypatch.setattr(
        "engine.walk_forward.stats.wilcoxon",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("not enough data")),
    )
    result = check_degradation_significance(
        [0.11, 0.10, 0.12, 0.09, 0.13],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        n_bootstrap=20,
        random_state=42,
    )
    assert result.wilcoxon_statistic == 0.0
    assert result.wilcoxon_p_value == 1.0


def test_check_degradation_significance_negligible_and_medium_effects(monkeypatch):
    monkeypatch.setattr("engine.walk_forward.stats.wilcoxon", lambda *_args, **_kwargs: (7.5, 0.5))
    negligible = check_degradation_significance(
        [0.10, 0.10, 0.10, 0.10, 0.10],
        [0.09, 0.09, 0.09, 0.09, 0.09],
        n_bootstrap=20,
        random_state=42,
    )
    assert negligible.effect_magnitude == "negligible"

    monkeypatch.setattr("engine.walk_forward.stats.wilcoxon", lambda *_args, **_kwargs: (4.5, 0.6))
    medium = check_degradation_significance(
        [0.10, 0.10, 0.10, 0.10, 0.10],
        [0.09, 0.09, 0.09, 0.09, 0.09],
        n_bootstrap=20,
        random_state=42,
    )
    assert medium.effect_magnitude == "medium"


def test_sharpe_ci_fallback_when_bootstrap_samples_empty():
    returns = np.array([0.0] * 20)
    sharpe, ci_lower, ci_upper = calculate_sharpe_confidence_interval(returns, n_bootstrap=20)
    assert ci_lower < ci_upper
    assert sharpe == 0.0


@pytest.mark.asyncio
async def test_validate_raises_when_no_splits(monkeypatch):
    validator = WalkForwardValidator()
    monkeypatch.setattr(validator, "create_time_splits", lambda *_args, **_kwargs: [])

    async def _backtest(*_args, **_kwargs):
        return {}

    with pytest.raises(ValueError, match="Could not create valid train/test splits"):
        await validator.validate(_backtest, ["AAPL"], "2024-01-01", "2024-12-31")


@pytest.mark.asyncio
async def test_validate_handles_negative_is_positive_oos_ratio_branch():
    validator = WalkForwardValidator(n_splits=1, min_train_days=20)
    calls = {"count": 0}

    async def _backtest(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"total_return": -0.10, "sharpe_ratio": -0.5, "num_trades": 10, "win_rate": 0.4}
        return {"total_return": 0.05, "sharpe_ratio": 0.6, "num_trades": 8, "win_rate": 0.5}

    result = await validator.validate(_backtest, ["AAPL"], "2024-01-01", "2024-12-31")
    assert result["avg_overfit_ratio"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_walk_forward_validation_convenience_function(monkeypatch):
    fake_simple_backtest_mod = types.ModuleType("simple_backtest")

    async def _simple_backtest(*_args, **_kwargs):
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "num_trades": 0, "win_rate": 0.0}

    fake_simple_backtest_mod.simple_backtest = _simple_backtest
    monkeypatch.setitem(sys.modules, "simple_backtest", fake_simple_backtest_mod)

    async def _fake_validate(self, backtest_fn, symbols, start_date, end_date, **kwargs):
        assert callable(backtest_fn)
        return {"passes_validation": True, "n_folds": 1}

    monkeypatch.setattr(WalkForwardValidator, "validate", _fake_validate)

    result = await run_walk_forward_validation(
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    assert result["passes_validation"] is True
