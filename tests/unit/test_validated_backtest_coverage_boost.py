from __future__ import annotations

import builtins
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from engine.validated_backtest import (
    RegimePerformance,
    ValidatedBacktestResult,
    ValidatedBacktestRunner,
    _coerce_float,
    _coerce_int,
    format_validated_backtest_report,
)


class _Strategy:
    def __init__(self, broker=None, **kwargs):
        self.broker = broker
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_run_backtest_success_and_failure_paths(monkeypatch):
    runner = ValidatedBacktestRunner(broker=None)

    class _FakeBacktestBroker:
        def __init__(self, initial_balance):
            self.initial_balance = initial_balance

    class _FakeBacktestEngine:
        def __init__(self, broker):
            self.broker = broker

        async def run(self, strategies, start_date, end_date):
            dates = pd.date_range(start=start_date, periods=3, freq="D")
            df = pd.DataFrame(
                {
                    "equity": [100000.0, 101000.0, 102000.0],
                    "returns": [0.0, 0.01, 0.0099],
                    "drawdown": [0.0, -0.002, -0.001],
                    "trades": [0, 1, 1],
                },
                index=dates,
            )
            return [df]

    fake_broker_mod = types.ModuleType("brokers.backtest_broker")
    fake_broker_mod.BacktestBroker = _FakeBacktestBroker
    fake_engine_mod = types.ModuleType("engine.backtest_engine")
    fake_engine_mod.BacktestEngine = _FakeBacktestEngine
    monkeypatch.setitem(sys.modules, "brokers.backtest_broker", fake_broker_mod)
    monkeypatch.setitem(sys.modules, "engine.backtest_engine", fake_engine_mod)

    ok = await runner._run_backtest(
        _Strategy,
        ["AAPL"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 3),
        100000,
    )
    assert ok["num_trades"] == 2
    assert ok["sharpe_ratio"] != 0

    class _ExplodingBacktestEngine(_FakeBacktestEngine):
        async def run(self, strategies, start_date, end_date):
            raise RuntimeError("boom")

    fake_engine_mod.BacktestEngine = _ExplodingBacktestEngine
    failed = await runner._run_backtest(
        _Strategy,
        ["AAPL"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 3),
        100000,
    )
    assert failed == {}


@pytest.mark.asyncio
async def test_run_walk_forward_branches(monkeypatch):
    class _Validator:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def create_time_splits(self, start_date, end_date):
            return [(start_date, start_date, end_date, end_date)]

    monkeypatch.setattr("engine.validated_backtest.WalkForwardValidator", _Validator)

    runner = ValidatedBacktestRunner(broker=None, n_splits=1)

    sequence = iter(
        [
            {"total_return": 0.2, "sharpe_ratio": 1.0, "num_trades": 10, "win_rate": 0.5},
            {"total_return": 0.0, "sharpe_ratio": 0.0, "num_trades": 5, "win_rate": 0.5},
        ]
    )

    async def _fake_run_backtest(*_args, **_kwargs):
        return next(sequence)

    monkeypatch.setattr(runner, "_run_backtest", _fake_run_backtest)

    result = await runner._run_walk_forward(
        _Strategy,
        ["AAPL"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        100000,
    )
    assert result is not None
    assert result["overfit_warning"] is True
    assert result["overfit_ratio"] == float("inf")

    sequence2 = iter(
        [
            {"total_return": -0.2, "sharpe_ratio": -1.0, "num_trades": 10, "win_rate": 0.4},
            {"total_return": 0.0, "sharpe_ratio": 0.0, "num_trades": 5, "win_rate": 0.5},
        ]
    )

    async def _fake_run_backtest2(*_args, **_kwargs):
        return next(sequence2)

    monkeypatch.setattr(runner, "_run_backtest", _fake_run_backtest2)
    result2 = await runner._run_walk_forward(
        _Strategy,
        ["AAPL"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        100000,
    )
    assert result2 is not None
    assert result2["overfit_ratio"] == 1.0


@pytest.mark.asyncio
async def test_run_walk_forward_exception_returns_none(monkeypatch):
    class _BadValidator:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def create_time_splits(self, start_date, end_date):
            raise RuntimeError("bad splits")

    monkeypatch.setattr("engine.validated_backtest.WalkForwardValidator", _BadValidator)
    runner = ValidatedBacktestRunner(broker=None)
    result = await runner._run_walk_forward(
        _Strategy,
        ["AAPL"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        100000,
    )
    assert result is None


@pytest.mark.asyncio
async def test_calculate_regime_metrics_success_and_errors(monkeypatch):
    runner = ValidatedBacktestRunner(broker=object())
    returns = pd.Series(
        [0.01, np.nan, -0.02, 0.03],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    class _Detector:
        def __init__(self, broker):
            self.calls = 0

        async def detect_regime(self, _date):
            self.calls += 1
            if self.calls == 1:
                return {"regime": "BULL"}
            if self.calls == 2:
                raise RuntimeError("detect failed")
            return "unexpected"

    fake_market_regime = types.ModuleType("utils.market_regime")
    fake_market_regime.MarketRegimeDetector = _Detector
    monkeypatch.setitem(sys.modules, "utils.market_regime", fake_market_regime)

    metrics = await runner._calculate_regime_metrics(
        pd.Series([100, 101, 99, 102], index=returns.index),
        returns,
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
    )
    assert "BULL" in metrics

    original_import = builtins.__import__

    def _import_error(name, *args, **kwargs):
        if name == "utils.market_regime":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_error)
    metrics_import_error = await runner._calculate_regime_metrics(
        pd.Series([100, 101], index=returns.index[:2]),
        returns.iloc[:2],
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
    )
    assert metrics_import_error == {}

    monkeypatch.setattr(builtins, "__import__", original_import)

    class _ExplodingDetector:
        def __init__(self, broker):
            raise RuntimeError("init fail")

    fake_market_regime_bad = types.ModuleType("utils.market_regime")
    fake_market_regime_bad.MarketRegimeDetector = _ExplodingDetector
    monkeypatch.setitem(sys.modules, "utils.market_regime", fake_market_regime_bad)
    metrics_runtime_error = await runner._calculate_regime_metrics(
        pd.Series([100, 101], index=returns.index[:2]),
        returns.iloc[:2].fillna(0.0),
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
    )
    assert metrics_runtime_error == {}


def test_sharpe_and_significance_fallbacks(monkeypatch):
    runner = ValidatedBacktestRunner(broker=None)

    assert runner._calculate_sharpe(pd.Series([0.01])) == 0
    assert runner._calculate_sharpe(pd.Series([np.nan, 0.01])) == 0
    assert runner._calculate_sharpe(pd.Series([0.01, 0.01, 0.01])) == 0
    assert runner._calculate_sharpe(pd.Series([0.01, -0.01, 0.02, -0.005])) != 0

    fake_scipy = types.ModuleType("scipy")
    fake_stats = types.ModuleType("stats")
    fake_stats.ttest_1samp = lambda data, mean: (2.0, 0.01)
    fake_scipy.stats = fake_stats
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    assert runner._calculate_significance(pd.Series([0.01] * 40)) == (True, 0.01)

    original_import = builtins.__import__

    def _import_error(name, *args, **kwargs):
        if name == "scipy":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_error)
    assert runner._calculate_significance(pd.Series([0.01] * 40)) == (False, None)
    monkeypatch.setattr(builtins, "__import__", original_import)

    fake_stats.ttest_1samp = lambda data, mean: (_ for _ in ()).throw(RuntimeError("ttest fail"))
    assert runner._calculate_significance(pd.Series([0.01] * 40)) == (False, None)


@pytest.mark.asyncio
async def test_run_validated_backtest_with_walk_forward(monkeypatch):
    runner = ValidatedBacktestRunner(
        broker=None, walk_forward_enabled=True, regime_analysis_enabled=True
    )

    async def _fake_run_backtest(*_args, **_kwargs):
        return {
            "total_return": 0.1,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.05,
            "num_trades": 20,
            "win_rate": 0.55,
            "equity_curve": pd.Series([100000, 101000]),
            "daily_returns": pd.Series([0.01, 0.0]),
        }

    async def _fake_walk_forward(*_args, **_kwargs):
        return {
            "overfit_warning": False,
            "overfit_ratio": 1.1,
            "is_return": 0.08,
            "oos_return": 0.06,
            "consistency": 0.75,
            "folds": [],
        }

    async def _fake_regime(*_args, **_kwargs):
        return {
            "BULL": RegimePerformance(
                regime="BULL",
                num_days=2,
                total_return=0.02,
                sharpe_ratio=1.0,
                max_drawdown=-0.01,
                win_rate=0.5,
                num_trades=0,
            )
        }

    monkeypatch.setattr(runner, "_run_backtest", _fake_run_backtest)
    monkeypatch.setattr(runner, "_run_walk_forward", _fake_walk_forward)
    monkeypatch.setattr(runner, "_calculate_regime_metrics", _fake_regime)
    monkeypatch.setattr(runner, "_calculate_significance", lambda _r: (True, 0.01))
    monkeypatch.setattr(runner, "_evaluate_validation_gates", lambda _r: ({"blockers": []}, True))

    result = await runner.run_validated_backtest(
        strategy_class=_Strategy,
        symbols=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-01-05",
    )
    assert result.walk_forward_validated is True
    assert result.validation_gates == {"blockers": []}


def test_log_result_summary_and_report_branches(monkeypatch):
    runner = ValidatedBacktestRunner(broker=None)
    fake_logger = MagicMock()
    monkeypatch.setattr("engine.validated_backtest.logger", fake_logger)

    result = ValidatedBacktestResult(
        strategy_name="TestStrategy",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        total_return=0.12,
        sharpe_ratio=1.3,
        max_drawdown=-0.06,
        num_trades=25,
        win_rate=0.52,
        walk_forward_validated=True,
        overfit_warning=True,
        overfit_ratio=2.1,
        is_return=0.15,
        oos_return=0.05,
        consistency_score=0.4,
        walk_forward_folds=[],
        regime_metrics={
            "BULL": RegimePerformance(
                regime="BULL",
                num_days=10,
                total_return=0.08,
                sharpe_ratio=1.1,
                max_drawdown=-0.03,
                win_rate=0.6,
                num_trades=0,
            )
        },
        statistically_significant=True,
        p_value=0.01,
    )
    result.validation_gates = {"blockers": ["min_sharpe"]}
    result.eligible_for_trading = False

    runner._log_result_summary(result)
    assert fake_logger.warning.called

    result2 = ValidatedBacktestResult(
        strategy_name="TestStrategy",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        total_return=0.01,
        sharpe_ratio=0.1,
        max_drawdown=-0.2,
        num_trades=5,
        win_rate=0.2,
        walk_forward_validated=False,
        overfit_warning=False,
        overfit_ratio=1.0,
        is_return=0.0,
        oos_return=0.0,
        consistency_score=0.0,
        statistically_significant=False,
        p_value=0.4,
    )
    result2.validation_gates = {
        "min_trades": {"passed": False, "value": 5, "threshold": 50},
        "blockers": ["min_trades"],
    }

    report = format_validated_backtest_report(result2)
    assert "NOT significant" in report
    assert "min_trades: FAIL" in report
    assert "Blockers: min_trades" in report


def test_coerce_helpers():
    assert _coerce_float(True, 1.5) == 1.5
    assert _coerce_float("abc", 2.5) == 2.5
    assert _coerce_float(object(), 3.5) == 3.5

    assert _coerce_int(True, 7) == 7
    assert _coerce_int(3.9, 0) == 3
    assert _coerce_int("bad", 11) == 11
    assert _coerce_int(object(), 13) == 13
