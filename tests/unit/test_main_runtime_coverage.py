from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import main


def _live_args(**overrides) -> Namespace:
    values = {
        "run_id": "live_test_run",
        "artifacts_dir": "results/runs",
        "skip_validation": False,
        "real": False,
        "enforce_governance_gate": True,
        "governance_approval_path": None,
        "governance_policy_doc_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.asyncio
async def test_run_live_warns_when_skip_validation_flag_is_used(monkeypatch) -> None:
    warning = MagicMock()

    monkeypatch.setattr(main.logger, "warning", warning)
    monkeypatch.setattr(main, "generate_run_id", lambda prefix: f"{prefix}_run")
    monkeypatch.setattr(main, "ensure_run_directory", lambda *_args: "results/runs/live_run")
    monkeypatch.setattr("builtins.input", lambda _prompt: "cancel")

    await main.run_live(_live_args(skip_validation=True, real=True))

    warning.assert_any_call(
        "--skip-validation is currently a no-op; "
        "main.py live does not run a pre-trade validation pass yet"
    )


@pytest.mark.asyncio
async def test_optimize_parameters_logs_combination_counter(monkeypatch) -> None:
    info = MagicMock()

    class _FakeStrategy:
        def __init__(self, broker=None, symbols=None):
            self.broker = broker
            self.symbols = symbols or []

        def default_parameters(self):
            return {"lookback": 1}

    class _FakeStrategyManager:
        def __init__(self, broker=None):
            self.broker = broker
            self.available_strategies = {"momentum": _FakeStrategy}
            self.backtest_engine = SimpleNamespace(
                run_backtest=AsyncMock(return_value={"equity_curve": [100000.0, 101000.0]})
            )
            self.perf_metrics = SimpleNamespace(
                calculate_metrics=lambda _result: {
                    "sharpe_ratio": 1.0,
                    "total_return": 0.01,
                    "max_drawdown": -0.01,
                }
            )

        def get_available_strategy_names(self):
            return list(self.available_strategies.keys())

        def close(self):
            return None

    monkeypatch.setattr(main.logger, "info", info)
    monkeypatch.setattr(main, "AlpacaBroker", lambda paper=True: object())
    monkeypatch.setattr(main, "StrategyManager", _FakeStrategyManager)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    args = Namespace(
        strategy="momentum",
        start_date="2024-01-01",
        end_date="2024-01-31",
        symbols="AAPL",
        param_ranges='{"lookback": {"min": 1, "max": 1, "step": 1}}',
        capital=100000,
        execution_profile="realistic",
        optimize_for="sharpe",
    )

    await main.optimize_parameters(args)

    info.assert_any_call("Testing combination 1/1: {'lookback': 1}")
