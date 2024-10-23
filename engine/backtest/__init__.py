"""
BacktestEngine sub-package.

The original ``engine/backtest_engine.py`` (~1,380 LOC) was split into
focused mixin modules that the top-level ``BacktestEngine`` class composes
via multiple inheritance:

    - core.py          — session resolution, run() loop, per-iteration
                         hook, inline metrics, per-symbol signal
                         processing, signed-qty PnL accounting
    - runner.py        — comprehensive run_backtest() driver, data
                         loading, OrderGateway wiring, end-of-period
                         liquidation, result assembly
    - walk_forward.py  — run_walk_forward_backtest(), fold setup,
                         train/test split, Sharpe-from-equity helper

External callers continue to import ``BacktestEngine`` from
``engine.backtest_engine`` — this sub-package is an internal
implementation detail.
"""

from engine.backtest.core import BacktestCoreMixin
from engine.backtest.runner import BacktestRunnerMixin
from engine.backtest.walk_forward import BacktestWalkForwardMixin

__all__ = [
    "BacktestCoreMixin",
    "BacktestRunnerMixin",
    "BacktestWalkForwardMixin",
]
