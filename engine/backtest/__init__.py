"""
BacktestEngine sub-package.

The original ``engine/backtest_engine.py`` (~1,380 LOC) was split into
focused mixin modules that the top-level ``BacktestEngine`` class composes
via multiple inheritance:

    - core.py          — session resolution, per-symbol signal
                         processing, signed-qty PnL accounting
    - runner.py        — comprehensive run_backtest() driver, data
                         loading, OrderSubmission wiring, end-of-period
                         liquidation, result assembly

External callers continue to import ``BacktestEngine`` from
``engine.backtest_engine`` — this sub-package is an internal
implementation detail.
"""

from engine.backtest.core import BacktestCoreMixin
from engine.backtest.runner import BacktestRunnerMixin

__all__ = [
    "BacktestCoreMixin",
    "BacktestRunnerMixin",
]
