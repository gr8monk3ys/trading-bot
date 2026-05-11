"""
BacktestEngine — thin facade composing the backtest sub-package mixins.

The implementation lives in ``engine/backtest/``:

    - engine/backtest/core.py          — session resolution, run() loop,
                                         per-iteration hook, inline
                                         performance metrics, per-symbol
                                         signal processing, signed-qty
                                         PnL accounting (Step 2B fix).
    - engine/backtest/runner.py        — comprehensive run_backtest()
                                         driver, parallel data loading,
                                         OrderGateway wiring (Step 2A),
                                         end-of-period liquidation
                                         (Step 2C), result assembly.
    - engine/backtest/walk_forward.py  — run_walk_forward_backtest(), fold
                                         setup, train/test split with
                                         embargo, IS-vs-OOS overfit
                                         detection.

External callers continue to do ``from engine.backtest_engine import
BacktestEngine`` — the sub-package is an internal implementation detail.
"""

import logging

from engine.backtest import (
    BacktestCoreMixin,
    BacktestRunnerMixin,
    BacktestWalkForwardMixin,
)

logger = logging.getLogger(__name__)


__all__ = ["BacktestEngine"]


class BacktestEngine(
    BacktestCoreMixin,
    BacktestRunnerMixin,
    BacktestWalkForwardMixin,
):
    """
    Engine for backtesting trading strategies using historical data.

    Composed from three mixins (see module docstring). The public API is
    unchanged from the pre-split monolith — ``engine.run()``,
    ``engine.run_backtest()``, and ``engine.run_walk_forward_backtest()``
    all work exactly as before, as do the private helpers
    (``_calculate_trade_pnl``, ``_calculate_performance_metrics``,
    ``_run_strategy_iteration``, ``_calculate_sharpe_from_equity``) that
    tests bind to instances.
    """

    def __init__(self, broker=None):
        """
        Initialize the backtest engine.

        Args:
            broker: The broker instance to use for market data. If None, create a new one.
        """
        self.broker = broker
        self.current_date = None
        self.strategies = []
        self.results = {}
