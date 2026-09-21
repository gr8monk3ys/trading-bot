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

External callers continue to do ``from engine.backtest_engine import
BacktestEngine`` — the sub-package is an internal implementation detail.
"""

import logging

from engine.backtest import (
    BacktestCoreMixin,
    BacktestRunnerMixin,
)

logger = logging.getLogger(__name__)


__all__ = ["BacktestEngine"]


class BacktestEngine(
    BacktestCoreMixin,
    BacktestRunnerMixin,
):
    """
    Engine for backtesting trading strategies using historical data.

    Composed from two mixins (see module docstring). The public API is
    ``engine.run_backtest()``; the private helper ``_calculate_trade_pnl``
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
