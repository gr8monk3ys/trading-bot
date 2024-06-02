"""
Mock broker for backtesting purposes with institutional-grade gap risk modeling.

Public API (re-exported here):
- :class:`BacktestBroker`
- :class:`ExecutionProfile`
- :class:`GapEvent`, :class:`GapStatistics`

This module is a thin facade. Implementation is split across:
- :mod:`brokers.backtest.core` — init, price data, position/balance queries
- :mod:`brokers.backtest.execution` — order placement, slippage, partial fills
- :mod:`brokers.backtest.gaps` — gap event dataclasses, gap-risk simulation
"""

from brokers.backtest.core import (
    EXECUTION_PROFILE_PRESETS,
    BacktestBrokerCore,
    ExecutionProfile,
)
from brokers.backtest.execution import BacktestBrokerExecutionMixin
from brokers.backtest.gaps import (
    BacktestBrokerGapsMixin,
    GapEvent,
    GapStatistics,
)


class BacktestBroker(
    BacktestBrokerCore,
    BacktestBrokerExecutionMixin,
    BacktestBrokerGapsMixin,
):
    """
    Simple broker for backtesting purposes with realistic slippage modeling.

    Combines:
    - :class:`brokers.backtest.core.BacktestBrokerCore` — state, price data,
      position/balance queries, async wrappers.
    - :class:`brokers.backtest.execution.BacktestBrokerExecutionMixin` —
      slippage, partial fills, :meth:`place_order`.
    - :class:`brokers.backtest.gaps.BacktestBrokerGapsMixin` — overnight
      gap simulation, stop-order tracking, gap statistics.
    """

    pass


__all__ = [
    "BacktestBroker",
    "ExecutionProfile",
    "EXECUTION_PROFILE_PRESETS",
    "GapEvent",
    "GapStatistics",
]
