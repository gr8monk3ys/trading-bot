"""
Thin re-export shim for the BacktestBroker package.

This module preserves the historical import path
``from brokers.backtest_broker import BacktestBroker, ...`` after the
Phase 11 refactor that moved the implementation into the
:mod:`brokers.backtest` sub-package.

New code should prefer importing directly from :mod:`brokers.backtest`.
"""

from brokers.backtest import (
    EXECUTION_PROFILE_PRESETS,
    BacktestBroker,
    ExecutionProfile,
    GapEvent,
    GapStatistics,
)

__all__ = [
    "BacktestBroker",
    "ExecutionProfile",
    "EXECUTION_PROFILE_PRESETS",
    "GapEvent",
    "GapStatistics",
]
