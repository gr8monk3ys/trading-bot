"""
MomentumStrategy configured for backtesting with daily data.

This version disables features that require intraday data and uses
less strict parameters to generate realistic trade signals.
"""

import logging

from strategies.momentum_strategy import MomentumStrategy
from strategies.params import MomentumBacktestParams

logger = logging.getLogger(__name__)


class MomentumStrategyBacktest(MomentumStrategy):
    Params = MomentumBacktestParams
    """
    MomentumStrategy variant optimized for daily-data backtesting.

    Key differences from production MomentumStrategy:
    - Uses standard RSI-14 with 30/70 thresholds (not RSI-2 aggressive)
    - Disables multi-timeframe filtering (requires intraday data)
    - Lower volume threshold for confirmation
    - Disabled strict timeframe alignment

    This allows realistic backtesting while maintaining the core momentum logic.
    """

    NAME = "MomentumStrategyBacktest"

    def default_parameters(self):
        """The defaults live once, on ``Params`` (strategies/params.py)."""
        return dict(self.Params.defaults())
