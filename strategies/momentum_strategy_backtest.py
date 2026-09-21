"""
MomentumStrategy configured for backtesting with daily data.

This version disables features that require intraday data and uses
less strict parameters to generate realistic trade signals.
"""

import logging

from strategies.momentum_strategy import MomentumStrategy

logger = logging.getLogger(__name__)


class MomentumStrategyBacktest(MomentumStrategy):
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
        """Override parameters for backtesting with daily data."""
        # Get base parameters from parent class
        params = MomentumStrategy.default_parameters(self)

        # === BACKTEST-FRIENDLY OVERRIDES ===

        # Use standard RSI-14 with 30/70 thresholds
        # RSI-2 with 10/90 is too aggressive for daily data
        params["rsi_mode"] = "standard"
        params["rsi_period"] = 14
        params["rsi_overbought"] = 70
        params["rsi_oversold"] = 30

        # Disable multi-timeframe filtering (requires intraday data)
        params["use_multi_timeframe"] = False

        # Lower volume threshold (1.5x is too strict for daily data)
        params["volume_factor"] = 1.2

        # Lower ADX threshold for more signals
        params["adx_threshold"] = 20

        # Disable strict alignment (not available in daily backtest)
        params["mtf_require_alignment"] = False

        # Keep other features enabled
        params["use_bollinger_filter"] = True
        params["enable_short_selling"] = True
        params["use_kelly_criterion"] = False  # Disable for clean backtest
        params["use_volatility_regime"] = False  # Simpler for backtesting

        logger.info("MomentumStrategyBacktest: Using daily-data optimized parameters")
        logger.info("  RSI mode: standard (14-period, 30/70 thresholds)")
        logger.info("  Multi-timeframe: disabled")
        logger.info("  Volume factor: 1.2x")
        logger.info("  ADX threshold: 20")

        return params
