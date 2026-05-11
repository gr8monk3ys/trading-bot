"""
MomentumStrategy indicators mixin.

Contains the TA-Lib indicator calculation pipeline used by the momentum
strategy:

    - ``_safe_last`` — NaN-aware last-value extractor for TA-Lib arrays
    - ``_calculate_indicators_from_arrays`` — pure-function indicator math
      that accepts numpy arrays (used both by live ``on_bar`` and the
      backtest-mode ``generate_signals`` path)
    - ``_update_indicators`` — wraps the above and stores results in
      ``self.indicators[symbol]``

These methods rely on attributes initialized by
``strategies/momentum/strategy.py``
(``self.indicators``, ``self.price_history``, ``self.rsi_period``,
``self.macd_fast``, ``self.macd_slow``, ``self.macd_signal``,
``self.adx_period``, ``self.fast_ma``, ``self.medium_ma``, ``self.slow_ma``,
``self.volume_ma_period``, ``self.atr_period``,
``self.use_bollinger_filter``, ``self.bb_period``, ``self.bb_std``) and
therefore must be mixed into the same concrete class.
"""

import logging

import numpy as np
import talib

logger = logging.getLogger(__name__)


class MomentumIndicatorsMixin:
    """TA-Lib indicator calculation for the momentum strategy."""

    def _safe_last(self, arr):
        """Extract last value from array, returning None if empty or NaN."""
        if arr is None or len(arr) == 0:
            return None
        val = arr[-1]
        if isinstance(val, float) and np.isnan(val):
            return None
        return float(val) if not np.isnan(val) else None

    def _calculate_indicators_from_arrays(self, closes, highs, lows, volumes):
        """
        Calculate all technical indicators from price arrays.

        Args:
            closes: numpy array of close prices
            highs: numpy array of high prices
            lows: numpy array of low prices
            volumes: numpy array of volumes

        Returns:
            dict: Dictionary of calculated indicator values
        """
        # TA-Lib expects contiguous float64 ("double") arrays.
        closes = np.asarray(closes, dtype=np.float64)
        highs = np.asarray(highs, dtype=np.float64)
        lows = np.asarray(lows, dtype=np.float64)
        volumes = np.asarray(volumes, dtype=np.float64)

        # Calculate RSI
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)

        # Calculate MACD
        macd, signal, hist = talib.MACD(
            closes,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        # Calculate ADX
        adx = talib.ADX(highs, lows, closes, timeperiod=self.adx_period)

        # Calculate moving averages
        fast_ma = talib.SMA(closes, timeperiod=self.fast_ma)
        medium_ma = talib.SMA(closes, timeperiod=self.medium_ma)
        slow_ma = talib.SMA(closes, timeperiod=self.slow_ma)

        # Calculate volume moving average
        volume_ma = talib.SMA(volumes, timeperiod=self.volume_ma_period)

        # Calculate ATR for stop loss
        atr = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)

        # Calculate Bollinger Bands for mean reversion filter
        bb_upper, bb_middle, bb_lower = None, None, None
        bb_position = None

        if self.use_bollinger_filter and len(closes) >= self.bb_period:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                closes,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0,
            )

            # Calculate Bollinger Band position (0 = at lower band, 1 = at upper band)
            current_close = self._safe_last(closes)
            bb_lower_val = self._safe_last(bb_lower)
            bb_upper_val = self._safe_last(bb_upper)

            if current_close and bb_lower_val and bb_upper_val and bb_upper_val != bb_lower_val:
                bb_position = (current_close - bb_lower_val) / (bb_upper_val - bb_lower_val)

        return {
            "rsi": self._safe_last(rsi),
            "macd": self._safe_last(macd),
            "macd_signal": self._safe_last(signal),
            "macd_hist": self._safe_last(hist),
            "adx": self._safe_last(adx),
            "fast_ma": self._safe_last(fast_ma),
            "medium_ma": self._safe_last(medium_ma),
            "slow_ma": self._safe_last(slow_ma),
            "volume": self._safe_last(volumes),
            "volume_ma": self._safe_last(volume_ma),
            "atr": self._safe_last(atr),
            "close": self._safe_last(closes),
            "bb_upper": self._safe_last(bb_upper) if bb_upper is not None else None,
            "bb_middle": self._safe_last(bb_middle) if bb_middle is not None else None,
            "bb_lower": self._safe_last(bb_lower) if bb_lower is not None else None,
            "bb_position": bb_position,
        }

    async def _update_indicators(self, symbol):
        """Update technical indicators for a symbol."""
        try:
            # Ensure we have enough price history
            if len(self.price_history[symbol]) < self.slow_ma:
                return

            # Extract price data into arrays
            closes = np.array([bar["close"] for bar in self.price_history[symbol]])
            highs = np.array([bar["high"] for bar in self.price_history[symbol]])
            lows = np.array([bar["low"] for bar in self.price_history[symbol]])
            volumes = np.array([bar["volume"] for bar in self.price_history[symbol]])

            # Calculate and store indicators using shared method
            self.indicators[symbol] = self._calculate_indicators_from_arrays(
                closes, highs, lows, volumes
            )

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}", exc_info=True)
