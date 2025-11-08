#!/usr/bin/env python3
"""
Multi-Timeframe Analysis Utility

Allows strategies to analyze price action across multiple timeframes for better signals.

Example usage:
    # In strategy initialize():
    self.mtf = MultiTimeframeAnalyzer(
        timeframes=['1Min', '5Min', '1Hour'],
        history_length=100
    )

    # In on_bar():
    await self.mtf.update(symbol, timestamp, price, volume)

    # Get multi-timeframe signal:
    signal = self.mtf.get_aligned_signal(symbol)
    if signal == 'bullish':  # All timeframes bullish
        # Strong buy signal
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class TimeframeData:
    """Stores price data for a single timeframe."""

    def __init__(self, timeframe: str, max_bars: int = 200):
        """
        Initialize timeframe data.

        Args:
            timeframe: Timeframe string ('1Min', '5Min', '15Min', '1Hour', '1Day')
            max_bars: Maximum number of bars to keep in memory
        """
        self.timeframe = timeframe
        self.max_bars = max_bars

        # Price data
        self.timestamps = deque(maxlen=max_bars)
        self.opens = deque(maxlen=max_bars)
        self.highs = deque(maxlen=max_bars)
        self.lows = deque(maxlen=max_bars)
        self.closes = deque(maxlen=max_bars)
        self.volumes = deque(maxlen=max_bars)

        # Current bar being built
        self.current_bar = None
        self.bar_duration = self._parse_timeframe(timeframe)

    def _parse_timeframe(self, timeframe: str) -> timedelta:
        """Parse timeframe string to timedelta."""
        if timeframe.endswith('Min'):
            minutes = int(timeframe[:-3])
            return timedelta(minutes=minutes)
        elif timeframe.endswith('Hour'):
            hours = int(timeframe[:-4])
            return timedelta(hours=hours)
        elif timeframe.endswith('Day'):
            days = int(timeframe[:-3])
            return timedelta(days=days)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    def update(self, timestamp: datetime, price: float, volume: float):
        """
        Update timeframe with new tick data.

        Args:
            timestamp: Current timestamp
            price: Current price
            volume: Current volume
        """
        # Initialize current bar if needed
        if self.current_bar is None:
            self._start_new_bar(timestamp, price, volume)
            return

        # Check if we need to close current bar and start new one
        bar_end_time = self.current_bar['start_time'] + self.bar_duration

        if timestamp >= bar_end_time:
            # Close current bar
            self._close_current_bar()
            # Start new bar
            self._start_new_bar(timestamp, price, volume)
        else:
            # Update current bar
            self.current_bar['high'] = max(self.current_bar['high'], price)
            self.current_bar['low'] = min(self.current_bar['low'], price)
            self.current_bar['close'] = price
            self.current_bar['volume'] += volume

    def _start_new_bar(self, timestamp: datetime, price: float, volume: float):
        """Start a new bar."""
        self.current_bar = {
            'start_time': timestamp,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume
        }

    def _close_current_bar(self):
        """Close current bar and add to history."""
        if self.current_bar:
            self.timestamps.append(self.current_bar['start_time'])
            self.opens.append(self.current_bar['open'])
            self.highs.append(self.current_bar['high'])
            self.lows.append(self.current_bar['low'])
            self.closes.append(self.current_bar['close'])
            self.volumes.append(self.current_bar['volume'])

    def get_closes(self, count: Optional[int] = None) -> np.ndarray:
        """Get close prices as numpy array."""
        closes = list(self.closes)
        if count:
            closes = closes[-count:]
        return np.array(closes)

    def get_highs(self, count: Optional[int] = None) -> np.ndarray:
        """Get high prices as numpy array."""
        highs = list(self.highs)
        if count:
            highs = highs[-count:]
        return np.array(highs)

    def get_lows(self, count: Optional[int] = None) -> np.ndarray:
        """Get low prices as numpy array."""
        lows = list(self.lows)
        if count:
            lows = lows[-count:]
        return np.array(lows)

    def get_volumes(self, count: Optional[int] = None) -> np.ndarray:
        """Get volumes as numpy array."""
        volumes = list(self.volumes)
        if count:
            volumes = volumes[-count:]
        return np.array(volumes)

    def __len__(self):
        """Return number of completed bars."""
        return len(self.closes)


class MultiTimeframeAnalyzer:
    """
    Analyzes price action across multiple timeframes.

    Features:
    - Tracks multiple timeframes simultaneously
    - Calculates trend for each timeframe
    - Provides aligned signals across timeframes
    - Detects timeframe divergences
    """

    def __init__(self, timeframes: List[str], history_length: int = 200):
        """
        Initialize multi-timeframe analyzer.

        Args:
            timeframes: List of timeframes to track (e.g., ['1Min', '5Min', '1Hour'])
            history_length: Number of bars to keep for each timeframe
        """
        self.timeframes = sorted(timeframes, key=lambda x: self._timeframe_to_minutes(x))
        self.history_length = history_length

        # Data storage: symbol -> timeframe -> TimeframeData
        self.data: Dict[str, Dict[str, TimeframeData]] = {}

        logger.info(f"Multi-timeframe analyzer initialized: {', '.join(self.timeframes)}")

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes for sorting."""
        if timeframe.endswith('Min'):
            return int(timeframe[:-3])
        elif timeframe.endswith('Hour'):
            return int(timeframe[:-4]) * 60
        elif timeframe.endswith('Day'):
            return int(timeframe[:-3]) * 1440
        return 0

    async def update(self, symbol: str, timestamp: datetime, price: float, volume: float = 0):
        """
        Update all timeframes with new tick data.

        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            price: Current price
            volume: Current volume
        """
        # Initialize symbol if needed
        if symbol not in self.data:
            self.data[symbol] = {
                tf: TimeframeData(tf, self.history_length)
                for tf in self.timeframes
            }

        # Update all timeframes
        for tf in self.timeframes:
            self.data[symbol][tf].update(timestamp, price, volume)

    def get_trend(self, symbol: str, timeframe: str, period: int = 20) -> str:
        """
        Get trend for a specific timeframe using SMA.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe to analyze
            period: SMA period for trend detection

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return 'neutral'

        tf_data = self.data[symbol][timeframe]

        if len(tf_data) < period + 1:
            return 'neutral'

        closes = tf_data.get_closes()

        # Calculate SMA
        sma = np.mean(closes[-period:])
        current_price = closes[-1]

        # Calculate trend strength
        price_above_sma = (current_price - sma) / sma

        # Strong trend thresholds
        if price_above_sma > 0.01:  # 1% above SMA
            return 'bullish'
        elif price_above_sma < -0.01:  # 1% below SMA
            return 'bearish'
        else:
            return 'neutral'

    def get_aligned_signal(self, symbol: str) -> str:
        """
        Get signal when all timeframes are aligned.

        Returns:
            'bullish' - All timeframes bullish
            'bearish' - All timeframes bearish
            'neutral' - Mixed or neutral signals
        """
        if symbol not in self.data:
            return 'neutral'

        trends = []
        for tf in self.timeframes:
            trend = self.get_trend(symbol, tf)
            trends.append(trend)

        # Check for alignment
        if all(t == 'bullish' for t in trends):
            return 'bullish'
        elif all(t == 'bearish' for t in trends):
            return 'bearish'
        else:
            return 'neutral'

    def get_timeframe_momentum(self, symbol: str, timeframe: str, period: int = 14) -> float:
        """
        Calculate momentum (rate of change) for a timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe to analyze
            period: Lookback period

        Returns:
            Momentum as percentage change
        """
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return 0.0

        tf_data = self.data[symbol][timeframe]

        if len(tf_data) < period + 1:
            return 0.0

        closes = tf_data.get_closes()

        # Calculate rate of change
        old_price = closes[-period]
        current_price = closes[-1]

        momentum = ((current_price - old_price) / old_price) * 100

        return momentum

    def detect_divergence(self, symbol: str) -> Optional[str]:
        """
        Detect divergences between timeframes.

        Returns:
            'bullish_divergence' - Short-term weak but long-term strong
            'bearish_divergence' - Short-term strong but long-term weak
            None - No significant divergence
        """
        if symbol not in self.data or len(self.timeframes) < 2:
            return None

        # Get shortest and longest timeframe trends
        short_tf = self.timeframes[0]
        long_tf = self.timeframes[-1]

        short_trend = self.get_trend(symbol, short_tf)
        long_trend = self.get_trend(symbol, long_tf)

        # Detect divergences
        if short_trend == 'bearish' and long_trend == 'bullish':
            return 'bullish_divergence'  # Short-term pullback in long-term uptrend
        elif short_trend == 'bullish' and long_trend == 'bearish':
            return 'bearish_divergence'  # Short-term rally in long-term downtrend

        return None

    def get_timeframe_data(self, symbol: str, timeframe: str) -> Optional[TimeframeData]:
        """Get raw timeframe data for custom analysis."""
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return None
        return self.data[symbol][timeframe]

    def get_status(self, symbol: str) -> Dict:
        """
        Get complete multi-timeframe status for a symbol.

        Returns:
            Dict with trends, alignment, and divergences
        """
        if symbol not in self.data:
            return {'error': 'No data for symbol'}

        status = {
            'timeframes': {},
            'aligned_signal': self.get_aligned_signal(symbol),
            'divergence': self.detect_divergence(symbol)
        }

        for tf in self.timeframes:
            status['timeframes'][tf] = {
                'trend': self.get_trend(symbol, tf),
                'momentum': self.get_timeframe_momentum(symbol, tf),
                'bar_count': len(self.data[symbol][tf])
            }

        return status
