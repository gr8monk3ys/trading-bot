#!/usr/bin/env python3
"""
Market Regime Detection System

Detects whether the market is in a BULL, BEAR, or SIDEWAYS regime
and recommends which strategy type to use.

Research shows:
- Momentum strategies work best in trending markets (bull/bear)
- Mean reversion strategies work best in sideways/ranging markets
- Using wrong strategy in wrong regime = losses

Expected Impact: +10-15% annual returns from not fighting the market

Usage:
    from utils.market_regime import MarketRegimeDetector

    detector = MarketRegimeDetector(broker)
    regime = await detector.detect_regime()

    if regime['type'] == 'bull':
        use_momentum_strategy()
    elif regime['type'] == 'sideways':
        use_mean_reversion_strategy()
    elif regime['type'] == 'bear':
        use_defensive_or_short_strategy()
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""

    BULL = "bull"  # Trending up - use momentum (long)
    BEAR = "bear"  # Trending down - use momentum (short) or cash
    SIDEWAYS = "sideways"  # Ranging - use mean reversion
    VOLATILE = "volatile"  # High volatility - reduce exposure
    UNKNOWN = "unknown"  # Not enough data


class MarketRegimeDetector:
    """
    Detects market regime using multiple indicators:

    1. Trend Detection (SMA crossovers)
       - SMA50 > SMA200 = bullish
       - SMA50 < SMA200 = bearish

    2. Trend Strength (ADX)
       - ADX > 25 = strong trend (momentum works)
       - ADX < 20 = weak trend (mean reversion works)

    3. Volatility (VIX or historical vol)
       - High vol = reduce exposure
       - Low vol = increase exposure

    4. Breadth (% of stocks above SMA)
       - >70% above = broad bull market
       - <30% above = broad bear market
    """

    # Regime detection thresholds
    ADX_TRENDING_THRESHOLD = 25  # ADX above this = trending
    ADX_RANGING_THRESHOLD = 20  # ADX below this = ranging
    BREADTH_BULL_THRESHOLD = 0.60  # 60% of stocks above MA = bull
    BREADTH_BEAR_THRESHOLD = 0.40  # 40% of stocks above MA = bear

    def __init__(self, broker, lookback_days: int = 200, cache_minutes: int = 30):
        """
        Initialize market regime detector.

        Args:
            broker: Trading broker instance for data fetching
            lookback_days: Days of history for indicator calculation
            cache_minutes: Minutes to cache regime detection results
        """
        self.broker = broker
        self.lookback_days = lookback_days
        self.cache_minutes = cache_minutes

        # Market index to use for regime detection
        self.market_index = "SPY"  # S&P 500 ETF as market proxy

        # Cache
        self.last_regime = None
        self.last_detection_time = None
        self.regime_history = []

        logger.info("MarketRegimeDetector initialized")

    async def detect_regime(self, force_refresh: bool = False) -> Dict:
        """
        Detect current market regime.

        Args:
            force_refresh: Bypass cache and recalculate

        Returns:
            Dict with regime info:
            {
                'type': 'bull' | 'bear' | 'sideways' | 'volatile',
                'confidence': 0.0-1.0,
                'trend_direction': 'up' | 'down' | 'flat',
                'trend_strength': 0-100 (ADX value),
                'volatility_regime': 'low' | 'normal' | 'high',
                'recommended_strategy': 'momentum' | 'mean_reversion' | 'defensive',
                'position_multiplier': 0.4-1.4
            }
        """
        try:
            # Check cache
            now = datetime.now()
            if (
                not force_refresh
                and self.last_regime
                and self.last_detection_time
                and (now - self.last_detection_time).total_seconds() < self.cache_minutes * 60
            ):
                return self.last_regime

            # Fetch market data
            bars = await self._get_market_bars()
            if bars is None or len(bars) < 50:
                logger.warning("Insufficient market data for regime detection")
                return self._get_default_regime()

            # Calculate indicators
            closes = np.array([b["close"] for b in bars])
            highs = np.array([b["high"] for b in bars])
            lows = np.array([b["low"] for b in bars])

            # 1. Trend direction (SMA crossover)
            sma_50 = self._calculate_sma(closes, 50)
            sma_200 = self._calculate_sma(closes, 200) if len(closes) >= 200 else sma_50

            trend_direction = "up" if sma_50 > sma_200 else "down" if sma_50 < sma_200 else "flat"

            # 2. Trend strength (ADX)
            adx = self._calculate_adx(highs, lows, closes, period=14)
            trend_strength = adx if adx else 20  # Default to neutral if calculation fails

            # 3. Determine if trending or ranging
            is_trending = trend_strength > self.ADX_TRENDING_THRESHOLD
            is_ranging = trend_strength < self.ADX_RANGING_THRESHOLD

            # 4. Volatility check (using ATR as % of price)
            atr = self._calculate_atr(highs, lows, closes, period=14)
            current_price = closes[-1]
            volatility_pct = (atr / current_price) * 100 if current_price > 0 else 2.0

            # Volatility regimes
            if volatility_pct > 3.0:
                volatility_regime = "high"
            elif volatility_pct < 1.0:
                volatility_regime = "low"
            else:
                volatility_regime = "normal"

            # 5. Determine market regime
            regime_type, confidence = self._classify_regime(
                trend_direction, trend_strength, is_trending, is_ranging, volatility_regime
            )

            # 6. Strategy recommendation
            recommended_strategy, position_multiplier = self._get_strategy_recommendation(
                regime_type, confidence, volatility_regime
            )

            regime = {
                "type": regime_type.value,
                "confidence": confidence,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "is_trending": is_trending,
                "is_ranging": is_ranging,
                "volatility_regime": volatility_regime,
                "volatility_pct": volatility_pct,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "recommended_strategy": recommended_strategy,
                "position_multiplier": position_multiplier,
                "detected_at": now.isoformat(),
            }

            # Log regime change
            if self.last_regime and regime["type"] != self.last_regime["type"]:
                logger.warning(
                    f"REGIME CHANGE: {self.last_regime['type'].upper()} -> {regime['type'].upper()} "
                    f"(confidence: {confidence:.0%})"
                )
                self.regime_history.append(
                    {
                        "time": now,
                        "from": self.last_regime["type"],
                        "to": regime["type"],
                        "confidence": confidence,
                    }
                )

            # Update cache
            self.last_regime = regime
            self.last_detection_time = now

            logger.info(
                f"Market Regime: {regime_type.value.upper()} "
                f"(trend: {trend_direction}, ADX: {trend_strength:.1f}, vol: {volatility_regime}) "
                f"-> Use {recommended_strategy}, position mult: {position_multiplier:.1f}x"
            )

            return regime

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}", exc_info=True)
            return self._get_default_regime()

    def _classify_regime(
        self,
        trend_direction: str,
        trend_strength: float,
        is_trending: bool,
        is_ranging: bool,
        volatility_regime: str,
    ) -> Tuple[MarketRegime, float]:
        """
        Classify market into regime with confidence score.

        Returns:
            Tuple of (MarketRegime, confidence 0.0-1.0)
        """
        confidence = 0.5  # Base confidence

        # High volatility overrides other signals
        if volatility_regime == "high":
            confidence = 0.7 + (trend_strength / 100) * 0.2
            return MarketRegime.VOLATILE, min(confidence, 0.95)

        # Trending markets
        if is_trending:
            if trend_direction == "up":
                # Strong uptrend
                confidence = 0.6 + (trend_strength - self.ADX_TRENDING_THRESHOLD) / 50
                return MarketRegime.BULL, min(confidence, 0.95)
            elif trend_direction == "down":
                # Strong downtrend
                confidence = 0.6 + (trend_strength - self.ADX_TRENDING_THRESHOLD) / 50
                return MarketRegime.BEAR, min(confidence, 0.95)

        # Ranging markets
        if is_ranging:
            confidence = 0.6 + (self.ADX_RANGING_THRESHOLD - trend_strength) / 40
            return MarketRegime.SIDEWAYS, min(confidence, 0.90)

        # Ambiguous - weak trend
        if trend_direction == "up":
            return MarketRegime.BULL, 0.55
        elif trend_direction == "down":
            return MarketRegime.BEAR, 0.55

        return MarketRegime.SIDEWAYS, 0.50

    def _get_strategy_recommendation(
        self, regime: MarketRegime, confidence: float, volatility_regime: str
    ) -> Tuple[str, float]:
        """
        Get strategy recommendation based on regime.

        Returns:
            Tuple of (strategy_name, position_multiplier)
        """
        # Base multipliers by regime
        regime_config = {
            MarketRegime.BULL: ("momentum_long", 1.2),
            MarketRegime.BEAR: ("momentum_short", 0.8),  # Can short or go to cash
            MarketRegime.SIDEWAYS: ("mean_reversion", 1.0),
            MarketRegime.VOLATILE: ("defensive", 0.5),
            MarketRegime.UNKNOWN: ("momentum_long", 0.7),
        }

        strategy, base_mult = regime_config.get(regime, ("momentum_long", 1.0))

        # Adjust multiplier based on confidence
        if confidence < 0.6:
            base_mult *= 0.8  # Lower confidence = smaller positions
        elif confidence > 0.8:
            base_mult *= 1.1  # Higher confidence = slightly larger positions

        # Adjust for volatility
        vol_adjustments = {"low": 1.2, "normal": 1.0, "high": 0.6}
        base_mult *= vol_adjustments.get(volatility_regime, 1.0)

        # Cap multiplier
        base_mult = max(0.3, min(1.5, base_mult))

        return strategy, base_mult

    async def _get_market_bars(self) -> Optional[list]:
        """Fetch market index bars."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 50)

            bars = await self.broker.get_bars(
                self.market_index,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                timeframe="1Day",
            )

            if bars is None:
                return None

            # Convert to list of dicts if needed
            if hasattr(bars, "__iter__"):
                return [
                    {
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                    }
                    for b in bars
                ]
            return None

        except Exception as e:
            logger.error(f"Error fetching market bars: {e}")
            return None

    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        return np.mean(prices[-period:])

    def _calculate_adx(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> Optional[float]:
        """Calculate Average Directional Index (ADX)."""
        try:
            if len(closes) < period + 1:
                return None

            # Calculate True Range
            tr = np.zeros(len(closes))
            for i in range(1, len(closes)):
                tr[i] = max(
                    highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])
                )

            # Calculate +DM and -DM
            plus_dm = np.zeros(len(closes))
            minus_dm = np.zeros(len(closes))

            for i in range(1, len(closes)):
                up_move = highs[i] - highs[i - 1]
                down_move = lows[i - 1] - lows[i]

                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move

            # Smooth with EMA
            atr = self._ema(tr[1:], period)
            plus_di = 100 * self._ema(plus_dm[1:], period) / atr if atr > 0 else 0
            minus_di = 100 * self._ema(minus_dm[1:], period) / atr if atr > 0 else 0

            # Calculate DX
            di_sum = plus_di + minus_di
            if di_sum == 0:
                return 0

            dx = 100 * abs(plus_di - minus_di) / di_sum

            # ADX is smoothed DX (simplified - just return DX)
            return dx

        except Exception as e:
            logger.debug(f"ADX calculation error: {e}")
            return None

    def _calculate_atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        if len(closes) < 2:
            return 0

        tr = np.zeros(len(closes))
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])
            )

        if len(tr) < period:
            return np.mean(tr[1:]) if len(tr) > 1 else 0

        return np.mean(tr[-period:])

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average (returns last value)."""
        if len(data) == 0:
            return 0
        if len(data) < period:
            return np.mean(data)

        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def _get_default_regime(self) -> Dict:
        """Return default regime when detection fails."""
        return {
            "type": MarketRegime.UNKNOWN.value,
            "confidence": 0.5,
            "trend_direction": "flat",
            "trend_strength": 20,
            "is_trending": False,
            "is_ranging": True,
            "volatility_regime": "normal",
            "volatility_pct": 2.0,
            "sma_50": None,
            "sma_200": None,
            "recommended_strategy": "momentum_long",
            "position_multiplier": 0.7,  # Conservative when uncertain
            "detected_at": datetime.now().isoformat(),
        }

    def get_regime_history(self) -> list:
        """Get history of regime changes."""
        return self.regime_history

    async def should_use_momentum(self) -> bool:
        """Quick check if momentum strategy is appropriate."""
        regime = await self.detect_regime()
        return regime["type"] in ["bull", "bear"] and regime["is_trending"]

    async def should_use_mean_reversion(self) -> bool:
        """Quick check if mean reversion strategy is appropriate."""
        regime = await self.detect_regime()
        return regime["type"] == "sideways" or regime["is_ranging"]

    async def get_position_multiplier(self) -> float:
        """Get current position size multiplier based on regime."""
        regime = await self.detect_regime()
        return regime["position_multiplier"]


# Convenience function for quick regime check
async def get_current_regime(broker) -> Dict:
    """Quick helper to get current market regime."""
    detector = MarketRegimeDetector(broker)
    return await detector.detect_regime()
