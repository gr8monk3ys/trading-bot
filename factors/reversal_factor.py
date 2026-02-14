"""
Short-Term Reversal Factor - 1-Month Reversal

Implements the short-term reversal anomaly:
- Stocks that performed poorly in the last month tend to outperform
- Stocks that performed well in the last month tend to underperform
- This is the opposite of momentum and provides diversification

Research shows:
- 1-month reversal is well-documented in academic literature
- Negatively correlated with 12-1 momentum
- Works best at the individual stock level
- Driven by investor overreaction and liquidity effects

Expected Alpha: +3-5% annually (uncorrelated with momentum)

Usage:
    from factors.reversal_factor import ReversalFactor

    factor = ReversalFactor(broker)
    score = await factor.calculate_score("AAPL")
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class ReversalFactor(BaseFactor):
    """
    1-month reversal factor (losers outperform).

    This factor exploits the short-term reversal effect:
    - Recent losers score higher (expected to rebound)
    - Recent winners score lower (expected to mean-revert)

    Negatively correlated with momentum, provides diversification.
    """

    def __init__(
        self,
        broker,
        lookback_days: int = 22,  # ~1 month of trading days
        cache_ttl_seconds: int = 1800,
    ):
        """
        Initialize reversal factor.

        Args:
            broker: Trading broker instance
            lookback_days: Days to calculate reversal over (default 22)
            cache_ttl_seconds: Cache TTL (default 30 minutes)
        """
        super().__init__(broker, cache_ttl_seconds)
        self.lookback_days = lookback_days

    @property
    def factor_name(self) -> str:
        return f"Reversal_{self.lookback_days}D"

    @property
    def higher_is_better(self) -> bool:
        # Higher score = more attractive (losers are more attractive)
        return True

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate 1-month reversal score.

        Lower 1-month return = higher score (losers expected to outperform).

        Returns:
            Tuple of (reversal_score, metadata)
        """
        # Get price data if not provided
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 10)

        if price_data is None or len(price_data) < self.lookback_days:
            logger.debug(f"Insufficient data for {symbol} reversal calculation")
            return 0.5, {"error": "insufficient_data"}

        closes = np.array([d["close"] for d in price_data])

        if len(closes) < self.lookback_days:
            return 0.5, {"error": "insufficient_data"}

        try:
            # Calculate 1-month return
            current_price = closes[-1]
            lookback_price = closes[-self.lookback_days]

            if lookback_price <= 0:
                return 0.5, {"error": "invalid_price"}

            return_1m = (current_price - lookback_price) / lookback_price

            # Invert: lower 1-month return = higher score
            # Range: -15% to +15% maps to 1.0 to 0.0
            reversal_score = self._score_inverse(return_1m, low=-0.15, high=0.15)

            # Also calculate additional metrics for analysis
            # Weekly returns for pattern detection
            weekly_returns = []
            for i in range(0, min(self.lookback_days, len(closes) - 5), 5):
                if len(closes) > i + 5:
                    w_return = (closes[-(i + 1)] - closes[-(i + 6)]) / closes[-(i + 6)]
                    weekly_returns.append(w_return)

            # Calculate volatility of returns (higher vol = stronger reversal signal)
            if len(closes) > 5:
                daily_returns = (
                    np.diff(closes[-self.lookback_days :]) / closes[-self.lookback_days : -1]
                )
                return_volatility = np.std(daily_returns) * np.sqrt(252)
            else:
                return_volatility = 0.0

            metadata = {
                "return_1m": return_1m,
                "return_1m_pct": return_1m * 100,
                "reversal_score": reversal_score,
                "current_price": current_price,
                "lookback_price": lookback_price,
                "return_volatility": return_volatility,
                "weekly_returns": weekly_returns[:4] if weekly_returns else [],
                "is_loser": return_1m < 0,
                "is_winner": return_1m > 0,
            }

            return reversal_score, metadata

        except Exception as e:
            logger.error(f"Error calculating reversal for {symbol}: {e}")
            return 0.5, {"error": str(e)}

    def _score_inverse(self, value: float, low: float, high: float) -> float:
        """
        Score where lower value = higher score.

        Used for reversal where losers get higher scores.
        """
        if value >= high:
            return 0.0
        if value <= low:
            return 1.0
        return 1.0 - (value - low) / (high - low)


class WeeklyReversalFactor(BaseFactor):
    """
    Weekly reversal factor (5-day reversal).

    Even shorter-term reversal for higher-frequency strategies.
    More driven by liquidity effects and bid-ask bounce.
    """

    def __init__(
        self,
        broker,
        lookback_days: int = 5,
        cache_ttl_seconds: int = 900,  # 15 minutes
    ):
        super().__init__(broker, cache_ttl_seconds)
        self.lookback_days = lookback_days

    @property
    def factor_name(self) -> str:
        return f"WeeklyReversal_{self.lookback_days}D"

    @property
    def higher_is_better(self) -> bool:
        return True

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate 5-day reversal score."""
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 5)

        if price_data is None or len(price_data) < self.lookback_days:
            return 0.5, {"error": "insufficient_data"}

        closes = np.array([d["close"] for d in price_data])

        try:
            current_price = closes[-1]
            lookback_price = closes[-self.lookback_days]

            if lookback_price <= 0:
                return 0.5, {"error": "invalid_price"}

            return_5d = (current_price - lookback_price) / lookback_price

            # Invert: lower return = higher score
            # Range: -5% to +5% for weekly
            score = self._score_inverse(return_5d, low=-0.05, high=0.05)

            metadata = {
                "return_5d": return_5d,
                "return_5d_pct": return_5d * 100,
                "current_price": current_price,
                "lookback_price": lookback_price,
            }

            return score, metadata

        except Exception as e:
            logger.error(f"Error calculating weekly reversal for {symbol}: {e}")
            return 0.5, {"error": str(e)}

    def _score_inverse(self, value: float, low: float, high: float) -> float:
        """Score where lower value = higher score."""
        if value >= high:
            return 0.0
        if value <= low:
            return 1.0
        return 1.0 - (value - low) / (high - low)
