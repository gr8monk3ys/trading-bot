"""
Earnings Surprise Factor (SUE) - Standardized Unexpected Earnings

Measures the magnitude and direction of earnings surprises.
Based on Post-Earnings Announcement Drift (PEAD) anomaly.

Positive surprise = stock continues to outperform for 3-6 months.
Negative surprise = stock continues to underperform.

Expected Alpha: +5-10% annually (PEAD effect is well-documented).

Usage:
    from factors.earnings_factor import EarningsSurpriseFactor

    factor = EarningsSurpriseFactor(broker)
    score = await factor.calculate_score("AAPL")
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)

# Cache for earnings data (refreshed daily)
_earnings_cache: Dict[str, Tuple[Dict, datetime]] = {}
_CACHE_TTL_HOURS = 24


class EarningsSurpriseFactor(BaseFactor):
    """
    Earnings Surprise Factor using Standardized Unexpected Earnings (SUE).

    SUE = (Actual EPS - Expected EPS) / Std(Surprises)

    Higher SUE = more positive surprise = higher score.

    Components:
    - Latest quarter surprise (50% weight)
    - Average of last 4 quarters (30% weight)
    - Consistency of beats (20% weight)
    """

    @property
    def factor_name(self) -> str:
        return "earnings_surprise"

    @property
    def higher_is_better(self) -> bool:
        return True

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate earnings surprise score.

        Returns composite score where higher = more positive surprises.
        """
        try:
            # Get earnings data
            earnings_data = await self._get_earnings_data(symbol)

            if earnings_data is None or not earnings_data.get("surprises"):
                return 0.5, {"error": "no_data"}

            surprises = earnings_data["surprises"]

            if len(surprises) < 1:
                return 0.5, {"error": "insufficient_data"}

            # Component 1: Latest quarter surprise
            latest_surprise = surprises[0]

            # Component 2: Average of last 4 quarters
            avg_surprise = np.mean(surprises[:4]) if len(surprises) >= 1 else latest_surprise

            # Component 3: Consistency (% of beats)
            beats = sum(1 for s in surprises[:4] if s > 0)
            beat_rate = beats / min(len(surprises), 4)

            # Calculate SUE (Standardized Unexpected Earnings)
            surprise_std = np.std(surprises) if len(surprises) > 1 else 1.0
            surprise_std = max(surprise_std, 0.01)  # Avoid division by zero
            sue = avg_surprise / surprise_std

            # Score each component
            latest_score = self._score_surprise(latest_surprise)
            avg_score = self._score_surprise(avg_surprise)
            consistency_score = beat_rate

            # Weighted composite
            composite = (
                0.50 * latest_score +
                0.30 * avg_score +
                0.20 * consistency_score
            )

            metadata = {
                "latest_surprise_pct": latest_surprise * 100,
                "avg_surprise_pct": avg_surprise * 100,
                "beat_rate": beat_rate,
                "sue": sue,
                "num_quarters": len(surprises),
                "latest_score": latest_score,
                "avg_score": avg_score,
            }

            return composite, metadata

        except Exception as e:
            logger.warning(f"Error calculating earnings surprise for {symbol}: {e}")
            return 0.5, {"error": str(e)}

    def _score_surprise(self, surprise_pct: float) -> float:
        """
        Score a surprise percentage.

        Maps -30% to +30% surprise to 0-1 score.
        """
        # Surprise is typically expressed as (actual-expected)/expected
        # Range from -0.3 to +0.3 (30% miss to 30% beat)
        if surprise_pct <= -0.3:
            return 0.0
        if surprise_pct >= 0.3:
            return 1.0
        return (surprise_pct + 0.3) / 0.6

    async def _get_earnings_data(self, symbol: str) -> Optional[Dict]:
        """
        Get earnings surprise data using yfinance.

        Uses 24h cache to avoid rate limits.
        """
        global _earnings_cache

        # Check cache
        if symbol in _earnings_cache:
            data, cached_time = _earnings_cache[symbol]
            if (datetime.now() - cached_time).total_seconds() < _CACHE_TTL_HOURS * 3600:
                return data

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)

            # Try to get earnings dates with surprises
            surprises = []

            try:
                # Method 1: Try earnings_dates (has Surprise column)
                earnings = ticker.earnings_dates
                if earnings is not None and len(earnings) > 0:
                    for _, row in earnings.head(8).iterrows():
                        if "Surprise(%)" in earnings.columns:
                            surprise = row.get("Surprise(%)")
                            if surprise is not None and not np.isnan(surprise):
                                surprises.append(surprise / 100)  # Convert to decimal
            except Exception:
                pass

            # Method 2: If no surprise data, try to calculate from EPS
            if not surprises:
                try:
                    info = ticker.info
                    actual_eps = info.get("trailingEps")
                    forward_eps = info.get("forwardEps")

                    if actual_eps and forward_eps and forward_eps > 0:
                        # Rough approximation of recent performance
                        implied_surprise = (actual_eps / forward_eps) - 1
                        surprises = [implied_surprise]
                except Exception:
                    pass

            earnings_data = {
                "surprises": surprises,
                "num_quarters": len(surprises),
            }

            # Cache result
            _earnings_cache[symbol] = (earnings_data, datetime.now())

            logger.debug(f"Fetched earnings for {symbol}: {len(surprises)} quarters")
            return earnings_data

        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            logger.warning(f"Error fetching earnings for {symbol}: {e}")
            return None


def clear_earnings_cache():
    """Clear the earnings data cache."""
    global _earnings_cache
    _earnings_cache.clear()
    logger.info("Earnings cache cleared")
