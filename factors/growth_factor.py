"""
Growth Factor - Earnings and revenue growth metrics

Measures how fast a company is growing.
Uses earnings growth, revenue growth, and quarterly earnings growth.

Higher growth = higher factor score.

Expected Alpha: +3-5% annually (especially in bull markets).

Usage:
    from factors.growth_factor import GrowthFactor

    factor = GrowthFactor(broker)
    score = await factor.calculate_score("AAPL")
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)

# Shared cache with other fundamental factors
from factors.value_factor import _fundamental_cache, _CACHE_TTL_HOURS


class GrowthFactor(BaseFactor):
    """
    Growth factor using earnings and revenue growth rates.

    Components:
    - Earnings Growth (40% weight): YoY earnings growth
    - Revenue Growth (35% weight): YoY revenue growth
    - Quarterly Earnings Growth (25% weight): Recent quarter growth

    Higher growth = higher scores.
    """

    @property
    def factor_name(self) -> str:
        return "growth"

    @property
    def higher_is_better(self) -> bool:
        return True

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate growth score using fundamental metrics.

        Returns composite score where higher = faster growth.
        """
        try:
            # Get fundamental data
            fundamentals = await self._get_fundamentals(symbol)

            if fundamentals is None:
                return 0.5, {"error": "no_data"}

            # Extract metrics (use safe defaults)
            earnings_growth = fundamentals.get("earnings_growth", 0.0)
            revenue_growth = fundamentals.get("revenue_growth", 0.0)
            quarterly_earnings = fundamentals.get("quarterly_earnings_growth", 0.0)

            # Handle edge cases and None values
            if earnings_growth is None:
                earnings_growth = 0.0
            if revenue_growth is None:
                revenue_growth = 0.0
            if quarterly_earnings is None:
                quarterly_earnings = 0.0

            # Clip extreme values
            earnings_growth = np.clip(earnings_growth, -0.5, 1.0)
            revenue_growth = np.clip(revenue_growth, -0.3, 0.8)
            quarterly_earnings = np.clip(quarterly_earnings, -0.5, 1.0)

            # Score each component (growth from -20% to +50% maps to 0-1)
            eg_score = self._score_ratio(earnings_growth, low=-0.2, high=0.5)
            rg_score = self._score_ratio(revenue_growth, low=-0.1, high=0.4)
            qe_score = self._score_ratio(quarterly_earnings, low=-0.3, high=0.5)

            # Weighted composite
            composite = (
                0.40 * eg_score +
                0.35 * rg_score +
                0.25 * qe_score
            )

            metadata = {
                "earnings_growth": earnings_growth,
                "revenue_growth": revenue_growth,
                "quarterly_earnings_growth": quarterly_earnings,
                "eg_score": eg_score,
                "rg_score": rg_score,
                "qe_score": qe_score,
            }

            return composite, metadata

        except Exception as e:
            logger.warning(f"Error calculating growth for {symbol}: {e}")
            return 0.5, {"error": str(e)}

    def _score_ratio(self, value: float, low: float, high: float) -> float:
        """Score where higher value = higher score."""
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)

    async def _get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Get fundamental data for a symbol using yfinance.

        Uses shared cache with other fundamental factors.
        """
        global _fundamental_cache

        # Check cache
        cache_key = f"{symbol}_growth"
        if cache_key in _fundamental_cache:
            data, cached_time = _fundamental_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < _CACHE_TTL_HOURS * 3600:
                return data

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            # Extract growth metrics
            fundamentals = {
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
                "quarterly_earnings_growth": info.get("earningsQuarterlyGrowth"),
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
            }

            # Cache result
            _fundamental_cache[cache_key] = (fundamentals, datetime.now())

            logger.debug(
                f"Fetched growth data for {symbol}: "
                f"Earnings={fundamentals['earnings_growth']}, Revenue={fundamentals['revenue_growth']}"
            )
            return fundamentals

        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            logger.warning(f"Error fetching growth data for {symbol}: {e}")
            return None
