"""
Value Factor - Fundamental value metrics

Measures how cheap a stock is relative to its fundamentals.
Uses P/E, P/B, P/S, and Dividend Yield.

Lower valuation ratios = higher factor score (more value).

Expected Alpha: +8-12% annually (academic research on value premium).

Usage:
    from factors.value_factor import ValueFactor

    factor = ValueFactor(broker)
    score = await factor.calculate_score("AAPL")
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)

# Cache for fundamental data (24h TTL to avoid rate limits)
_fundamental_cache: Dict[str, Tuple[Dict, datetime]] = {}
_CACHE_TTL_HOURS = 24


class ValueFactor(BaseFactor):
    """
    Value factor using fundamental valuation ratios.

    Components:
    - P/E ratio (30% weight): Price to Earnings
    - P/B ratio (25% weight): Price to Book
    - P/S ratio (25% weight): Price to Sales
    - Dividend Yield (20% weight): Annual dividend / price

    Lower valuations = higher scores (value stocks).
    """

    @property
    def factor_name(self) -> str:
        return "value"

    @property
    def higher_is_better(self) -> bool:
        # For value, lower ratios are better, but we invert in calculate_raw_score
        # so higher scores = more value
        return True

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate value score using fundamental ratios.

        Returns composite score where higher = more value (cheaper).
        """
        try:
            # Get fundamental data
            fundamentals = await self._get_fundamentals(symbol)

            if fundamentals is None:
                return 0.5, {"error": "no_data"}

            # Extract ratios (use safe defaults)
            pe = fundamentals.get("trailing_pe", 50.0)
            pb = fundamentals.get("price_to_book", 5.0)
            ps = fundamentals.get("price_to_sales", 5.0)
            dy = fundamentals.get("dividend_yield", 0.0) or 0.0

            # Handle edge cases
            if pe is None or pe <= 0 or pe > 200:
                pe = 50.0  # Neutral
            if pb is None or pb <= 0 or pb > 50:
                pb = 5.0
            if ps is None or ps <= 0 or ps > 50:
                ps = 5.0
            if dy is None or dy < 0:
                dy = 0.0

            # Score each component (lower ratio = higher score)
            pe_score = self._score_ratio_inverse(pe, low=5, high=50)
            pb_score = self._score_ratio_inverse(pb, low=0.5, high=10)
            ps_score = self._score_ratio_inverse(ps, low=0.5, high=10)
            dy_score = self._score_ratio(dy, low=0.0, high=0.08)

            # Weighted composite
            composite = (
                0.30 * pe_score +
                0.25 * pb_score +
                0.25 * ps_score +
                0.20 * dy_score
            )

            metadata = {
                "trailing_pe": pe,
                "price_to_book": pb,
                "price_to_sales": ps,
                "dividend_yield": dy,
                "pe_score": pe_score,
                "pb_score": pb_score,
                "ps_score": ps_score,
                "dy_score": dy_score,
            }

            return composite, metadata

        except Exception as e:
            logger.warning(f"Error calculating value for {symbol}: {e}")
            return 0.5, {"error": str(e)}

    def _score_ratio(self, value: float, low: float, high: float) -> float:
        """Score where higher value = higher score."""
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)

    def _score_ratio_inverse(self, value: float, low: float, high: float) -> float:
        """Score where lower value = higher score (for P/E, P/B, P/S)."""
        if value >= high:
            return 0.0
        if value <= low:
            return 1.0
        return 1.0 - (value - low) / (high - low)

    async def _get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Get fundamental data for a symbol using yfinance.

        Uses 24h cache to avoid rate limits.
        """
        global _fundamental_cache

        # Check cache
        if symbol in _fundamental_cache:
            data, cached_time = _fundamental_cache[symbol]
            if (datetime.now() - cached_time).total_seconds() < _CACHE_TTL_HOURS * 3600:
                return data

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            # Extract key metrics
            fundamentals = {
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield"),
                "enterprise_value": info.get("enterpriseValue"),
                "market_cap": info.get("marketCap"),
            }

            # Cache result
            _fundamental_cache[symbol] = (fundamentals, datetime.now())

            logger.debug(f"Fetched fundamentals for {symbol}: PE={fundamentals['trailing_pe']}")
            return fundamentals

        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
            return None


# Utility function to clear the fundamental cache
def clear_fundamental_cache():
    """Clear the global fundamental data cache."""
    global _fundamental_cache
    _fundamental_cache.clear()
    logger.info("Fundamental cache cleared")
