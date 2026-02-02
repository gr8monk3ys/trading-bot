"""
Quality Factor - Financial quality metrics

Measures the quality of a company's financials.
Uses ROE, profit margins, debt levels, and liquidity.

Higher quality = higher factor score.

Expected Alpha: +5-8% annually (defensive in bear markets).

Usage:
    from factors.quality_factor import QualityFactor

    factor = QualityFactor(broker)
    score = await factor.calculate_score("AAPL")
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)

# Shared cache with value_factor (same yfinance data)
from factors.value_factor import _fundamental_cache, _CACHE_TTL_HOURS


class QualityFactor(BaseFactor):
    """
    Quality factor using profitability and balance sheet metrics.

    Components:
    - ROE (35% weight): Return on Equity
    - Profit Margins (25% weight): Net profit margin
    - Debt/Equity (25% weight): Lower is better
    - Current Ratio (15% weight): Liquidity measure

    Higher quality = higher scores.
    """

    @property
    def factor_name(self) -> str:
        return "quality"

    @property
    def higher_is_better(self) -> bool:
        return True

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate quality score using fundamental metrics.

        Returns composite score where higher = better quality.
        """
        try:
            # Get fundamental data
            fundamentals = await self._get_fundamentals(symbol)

            if fundamentals is None:
                return 0.5, {"error": "no_data"}

            # Extract metrics (use safe defaults)
            roe = fundamentals.get("roe", 0.10)
            margin = fundamentals.get("profit_margin", 0.05)
            debt_equity = fundamentals.get("debt_equity", 100.0)
            current_ratio = fundamentals.get("current_ratio", 1.0)

            # Handle edge cases
            if roe is None or roe < -1.0:
                roe = 0.0
            if margin is None or margin < -1.0:
                margin = 0.0
            if debt_equity is None or debt_equity < 0:
                debt_equity = 100.0
            if current_ratio is None or current_ratio < 0:
                current_ratio = 1.0

            # Score each component
            roe_score = self._score_ratio(roe, low=0.0, high=0.30)
            margin_score = self._score_ratio(margin, low=0.0, high=0.25)
            debt_score = self._score_ratio_inverse(debt_equity, low=0.0, high=200.0)
            liquidity_score = self._score_ratio(current_ratio, low=0.5, high=3.0)

            # Weighted composite
            composite = (
                0.35 * roe_score +
                0.25 * margin_score +
                0.25 * debt_score +
                0.15 * liquidity_score
            )

            metadata = {
                "roe": roe,
                "profit_margin": margin,
                "debt_equity": debt_equity,
                "current_ratio": current_ratio,
                "roe_score": roe_score,
                "margin_score": margin_score,
                "debt_score": debt_score,
                "liquidity_score": liquidity_score,
            }

            return composite, metadata

        except Exception as e:
            logger.warning(f"Error calculating quality for {symbol}: {e}")
            return 0.5, {"error": str(e)}

    def _score_ratio(self, value: float, low: float, high: float) -> float:
        """Score where higher value = higher score."""
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)

    def _score_ratio_inverse(self, value: float, low: float, high: float) -> float:
        """Score where lower value = higher score (for debt/equity)."""
        if value >= high:
            return 0.0
        if value <= low:
            return 1.0
        return 1.0 - (value - low) / (high - low)

    async def _get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Get fundamental data for a symbol using yfinance.

        Uses shared cache with ValueFactor.
        """
        global _fundamental_cache

        # Check cache for full data
        cache_key = f"{symbol}_quality"
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

            # Extract quality metrics
            fundamentals = {
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "debt_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
            }

            # Cache result
            _fundamental_cache[cache_key] = (fundamentals, datetime.now())

            logger.debug(f"Fetched quality data for {symbol}: ROE={fundamentals['roe']}")
            return fundamentals

        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            logger.warning(f"Error fetching quality data for {symbol}: {e}")
            return None
