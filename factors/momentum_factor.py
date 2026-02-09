"""
Momentum Factor - 12-1 Month Momentum (Jegadeesh-Titman)

Implements the classic academic momentum factor:
- Calculate 12-month return
- Subtract most recent 1-month return (skip month)
- This avoids short-term reversal and captures medium-term momentum

Research shows:
- Winner stocks tend to continue winning
- Skip month reduces reversal effect
- Works best with cross-sectional ranking

Expected Impact: 5-10% annual alpha from momentum timing
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class MomentumFactor(BaseFactor):
    """
    Classic 12-1 month momentum factor.

    Calculates: 12-month return minus 1-month return
    This captures intermediate momentum while avoiding short-term reversal.
    """

    def __init__(
        self,
        broker,
        lookback_months: int = 12,
        skip_months: int = 1,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize momentum factor.

        Args:
            broker: Trading broker instance
            lookback_months: Total lookback period (default 12)
            skip_months: Recent months to skip (default 1)
            cache_ttl_seconds: Cache TTL (default 1 hour)
        """
        super().__init__(broker, cache_ttl_seconds)
        self.lookback_months = lookback_months
        self.skip_months = skip_months

        # Convert months to trading days (approx 21 days/month)
        self.lookback_days = lookback_months * 21
        self.skip_days = skip_months * 21

    @property
    def factor_name(self) -> str:
        return f"Momentum_{self.lookback_months}M_skip{self.skip_months}M"

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate 12-1 month momentum score.

        Args:
            symbol: Stock symbol
            price_data: Optional pre-fetched price data

        Returns:
            Tuple of (momentum_score, metadata)
        """
        # Get price data if not provided
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 30)

        if price_data is None or len(price_data) < self.lookback_days:
            logger.debug(f"Insufficient data for {symbol} momentum calculation")
            return (np.nan, {"error": "insufficient_data"})

        closes = np.array([d["close"] for d in price_data])

        # Ensure we have enough data
        if len(closes) < self.lookback_days:
            return (np.nan, {"error": "insufficient_data"})

        try:
            # Current price (after skipping recent period)
            # Skip the most recent skip_days
            current_idx = -1 - self.skip_days
            if abs(current_idx) > len(closes):
                current_idx = -len(closes) // 2

            current_price = closes[current_idx]

            # Price lookback_days ago from the skip point
            lookback_idx = current_idx - self.lookback_days + self.skip_days
            if abs(lookback_idx) > len(closes):
                lookback_idx = 0

            lookback_price = closes[lookback_idx]

            # Calculate 12-month return (excluding skip period)
            if lookback_price <= 0:
                return (np.nan, {"error": "invalid_price"})

            momentum_12m = (current_price - lookback_price) / lookback_price

            # Calculate 1-month return (the skip period we're excluding)
            recent_price = closes[-1]
            skip_start_price = closes[-1 - self.skip_days] if len(closes) > self.skip_days else closes[0]

            if skip_start_price <= 0:
                return (np.nan, {"error": "invalid_price"})

            momentum_1m = (recent_price - skip_start_price) / skip_start_price

            # 12-1 momentum: total momentum minus recent momentum
            momentum_12_1 = momentum_12m - momentum_1m

            metadata = {
                "momentum_12m": momentum_12m,
                "momentum_1m": momentum_1m,
                "momentum_12_1": momentum_12_1,
                "current_price": current_price,
                "lookback_price": lookback_price,
                "data_points": len(closes),
            }

            return (momentum_12_1, metadata)

        except Exception as e:
            logger.error(f"Error calculating momentum for {symbol}: {e}")
            return (np.nan, {"error": str(e)})


class ShortTermMomentumFactor(BaseFactor):
    """
    Short-term momentum factor (1-3 months).

    Useful for faster-moving strategies or combining with longer-term momentum.
    """

    def __init__(
        self,
        broker,
        lookback_days: int = 63,  # ~3 months
        cache_ttl_seconds: int = 1800,
    ):
        super().__init__(broker, cache_ttl_seconds)
        self.lookback_days = lookback_days

    @property
    def factor_name(self) -> str:
        return f"ShortMomentum_{self.lookback_days}D"

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate short-term momentum score."""
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 10)

        if price_data is None or len(price_data) < self.lookback_days:
            return (np.nan, {"error": "insufficient_data"})

        closes = np.array([d["close"] for d in price_data])

        try:
            current_price = closes[-1]
            lookback_price = closes[-self.lookback_days]

            if lookback_price <= 0:
                return (np.nan, {"error": "invalid_price"})

            momentum = (current_price - lookback_price) / lookback_price

            metadata = {
                "momentum": momentum,
                "current_price": current_price,
                "lookback_price": lookback_price,
            }

            return (momentum, metadata)

        except Exception as e:
            logger.error(f"Error calculating short momentum for {symbol}: {e}")
            return (np.nan, {"error": str(e)})


class RelativeStrengthFactor(BaseFactor):
    """
    Relative strength factor vs benchmark (SPY).

    Measures how much a stock outperforms/underperforms the market.
    """

    def __init__(
        self,
        broker,
        benchmark: str = "SPY",
        lookback_days: int = 63,
        cache_ttl_seconds: int = 1800,
    ):
        super().__init__(broker, cache_ttl_seconds)
        self.benchmark = benchmark
        self.lookback_days = lookback_days
        self._benchmark_data = None
        self._benchmark_time = None

    @property
    def factor_name(self) -> str:
        return f"RelativeStrength_vs_{self.benchmark}"

    async def _get_benchmark_data(self) -> Optional[List[Dict]]:
        """Get cached benchmark data."""
        from datetime import datetime

        now = datetime.now()
        if (
            self._benchmark_data is not None
            and self._benchmark_time is not None
            and (now - self._benchmark_time).total_seconds() < 3600
        ):
            return self._benchmark_data

        self._benchmark_data = await self.get_price_data(
            self.benchmark, days=self.lookback_days + 10
        )
        self._benchmark_time = now
        return self._benchmark_data

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate relative strength vs benchmark."""
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 10)

        benchmark_data = await self._get_benchmark_data()

        if price_data is None or benchmark_data is None:
            return (np.nan, {"error": "insufficient_data"})

        if len(price_data) < self.lookback_days or len(benchmark_data) < self.lookback_days:
            return (np.nan, {"error": "insufficient_data"})

        try:
            # Stock return
            stock_closes = np.array([d["close"] for d in price_data])
            stock_return = (stock_closes[-1] - stock_closes[-self.lookback_days]) / stock_closes[-self.lookback_days]

            # Benchmark return
            bench_closes = np.array([d["close"] for d in benchmark_data])
            bench_return = (bench_closes[-1] - bench_closes[-self.lookback_days]) / bench_closes[-self.lookback_days]

            # Relative strength = stock return - benchmark return
            relative_strength = stock_return - bench_return

            metadata = {
                "stock_return": stock_return,
                "benchmark_return": bench_return,
                "relative_strength": relative_strength,
                "outperformance": relative_strength > 0,
            }

            return (relative_strength, metadata)

        except Exception as e:
            logger.error(f"Error calculating relative strength for {symbol}: {e}")
            return (np.nan, {"error": str(e)})


# Sector ETF mapping for sector-relative momentum
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    # Alternative mappings for common variations
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Basic Materials": "XLB",
    "Financial Services": "XLF",
}


class SectorRelativeMomentumFactor(BaseFactor):
    """
    Sector-Relative Momentum Factor.

    Measures momentum relative to the stock's sector, not the broad market.
    This is orthogonal to absolute momentum and provides additional alpha.

    A stock that's up 10% when its sector is up 15% has negative sector-relative
    momentum, even though absolute momentum is positive.

    Expected Alpha: +3-5% annually (orthogonal to absolute momentum)
    """

    def __init__(
        self,
        broker,
        lookback_days: int = 63,  # ~3 months
        cache_ttl_seconds: int = 1800,
    ):
        """
        Initialize sector-relative momentum factor.

        Args:
            broker: Trading broker instance
            lookback_days: Lookback period for momentum calculation
            cache_ttl_seconds: Cache TTL (default 30 minutes)
        """
        super().__init__(broker, cache_ttl_seconds)
        self.lookback_days = lookback_days
        self._sector_cache = {}  # symbol -> sector mapping
        self._sector_etf_data = {}  # etf -> price data cache
        self._sector_etf_time = {}  # etf -> cache time

    @property
    def factor_name(self) -> str:
        return f"SectorRelMom_{self.lookback_days}D"

    @property
    def higher_is_better(self) -> bool:
        return True

    async def _get_sector(self, symbol: str) -> Optional[str]:
        """
        Get the sector for a symbol using yfinance.

        Uses caching to avoid repeated API calls.
        """
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector = info.get("sector")
            if sector:
                self._sector_cache[symbol] = sector
                return sector

            return None

        except ImportError:
            logger.warning("yfinance not installed. Cannot determine sector.")
            return None
        except Exception as e:
            logger.debug(f"Could not get sector for {symbol}: {e}")
            return None

    async def _get_sector_etf_data(self, etf: str) -> Optional[List[Dict]]:
        """Get cached sector ETF price data."""
        from datetime import datetime

        now = datetime.now()
        cache_key = etf

        if (
            cache_key in self._sector_etf_data
            and cache_key in self._sector_etf_time
            and (now - self._sector_etf_time[cache_key]).total_seconds() < 3600
        ):
            return self._sector_etf_data[cache_key]

        data = await self.get_price_data(etf, days=self.lookback_days + 10)
        if data:
            self._sector_etf_data[cache_key] = data
            self._sector_etf_time[cache_key] = now

        return data

    async def _calculate_momentum(self, price_data: List[Dict]) -> Optional[float]:
        """Calculate simple momentum from price data."""
        if price_data is None or len(price_data) < self.lookback_days:
            return None

        closes = np.array([d["close"] for d in price_data])
        if len(closes) < self.lookback_days:
            return None

        current_price = closes[-1]
        lookback_price = closes[-self.lookback_days]

        if lookback_price <= 0:
            return None

        return (current_price - lookback_price) / lookback_price

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate sector-relative momentum.

        Sector-relative momentum = Stock momentum - Sector ETF momentum

        Returns:
            Tuple of (sector_relative_score, metadata)
        """
        # Get price data if not provided
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 10)

        if price_data is None or len(price_data) < self.lookback_days:
            return 0.5, {"error": "insufficient_data"}

        try:
            # Get stock momentum
            stock_momentum = await self._calculate_momentum(price_data)
            if stock_momentum is None:
                return 0.5, {"error": "could_not_calculate_momentum"}

            # Get sector
            sector = await self._get_sector(symbol)
            if sector is None:
                # Fall back to SPY if sector unknown
                sector_etf = "SPY"
                sector = "Unknown"
            else:
                sector_etf = SECTOR_ETFS.get(sector, "SPY")

            # Get sector ETF momentum
            sector_data = await self._get_sector_etf_data(sector_etf)
            sector_momentum = await self._calculate_momentum(sector_data)

            if sector_momentum is None:
                # If we can't get sector data, return neutral
                return 0.5, {"error": "could_not_get_sector_data", "sector": sector}

            # Sector-relative momentum
            relative_momentum = stock_momentum - sector_momentum

            # Score: -30% to +30% relative momentum maps to 0-1
            score = self._score_relative(relative_momentum, low=-0.30, high=0.30)

            metadata = {
                "stock_momentum": stock_momentum,
                "stock_momentum_pct": stock_momentum * 100,
                "sector": sector,
                "sector_etf": sector_etf,
                "sector_momentum": sector_momentum,
                "sector_momentum_pct": sector_momentum * 100,
                "relative_momentum": relative_momentum,
                "relative_momentum_pct": relative_momentum * 100,
                "outperforms_sector": relative_momentum > 0,
            }

            return score, metadata

        except Exception as e:
            logger.error(f"Error calculating sector-relative momentum for {symbol}: {e}")
            return 0.5, {"error": str(e)}

    def _score_relative(self, value: float, low: float, high: float) -> float:
        """Score where higher value = higher score."""
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)
