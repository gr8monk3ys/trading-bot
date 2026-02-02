"""
Fundamental Data Provider

Fetches fundamental data for value and quality factor calculations:
- P/E ratio (trailing and forward)
- P/B ratio (price-to-book)
- ROE (return on equity)
- ROA (return on assets)
- Debt/Equity ratio
- Revenue and earnings growth

Uses yfinance (free) as primary data source.

Usage:
    from utils.fundamental_data import FundamentalDataProvider

    provider = FundamentalDataProvider()
    data = await provider.get_fundamentals("AAPL")

    # data = {
    #     'pe_ratio': 28.5,
    #     'pb_ratio': 45.2,
    #     'roe': 0.145,
    #     ...
    # }
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import yfinance
yf = None


@dataclass
class FundamentalData:
    """Fundamental data for a stock."""

    symbol: str
    timestamp: datetime

    # Valuation metrics
    pe_ratio: Optional[float] = None  # Trailing P/E
    forward_pe: Optional[float] = None  # Forward P/E
    pb_ratio: Optional[float] = None  # Price-to-Book
    ps_ratio: Optional[float] = None  # Price-to-Sales
    peg_ratio: Optional[float] = None  # P/E to Growth

    # Profitability metrics
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None

    # Growth metrics
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    eps_growth: Optional[float] = None

    # Financial health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

    # Dividend metrics
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # Size metrics
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None

    # Quality score (computed)
    quality_score: Optional[float] = None
    value_score: Optional[float] = None

    # Raw data for debugging
    raw_info: Dict[str, Any] = field(default_factory=dict)


class FundamentalDataProvider:
    """
    Provides fundamental data for stocks using yfinance.

    Features:
    - Caching to reduce API calls
    - Batch fetching for efficiency
    - Fallback values for missing data
    - Quality and value score computation
    """

    def __init__(self, cache_ttl_hours: int = 24):
        """
        Initialize provider.

        Args:
            cache_ttl_hours: How long to cache fundamental data
        """
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Cache
        self._cache: Dict[str, FundamentalData] = {}
        self._cache_time: Dict[str, datetime] = {}

        # Initialize yfinance
        self._init_yfinance()

    def _init_yfinance(self):
        """Lazy-load yfinance."""
        global yf
        if yf is None:
            try:
                import yfinance

                yf = yfinance
                logger.info("yfinance initialized for fundamental data")
            except ImportError:
                logger.warning(
                    "yfinance not installed - fundamental data will not be available. "
                    "Install with: pip install yfinance"
                )

    async def get_fundamentals(
        self, symbol: str, force_refresh: bool = False
    ) -> Optional[FundamentalData]:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Stock symbol
            force_refresh: Force refresh even if cached

        Returns:
            FundamentalData or None if unavailable
        """
        if yf is None:
            return None

        # Check cache
        if not force_refresh and self._is_cache_valid(symbol):
            return self._cache[symbol]

        try:
            # Run yfinance in thread pool (it's synchronous)
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._fetch_from_yfinance, symbol)

            if data:
                self._cache[symbol] = data
                self._cache_time[symbol] = datetime.now()

            return data

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    def _fetch_from_yfinance(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch data from yfinance (synchronous)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "symbol" not in info:
                return None

            # Extract valuation metrics
            pe_ratio = self._safe_get(info, "trailingPE")
            forward_pe = self._safe_get(info, "forwardPE")
            pb_ratio = self._safe_get(info, "priceToBook")
            ps_ratio = self._safe_get(info, "priceToSalesTrailing12Months")
            peg_ratio = self._safe_get(info, "pegRatio")

            # Profitability metrics
            roe = self._safe_get(info, "returnOnEquity")
            roa = self._safe_get(info, "returnOnAssets")
            profit_margin = self._safe_get(info, "profitMargins")
            operating_margin = self._safe_get(info, "operatingMargins")
            gross_margin = self._safe_get(info, "grossMargins")

            # Growth metrics
            revenue_growth = self._safe_get(info, "revenueGrowth")
            earnings_growth = self._safe_get(info, "earningsGrowth")
            eps_growth = self._safe_get(info, "earningsQuarterlyGrowth")

            # Financial health
            debt_to_equity = self._safe_get(info, "debtToEquity")
            if debt_to_equity is not None:
                debt_to_equity = debt_to_equity / 100  # Convert from percentage
            current_ratio = self._safe_get(info, "currentRatio")
            quick_ratio = self._safe_get(info, "quickRatio")

            # Dividend metrics
            dividend_yield = self._safe_get(info, "dividendYield")
            payout_ratio = self._safe_get(info, "payoutRatio")

            # Size metrics
            market_cap = self._safe_get(info, "marketCap")
            enterprise_value = self._safe_get(info, "enterpriseValue")

            # Create data object
            data = FundamentalData(
                symbol=symbol,
                timestamp=datetime.now(),
                pe_ratio=pe_ratio,
                forward_pe=forward_pe,
                pb_ratio=pb_ratio,
                ps_ratio=ps_ratio,
                peg_ratio=peg_ratio,
                roe=roe,
                roa=roa,
                profit_margin=profit_margin,
                operating_margin=operating_margin,
                gross_margin=gross_margin,
                revenue_growth=revenue_growth,
                earnings_growth=earnings_growth,
                eps_growth=eps_growth,
                debt_to_equity=debt_to_equity,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                dividend_yield=dividend_yield,
                payout_ratio=payout_ratio,
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                raw_info=info,
            )

            # Calculate quality and value scores
            data.quality_score = self._calculate_quality_score(data)
            data.value_score = self._calculate_value_score(data)

            return data

        except Exception as e:
            logger.debug(f"yfinance error for {symbol}: {e}")
            return None

    def _safe_get(self, info: Dict, key: str) -> Optional[float]:
        """Safely extract a numeric value from info dict."""
        try:
            value = info.get(key)
            if value is None:
                return None
            if isinstance(value, (int, float)) and value != float("inf"):
                return float(value)
            return None
        except (ValueError, TypeError):
            return None

    def _calculate_quality_score(self, data: FundamentalData) -> float:
        """
        Calculate quality score based on profitability and financial health.

        Score is 0-100, higher is better.
        """
        scores = []
        weights = []

        # ROE score (higher is better, typical range 0-30%)
        if data.roe is not None:
            roe_score = min(100, max(0, (data.roe / 0.25) * 100))
            scores.append(roe_score)
            weights.append(0.25)

        # ROA score (higher is better, typical range 0-15%)
        if data.roa is not None:
            roa_score = min(100, max(0, (data.roa / 0.12) * 100))
            scores.append(roa_score)
            weights.append(0.20)

        # Profit margin score (higher is better)
        if data.profit_margin is not None:
            margin_score = min(100, max(0, (data.profit_margin / 0.20) * 100))
            scores.append(margin_score)
            weights.append(0.20)

        # Debt/Equity score (lower is better)
        if data.debt_to_equity is not None:
            # D/E < 0.5 = good, > 2.0 = bad
            de_score = max(0, min(100, 100 - (data.debt_to_equity - 0.5) * 40))
            scores.append(de_score)
            weights.append(0.20)

        # Earnings growth score
        if data.earnings_growth is not None:
            growth_score = min(100, max(0, 50 + data.earnings_growth * 100))
            scores.append(growth_score)
            weights.append(0.15)

        if not scores:
            return 50.0  # Neutral if no data

        # Weighted average
        total_weight = sum(weights)
        quality_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return quality_score

    def _calculate_value_score(self, data: FundamentalData) -> float:
        """
        Calculate value score based on valuation metrics.

        Score is 0-100, higher is better (more undervalued).
        """
        scores = []
        weights = []

        # P/E score (lower is better, typical range 10-40)
        if data.pe_ratio is not None and data.pe_ratio > 0:
            # P/E of 15 = 100, P/E of 40 = 0
            pe_score = max(0, min(100, 100 - (data.pe_ratio - 10) * 4))
            scores.append(pe_score)
            weights.append(0.35)

        # P/B score (lower is better)
        if data.pb_ratio is not None and data.pb_ratio > 0:
            # P/B of 1 = 100, P/B of 6 = 0
            pb_score = max(0, min(100, 100 - (data.pb_ratio - 1) * 20))
            scores.append(pb_score)
            weights.append(0.25)

        # PEG score (lower is better)
        if data.peg_ratio is not None and 0 < data.peg_ratio < 5:
            # PEG of 1 = 70, PEG of 2 = 30
            peg_score = max(0, min(100, 100 - (data.peg_ratio - 0.5) * 40))
            scores.append(peg_score)
            weights.append(0.25)

        # P/S score (lower is better)
        if data.ps_ratio is not None and data.ps_ratio > 0:
            # P/S of 2 = 80, P/S of 10 = 0
            ps_score = max(0, min(100, 100 - (data.ps_ratio - 1) * 11))
            scores.append(ps_score)
            weights.append(0.15)

        if not scores:
            return 50.0  # Neutral if no data

        # Weighted average
        total_weight = sum(weights)
        value_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return value_score

    async def get_fundamentals_batch(
        self, symbols: List[str], max_concurrent: int = 5
    ) -> Dict[str, FundamentalData]:
        """
        Get fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols
            max_concurrent: Maximum concurrent requests

        Returns:
            Dict of symbol -> FundamentalData
        """
        if yf is None:
            return {}

        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(symbol):
            async with semaphore:
                return symbol, await self.get_fundamentals(symbol)

        tasks = [fetch_with_semaphore(s) for s in symbols]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, Exception):
                continue
            symbol, data = result
            if data:
                results[symbol] = data

        logger.info(
            f"Fetched fundamentals for {len(results)}/{len(symbols)} symbols"
        )

        return results

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self._cache or symbol not in self._cache_time:
            return False

        age = datetime.now() - self._cache_time[symbol]
        return age < self.cache_ttl

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_time.clear()


# Value and Quality factors using fundamental data
from factors.base_factor import BaseFactor, FactorScore
from typing import Tuple


class ValueFactor(BaseFactor):
    """
    Value factor based on fundamental data.

    Uses P/E, P/B, and other valuation metrics.
    """

    def __init__(
        self,
        broker,
        fundamental_provider: Optional[FundamentalDataProvider] = None,
        cache_ttl_seconds: int = 86400,  # 24 hours
    ):
        super().__init__(broker, cache_ttl_seconds)
        self.fundamental_provider = fundamental_provider or FundamentalDataProvider()

    @property
    def factor_name(self) -> str:
        return "Value"

    async def calculate_raw_score(
        self, symbol: str, price_data=None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate value score from fundamentals."""
        data = await self.fundamental_provider.get_fundamentals(symbol)

        if data is None or data.value_score is None:
            return (float("nan"), {"error": "no_fundamental_data"})

        metadata = {
            "value_score": data.value_score,
            "pe_ratio": data.pe_ratio,
            "pb_ratio": data.pb_ratio,
            "peg_ratio": data.peg_ratio,
            "ps_ratio": data.ps_ratio,
        }

        return (data.value_score, metadata)


class QualityFactor(BaseFactor):
    """
    Quality factor based on fundamental data.

    Uses ROE, margins, debt levels, and growth metrics.
    """

    def __init__(
        self,
        broker,
        fundamental_provider: Optional[FundamentalDataProvider] = None,
        cache_ttl_seconds: int = 86400,
    ):
        super().__init__(broker, cache_ttl_seconds)
        self.fundamental_provider = fundamental_provider or FundamentalDataProvider()

    @property
    def factor_name(self) -> str:
        return "Quality"

    async def calculate_raw_score(
        self, symbol: str, price_data=None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate quality score from fundamentals."""
        data = await self.fundamental_provider.get_fundamentals(symbol)

        if data is None or data.quality_score is None:
            return (float("nan"), {"error": "no_fundamental_data"})

        metadata = {
            "quality_score": data.quality_score,
            "roe": data.roe,
            "roa": data.roa,
            "profit_margin": data.profit_margin,
            "debt_to_equity": data.debt_to_equity,
            "earnings_growth": data.earnings_growth,
        }

        return (data.quality_score, metadata)
