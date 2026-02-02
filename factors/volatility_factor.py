"""
Volatility Factor - Low-Volatility Anomaly

Implements the low-volatility anomaly factor:
- Lower volatility stocks tend to have higher risk-adjusted returns
- This contradicts CAPM but is empirically robust

Research shows:
- Low-vol stocks outperform on risk-adjusted basis
- High-vol stocks often underperform (lottery effect)
- Works especially well in down markets

Expected Impact: 3-5% annual alpha from volatility timing
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factors.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class VolatilityFactor(BaseFactor):
    """
    Low-volatility factor.

    IMPORTANT: For this factor, LOWER is BETTER (low-vol anomaly).
    We override higher_is_better = False.
    """

    def __init__(
        self,
        broker,
        lookback_days: int = 252,
        annualize: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize volatility factor.

        Args:
            broker: Trading broker instance
            lookback_days: Days to calculate volatility (default 252 = 1 year)
            annualize: Whether to annualize volatility
            cache_ttl_seconds: Cache TTL
        """
        super().__init__(broker, cache_ttl_seconds)
        self.lookback_days = lookback_days
        self.annualize = annualize

    @property
    def factor_name(self) -> str:
        return f"Volatility_{self.lookback_days}D"

    @property
    def higher_is_better(self) -> bool:
        """Lower volatility is better (low-vol anomaly)."""
        return False

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate historical volatility.

        Args:
            symbol: Stock symbol
            price_data: Optional pre-fetched price data

        Returns:
            Tuple of (volatility, metadata)
        """
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 30)

        if price_data is None or len(price_data) < 20:
            return (np.nan, {"error": "insufficient_data"})

        closes = np.array([d["close"] for d in price_data])

        try:
            # Calculate daily returns
            returns = np.diff(np.log(closes))

            # Calculate standard deviation
            volatility = np.std(returns)

            # Annualize if requested
            if self.annualize:
                volatility = volatility * np.sqrt(252)

            # Calculate additional metrics
            downside_returns = returns[returns < 0]
            downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0

            # Mean return for Sharpe-like metric
            mean_return = np.mean(returns) * 252  # Annualized

            metadata = {
                "volatility": volatility,
                "downside_volatility": downside_vol,
                "annualized_return": mean_return,
                "return_per_vol": mean_return / volatility if volatility > 0 else 0,
                "data_points": len(returns),
                "max_drawdown": self._calculate_max_drawdown(closes),
            }

            return (volatility, metadata)

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return (np.nan, {"error": str(e)})

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown from price series."""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return float(np.min(drawdown))


class DownsideVolatilityFactor(BaseFactor):
    """
    Downside volatility factor (semi-deviation).

    Only measures volatility of negative returns, which is more
    relevant for risk-averse investors.
    """

    def __init__(
        self,
        broker,
        lookback_days: int = 252,
        cache_ttl_seconds: int = 3600,
    ):
        super().__init__(broker, cache_ttl_seconds)
        self.lookback_days = lookback_days

    @property
    def factor_name(self) -> str:
        return f"DownsideVol_{self.lookback_days}D"

    @property
    def higher_is_better(self) -> bool:
        return False  # Lower downside vol is better

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate downside volatility."""
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 30)

        if price_data is None or len(price_data) < 20:
            return (np.nan, {"error": "insufficient_data"})

        closes = np.array([d["close"] for d in price_data])

        try:
            returns = np.diff(np.log(closes))

            # Only negative returns
            downside_returns = returns[returns < 0]

            if len(downside_returns) < 5:
                return (np.nan, {"error": "insufficient_downside_data"})

            downside_vol = np.std(downside_returns) * np.sqrt(252)

            # Calculate Sortino-like ratio
            mean_return = np.mean(returns) * 252
            sortino_ratio = mean_return / downside_vol if downside_vol > 0 else 0

            metadata = {
                "downside_volatility": downside_vol,
                "total_volatility": np.std(returns) * np.sqrt(252),
                "annualized_return": mean_return,
                "sortino_ratio": sortino_ratio,
                "downside_observations": len(downside_returns),
            }

            return (downside_vol, metadata)

        except Exception as e:
            logger.error(f"Error calculating downside vol for {symbol}: {e}")
            return (np.nan, {"error": str(e)})


class BetaFactor(BaseFactor):
    """
    Beta factor - sensitivity to market movements.

    Low-beta stocks often outperform on a risk-adjusted basis
    (similar to low-vol anomaly).
    """

    def __init__(
        self,
        broker,
        benchmark: str = "SPY",
        lookback_days: int = 252,
        cache_ttl_seconds: int = 3600,
    ):
        super().__init__(broker, cache_ttl_seconds)
        self.benchmark = benchmark
        self.lookback_days = lookback_days
        self._benchmark_data = None
        self._benchmark_time = None

    @property
    def factor_name(self) -> str:
        return f"Beta_vs_{self.benchmark}"

    @property
    def higher_is_better(self) -> bool:
        return False  # Lower beta often better (low-beta anomaly)

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
            self.benchmark, days=self.lookback_days + 30
        )
        self._benchmark_time = now
        return self._benchmark_data

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate beta vs benchmark."""
        if price_data is None:
            price_data = await self.get_price_data(symbol, days=self.lookback_days + 30)

        benchmark_data = await self._get_benchmark_data()

        if price_data is None or benchmark_data is None:
            return (np.nan, {"error": "insufficient_data"})

        # Align data lengths
        min_len = min(len(price_data), len(benchmark_data))
        if min_len < 20:
            return (np.nan, {"error": "insufficient_data"})

        try:
            stock_closes = np.array([d["close"] for d in price_data[-min_len:]])
            bench_closes = np.array([d["close"] for d in benchmark_data[-min_len:]])

            # Calculate returns
            stock_returns = np.diff(np.log(stock_closes))
            bench_returns = np.diff(np.log(bench_closes))

            # Calculate beta using covariance/variance
            covariance = np.cov(stock_returns, bench_returns)[0, 1]
            benchmark_variance = np.var(bench_returns)

            if benchmark_variance == 0:
                return (np.nan, {"error": "zero_benchmark_variance"})

            beta = covariance / benchmark_variance

            # Calculate alpha
            mean_stock_return = np.mean(stock_returns) * 252
            mean_bench_return = np.mean(bench_returns) * 252
            alpha = mean_stock_return - beta * mean_bench_return

            # Correlation
            correlation = np.corrcoef(stock_returns, bench_returns)[0, 1]

            metadata = {
                "beta": beta,
                "alpha": alpha,
                "correlation": correlation,
                "r_squared": correlation ** 2,
                "stock_vol": np.std(stock_returns) * np.sqrt(252),
                "benchmark_vol": np.std(bench_returns) * np.sqrt(252),
            }

            return (beta, metadata)

        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
            return (np.nan, {"error": str(e)})
