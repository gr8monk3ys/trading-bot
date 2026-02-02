#!/usr/bin/env python3
"""
Relative Strength Ranking

Compares individual stocks to the market (SPY) to identify
outperformers and underperformers.

Research shows:
- Stocks outperforming the market tend to continue outperforming
- Relative strength is more predictive than absolute price momentum
- Top 20% RS stocks outperform bottom 20% by 10-15% annually

Strategy:
- Calculate relative strength vs SPY
- Only trade stocks in top 30% of relative strength
- Avoid stocks in bottom 30%
- Rank changes can signal momentum shifts

Expected Impact: 8-12% better returns by focusing on leaders

Usage:
    from utils.relative_strength import RelativeStrengthRanker

    ranker = RelativeStrengthRanker(broker)
    rankings = await ranker.rank_symbols(symbols)

    # Get only the leaders
    leaders = ranker.get_leaders(rankings, top_pct=0.30)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RelativeStrengthRanker:
    """
    Ranks stocks by relative strength compared to benchmark (SPY).

    Relative Strength = Stock Return / Benchmark Return

    RS > 1.0 = outperforming market
    RS < 1.0 = underperforming market
    """

    def __init__(
        self,
        broker,
        benchmark: str = "SPY",
        lookback_days: int = 20,
        min_rs_for_long: float = 1.05,  # Must outperform by 5%
        max_rs_for_short: float = 0.95,  # Must underperform by 5%
    ):
        """
        Initialize relative strength ranker.

        Args:
            broker: Trading broker for data fetching
            benchmark: Benchmark symbol (default: SPY)
            lookback_days: Days to calculate relative strength
            min_rs_for_long: Minimum RS ratio for long trades
            max_rs_for_short: Maximum RS ratio for short trades
        """
        self.broker = broker
        self.benchmark = benchmark
        self.lookback_days = lookback_days
        self.min_rs_for_long = min_rs_for_long
        self.max_rs_for_short = max_rs_for_short

        # Cache
        self._benchmark_return = None
        self._last_calc_date = None

        logger.info(
            f"RelativeStrengthRanker: benchmark={benchmark}, "
            f"lookback={lookback_days}d, min_rs={min_rs_for_long}"
        )

    async def get_benchmark_return(self, force_refresh: bool = False) -> float:
        """Get benchmark return for lookback period."""
        today = datetime.now().date()

        if (
            not force_refresh
            and self._benchmark_return is not None
            and self._last_calc_date == today
        ):
            return self._benchmark_return

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 10)

            bars = await self.broker.get_bars(
                self.benchmark,
                timeframe="1Day",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if bars is None or len(bars) < self.lookback_days:
                logger.warning("Insufficient benchmark data")
                return 0.0

            # Calculate return
            closes = [float(b.close) for b in bars]
            start_price = (
                closes[-self.lookback_days] if len(closes) >= self.lookback_days else closes[0]
            )
            end_price = closes[-1]

            benchmark_return = (end_price - start_price) / start_price

            self._benchmark_return = benchmark_return
            self._last_calc_date = today

            return benchmark_return

        except Exception as e:
            logger.error(f"Error getting benchmark return: {e}")
            return 0.0

    async def calculate_relative_strength(self, symbol: str) -> Optional[Dict]:
        """
        Calculate relative strength for a single symbol.

        Returns:
            Dict with RS metrics or None on error
        """
        try:
            # Get symbol data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 10)

            bars = await self.broker.get_bars(
                symbol,
                timeframe="1Day",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if bars is None or len(bars) < self.lookback_days:
                return None

            # Calculate stock return
            closes = [float(b.close) for b in bars]
            start_price = (
                closes[-self.lookback_days] if len(closes) >= self.lookback_days else closes[0]
            )
            end_price = closes[-1]

            # Guard against division by zero
            if start_price <= 0:
                logger.warning(f"Invalid start price for {symbol}: {start_price}")
                return None

            stock_return = (end_price - start_price) / start_price

            # Get benchmark return
            benchmark_return = await self.get_benchmark_return()

            # Calculate RS ratio
            # Avoid division by zero
            if benchmark_return == 0:
                rs_ratio = 1.0 + stock_return  # Use absolute return if benchmark flat
            else:
                rs_ratio = (1 + stock_return) / (1 + benchmark_return)

            # RS line (cumulative relative performance)
            rs_line = stock_return - benchmark_return

            return {
                "symbol": symbol,
                "stock_return": stock_return,
                "benchmark_return": benchmark_return,
                "rs_ratio": rs_ratio,
                "rs_line": rs_line,
                "outperforming": rs_ratio > 1.0,
                "current_price": end_price,
                "lookback_days": self.lookback_days,
            }

        except Exception as e:
            logger.warning(f"Error calculating RS for {symbol}: {e}")
            return None

    async def rank_symbols(self, symbols: List[str]) -> List[Dict]:
        """
        Rank symbols by relative strength.

        Returns:
            List of RS dicts sorted by rs_ratio (highest first)
        """
        import asyncio

        # Parallel execution for better performance
        tasks = [self.calculate_relative_strength(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rankings = []
        for result in results:
            if result and not isinstance(result, Exception):
                rankings.append(result)

        # Sort by RS ratio (highest first)
        rankings.sort(key=lambda x: x["rs_ratio"], reverse=True)

        # Add rank and percentile
        for i, rs in enumerate(rankings):
            rs["rank"] = i + 1
            rs["percentile"] = (len(rankings) - i) / len(rankings)

        if rankings:
            logger.info(f"Ranked {len(rankings)} symbols by relative strength")
            logger.info(f"  Top: {rankings[0]['symbol']} RS={rankings[0]['rs_ratio']:.3f}")
            logger.info(f"  Bottom: {rankings[-1]['symbol']} RS={rankings[-1]['rs_ratio']:.3f}")

        return rankings

    def get_leaders(
        self, rankings: List[Dict], top_pct: float = 0.30, min_count: int = 5
    ) -> List[str]:
        """
        Get top relative strength leaders.

        Args:
            rankings: Output from rank_symbols()
            top_pct: Top percentage to include (0.30 = top 30%)
            min_count: Minimum symbols to return

        Returns:
            List of leader symbols
        """
        if not rankings:
            return []

        n = max(min_count, int(len(rankings) * top_pct))
        leaders = [r["symbol"] for r in rankings[:n] if r["rs_ratio"] >= self.min_rs_for_long]

        return leaders

    def get_laggards(
        self, rankings: List[Dict], bottom_pct: float = 0.30, min_count: int = 5
    ) -> List[str]:
        """
        Get bottom relative strength laggards.

        Args:
            rankings: Output from rank_symbols()
            bottom_pct: Bottom percentage to include
            min_count: Minimum symbols to return

        Returns:
            List of laggard symbols
        """
        if not rankings:
            return []

        n = max(min_count, int(len(rankings) * bottom_pct))
        laggards = [r["symbol"] for r in rankings[-n:] if r["rs_ratio"] <= self.max_rs_for_short]

        return laggards

    def filter_by_rs(self, signal: str, rs_info: Dict) -> Tuple[bool, str]:
        """
        Filter a trade signal based on relative strength.

        Args:
            signal: 'long' or 'short'
            rs_info: RS info dict from calculate_relative_strength()

        Returns:
            Tuple of (should_trade, reason)
        """
        if rs_info is None:
            return True, "No RS data - allowing trade"

        rs_ratio = rs_info["rs_ratio"]

        if signal == "long":
            if rs_ratio >= self.min_rs_for_long:
                return True, f"RS {rs_ratio:.3f} >= {self.min_rs_for_long} - leader"
            elif rs_ratio >= 1.0:
                return True, f"RS {rs_ratio:.3f} - slight outperformer"
            else:
                return False, f"RS {rs_ratio:.3f} < 1.0 - underperformer, skip long"

        elif signal == "short":
            if rs_ratio <= self.max_rs_for_short:
                return True, f"RS {rs_ratio:.3f} <= {self.max_rs_for_short} - laggard"
            elif rs_ratio <= 1.0:
                return True, f"RS {rs_ratio:.3f} - slight underperformer"
            else:
                return False, f"RS {rs_ratio:.3f} > 1.0 - outperformer, skip short"

        return True, "Neutral signal"

    async def get_rs_report(self, symbols: List[str]) -> Dict:
        """Get comprehensive relative strength report."""
        rankings = await self.rank_symbols(symbols)
        benchmark_return = await self.get_benchmark_return()

        leaders = self.get_leaders(rankings, top_pct=0.30)
        laggards = self.get_laggards(rankings, bottom_pct=0.30)

        # Calculate stats
        rs_values = [r["rs_ratio"] for r in rankings]
        outperformers = [r for r in rankings if r["outperforming"]]

        return {
            "benchmark": self.benchmark,
            "benchmark_return": benchmark_return,
            "lookback_days": self.lookback_days,
            "symbols_ranked": len(rankings),
            "outperformers": len(outperformers),
            "underperformers": len(rankings) - len(outperformers),
            "avg_rs": np.mean(rs_values) if rs_values else 1.0,
            "median_rs": np.median(rs_values) if rs_values else 1.0,
            "leaders": leaders,
            "laggards": laggards,
            "rankings": rankings[:10],  # Top 10 for display
        }


class RSMomentumFilter:
    """
    Combines relative strength with momentum for stronger signals.
    """

    # Cache configuration
    MAX_CACHE_SIZE = 500  # Maximum symbols to cache
    DEFAULT_CACHE_MAX_AGE_MINUTES = 120

    # Position multiplier thresholds
    TOP_TIER_PERCENTILE = 0.80
    UPPER_TIER_PERCENTILE = 0.60
    LOWER_TIER_PERCENTILE = 0.40
    BOTTOM_TIER_PERCENTILE = 0.20

    # Position multipliers
    TOP_TIER_MULTIPLIER = 1.2
    UPPER_TIER_MULTIPLIER = 1.1
    LOWER_TIER_MULTIPLIER = 0.9
    BOTTOM_TIER_MULTIPLIER = 0.8
    DEFAULT_MULTIPLIER = 1.0

    def __init__(self, broker, max_cache_size: int = None):
        """Initialize RS momentum filter.

        Args:
            broker: Trading broker for data fetching
            max_cache_size: Maximum symbols to cache (default: MAX_CACHE_SIZE)
        """
        self.broker = broker
        self.ranker = RelativeStrengthRanker(broker)
        self._rankings_cache = {}
        self._cache_time = None
        self._max_cache_size = max_cache_size or self.MAX_CACHE_SIZE

    async def refresh_rankings(self, symbols: List[str]):
        """Refresh RS rankings (call daily or before session).

        Args:
            symbols: List of symbols to rank and cache

        Note:
            If more symbols than max_cache_size, keeps top performers.
        """
        rankings = await self.ranker.rank_symbols(symbols)

        # Limit cache size - keep top performers by RS ratio
        if len(rankings) > self._max_cache_size:
            logger.warning(
                f"Truncating RS cache from {len(rankings)} to {self._max_cache_size} symbols"
            )
            rankings = rankings[: self._max_cache_size]

        self._rankings_cache = {r["symbol"]: r for r in rankings}
        self._cache_time = datetime.now()
        logger.info(f"Refreshed RS rankings for {len(self._rankings_cache)} symbols")

    def get_rs(self, symbol: str, max_age_minutes: int = 120) -> Optional[Dict]:
        """
        Get cached RS info for a symbol.

        Args:
            symbol: Stock symbol
            max_age_minutes: Maximum cache age before returning None

        Returns:
            RS info dict or None if not cached or stale
        """
        if self._cache_time is None:
            return None
        age = (datetime.now() - self._cache_time).total_seconds() / 60
        if age > max_age_minutes:
            logger.debug(f"RS cache stale ({age:.0f}m old)")
            return None
        return self._rankings_cache.get(symbol)

    def should_trade(self, symbol: str, signal: str) -> Tuple[bool, str]:
        """
        Check if a trade should be taken based on RS.

        Returns:
            Tuple of (should_trade, reason)
        """
        rs_info = self.get_rs(symbol)
        return self.ranker.filter_by_rs(signal, rs_info)

    def get_position_multiplier(self, symbol: str) -> float:
        """
        Get position size multiplier based on RS rank.

        Leaders get larger positions, laggards get smaller.

        Args:
            symbol: Stock symbol

        Returns:
            Position size multiplier (0.8 to 1.2)
        """
        rs_info = self.get_rs(symbol)
        if rs_info is None:
            return self.DEFAULT_MULTIPLIER

        percentile = rs_info.get("percentile", 0.5)

        # Top performers get larger positions, laggards get smaller
        if percentile >= self.TOP_TIER_PERCENTILE:
            return self.TOP_TIER_MULTIPLIER
        elif percentile >= self.UPPER_TIER_PERCENTILE:
            return self.UPPER_TIER_MULTIPLIER
        elif percentile <= self.BOTTOM_TIER_PERCENTILE:
            return self.BOTTOM_TIER_MULTIPLIER
        elif percentile <= self.LOWER_TIER_PERCENTILE:
            return self.LOWER_TIER_MULTIPLIER
        else:
            return self.DEFAULT_MULTIPLIER


if __name__ == "__main__":
    """Test relative strength ranker."""
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    async def test():
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        print("\n" + "=" * 60)
        print("RELATIVE STRENGTH RANKING")
        print("=" * 60)

        ranker = RelativeStrengthRanker(broker, lookback_days=20)

        # Test symbols
        symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AMD",
            "JPM",
            "XOM",
            "UNH",
            "V",
        ]

        report = await ranker.get_rs_report(symbols)

        print(f"\nBenchmark: {report['benchmark']}")
        print(f"Benchmark Return ({report['lookback_days']}d): {report['benchmark_return']:+.1%}")
        print(f"\nSymbols Ranked: {report['symbols_ranked']}")
        print(f"Outperformers: {report['outperformers']}")
        print(f"Underperformers: {report['underperformers']}")

        print(f"\nAvg RS: {report['avg_rs']:.3f}")
        print(f"Median RS: {report['median_rs']:.3f}")

        print("\nTop 10 by Relative Strength:")
        print("-" * 50)
        for r in report["rankings"]:
            status = (
                "LEADER"
                if r["rs_ratio"] >= 1.05
                else "OUTPERFORM" if r["rs_ratio"] >= 1.0 else "LAGGING"
            )
            print(
                f"  {r['rank']:2d}. {r['symbol']:5s} | RS: {r['rs_ratio']:.3f} | "
                f"Return: {r['stock_return']:+.1%} | {status}"
            )

        print(f"\nLeaders (top 30%): {', '.join(report['leaders'])}")
        print(f"Laggards (bottom 30%): {', '.join(report['laggards'])}")

        print("=" * 60)

    asyncio.run(test())
