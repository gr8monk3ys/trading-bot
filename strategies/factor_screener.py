"""
Factor-Based Stock Screener - Cross-Sectional Stock Selection

The critical missing piece that transforms this from "trade configured symbols"
to "systematically select the best stocks from the universe."

Uses composite factor scores to rank and select stocks.

Usage:
    from strategies.factor_screener import FactorScreener
    from factors.factor_portfolio import FactorPortfolio

    portfolio = FactorPortfolio(broker)
    screener = FactorScreener(portfolio, broker)

    # Get top 20 stocks from S&P 500
    top_stocks = await screener.get_top_stocks(sp500_symbols, top_n=20)

    # Get long/short portfolio
    longs, shorts = await screener.get_long_short_portfolio(universe, long_n=20, short_n=20)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScreenerResult:
    """Result of stock screening."""

    symbol: str
    composite_score: float
    factor_scores: Dict[str, float]
    rank: int
    percentile: float
    avg_daily_volume: float
    passes_liquidity: bool
    timestamp: datetime


@dataclass
class PortfolioRecommendation:
    """Recommended portfolio from screener."""

    long_positions: List[str]
    short_positions: List[str]
    long_scores: Dict[str, float]
    short_scores: Dict[str, float]
    universe_size: int
    screened_at: datetime
    liquidity_filter_applied: bool
    min_liquidity_threshold: float


class FactorScreener:
    """
    Cross-sectional stock selection using factor rankings.

    Ranks an entire universe by composite factor score and selects
    the top N stocks for long positions (and optionally bottom N for shorts).
    """

    def __init__(
        self,
        factor_portfolio,
        broker,
        cache_ttl_seconds: int = 1800,  # 30 minutes
    ):
        """
        Initialize factor screener.

        Args:
            factor_portfolio: FactorPortfolio instance for scoring
            broker: Trading broker for volume data
            cache_ttl_seconds: How long to cache screening results
        """
        self.factor_portfolio = factor_portfolio
        self.broker = broker
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[List[ScreenerResult], datetime]] = {}
        self._volume_cache: Dict[str, Tuple[float, datetime]] = {}

    async def get_top_stocks(
        self,
        universe: List[str],
        top_n: int = 20,
        min_liquidity_adv: float = 1_000_000,  # Min $1M average daily volume
        exclude_symbols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Rank universe by composite factor score, return top N.

        Args:
            universe: List of symbols to screen
            top_n: Number of top stocks to return
            min_liquidity_adv: Minimum average daily dollar volume
            exclude_symbols: Symbols to exclude from results

        Returns:
            List of top N symbols by composite score
        """
        exclude_symbols = exclude_symbols or []

        # Get full screening results
        results = await self._screen_universe(universe, min_liquidity_adv)

        # Filter exclusions and sort by score
        filtered = [
            r for r in results
            if r.symbol not in exclude_symbols and r.passes_liquidity
        ]

        # Already sorted by score in _screen_universe
        top_symbols = [r.symbol for r in filtered[:top_n]]

        logger.info(
            f"Factor screener: Top {len(top_symbols)} stocks from {len(universe)} universe "
            f"(liquidity filter: ${min_liquidity_adv:,.0f})"
        )

        for i, result in enumerate(filtered[:min(10, top_n)], 1):
            logger.debug(
                f"  {i}. {result.symbol}: {result.composite_score:.1f} "
                f"(rank {result.rank}/{len(results)})"
            )

        return top_symbols

    async def get_long_short_portfolio(
        self,
        universe: List[str],
        long_n: int = 20,
        short_n: int = 20,
        min_liquidity_adv: float = 1_000_000,
    ) -> PortfolioRecommendation:
        """
        Get top N longs and bottom N shorts for market-neutral portfolio.

        Args:
            universe: List of symbols to screen
            long_n: Number of long positions
            short_n: Number of short positions
            min_liquidity_adv: Minimum average daily dollar volume

        Returns:
            PortfolioRecommendation with longs and shorts
        """
        results = await self._screen_universe(universe, min_liquidity_adv)

        # Filter by liquidity
        liquid_results = [r for r in results if r.passes_liquidity]

        if len(liquid_results) < long_n + short_n:
            logger.warning(
                f"Insufficient liquid stocks: {len(liquid_results)} < {long_n + short_n} required"
            )

        # Top N for longs (highest scores)
        longs = liquid_results[:long_n]

        # Bottom N for shorts (lowest scores)
        shorts = liquid_results[-short_n:] if short_n > 0 else []

        recommendation = PortfolioRecommendation(
            long_positions=[r.symbol for r in longs],
            short_positions=[r.symbol for r in shorts],
            long_scores={r.symbol: r.composite_score for r in longs},
            short_scores={r.symbol: r.composite_score for r in shorts},
            universe_size=len(universe),
            screened_at=datetime.now(),
            liquidity_filter_applied=True,
            min_liquidity_threshold=min_liquidity_adv,
        )

        logger.info(
            f"Long/Short Portfolio: {len(longs)} longs, {len(shorts)} shorts "
            f"from {len(liquid_results)} liquid stocks"
        )

        return recommendation

    async def get_sector_top_stocks(
        self,
        universe: List[str],
        sector_map: Dict[str, str],  # symbol -> sector
        top_per_sector: int = 3,
        min_liquidity_adv: float = 500_000,
    ) -> Dict[str, List[str]]:
        """
        Get top N stocks per sector for sector-balanced portfolio.

        Args:
            universe: List of symbols to screen
            sector_map: Mapping of symbol to sector
            top_per_sector: Number of stocks to pick per sector
            min_liquidity_adv: Minimum average daily dollar volume

        Returns:
            Dict of sector -> list of top symbols
        """
        results = await self._screen_universe(universe, min_liquidity_adv)

        # Group by sector
        sector_results: Dict[str, List[ScreenerResult]] = {}
        for result in results:
            if not result.passes_liquidity:
                continue

            sector = sector_map.get(result.symbol, "Unknown")
            if sector not in sector_results:
                sector_results[sector] = []
            sector_results[sector].append(result)

        # Get top N per sector
        sector_picks = {}
        for sector, sector_stocks in sector_results.items():
            # Already sorted by score
            sector_picks[sector] = [r.symbol for r in sector_stocks[:top_per_sector]]

        logger.info(
            f"Sector-balanced selection: {sum(len(v) for v in sector_picks.values())} "
            f"stocks across {len(sector_picks)} sectors"
        )

        return sector_picks

    async def get_detailed_rankings(
        self,
        universe: List[str],
        min_liquidity_adv: float = 0,  # No filter by default
    ) -> List[ScreenerResult]:
        """
        Get detailed screening results for all symbols.

        Args:
            universe: List of symbols to screen
            min_liquidity_adv: Minimum liquidity filter

        Returns:
            List of ScreenerResult sorted by score descending
        """
        return await self._screen_universe(universe, min_liquidity_adv)

    async def _screen_universe(
        self,
        universe: List[str],
        min_liquidity_adv: float,
    ) -> List[ScreenerResult]:
        """
        Internal method to screen universe and return sorted results.

        Uses caching to avoid repeated expensive calculations.
        """
        cache_key = f"{','.join(sorted(universe))}_{min_liquidity_adv}"

        # Check cache
        if cache_key in self._cache:
            results, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                logger.debug(f"Using cached screening results ({len(results)} stocks)")
                return results

        logger.info(f"Screening {len(universe)} stocks...")

        # Get composite rankings from factor portfolio
        rankings = await self.factor_portfolio.get_composite_rankings(universe)

        if not rankings:
            logger.warning("No rankings returned from factor portfolio")
            return []

        # Get volume data for liquidity filtering
        volume_tasks = [self._get_avg_daily_volume(symbol) for symbol in universe]
        volumes = await asyncio.gather(*volume_tasks, return_exceptions=True)

        volume_map = {}
        for symbol, vol in zip(universe, volumes, strict=False):
            if isinstance(vol, Exception):
                volume_map[symbol] = 0
            else:
                volume_map[symbol] = vol

        # Build results
        results = []
        for symbol, composite in rankings.items():
            adv = volume_map.get(symbol, 0)

            # Extract individual factor scores
            factor_scores = {}
            for factor_name, score in composite.factor_scores.items():
                factor_scores[factor_name] = score.normalized_score

            result = ScreenerResult(
                symbol=symbol,
                composite_score=composite.composite_score,
                factor_scores=factor_scores,
                rank=composite.rank,
                percentile=composite.percentile,
                avg_daily_volume=adv,
                passes_liquidity=adv >= min_liquidity_adv,
                timestamp=datetime.now(),
            )
            results.append(result)

        # Sort by composite score descending
        results.sort(key=lambda r: r.composite_score, reverse=True)

        # Update ranks after sorting
        for i, result in enumerate(results, 1):
            result.rank = i

        # Cache results
        self._cache[cache_key] = (results, datetime.now())

        liquid_count = sum(1 for r in results if r.passes_liquidity)
        logger.info(
            f"Screening complete: {len(results)} total, {liquid_count} pass liquidity filter"
        )

        return results

    async def _get_avg_daily_volume(self, symbol: str) -> float:
        """
        Get average daily dollar volume for a symbol.

        Uses 20-day average with caching.
        """
        # Check volume cache (1 hour TTL)
        if symbol in self._volume_cache:
            vol, cached_time = self._volume_cache[symbol]
            if (datetime.now() - cached_time).total_seconds() < 3600:
                return vol

        try:
            bars = await self.broker.get_bars(symbol, timeframe="1Day", limit=20)
            if not bars or len(bars) < 5:
                return 0

            # Calculate dollar volume (price * volume)
            dollar_volumes = [b.close * b.volume for b in bars]
            adv = np.mean(dollar_volumes)

            self._volume_cache[symbol] = (adv, datetime.now())
            return adv

        except Exception as e:
            logger.debug(f"Could not get volume for {symbol}: {e}")
            return 0

    def clear_cache(self):
        """Clear all caches."""
        self._cache.clear()
        self._volume_cache.clear()
        logger.info("Factor screener cache cleared")


class RebalanceRecommendation:
    """
    Generates rebalance recommendations comparing current positions to screener output.
    """

    def __init__(self, screener: FactorScreener):
        self.screener = screener

    async def get_rebalance_actions(
        self,
        current_positions: Dict[str, float],  # symbol -> value
        universe: List[str],
        target_positions: int = 20,
        min_liquidity_adv: float = 1_000_000,
    ) -> Dict[str, Any]:
        """
        Compare current positions to screener recommendations.

        Returns:
            Dict with 'to_buy', 'to_sell', 'to_hold' lists
        """
        # Get top stocks from screener
        top_stocks = await self.screener.get_top_stocks(
            universe,
            top_n=target_positions,
            min_liquidity_adv=min_liquidity_adv,
        )

        current_symbols = set(current_positions.keys())
        target_symbols = set(top_stocks)

        to_buy = list(target_symbols - current_symbols)
        to_sell = list(current_symbols - target_symbols)
        to_hold = list(current_symbols & target_symbols)

        turnover = len(to_buy) + len(to_sell)
        turnover_pct = turnover / (2 * max(len(current_symbols), len(target_symbols), 1))

        return {
            "to_buy": to_buy,
            "to_sell": to_sell,
            "to_hold": to_hold,
            "turnover": turnover,
            "turnover_pct": turnover_pct,
            "current_count": len(current_symbols),
            "target_count": len(target_symbols),
        }
