"""
Alternative Data Provider - Abstract base class and aggregator.

This module provides the foundation for integrating alternative data sources
into the trading system. Each provider inherits from AlternativeDataProvider
and implements source-specific logic.

Usage:
    from data.alternative_data_provider import AltDataAggregator
    from data.social_sentiment_advanced import RedditSentimentProvider

    aggregator = AltDataAggregator()
    aggregator.register_provider(RedditSentimentProvider())

    signals = await aggregator.get_signals(["AAPL", "TSLA", "GME"])
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from data.alt_data_types import (
    AggregatedSignal,
    AltDataProviderStatus,
    AltDataSource,
    AlternativeSignal,
)

logger = logging.getLogger(__name__)


class AltDataCache:
    """
    TTL-based cache for alternative data signals.

    Prevents excessive API calls and provides consistent signals
    within the cache window.
    """

    def __init__(self, default_ttl_seconds: int = 300):  # 5 minute default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, source: AltDataSource, symbol: str) -> str:
        """Generate cache key."""
        return f"{source.value}:{symbol}"

    def get(self, source: AltDataSource, symbol: str) -> Optional[AlternativeSignal]:
        """Get cached signal if valid."""
        key = self._make_key(source, symbol)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if datetime.now() > entry["expires"]:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry["signal"]

    def set(
        self, source: AltDataSource, symbol: str, signal: AlternativeSignal, ttl_seconds: Optional[int] = None
    ):
        """Cache a signal."""
        key = self._make_key(source, symbol)
        ttl = ttl_seconds or self._default_ttl

        self._cache[key] = {
            "signal": signal,
            "expires": datetime.now() + timedelta(seconds=ttl),
            "cached_at": datetime.now(),
        }

    def invalidate(self, source: AltDataSource, symbol: Optional[str] = None):
        """Invalidate cache entries."""
        if symbol:
            key = self._make_key(source, symbol)
            self._cache.pop(key, None)
        else:
            # Invalidate all entries for this source
            keys_to_remove = [k for k in self._cache if k.startswith(f"{source.value}:")]
            for key in keys_to_remove:
                del self._cache[key]

    def clear(self):
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class AlternativeDataProvider(ABC):
    """
    Abstract base class for alternative data providers.

    Each provider is responsible for:
    1. Connecting to its data source (API, scraping, etc.)
    2. Fetching raw data for requested symbols
    3. Processing raw data into normalized signals
    4. Handling rate limiting and errors gracefully
    """

    def __init__(
        self,
        source: AltDataSource,
        cache_ttl_seconds: int = 300,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ):
        self.source = source
        self.cache_ttl = cache_ttl_seconds
        self.max_retries = max_retries
        self.timeout = timeout_seconds

        # Status tracking
        self._error_count = 0
        self._last_update: Optional[datetime] = None
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[datetime] = None
        self._latencies: List[float] = []

        # Initialization flag
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider (authenticate, load models, etc.).

        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    async def fetch_signal(self, symbol: str) -> Optional[AlternativeSignal]:
        """
        Fetch alternative data signal for a single symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            AlternativeSignal if data available, None otherwise.
        """
        pass

    async def get_signals(self, symbols: List[str]) -> Dict[str, AlternativeSignal]:
        """
        Fetch signals for multiple symbols.

        Default implementation fetches in parallel with rate limiting.
        Override for providers that support batch requests.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dict mapping symbol to signal (symbols without signals omitted)
        """
        if not self._initialized:
            success = await self.initialize()
            if not success:
                logger.error(f"{self.source.value}: Failed to initialize provider")
                return {}

        results: Dict[str, AlternativeSignal] = {}

        # Fetch in parallel with semaphore for rate limiting
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

        async def fetch_with_limit(sym: str) -> tuple[str, Optional[AlternativeSignal]]:
            async with semaphore:
                try:
                    start = datetime.now()
                    signal = await asyncio.wait_for(self.fetch_signal(sym), timeout=self.timeout)
                    latency = (datetime.now() - start).total_seconds() * 1000
                    self._latencies.append(latency)

                    # Keep only last 100 latencies
                    if len(self._latencies) > 100:
                        self._latencies = self._latencies[-100:]

                    return (sym, signal)
                except asyncio.TimeoutError:
                    logger.warning(f"{self.source.value}: Timeout fetching {sym}")
                    self._error_count += 1
                    return (sym, None)
                except Exception as e:
                    logger.error(f"{self.source.value}: Error fetching {sym}: {e}")
                    self._error_count += 1
                    return (sym, None)

        tasks = [fetch_with_limit(sym) for sym in symbols]
        responses = await asyncio.gather(*tasks)

        for symbol, signal in responses:
            if signal is not None:
                results[symbol] = signal
                self._last_update = datetime.now()

        return results

    def get_status(self) -> AltDataProviderStatus:
        """Get current provider status."""
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

        return AltDataProviderStatus(
            source=self.source,
            is_healthy=self._error_count < 5,
            last_update=self._last_update,
            error_count=self._error_count,
            rate_limit_remaining=self._rate_limit_remaining,
            rate_limit_reset=self._rate_limit_reset,
            avg_latency_ms=avg_latency,
        )

    def reset_errors(self):
        """Reset error count (call after successful recovery)."""
        self._error_count = 0


class AltDataAggregator:
    """
    Aggregates signals from multiple alternative data providers.

    Provides a unified interface for fetching and combining signals
    from various sources (social, order flow, web scraping, etc.).
    """

    # Weights for different sources (can be adjusted based on backtesting)
    DEFAULT_SOURCE_WEIGHTS = {
        AltDataSource.REDDIT: 0.15,
        AltDataSource.TWITTER: 0.15,
        AltDataSource.STOCKTWITS: 0.10,
        AltDataSource.DARK_POOL: 0.20,
        AltDataSource.OPTIONS_FLOW: 0.20,
        AltDataSource.JOB_POSTINGS: 0.05,
        AltDataSource.GLASSDOOR: 0.05,
        AltDataSource.APP_STORE: 0.05,
        AltDataSource.NEWS_ADVANCED: 0.05,
    }

    def __init__(
        self,
        cache_ttl_seconds: int = 300,
        source_weights: Optional[Dict[AltDataSource, float]] = None,
    ):
        self._providers: Dict[AltDataSource, AlternativeDataProvider] = {}
        self._cache = AltDataCache(default_ttl_seconds=cache_ttl_seconds)
        self._source_weights = source_weights or self.DEFAULT_SOURCE_WEIGHTS

        # Performance tracking
        self._killed_sources: Set[AltDataSource] = set()
        self._source_performance: Dict[AltDataSource, List[float]] = {}

    def register_provider(self, provider: AlternativeDataProvider):
        """Register an alternative data provider."""
        self._providers[provider.source] = provider
        self._source_performance[provider.source] = []
        logger.info(f"Registered alternative data provider: {provider.source.value}")

    def unregister_provider(self, source: AltDataSource):
        """Unregister a provider."""
        if source in self._providers:
            del self._providers[source]
            logger.info(f"Unregistered alternative data provider: {source.value}")

    async def initialize_all(self) -> Dict[AltDataSource, bool]:
        """Initialize all registered providers."""
        results = {}

        for source, provider in self._providers.items():
            try:
                success = await provider.initialize()
                results[source] = success
                if success:
                    logger.info(f"Initialized {source.value}")
                else:
                    logger.warning(f"Failed to initialize {source.value}")
            except Exception as e:
                logger.error(f"Error initializing {source.value}: {e}")
                results[source] = False

        return results

    async def get_signal(
        self,
        symbol: str,
        sources: Optional[List[AltDataSource]] = None,
        use_cache: bool = True,
    ) -> Optional[AggregatedSignal]:
        """
        Get aggregated signal for a single symbol.

        Args:
            symbol: Stock ticker symbol
            sources: Specific sources to query (None = all registered)
            use_cache: Whether to use cached signals

        Returns:
            AggregatedSignal combining all available signals
        """
        signals = await self.get_signals([symbol], sources, use_cache)
        return signals.get(symbol)

    async def get_signals(
        self,
        symbols: List[str],
        sources: Optional[List[AltDataSource]] = None,
        use_cache: bool = True,
    ) -> Dict[str, AggregatedSignal]:
        """
        Get aggregated signals for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            sources: Specific sources to query (None = all registered)
            use_cache: Whether to use cached signals

        Returns:
            Dict mapping symbol to AggregatedSignal
        """
        target_sources = sources or list(self._providers.keys())

        # Filter out killed sources
        active_sources = [s for s in target_sources if s not in self._killed_sources]

        if not active_sources:
            logger.warning("No active alternative data sources available")
            return {}

        # Collect signals from all providers
        all_signals: Dict[str, List[AlternativeSignal]] = {sym: [] for sym in symbols}

        for source in active_sources:
            if source not in self._providers:
                continue

            provider = self._providers[source]

            # Check cache first
            if use_cache:
                cached_symbols = []
                uncached_symbols = []

                for sym in symbols:
                    cached = self._cache.get(source, sym)
                    if cached:
                        all_signals[sym].append(cached)
                        cached_symbols.append(sym)
                    else:
                        uncached_symbols.append(sym)

                symbols_to_fetch = uncached_symbols
            else:
                symbols_to_fetch = symbols

            # Fetch uncached signals
            if symbols_to_fetch:
                try:
                    fetched = await provider.get_signals(symbols_to_fetch)

                    for sym, signal in fetched.items():
                        all_signals[sym].append(signal)

                        # Cache the signal
                        if use_cache:
                            self._cache.set(source, sym, signal)

                except Exception as e:
                    logger.error(f"Error fetching from {source.value}: {e}")

        # Aggregate signals for each symbol
        results: Dict[str, AggregatedSignal] = {}

        for symbol, signals in all_signals.items():
            if not signals:
                continue

            # Apply source weights
            weighted_signals = []
            for sig in signals:
                weight = self._source_weights.get(sig.source, 0.1)
                sig.confidence *= weight  # Adjust confidence by source weight
                weighted_signals.append(sig)

            aggregated = AggregatedSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                sources=[s.source for s in weighted_signals],
                individual_signals=weighted_signals,
            )

            results[symbol] = aggregated

        return results

    def update_source_weight(self, source: AltDataSource, weight: float):
        """Update weight for a source based on performance."""
        self._source_weights[source] = max(0.0, min(1.0, weight))
        logger.info(f"Updated {source.value} weight to {weight:.2f}")

    def kill_source(self, source: AltDataSource, reason: str):
        """Kill a source that's performing poorly."""
        self._killed_sources.add(source)
        logger.warning(f"KILLED source {source.value}: {reason}")

    def revive_source(self, source: AltDataSource):
        """Revive a previously killed source."""
        self._killed_sources.discard(source)
        logger.info(f"Revived source {source.value}")

    def record_performance(self, source: AltDataSource, actual_return: float, predicted_direction: float):
        """
        Record performance of a source for adaptive weighting.

        Args:
            source: The data source
            actual_return: Actual return achieved
            predicted_direction: Signal value at time of prediction
        """
        # Calculate if prediction was correct
        correct = (actual_return > 0 and predicted_direction > 0) or (
            actual_return < 0 and predicted_direction < 0
        )

        if source not in self._source_performance:
            self._source_performance[source] = []

        self._source_performance[source].append(1.0 if correct else 0.0)

        # Keep last 100 predictions
        if len(self._source_performance[source]) > 100:
            self._source_performance[source] = self._source_performance[source][-100:]

        # Check for kill switch (accuracy below 40% on 50+ predictions)
        if len(self._source_performance[source]) >= 50:
            accuracy = sum(self._source_performance[source]) / len(self._source_performance[source])
            if accuracy < 0.40:
                self.kill_source(source, f"Accuracy {accuracy:.1%} below 40% threshold")

    def get_source_accuracy(self, source: AltDataSource) -> Optional[float]:
        """Get historical accuracy for a source."""
        if source not in self._source_performance or not self._source_performance[source]:
            return None
        return sum(self._source_performance[source]) / len(self._source_performance[source])

    def get_all_statuses(self) -> Dict[AltDataSource, AltDataProviderStatus]:
        """Get status of all providers."""
        statuses = {}
        for source, provider in self._providers.items():
            status = provider.get_status()
            status.cache_hit_rate = self._cache.hit_rate
            statuses[source] = status
        return statuses

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_rate": self._cache.hit_rate,
            "hits": self._cache._hits,
            "misses": self._cache._misses,
        }
