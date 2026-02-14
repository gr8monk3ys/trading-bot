"""
Base Factor - Abstract interface for all quantitative factors.

All factor implementations inherit from this base class and implement
the calculate_score() method.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactorScore:
    """Result of factor calculation for a single symbol."""

    symbol: str
    factor_name: str
    raw_score: float  # Unnormalized score
    normalized_score: float  # Normalized to 0-100 scale
    percentile: float  # Percentile rank (0-100)
    timestamp: datetime
    metadata: Dict[str, Any]  # Additional factor-specific data


class BaseFactor(ABC):
    """
    Abstract base class for quantitative factors.

    All factors must implement:
    - calculate_raw_score(): Compute raw factor value for a symbol
    - factor_name: Name of the factor

    The base class provides:
    - Normalization to 0-100 scale
    - Percentile ranking across symbols
    - Caching for efficiency
    """

    def __init__(self, broker, cache_ttl_seconds: int = 300):
        """
        Initialize factor.

        Args:
            broker: Trading broker instance for data fetching
            cache_ttl_seconds: How long to cache factor scores
        """
        self.broker = broker
        self.cache_ttl = cache_ttl_seconds

        # Cache for factor scores
        self._cache: Dict[str, FactorScore] = {}
        self._cache_time: Dict[str, datetime] = {}

        # Cross-sectional normalization parameters
        self._last_cross_section_scores: Optional[Dict[str, float]] = None
        self._last_cross_section_time: Optional[datetime] = None

    @property
    @abstractmethod
    def factor_name(self) -> str:
        """Return the name of this factor."""
        pass

    @property
    def higher_is_better(self) -> bool:
        """
        Return True if higher raw scores are better.

        Override in subclasses where lower is better (e.g., volatility factor).
        """
        return True

    @abstractmethod
    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calculate raw factor score for a symbol.

        Args:
            symbol: Stock symbol
            price_data: Optional pre-fetched price data

        Returns:
            Tuple of (raw_score, metadata_dict)
        """
        pass

    async def calculate_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Optional[FactorScore]:
        """
        Calculate factor score with normalization.

        Args:
            symbol: Stock symbol
            price_data: Optional pre-fetched price data

        Returns:
            FactorScore or None if calculation fails
        """
        try:
            # Check cache
            if self._is_cache_valid(symbol):
                return self._cache[symbol]

            # Calculate raw score
            raw_score, metadata = await self.calculate_raw_score(symbol, price_data)

            if raw_score is None or np.isnan(raw_score):
                return None

            # Normalize score (will use cross-sectional data if available)
            normalized = self._normalize_score(raw_score)
            percentile = self._get_percentile(raw_score)

            # Create result
            result = FactorScore(
                symbol=symbol,
                factor_name=self.factor_name,
                raw_score=raw_score,
                normalized_score=normalized,
                percentile=percentile,
                timestamp=datetime.now(),
                metadata=metadata,
            )

            # Cache result
            self._cache[symbol] = result
            self._cache_time[symbol] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Error calculating {self.factor_name} for {symbol}: {e}")
            return None

    async def calculate_scores_batch(self, symbols: List[str]) -> Dict[str, FactorScore]:
        """
        Calculate factor scores for multiple symbols.

        Performs cross-sectional normalization using all symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict of symbol -> FactorScore
        """
        results = {}
        raw_scores = {}

        # First pass: calculate raw scores for all symbols
        for symbol in symbols:
            try:
                raw_score, metadata = await self.calculate_raw_score(symbol)
                if raw_score is not None and not np.isnan(raw_score):
                    raw_scores[symbol] = (raw_score, metadata)
            except Exception as e:
                logger.debug(f"Error calculating {self.factor_name} for {symbol}: {e}")

        if not raw_scores:
            return results

        # Update cross-sectional data for normalization
        self._last_cross_section_scores = {s: r for s, (r, _) in raw_scores.items()}
        self._last_cross_section_time = datetime.now()

        # Second pass: normalize and create FactorScore objects
        all_raw = [r for r, _ in raw_scores.values()]
        mean = np.mean(all_raw)
        std = np.std(all_raw) if len(all_raw) > 1 else 1.0

        for symbol, (raw_score, metadata) in raw_scores.items():
            # Z-score normalization, then scale to 0-100
            if std > 0:
                z_score = (raw_score - mean) / std
                # Clip to +/- 3 std, then scale to 0-100
                normalized = 50 + (np.clip(z_score, -3, 3) / 3) * 50
                if not self.higher_is_better:
                    normalized = 100 - normalized
            else:
                normalized = 50

            # Calculate percentile
            rank = sum(1 for r in all_raw if r <= raw_score)
            percentile = (rank / len(all_raw)) * 100
            if not self.higher_is_better:
                percentile = 100 - percentile

            results[symbol] = FactorScore(
                symbol=symbol,
                factor_name=self.factor_name,
                raw_score=raw_score,
                normalized_score=normalized,
                percentile=percentile,
                timestamp=datetime.now(),
                metadata=metadata,
            )

            # Update cache
            self._cache[symbol] = results[symbol]
            self._cache_time[symbol] = datetime.now()

        logger.info(
            f"{self.factor_name}: Calculated scores for {len(results)}/{len(symbols)} symbols"
        )

        return results

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached score is still valid."""
        if symbol not in self._cache or symbol not in self._cache_time:
            return False

        age = (datetime.now() - self._cache_time[symbol]).total_seconds()
        return age < self.cache_ttl

    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize raw score to 0-100 scale.

        Uses cross-sectional data if available, otherwise uses heuristics.
        """
        if self._last_cross_section_scores and len(self._last_cross_section_scores) > 1:
            all_scores = list(self._last_cross_section_scores.values())
            mean = np.mean(all_scores)
            std = np.std(all_scores)

            if std > 0:
                z_score = (raw_score - mean) / std
                normalized = 50 + (np.clip(z_score, -3, 3) / 3) * 50
            else:
                normalized = 50
        else:
            # Fallback: simple linear mapping (factor-specific override recommended)
            normalized = np.clip(raw_score * 50 + 50, 0, 100)

        if not self.higher_is_better:
            normalized = 100 - normalized

        return normalized

    def _get_percentile(self, raw_score: float) -> float:
        """Get percentile rank of score within cross-section."""
        if not self._last_cross_section_scores:
            return 50  # Unknown percentile

        all_scores = list(self._last_cross_section_scores.values())
        rank = sum(1 for s in all_scores if s <= raw_score)
        percentile = (rank / len(all_scores)) * 100

        if not self.higher_is_better:
            percentile = 100 - percentile

        return percentile

    async def get_price_data(self, symbol: str, days: int = 252) -> Optional[List[Dict]]:
        """
        Fetch historical price data for a symbol.

        Args:
            symbol: Stock symbol
            days: Number of trading days of history

        Returns:
            List of price dictionaries or None
        """
        try:
            from datetime import timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days * 1.5))  # Buffer for weekends

            bars = await self.broker.get_bars(
                symbol,
                timeframe="1Day",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if bars is None or len(bars) < 10:
                return None

            return [
                {
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "timestamp": b.timestamp if hasattr(b, "timestamp") else None,
                }
                for b in bars
            ]

        except Exception as e:
            logger.debug(f"Error fetching price data for {symbol}: {e}")
            return None

    def clear_cache(self):
        """Clear all cached scores."""
        self._cache.clear()
        self._cache_time.clear()
        self._last_cross_section_scores = None
        self._last_cross_section_time = None
