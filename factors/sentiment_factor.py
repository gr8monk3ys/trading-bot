"""
News Sentiment Factor - Alpha signal from news sentiment analysis.

Uses FinBERT-based sentiment analysis of financial news to generate
trading signals. Positive news sentiment indicates bullish outlook,
negative sentiment indicates bearish outlook.

Research shows:
- News sentiment is a leading indicator (precedes price moves by 1-3 days)
- Works best combined with other factors (confirmation)
- Most effective for event-driven moves (earnings, M&A, FDA approvals)

Expected Impact: +30-50 bps annually when combined with other factors.

Usage:
    from factors.sentiment_factor import NewsSentimentFactor

    factor = NewsSentimentFactor(broker, api_key, secret_key)
    score = await factor.calculate_score("AAPL")
    # score.normalized_score: 0-100 (50 = neutral, >50 = bullish, <50 = bearish)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from factors.base_factor import BaseFactor, FactorScore

logger = logging.getLogger(__name__)


class NewsSentimentFactor(BaseFactor):
    """
    News sentiment as a quantitative alpha factor.

    Maps sentiment scores (-1 to +1) from news analysis to a 0-100 factor score.
    Uses the NewsSentimentAnalyzer for FinBERT-based analysis.

    Scoring:
    - Raw score: -1.0 to +1.0 (news sentiment)
    - Normalized: 0-100 (50 = neutral, 100 = very bullish, 0 = very bearish)
    """

    def __init__(
        self,
        broker,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        lookback_hours: int = 24,
        cache_ttl_seconds: int = 900,  # 15 minutes cache
    ):
        """
        Initialize news sentiment factor.

        Args:
            broker: Trading broker instance
            api_key: Alpaca API key (defaults to env var)
            secret_key: Alpaca secret key (defaults to env var)
            lookback_hours: Hours of news to analyze
            cache_ttl_seconds: Cache duration for sentiment scores
        """
        super().__init__(broker, cache_ttl_seconds)

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.lookback_hours = lookback_hours

        # Lazy-loaded analyzer
        self._analyzer = None
        self._analyzer_error = None

    @property
    def factor_name(self) -> str:
        """Return the name of this factor."""
        return "NewsSentiment"

    @property
    def higher_is_better(self) -> bool:
        """Higher sentiment score = more bullish = better."""
        return True

    def _get_analyzer(self):
        """Lazy load the sentiment analyzer."""
        if self._analyzer is not None:
            return self._analyzer

        if self._analyzer_error is not None:
            # Already failed to load
            return None

        try:
            from utils.news_sentiment import NewsSentimentAnalyzer

            if not self.api_key or not self.secret_key:
                self._analyzer_error = "Missing API credentials"
                logger.warning("NewsSentimentFactor: Missing API credentials")
                return None

            self._analyzer = NewsSentimentAnalyzer(
                api_key=self.api_key,
                secret_key=self.secret_key,
                cache_ttl_minutes=15,
            )
            logger.info("NewsSentimentFactor: Analyzer initialized")
            return self._analyzer

        except ImportError as e:
            self._analyzer_error = str(e)
            logger.warning(f"NewsSentimentFactor: Could not import analyzer: {e}")
            return None
        except Exception as e:
            self._analyzer_error = str(e)
            logger.error(f"NewsSentimentFactor: Initialization failed: {e}")
            return None

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate raw sentiment score for a symbol.

        Args:
            symbol: Stock symbol
            price_data: Not used (sentiment doesn't need price data)

        Returns:
            Tuple of (raw_score, metadata_dict)
            raw_score: -1.0 to +1.0 (negative to positive sentiment)
        """
        analyzer = self._get_analyzer()

        if analyzer is None:
            # Return neutral if analyzer not available
            return 0.0, {
                "error": self._analyzer_error or "Analyzer not available",
                "news_count": 0,
                "sentiment": "neutral",
            }

        try:
            # Get sentiment from analyzer
            result = await analyzer.get_symbol_sentiment(
                symbol,
                lookback_hours=self.lookback_hours,
                include_summary=True,  # More accurate but slower
            )

            metadata = {
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "news_count": result.news_count,
                "headlines": result.headlines[:3],  # Top 3 headlines
                "timestamp": result.timestamp.isoformat(),
            }

            # Raw score is already -1 to +1
            raw_score = result.score

            if result.news_count == 0:
                logger.debug(f"{symbol}: No recent news found")
            else:
                logger.debug(
                    f"{symbol}: Sentiment {result.sentiment} "
                    f"(score: {raw_score:+.2f}, {result.news_count} articles)"
                )

            return raw_score, metadata

        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return 0.0, {"error": str(e), "news_count": 0}

    async def calculate_scores_batch(self, symbols: List[str]) -> Dict[str, FactorScore]:
        """
        Calculate sentiment scores for multiple symbols efficiently.

        Uses bulk sentiment analysis for better performance.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict of symbol -> FactorScore
        """
        import numpy as np

        analyzer = self._get_analyzer()
        results = {}

        if analyzer is None:
            logger.warning("NewsSentimentFactor: Analyzer not available for batch")
            return results

        try:
            # Get bulk sentiment
            sentiment_results = await analyzer.get_bulk_sentiment(
                symbols, lookback_hours=self.lookback_hours
            )

            if not sentiment_results:
                return results

            # Extract raw scores for normalization
            raw_scores = {}
            metadata_map = {}

            for symbol, sentiment in sentiment_results.items():
                raw_scores[symbol] = sentiment.score
                metadata_map[symbol] = {
                    "sentiment": sentiment.sentiment,
                    "confidence": sentiment.confidence,
                    "news_count": sentiment.news_count,
                    "headlines": sentiment.headlines[:3],
                }

            # Update cross-sectional data
            self._last_cross_section_scores = raw_scores
            self._last_cross_section_time = datetime.now()

            # Normalize scores
            all_raw = list(raw_scores.values())
            np.mean(all_raw)
            np.std(all_raw) if len(all_raw) > 1 else 1.0

            for symbol, raw_score in raw_scores.items():
                # Map -1 to +1 sentiment to 0-100 scale
                # Using linear mapping: -1 -> 0, 0 -> 50, +1 -> 100
                normalized = 50 + (raw_score * 50)
                normalized = np.clip(normalized, 0, 100)

                # Calculate percentile
                rank = sum(1 for r in all_raw if r <= raw_score)
                percentile = (rank / len(all_raw)) * 100

                results[symbol] = FactorScore(
                    symbol=symbol,
                    factor_name=self.factor_name,
                    raw_score=raw_score,
                    normalized_score=normalized,
                    percentile=percentile,
                    timestamp=datetime.now(),
                    metadata=metadata_map[symbol],
                )

                # Update cache
                self._cache[symbol] = results[symbol]
                self._cache_time[symbol] = datetime.now()

            logger.info(
                f"NewsSentiment: Calculated scores for {len(results)}/{len(symbols)} symbols"
            )

            # Log sentiment summary
            positive = sum(1 for r in results.values() if r.raw_score > 0.2)
            negative = sum(1 for r in results.values() if r.raw_score < -0.2)
            neutral = len(results) - positive - negative

            logger.info(
                f"NewsSentiment summary: {positive} positive, {neutral} neutral, {negative} negative"
            )

            return results

        except Exception as e:
            logger.error(f"Error in batch sentiment calculation: {e}")
            return results


class SentimentMomentumFactor(BaseFactor):
    """
    Sentiment momentum - change in sentiment over time.

    Measures whether sentiment is improving or deteriorating,
    which can be more predictive than current sentiment level.

    Scoring:
    - Compares current sentiment to sentiment from N days ago
    - Positive change = improving sentiment = bullish
    """

    def __init__(
        self,
        broker,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        current_hours: int = 24,
        comparison_hours: int = 72,
        cache_ttl_seconds: int = 900,
    ):
        """
        Initialize sentiment momentum factor.

        Args:
            broker: Trading broker instance
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            current_hours: Recent window for current sentiment
            comparison_hours: Historical window for comparison
            cache_ttl_seconds: Cache duration
        """
        super().__init__(broker, cache_ttl_seconds)

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.current_hours = current_hours
        self.comparison_hours = comparison_hours

        self._analyzer = None
        self._analyzer_error = None

    @property
    def factor_name(self) -> str:
        """Return the name of this factor."""
        return "SentimentMomentum"

    @property
    def higher_is_better(self) -> bool:
        """Improving sentiment = better."""
        return True

    def _get_analyzer(self):
        """Lazy load the sentiment analyzer."""
        if self._analyzer is not None:
            return self._analyzer

        if self._analyzer_error is not None:
            return None

        try:
            from utils.news_sentiment import NewsSentimentAnalyzer

            if not self.api_key or not self.secret_key:
                self._analyzer_error = "Missing API credentials"
                return None

            self._analyzer = NewsSentimentAnalyzer(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            return self._analyzer

        except Exception as e:
            self._analyzer_error = str(e)
            return None

    async def calculate_raw_score(
        self, symbol: str, price_data: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate sentiment momentum (change in sentiment).

        Args:
            symbol: Stock symbol
            price_data: Not used

        Returns:
            Tuple of (raw_score, metadata)
            raw_score: -2.0 to +2.0 (change in sentiment)
        """
        analyzer = self._get_analyzer()

        if analyzer is None:
            return 0.0, {"error": self._analyzer_error or "Analyzer not available"}

        try:
            # Get current sentiment (recent window)
            current = await analyzer.get_symbol_sentiment(symbol, lookback_hours=self.current_hours)

            # Get historical sentiment (longer window)
            historical = await analyzer.get_symbol_sentiment(
                symbol, lookback_hours=self.comparison_hours
            )

            # Calculate momentum (change in sentiment)
            momentum = current.score - historical.score

            metadata = {
                "current_sentiment": current.score,
                "historical_sentiment": historical.score,
                "current_period_hours": self.current_hours,
                "comparison_period_hours": self.comparison_hours,
                "current_news_count": current.news_count,
                "historical_news_count": historical.news_count,
            }

            # Determine direction
            if momentum > 0.1:
                metadata["direction"] = "improving"
            elif momentum < -0.1:
                metadata["direction"] = "deteriorating"
            else:
                metadata["direction"] = "stable"

            return momentum, metadata

        except Exception as e:
            logger.error(f"Error calculating sentiment momentum for {symbol}: {e}")
            return 0.0, {"error": str(e)}
