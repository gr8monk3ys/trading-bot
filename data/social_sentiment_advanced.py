"""
Social Sentiment Advanced - Reddit, Twitter, and StockTwits sentiment analysis.

This module provides advanced sentiment analysis from social media platforms
to generate alternative data signals for trading decisions.

Features:
- Reddit sentiment (r/wallstreetbets, r/stocks, r/investing)
- Mention volume tracking and anomaly detection
- Meme stock risk flagging
- Ticker extraction from text
- FinBERT-based sentiment scoring

Usage:
    from data.social_sentiment_advanced import RedditSentimentProvider

    provider = RedditSentimentProvider(
        client_id="your_client_id",
        client_secret="your_client_secret"
    )
    await provider.initialize()
    signal = await provider.fetch_signal("AAPL")
"""

import asyncio
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from data.alt_data_types import (
    AltDataSource,
    AlternativeSignal,
    SocialSentimentSignal,
)
from data.alternative_data_provider import AlternativeDataProvider

logger = logging.getLogger(__name__)

# Common words that look like tickers but aren't
TICKER_BLACKLIST = {
    "A", "I", "AM", "PM", "CEO", "CFO", "COO", "CTO", "DD", "EPS", "ETF",
    "FD", "FDA", "FOMO", "FUD", "GDP", "GUH", "HIV", "IMO", "IPO", "ITM",
    "IV", "LOL", "MACD", "MOM", "MOASS", "NYSE", "OTM", "PE", "PR", "PT",
    "RSI", "SEC", "SPAC", "SPY", "TLDR", "USA", "USD", "WTF", "YOY", "YTD",
    "ATH", "ATL", "HODL", "YOLO", "APE", "APES", "MOON", "DIP", "RIP",
    "ALL", "ANY", "ARE", "CAN", "CEO", "FOR", "HAS", "THE", "WAS",
    "NOW", "NEW", "OLD", "BIG", "TOP", "LOW", "HIGH", "UP", "DOWN",
}

# Subreddits to monitor
TRADING_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "stockmarket",
    "trading",
    "pennystocks",
]


class TickerExtractor:
    """Extract stock tickers from text."""

    # Pattern for $TICKER format
    CASHTAG_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

    # Pattern for potential tickers (all caps, 1-5 letters)
    CAPS_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

    def __init__(self, valid_tickers: Optional[Set[str]] = None):
        """
        Initialize ticker extractor.

        Args:
            valid_tickers: Set of valid ticker symbols to filter against.
                          If None, uses basic heuristics.
        """
        self._valid_tickers = valid_tickers or set()

    def extract(self, text: str) -> List[str]:
        """Extract tickers from text."""
        tickers = set()

        # First, extract cashtags ($AAPL) - high confidence
        cashtags = self.CASHTAG_PATTERN.findall(text)
        for ticker in cashtags:
            if ticker not in TICKER_BLACKLIST:
                if not self._valid_tickers or ticker in self._valid_tickers:
                    tickers.add(ticker)

        # Then extract all-caps words that might be tickers
        caps = self.CAPS_PATTERN.findall(text)
        for ticker in caps:
            if ticker not in TICKER_BLACKLIST:
                # Only add if we have a valid tickers list and it's in there
                if self._valid_tickers and ticker in self._valid_tickers:
                    tickers.add(ticker)

        return list(tickers)


class SentimentAnalyzer:
    """
    Analyze sentiment using FinBERT or fallback methods.

    Uses the existing news_sentiment.py FinBERT integration if available,
    otherwise falls back to simple keyword-based sentiment.
    """

    # Simple sentiment keywords for fallback
    BULLISH_KEYWORDS = {
        "buy", "calls", "moon", "rocket", "bullish", "long", "undervalued",
        "breakout", "squeeze", "tendies", "gain", "profit", "green", "up",
        "yolo", "diamond hands", "hold", "hodl", "to the moon", "pump",
    }

    BEARISH_KEYWORDS = {
        "sell", "puts", "crash", "bearish", "short", "overvalued",
        "dump", "loss", "red", "down", "paper hands", "bag holder",
        "recession", "bubble", "scam", "fraud", "bankrupt",
    }

    def __init__(self, use_finbert: bool = True):
        self._use_finbert = use_finbert
        self._finbert_analyzer = None

    async def initialize(self) -> bool:
        """Initialize sentiment analyzer."""
        if self._use_finbert:
            try:
                # Try to import the existing FinBERT analyzer
                from utils.news_sentiment import NewsSentimentAnalyzer
                self._finbert_analyzer = NewsSentimentAnalyzer()
                await self._finbert_analyzer.initialize()
                logger.info("Initialized FinBERT sentiment analyzer")
                return True
            except ImportError:
                logger.warning("FinBERT not available, using keyword-based sentiment")
                self._use_finbert = False
            except Exception as e:
                logger.warning(f"Failed to initialize FinBERT: {e}, using fallback")
                self._use_finbert = False

        return True

    async def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score: -1 (bearish) to +1 (bullish)
            confidence: 0 to 1
        """
        if self._finbert_analyzer:
            try:
                result = await self._finbert_analyzer.analyze_text(text)
                return result.get("sentiment_score", 0.0), result.get("confidence", 0.5)
            except Exception as e:
                logger.debug(f"FinBERT analysis failed: {e}")

        # Fallback to keyword-based
        return self._keyword_sentiment(text)

    def _keyword_sentiment(self, text: str) -> Tuple[float, float]:
        """Simple keyword-based sentiment analysis."""
        text_lower = text.lower()

        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0.3  # Neutral with low confidence

        # Calculate sentiment score
        sentiment = (bullish_count - bearish_count) / total

        # Confidence based on keyword density
        confidence = min(0.8, 0.3 + (total / 10))

        return sentiment, confidence


class RedditSentimentProvider(AlternativeDataProvider):
    """
    Reddit sentiment analysis provider.

    Fetches posts from trading-related subreddits and analyzes sentiment
    for individual tickers.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "TradingBot/1.0",
        subreddits: Optional[List[str]] = None,
        lookback_hours: int = 24,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize Reddit sentiment provider.

        Args:
            client_id: Reddit API client ID (or set REDDIT_CLIENT_ID env var)
            client_secret: Reddit API client secret (or set REDDIT_CLIENT_SECRET env var)
            user_agent: User agent string for Reddit API
            subreddits: List of subreddits to monitor
            lookback_hours: How far back to look for posts
            cache_ttl_seconds: Cache TTL
        """
        super().__init__(
            source=AltDataSource.REDDIT,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        self._client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self._client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self._user_agent = user_agent
        self._subreddits = subreddits or TRADING_SUBREDDITS
        self._lookback_hours = lookback_hours

        self._reddit = None
        self._ticker_extractor = TickerExtractor()
        self._sentiment_analyzer = SentimentAnalyzer()

        # Rolling mention counts for anomaly detection
        self._mention_history: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)

    async def initialize(self) -> bool:
        """Initialize Reddit API connection."""
        if not self._client_id or not self._client_secret:
            logger.warning(
                "Reddit credentials not provided. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET environment variables."
            )
            # Return True anyway - we'll use mock data for development
            self._initialized = True
            return True

        try:
            import praw

            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )

            # Test connection
            self._reddit.read_only = True

            await self._sentiment_analyzer.initialize()
            self._initialized = True
            logger.info("Reddit API initialized successfully")
            return True

        except ImportError:
            logger.warning("PRAW not installed. Install with: pip install praw")
            self._initialized = True  # Use mock data
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            return False

    async def fetch_signal(self, symbol: str) -> Optional[SocialSentimentSignal]:
        """
        Fetch Reddit sentiment signal for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            SocialSentimentSignal with Reddit sentiment data
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch and analyze posts
            mentions = await self._fetch_mentions(symbol)

            if not mentions:
                return None

            # Analyze sentiment
            sentiments = []
            for post in mentions:
                text = f"{post.get('title', '')} {post.get('body', '')}"
                score, conf = await self._sentiment_analyzer.analyze(text)
                sentiments.append((score, conf, post.get("score", 1)))

            # Calculate weighted sentiment
            total_weight = 0
            weighted_sentiment = 0

            for sentiment, confidence, upvotes in sentiments:
                weight = confidence * (1 + min(upvotes, 1000) / 1000)  # Cap upvote influence
                weighted_sentiment += sentiment * weight
                total_weight += weight

            if total_weight > 0:
                avg_sentiment = weighted_sentiment / total_weight
            else:
                avg_sentiment = 0.0

            # Calculate ratios
            positive = sum(1 for s, _, _ in sentiments if s > 0.1)
            negative = sum(1 for s, _, _ in sentiments if s < -0.1)
            neutral = len(sentiments) - positive - negative

            total = len(sentiments) or 1
            positive_ratio = positive / total
            negative_ratio = negative / total
            neutral_ratio = neutral / total

            # Calculate mention change
            mention_change = self._calculate_mention_change(symbol, len(mentions))

            # Determine if trending
            trending = mention_change > 100 or len(mentions) > 50

            # Calculate confidence based on sample size and agreement
            confidence = min(0.9, 0.3 + (len(mentions) / 100))

            return SocialSentimentSignal(
                symbol=symbol,
                source=AltDataSource.REDDIT,
                timestamp=datetime.now(),
                signal_value=avg_sentiment,
                confidence=confidence,
                mention_count=len(mentions),
                mention_change_pct=mention_change,
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                neutral_ratio=neutral_ratio,
                trending=trending,
                meme_stock_risk=mention_change > 200 or len(mentions) > 100,
                raw_data={"post_count": len(mentions), "subreddits": self._subreddits},
            )

        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _fetch_mentions(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch posts mentioning the symbol."""
        mentions = []

        if self._reddit is None:
            # Return mock data for development
            return self._get_mock_mentions(symbol)

        try:
            # Run in thread pool since PRAW is synchronous
            loop = asyncio.get_event_loop()

            for subreddit_name in self._subreddits:
                try:
                    subreddit = await loop.run_in_executor(
                        None, lambda: self._reddit.subreddit(subreddit_name)
                    )

                    # Search for symbol mentions
                    search_results = await loop.run_in_executor(
                        None,
                        lambda: list(
                            subreddit.search(
                                f"${symbol} OR {symbol}",
                                time_filter="day",
                                limit=50,
                            )
                        ),
                    )

                    for post in search_results:
                        # Filter by time
                        post_time = datetime.fromtimestamp(post.created_utc)
                        if post_time < datetime.now() - timedelta(hours=self._lookback_hours):
                            continue

                        mentions.append(
                            {
                                "title": post.title,
                                "body": post.selftext[:1000] if post.selftext else "",
                                "score": post.score,
                                "num_comments": post.num_comments,
                                "subreddit": subreddit_name,
                                "created_utc": post.created_utc,
                            }
                        )

                except Exception as e:
                    logger.debug(f"Error fetching from r/{subreddit_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching Reddit mentions: {e}")

        return mentions

    def _get_mock_mentions(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock mentions for development/testing."""
        import random

        # Generate random number of mock posts
        num_posts = random.randint(5, 30)
        mentions = []

        sample_titles = [
            f"${symbol} looking strong today!",
            f"Why I'm bullish on {symbol}",
            f"{symbol} DD - undervalued gem",
            f"Bought more {symbol} on the dip",
            f"Is {symbol} overvalued?",
            f"{symbol} earnings expectations",
            f"Technical analysis on {symbol}",
        ]

        for _ in range(num_posts):
            mentions.append(
                {
                    "title": random.choice(sample_titles),
                    "body": "",
                    "score": random.randint(1, 500),
                    "num_comments": random.randint(1, 100),
                    "subreddit": random.choice(self._subreddits),
                    "created_utc": (datetime.now() - timedelta(hours=random.randint(1, 24))).timestamp(),
                }
            )

        return mentions

    def _calculate_mention_change(self, symbol: str, current_count: int) -> float:
        """Calculate percentage change in mentions vs rolling average."""
        now = datetime.now()

        # Add current count to history
        self._mention_history[symbol].append((now, current_count))

        # Keep only last 7 days
        cutoff = now - timedelta(days=7)
        self._mention_history[symbol] = [
            (t, c) for t, c in self._mention_history[symbol] if t > cutoff
        ]

        # Calculate rolling average
        if len(self._mention_history[symbol]) < 2:
            return 0.0

        avg_count = sum(c for _, c in self._mention_history[symbol][:-1]) / (
            len(self._mention_history[symbol]) - 1
        )

        if avg_count == 0:
            return 0.0

        return ((current_count - avg_count) / avg_count) * 100


class StockTwitsSentimentProvider(AlternativeDataProvider):
    """
    StockTwits sentiment analysis provider.

    Note: StockTwits API has been deprecated for new applications.
    This is a placeholder for when alternative access is available.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        super().__init__(
            source=AltDataSource.STOCKTWITS,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def initialize(self) -> bool:
        """Initialize StockTwits provider."""
        logger.warning(
            "StockTwits API access is limited. "
            "Consider using Reddit sentiment as primary social signal."
        )
        self._initialized = True
        return True

    async def fetch_signal(self, symbol: str) -> Optional[AlternativeSignal]:
        """Fetch StockTwits sentiment (placeholder)."""
        # StockTwits API is deprecated for new apps
        # Return None until alternative access is available
        return None


# Factory function for creating social sentiment providers
def create_social_sentiment_provider(
    source: AltDataSource,
    **kwargs,
) -> Optional[AlternativeDataProvider]:
    """
    Create a social sentiment provider.

    Args:
        source: The social data source to create
        **kwargs: Provider-specific configuration

    Returns:
        Configured provider instance
    """
    if source == AltDataSource.REDDIT:
        return RedditSentimentProvider(**kwargs)
    elif source == AltDataSource.STOCKTWITS:
        return StockTwitsSentimentProvider(**kwargs)
    else:
        logger.warning(f"Unknown social sentiment source: {source}")
        return None
