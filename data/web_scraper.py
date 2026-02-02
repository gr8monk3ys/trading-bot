"""
Web Scraper - Job postings, employee reviews, and app rankings signals.

This module provides alternative data signals from web scraping sources:
- Job postings (hiring trends, layoff signals)
- Glassdoor employee sentiment
- App Store / Google Play rankings

Usage:
    from data.web_scraper import (
        JobPostingsProvider,
        GlassdoorSentimentProvider,
        AppRankingsProvider,
    )

    provider = JobPostingsProvider()
    await provider.initialize()
    signal = await provider.fetch_signal("AAPL")
"""

import asyncio
import logging
import re
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from data.alt_data_types import (
    AltDataSource,
    AlternativeSignal,
    WebScrapingSignal,
)
from data.alternative_data_provider import AlternativeDataProvider

logger = logging.getLogger(__name__)


# Company name to ticker mapping (subset - would be expanded)
COMPANY_TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "paypal": "PYPL",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "coinbase": "COIN",
    "shopify": "SHOP",
    "zoom": "ZM",
    "spotify": "SPOT",
    "snap": "SNAP",
    "twitter": "X",
    "disney": "DIS",
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "starbucks": "SBUX",
    "mcdonalds": "MCD",
    "nike": "NKE",
    "jpmorgan": "JPM",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "visa": "V",
    "mastercard": "MA",
}

# Reverse mapping
TICKER_COMPANY_MAP = {v: k for k, v in COMPANY_TICKER_MAP.items()}


@dataclass
class JobPostingData:
    """Data from job posting analysis."""

    total_postings: int = 0
    change_pct_30d: float = 0.0
    engineering_pct: float = 0.0
    sales_pct: float = 0.0
    layoff_mentions: int = 0
    hiring_surge: bool = False
    contraction_signal: bool = False


@dataclass
class GlassdoorData:
    """Data from Glassdoor analysis."""

    overall_rating: float = 0.0
    rating_change_90d: float = 0.0
    recommend_pct: float = 0.0
    ceo_approval: float = 0.0
    review_count_30d: int = 0
    sentiment_score: float = 0.0  # -1 to +1


@dataclass
class AppRankingData:
    """Data from app store ranking analysis."""

    ios_rank: Optional[int] = None
    android_rank: Optional[int] = None
    ios_rank_change_7d: int = 0
    android_rank_change_7d: int = 0
    combined_rank_score: float = 0.0  # 0 to 1, higher is better


class JobPostingsProvider(AlternativeDataProvider):
    """
    Job postings analysis provider.

    Analyzes hiring trends from job boards to generate signals:
    - Hiring surge = expansion signal (bullish)
    - Layoff keywords = contraction signal (bearish)
    - Engineering vs Sales ratio indicates R&D investment
    """

    # Keywords indicating expansion
    EXPANSION_KEYWORDS = {
        "growing team", "rapid growth", "scaling", "expansion",
        "new office", "new market", "hiring spree",
    }

    # Keywords indicating contraction
    CONTRACTION_KEYWORDS = {
        "restructuring", "streamlining", "layoff", "reduction",
        "downsizing", "workforce adjustment", "right-sizing",
    }

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,  # 1 hour cache (job data doesn't change fast)
    ):
        super().__init__(
            source=AltDataSource.JOB_POSTINGS,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._job_history: Dict[str, List[Tuple[datetime, int]]] = {}

    async def initialize(self) -> bool:
        """Initialize job postings provider."""
        # In production, would initialize API connections to:
        # - LinkedIn API (requires partnership)
        # - Indeed API
        # - Greenhouse, Lever APIs
        logger.info(
            "JobPostingsProvider initialized (using mock data - "
            "production would use LinkedIn/Indeed APIs)"
        )
        self._initialized = True
        return True

    async def fetch_signal(self, symbol: str) -> Optional[WebScrapingSignal]:
        """Fetch job posting signal for a symbol."""
        if not self._initialized:
            await self.initialize()

        try:
            job_data = await self._fetch_job_data(symbol)

            if job_data is None:
                return None

            # Calculate signal value
            signal_value = self._calculate_signal(job_data)

            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(job_data)

            # Build keywords lists
            hiring_keywords = ["hiring", "growth"] if job_data.hiring_surge else []
            layoff_keywords = ["layoff", "restructuring"] if job_data.contraction_signal else []

            return WebScrapingSignal(
                symbol=symbol,
                source=AltDataSource.JOB_POSTINGS,
                timestamp=datetime.now(),
                signal_value=signal_value,
                confidence=confidence,
                job_posting_count=job_data.total_postings,
                job_posting_change_pct=job_data.change_pct_30d,
                hiring_keywords=hiring_keywords,
                layoff_keywords=layoff_keywords,
                raw_data={
                    "total_postings": job_data.total_postings,
                    "change_pct_30d": job_data.change_pct_30d,
                    "engineering_pct": job_data.engineering_pct,
                    "sales_pct": job_data.sales_pct,
                    "hiring_surge": job_data.hiring_surge,
                    "contraction_signal": job_data.contraction_signal,
                },
            )

        except Exception as e:
            logger.error(f"Error fetching job postings for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _fetch_job_data(self, symbol: str) -> Optional[JobPostingData]:
        """Fetch job posting data for a company."""
        # In production, would scrape/API call to job boards
        # For now, return mock data based on symbol characteristics
        return self._get_mock_job_data(symbol)

    def _get_mock_job_data(self, symbol: str) -> JobPostingData:
        """Generate mock job posting data."""
        import random

        # Seed based on symbol for consistency
        random.seed(hash(symbol) % 2**32)

        # Large caps tend to have more postings
        base_postings = random.randint(50, 500)

        # Random change in postings
        change_pct = random.uniform(-20, 30)

        # Engineering vs sales ratio
        engineering_pct = random.uniform(0.3, 0.7)
        sales_pct = random.uniform(0.1, 0.3)

        # Hiring surge if change > 20%
        hiring_surge = change_pct > 20

        # Contraction if change < -15%
        contraction_signal = change_pct < -15

        return JobPostingData(
            total_postings=base_postings,
            change_pct_30d=change_pct,
            engineering_pct=engineering_pct,
            sales_pct=sales_pct,
            layoff_mentions=random.randint(0, 5) if contraction_signal else 0,
            hiring_surge=hiring_surge,
            contraction_signal=contraction_signal,
        )

    def _calculate_signal(self, data: JobPostingData) -> float:
        """Calculate signal value from job data."""
        signal = 0.0

        # Job posting change contributes most
        # Normalize to -1 to +1 range (assuming max +/- 50% change)
        change_signal = max(-1.0, min(1.0, data.change_pct_30d / 50.0))
        signal += change_signal * 0.6

        # High engineering ratio is positive
        eng_signal = (data.engineering_pct - 0.4) * 2  # Center at 40%
        signal += eng_signal * 0.2

        # Hiring surge bonus
        if data.hiring_surge:
            signal += 0.2

        # Contraction penalty
        if data.contraction_signal:
            signal -= 0.3

        return max(-1.0, min(1.0, signal))

    def _calculate_confidence(self, data: JobPostingData) -> float:
        """Calculate confidence based on data quality."""
        confidence = 0.5  # Base confidence

        # More postings = more reliable data
        if data.total_postings > 100:
            confidence += 0.2
        elif data.total_postings > 50:
            confidence += 0.1

        # Large changes are more reliable signals
        if abs(data.change_pct_30d) > 15:
            confidence += 0.1

        return min(0.9, confidence)


class GlassdoorSentimentProvider(AlternativeDataProvider):
    """
    Glassdoor employee sentiment provider.

    Analyzes employee reviews for signals:
    - Rising ratings = positive sentiment, potential outperformance
    - Falling ratings = internal issues, potential underperformance
    - CEO approval changes can indicate strategic shifts
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
    ):
        super().__init__(
            source=AltDataSource.GLASSDOOR,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def initialize(self) -> bool:
        """Initialize Glassdoor provider."""
        logger.info(
            "GlassdoorSentimentProvider initialized (using mock data - "
            "production would use Glassdoor scraping/API)"
        )
        self._initialized = True
        return True

    async def fetch_signal(self, symbol: str) -> Optional[WebScrapingSignal]:
        """Fetch Glassdoor sentiment signal."""
        if not self._initialized:
            await self.initialize()

        try:
            glassdoor_data = await self._fetch_glassdoor_data(symbol)

            if glassdoor_data is None:
                return None

            signal_value = self._calculate_signal(glassdoor_data)
            confidence = self._calculate_confidence(glassdoor_data)

            return WebScrapingSignal(
                symbol=symbol,
                source=AltDataSource.GLASSDOOR,
                timestamp=datetime.now(),
                signal_value=signal_value,
                confidence=confidence,
                avg_rating=glassdoor_data.overall_rating,
                rating_change=glassdoor_data.rating_change_90d,
                review_count=glassdoor_data.review_count_30d,
                review_sentiment=glassdoor_data.sentiment_score,
                raw_data={
                    "overall_rating": glassdoor_data.overall_rating,
                    "rating_change_90d": glassdoor_data.rating_change_90d,
                    "recommend_pct": glassdoor_data.recommend_pct,
                    "ceo_approval": glassdoor_data.ceo_approval,
                    "sentiment_score": glassdoor_data.sentiment_score,
                },
            )

        except Exception as e:
            logger.error(f"Error fetching Glassdoor data for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _fetch_glassdoor_data(self, symbol: str) -> Optional[GlassdoorData]:
        """Fetch Glassdoor data for a company."""
        return self._get_mock_glassdoor_data(symbol)

    def _get_mock_glassdoor_data(self, symbol: str) -> GlassdoorData:
        """Generate mock Glassdoor data."""
        import random

        random.seed(hash(symbol + "glassdoor") % 2**32)

        # Overall rating 1-5
        overall_rating = random.uniform(3.0, 4.5)

        # Rating change in last 90 days
        rating_change = random.uniform(-0.3, 0.3)

        # Recommend percentage
        recommend_pct = random.uniform(0.5, 0.9)

        # CEO approval
        ceo_approval = random.uniform(0.4, 0.95)

        # Recent reviews
        review_count = random.randint(10, 200)

        # Sentiment from recent reviews
        sentiment = random.uniform(-0.3, 0.5)

        return GlassdoorData(
            overall_rating=overall_rating,
            rating_change_90d=rating_change,
            recommend_pct=recommend_pct,
            ceo_approval=ceo_approval,
            review_count_30d=review_count,
            sentiment_score=sentiment,
        )

    def _calculate_signal(self, data: GlassdoorData) -> float:
        """Calculate signal from Glassdoor data."""
        signal = 0.0

        # Overall rating (centered at 3.5)
        rating_signal = (data.overall_rating - 3.5) / 1.5  # -1 to +1 range
        signal += rating_signal * 0.3

        # Rating change is important
        change_signal = data.rating_change_90d * 3  # Amplify small changes
        signal += change_signal * 0.3

        # Recommend percentage
        recommend_signal = (data.recommend_pct - 0.7) * 3  # Center at 70%
        signal += recommend_signal * 0.2

        # Recent sentiment
        signal += data.sentiment_score * 0.2

        return max(-1.0, min(1.0, signal))

    def _calculate_confidence(self, data: GlassdoorData) -> float:
        """Calculate confidence from Glassdoor data quality."""
        confidence = 0.4  # Lower base - employee reviews are noisy

        # More reviews = more reliable
        if data.review_count_30d > 50:
            confidence += 0.2
        elif data.review_count_30d > 20:
            confidence += 0.1

        # Extreme ratings are more reliable signals
        if data.overall_rating > 4.2 or data.overall_rating < 2.8:
            confidence += 0.1

        return min(0.8, confidence)


class AppRankingsProvider(AlternativeDataProvider):
    """
    App Store / Google Play rankings provider.

    Analyzes app rankings for consumer-facing companies:
    - Rising rankings = increased user engagement
    - Falling rankings = potential user churn
    """

    # Companies with major consumer apps
    CONSUMER_APP_COMPANIES = {
        "META", "GOOGL", "AMZN", "NFLX", "UBER", "LYFT",
        "ABNB", "SPOT", "SNAP", "DIS", "SBUX", "PYPL",
    }

    def __init__(
        self,
        cache_ttl_seconds: int = 1800,  # 30 min cache (rankings change faster)
    ):
        super().__init__(
            source=AltDataSource.APP_STORE,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def initialize(self) -> bool:
        """Initialize app rankings provider."""
        logger.info(
            "AppRankingsProvider initialized (using mock data - "
            "production would use App Annie/Sensor Tower APIs)"
        )
        self._initialized = True
        return True

    async def fetch_signal(self, symbol: str) -> Optional[WebScrapingSignal]:
        """Fetch app ranking signal."""
        if not self._initialized:
            await self.initialize()

        # Only relevant for consumer app companies
        if symbol not in self.CONSUMER_APP_COMPANIES:
            return None

        try:
            app_data = await self._fetch_app_data(symbol)

            if app_data is None:
                return None

            signal_value = self._calculate_signal(app_data)
            confidence = self._calculate_confidence(app_data)

            # Use iOS rank as primary (or average)
            primary_rank = app_data.ios_rank or app_data.android_rank or 0
            avg_rank_change = (app_data.ios_rank_change_7d + app_data.android_rank_change_7d) // 2

            return WebScrapingSignal(
                symbol=symbol,
                source=AltDataSource.APP_STORE,
                timestamp=datetime.now(),
                signal_value=signal_value,
                confidence=confidence,
                app_rank=primary_rank,
                app_rank_change=avg_rank_change,
                raw_data={
                    "ios_rank": app_data.ios_rank,
                    "android_rank": app_data.android_rank,
                    "ios_rank_change_7d": app_data.ios_rank_change_7d,
                    "android_rank_change_7d": app_data.android_rank_change_7d,
                    "combined_rank_score": app_data.combined_rank_score,
                },
            )

        except Exception as e:
            logger.error(f"Error fetching app rankings for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _fetch_app_data(self, symbol: str) -> Optional[AppRankingData]:
        """Fetch app ranking data."""
        return self._get_mock_app_data(symbol)

    def _get_mock_app_data(self, symbol: str) -> AppRankingData:
        """Generate mock app ranking data."""
        import random

        random.seed(hash(symbol + "appstore") % 2**32)

        # Top companies tend to have higher ranks
        base_rank = random.randint(1, 100)

        # Rank changes
        ios_change = random.randint(-20, 20)
        android_change = random.randint(-20, 20)

        # Combined score (higher = better)
        combined = 1.0 - (base_rank / 200)  # Normalize to 0-1

        return AppRankingData(
            ios_rank=base_rank,
            android_rank=base_rank + random.randint(-10, 10),
            ios_rank_change_7d=ios_change,
            android_rank_change_7d=android_change,
            combined_rank_score=max(0.0, min(1.0, combined)),
        )

    def _calculate_signal(self, data: AppRankingData) -> float:
        """Calculate signal from app ranking data."""
        signal = 0.0

        # Current rank score
        signal += (data.combined_rank_score - 0.5) * 0.4

        # Rank improvement (negative change = improvement)
        avg_change = (data.ios_rank_change_7d + data.android_rank_change_7d) / 2

        # Normalize: -20 change (improvement) = +1, +20 change (decline) = -1
        change_signal = -avg_change / 20.0
        signal += change_signal * 0.6

        return max(-1.0, min(1.0, signal))

    def _calculate_confidence(self, data: AppRankingData) -> float:
        """Calculate confidence from app data quality."""
        confidence = 0.5

        # Top 50 apps have more reliable data
        if data.ios_rank and data.ios_rank <= 50:
            confidence += 0.2

        # Large rank changes are more significant
        avg_change = abs(data.ios_rank_change_7d + data.android_rank_change_7d) / 2
        if avg_change > 10:
            confidence += 0.1

        return min(0.85, confidence)


class WebScraperAggregator:
    """
    Aggregates signals from all web scraping sources.

    Combines job postings, Glassdoor, and app rankings into
    a single composite web data signal.
    """

    def __init__(self):
        self._providers: Dict[AltDataSource, AlternativeDataProvider] = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all web scraping providers."""
        self._providers[AltDataSource.JOB_POSTINGS] = JobPostingsProvider()
        self._providers[AltDataSource.GLASSDOOR] = GlassdoorSentimentProvider()
        self._providers[AltDataSource.APP_STORE] = AppRankingsProvider()

        for provider in self._providers.values():
            await provider.initialize()

        self._initialized = True
        return True

    async def get_composite_signal(self, symbol: str) -> Optional[WebScrapingSignal]:
        """
        Get composite signal from all web scraping sources.

        Weights:
        - Job postings: 40% (most predictive of fundamentals)
        - Glassdoor: 30% (employee sentiment predicts issues)
        - App rankings: 30% (consumer engagement, if applicable)
        """
        if not self._initialized:
            await self.initialize()

        signals: List[Tuple[float, float, float]] = []  # (value, confidence, weight)

        # Fetch from all providers
        for source, provider in self._providers.items():
            signal = await provider.fetch_signal(symbol)

            if signal:
                weight = {
                    AltDataSource.JOB_POSTINGS: 0.4,
                    AltDataSource.GLASSDOOR: 0.3,
                    AltDataSource.APP_STORE: 0.3,
                }.get(source, 0.2)

                signals.append((signal.signal_value, signal.confidence, weight))

        if not signals:
            return None

        # Calculate weighted composite
        total_weight = sum(conf * weight for _, conf, weight in signals)

        if total_weight == 0:
            return None

        composite_value = sum(
            val * conf * weight for val, conf, weight in signals
        ) / total_weight

        # Composite confidence
        composite_confidence = min(
            0.9,
            sum(conf * weight for _, conf, weight in signals) / sum(w for _, _, w in signals)
        )

        return WebScrapingSignal(
            symbol=symbol,
            source=AltDataSource.JOB_POSTINGS,  # Primary source
            timestamp=datetime.now(),
            signal_value=composite_value,
            confidence=composite_confidence,
            raw_data={
                "source_count": len(signals),
                "sources": [str(s) for s in self._providers.keys()],
                "composite": True,
            },
        )


# Factory function
def create_web_scraper_provider(
    source: AltDataSource,
    **kwargs,
) -> Optional[AlternativeDataProvider]:
    """
    Create a web scraper provider.

    Args:
        source: The web scraping source to create
        **kwargs: Provider-specific configuration

    Returns:
        Configured provider instance
    """
    if source == AltDataSource.JOB_POSTINGS:
        return JobPostingsProvider(**kwargs)
    elif source == AltDataSource.GLASSDOOR:
        return GlassdoorSentimentProvider(**kwargs)
    elif source == AltDataSource.APP_STORE:
        return AppRankingsProvider(**kwargs)
    else:
        logger.warning(f"Unknown web scraper source: {source}")
        return None
