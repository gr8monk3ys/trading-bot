"""
Alternative Data Types - Dataclasses for alternative data signals.

This module defines the data structures used throughout the alternative
data framework for representing signals from various non-traditional sources.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AltDataSource(Enum):
    """Sources of alternative data signals."""

    # Social Media
    REDDIT = "reddit"
    TWITTER = "twitter"
    STOCKTWITS = "stocktwits"

    # Order Flow
    DARK_POOL = "dark_pool"
    OPTIONS_FLOW = "options_flow"
    BLOCK_TRADES = "block_trades"

    # Web Data
    JOB_POSTINGS = "job_postings"
    GLASSDOOR = "glassdoor"
    APP_STORE = "app_store"
    PRODUCT_REVIEWS = "product_reviews"

    # Satellite & Geolocation
    SATELLITE = "satellite"
    FOOT_TRAFFIC = "foot_traffic"

    # Other
    NEWS_ADVANCED = "news_advanced"
    SEC_FILINGS = "sec_filings"
    EARNINGS_CALLS = "earnings_calls"

    # LLM-powered analysis
    LLM_EARNINGS = "llm_earnings"
    LLM_FED_SPEECH = "llm_fed_speech"
    LLM_SEC_FILING = "llm_sec_filing"
    LLM_NEWS_THEME = "llm_news_theme"


class SignalDirection(Enum):
    """Direction of the trading signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Strength/confidence level of the signal."""

    VERY_STRONG = "very_strong"  # > 0.8 confidence
    STRONG = "strong"  # 0.6 - 0.8
    MODERATE = "moderate"  # 0.4 - 0.6
    WEAK = "weak"  # 0.2 - 0.4
    VERY_WEAK = "very_weak"  # < 0.2


@dataclass
class AlternativeSignal:
    """
    A single alternative data signal for a symbol.

    Attributes:
        symbol: Stock ticker symbol
        source: Source of the alternative data
        timestamp: When the signal was generated
        signal_value: Normalized signal value (-1 to +1, negative=bearish, positive=bullish)
        confidence: Confidence in the signal (0 to 1)
        direction: Bullish, bearish, or neutral
        strength: Categorical strength of the signal
        raw_data: Original raw data that produced this signal
        metadata: Additional context-specific data
    """

    symbol: str
    source: AltDataSource
    timestamp: datetime
    signal_value: float  # -1 to +1
    confidence: float  # 0 to 1

    direction: SignalDirection = SignalDirection.NEUTRAL
    strength: SignalStrength = SignalStrength.MODERATE
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and derive fields."""
        # Clamp signal_value to [-1, 1]
        self.signal_value = max(-1.0, min(1.0, self.signal_value))

        # Clamp confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Derive direction from signal_value if not set
        if self.signal_value > 0.1:
            self.direction = SignalDirection.BULLISH
        elif self.signal_value < -0.1:
            self.direction = SignalDirection.BEARISH
        else:
            self.direction = SignalDirection.NEUTRAL

        # Derive strength from confidence
        if self.confidence >= 0.8:
            self.strength = SignalStrength.VERY_STRONG
        elif self.confidence >= 0.6:
            self.strength = SignalStrength.STRONG
        elif self.confidence >= 0.4:
            self.strength = SignalStrength.MODERATE
        elif self.confidence >= 0.2:
            self.strength = SignalStrength.WEAK
        else:
            self.strength = SignalStrength.VERY_WEAK

    @property
    def weighted_signal(self) -> float:
        """Signal value weighted by confidence."""
        return self.signal_value * self.confidence

    @property
    def is_actionable(self) -> bool:
        """Whether signal is strong enough to act on."""
        return self.confidence >= 0.4 and abs(self.signal_value) >= 0.2


@dataclass
class SocialSentimentSignal(AlternativeSignal):
    """Signal from social media sentiment analysis."""

    mention_count: int = 0
    mention_change_pct: float = 0.0  # vs rolling average
    positive_ratio: float = 0.5
    negative_ratio: float = 0.5
    neutral_ratio: float = 0.0
    trending: bool = False
    meme_stock_risk: bool = False  # High retail attention, volatile

    def __post_init__(self):
        super().__post_init__()
        # Flag meme stock risk if mention spike is extreme
        if self.mention_change_pct > 200:  # 3x normal mentions
            self.meme_stock_risk = True


@dataclass
class OrderFlowSignal(AlternativeSignal):
    """Signal from order flow analysis (dark pools, options)."""

    # Dark pool metrics
    dark_pool_volume: float = 0.0
    dark_pool_pct_of_total: float = 0.0
    block_trade_count: int = 0
    avg_block_size: float = 0.0

    # Options metrics
    call_volume: int = 0
    put_volume: int = 0
    put_call_ratio: float = 1.0
    unusual_options_activity: bool = False
    sweep_count: int = 0  # Aggressive orders hitting multiple exchanges


@dataclass
class WebScrapingSignal(AlternativeSignal):
    """Signal from web scraping (jobs, reviews, app rankings)."""

    # Job posting metrics
    job_posting_count: int = 0
    job_posting_change_pct: float = 0.0
    hiring_keywords: List[str] = field(default_factory=list)
    layoff_keywords: List[str] = field(default_factory=list)

    # Review/rating metrics
    avg_rating: float = 0.0
    rating_change: float = 0.0
    review_count: int = 0
    review_sentiment: float = 0.0

    # App ranking metrics
    app_rank: int = 0
    app_rank_change: int = 0


@dataclass
class AggregatedSignal:
    """
    Aggregated signal from multiple alternative data sources.

    Combines signals from different sources into a single actionable signal.
    """

    symbol: str
    timestamp: datetime
    sources: List[AltDataSource]
    individual_signals: List[AlternativeSignal]

    # Aggregated metrics
    composite_signal: float = 0.0  # Weighted average of all signals
    composite_confidence: float = 0.0
    agreement_ratio: float = 0.0  # % of sources agreeing on direction

    def __post_init__(self):
        """Calculate aggregated metrics."""
        if not self.individual_signals:
            return

        # Calculate weighted average signal
        total_weight = 0.0
        weighted_sum = 0.0

        for sig in self.individual_signals:
            weight = sig.confidence
            weighted_sum += sig.signal_value * weight
            total_weight += weight

        if total_weight > 0:
            self.composite_signal = weighted_sum / total_weight

        # Calculate composite confidence (average of individual confidences)
        self.composite_confidence = sum(s.confidence for s in self.individual_signals) / len(
            self.individual_signals
        )

        # Calculate agreement ratio
        bullish = sum(1 for s in self.individual_signals if s.direction == SignalDirection.BULLISH)
        bearish = sum(1 for s in self.individual_signals if s.direction == SignalDirection.BEARISH)
        majority = max(bullish, bearish)
        self.agreement_ratio = majority / len(self.individual_signals)

    @property
    def direction(self) -> SignalDirection:
        """Overall signal direction."""
        if self.composite_signal > 0.1:
            return SignalDirection.BULLISH
        elif self.composite_signal < -0.1:
            return SignalDirection.BEARISH
        return SignalDirection.NEUTRAL

    @property
    def is_high_conviction(self) -> bool:
        """Whether this is a high-conviction signal."""
        return (
            self.composite_confidence >= 0.6
            and self.agreement_ratio >= 0.7
            and abs(self.composite_signal) >= 0.3
        )


@dataclass
class AltDataProviderStatus:
    """Status information for an alternative data provider."""

    source: AltDataSource
    is_healthy: bool
    last_update: Optional[datetime]
    error_count: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0

    def __post_init__(self):
        """Derive health status."""
        if self.error_count > 5:
            self.is_healthy = False
