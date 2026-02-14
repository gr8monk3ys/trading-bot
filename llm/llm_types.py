"""
LLM Data Types

Defines dataclasses for LLM responses, analysis results, and configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMClientConfig:
    """Configuration for LLM client."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Provider selection
    primary_provider: LLMProvider = LLMProvider.ANTHROPIC

    # Model selection
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 60.0

    # Rate limiting
    max_tokens_per_minute: int = 50000
    max_requests_per_minute: int = 10

    # Caching
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_enabled: bool = True

    # Cost management
    daily_cost_cap_usd: float = 50.0
    weekly_cost_cap_usd: float = 250.0
    monthly_cost_cap_usd: float = 750.0

    # Analysis temperature (0 = deterministic)
    temperature: float = 0.0


@dataclass
class LLMResponse:
    """Response from LLM API call."""

    content: str
    provider: LLMProvider
    model: str

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Cost tracking
    cost_usd: float

    # Performance
    latency_ms: float

    # Caching
    cached: bool = False
    cache_key: Optional[str] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None

    def __post_init__(self):
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class LLMAnalysisResult:
    """Base result from LLM analysis."""

    source_type: str  # "earnings", "fed", "sec", "news"
    symbol: Optional[str]
    timestamp: datetime

    # Core signal
    sentiment_score: float  # -1 to +1
    confidence: float  # 0 to 1

    # Structured insights
    key_insights: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

    # Reasoning (chain of thought)
    reasoning: str = ""

    # LLM metadata
    llm_provider: str = ""
    llm_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and clamp signal values."""
        self.sentiment_score = max(-1.0, min(1.0, self.sentiment_score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class EarningsAnalysis(LLMAnalysisResult):
    """Earnings call analysis result."""

    # Earnings-specific fields
    guidance_change: str = "no_guidance"  # "raised", "lowered", "maintained", "no_guidance"
    management_tone: str = "neutral"  # "confident", "cautious", "defensive", "optimistic"
    key_metrics_mentioned: Dict[str, str] = field(default_factory=dict)
    analyst_sentiment: str = "neutral"  # "bullish", "bearish", "mixed"
    surprise_factor: float = 0.0  # How surprising was the content (-1 to +1)

    # Transcript metadata
    fiscal_quarter: str = ""
    fiscal_year: int = 0
    transcript_date: Optional[datetime] = None
    word_count: int = 0

    def __post_init__(self):
        """Initialize and validate."""
        super().__post_init__()
        self.source_type = "earnings"
        self.surprise_factor = max(-1.0, min(1.0, self.surprise_factor))

        # Validate enums
        valid_guidance = ["raised", "lowered", "maintained", "no_guidance"]
        if self.guidance_change not in valid_guidance:
            self.guidance_change = "no_guidance"

        valid_tone = ["confident", "cautious", "defensive", "optimistic", "neutral"]
        if self.management_tone not in valid_tone:
            self.management_tone = "neutral"

        valid_sentiment = ["bullish", "bearish", "mixed", "neutral"]
        if self.analyst_sentiment not in valid_sentiment:
            self.analyst_sentiment = "neutral"


@dataclass
class FedSpeechAnalysis(LLMAnalysisResult):
    """Fed speech analysis result."""

    # Fed-specific fields
    rate_expectations: str = "neutral"  # "hawkish", "dovish", "neutral"
    key_themes: List[str] = field(default_factory=list)
    policy_signals: List[str] = field(default_factory=list)
    market_implications: Dict[str, str] = field(default_factory=dict)  # {"equities": "bearish"}
    deviation_from_consensus: float = 0.0  # -1 to +1

    # Speech metadata
    speaker: str = ""
    title: str = ""
    speech_date: Optional[datetime] = None
    event_type: str = ""  # "testimony", "speech", "press_conference"
    fomc_proximity_days: int = 0

    def __post_init__(self):
        """Initialize and validate."""
        super().__post_init__()
        self.source_type = "fed"
        self.symbol = None  # Fed analysis is market-wide
        self.deviation_from_consensus = max(-1.0, min(1.0, self.deviation_from_consensus))

        valid_expectations = ["hawkish", "dovish", "neutral"]
        if self.rate_expectations not in valid_expectations:
            self.rate_expectations = "neutral"


@dataclass
class SECFilingAnalysis(LLMAnalysisResult):
    """SEC filing analysis result."""

    # SEC-specific fields
    filing_type: str = ""  # "10-K", "10-Q", "8-K"
    material_changes: List[str] = field(default_factory=list)
    new_risk_factors: List[str] = field(default_factory=list)
    removed_risk_factors: List[str] = field(default_factory=list)
    litigation_mentions: List[str] = field(default_factory=list)
    related_party_concerns: List[str] = field(default_factory=list)
    going_concern_risk: bool = False

    # Filing metadata
    filing_date: Optional[datetime] = None
    accepted_date: Optional[datetime] = None
    cik: str = ""
    accession_number: str = ""

    def __post_init__(self):
        """Initialize and validate."""
        super().__post_init__()
        self.source_type = "sec"

        valid_types = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1", ""]
        if self.filing_type not in valid_types:
            self.filing_type = ""


@dataclass
class NewsThemeAnalysis(LLMAnalysisResult):
    """News theme extraction result."""

    # News-specific fields
    primary_theme: str = ""  # "product_launch", "acquisition", "earnings", etc.
    secondary_themes: List[str] = field(default_factory=list)
    catalysts_identified: List[str] = field(default_factory=list)
    time_sensitivity: str = "medium"  # "immediate", "short_term", "long_term"
    market_impact_estimate: str = "medium"  # "high", "medium", "low"

    # News metadata
    article_count: int = 0
    sources: List[str] = field(default_factory=list)
    date_range_hours: int = 24

    def __post_init__(self):
        """Initialize and validate."""
        super().__post_init__()
        self.source_type = "news"

        valid_sensitivity = ["immediate", "short_term", "long_term", "medium"]
        if self.time_sensitivity not in valid_sensitivity:
            self.time_sensitivity = "medium"

        valid_impact = ["high", "medium", "low"]
        if self.market_impact_estimate not in valid_impact:
            self.market_impact_estimate = "medium"


# Cost per million tokens (as of 2024)
LLM_COSTS = {
    "gpt-4o": {
        "input": 2.50,  # $2.50 per 1M input tokens
        "output": 10.00,  # $10.00 per 1M output tokens
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "gpt-4": {
        "input": 30.00,
        "output": 60.00,
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given model and token count."""
    if model not in LLM_COSTS:
        # Default to moderate cost estimate
        return (input_tokens * 5.0 + output_tokens * 15.0) / 1_000_000

    costs = LLM_COSTS[model]
    input_cost = (input_tokens * costs["input"]) / 1_000_000
    output_cost = (output_tokens * costs["output"]) / 1_000_000
    return input_cost + output_cost
