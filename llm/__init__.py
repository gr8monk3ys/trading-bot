"""
LLM Module - Language Model Integration for Trading Signals

Provides unified interface for OpenAI GPT-4 and Anthropic Claude with:
- Automatic fallback on failure
- Token counting and cost tracking
- Rate limiting and retry logic
- Response caching
"""

from llm.llm_types import (
    LLMProvider,
    LLMResponse,
    LLMClientConfig,
    LLMAnalysisResult,
    EarningsAnalysis,
    FedSpeechAnalysis,
    SECFilingAnalysis,
    NewsThemeAnalysis,
)

from llm.llm_client import (
    BaseLLMClient,
    OpenAIClient,
    AnthropicClient,
    LLMClientWithFallback,
    RateLimiter,
    CostTracker,
    create_llm_client,
)

__all__ = [
    # Types
    "LLMProvider",
    "LLMResponse",
    "LLMClientConfig",
    "LLMAnalysisResult",
    "EarningsAnalysis",
    "FedSpeechAnalysis",
    "SECFilingAnalysis",
    "NewsThemeAnalysis",
    # Clients
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "LLMClientWithFallback",
    "RateLimiter",
    "CostTracker",
    "create_llm_client",
]
