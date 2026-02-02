"""
LLM Prompts for Trading Signal Analysis

Structured prompts for extracting trading signals from various text sources.
"""

from llm.prompts.earnings_prompts import (
    EARNINGS_SYSTEM_PROMPT,
    EARNINGS_ANALYSIS_PROMPT,
    EARNINGS_ANALYSIS_SCHEMA,
    format_earnings_prompt,
)
from llm.prompts.fed_speech_prompts import (
    FED_SYSTEM_PROMPT,
    FED_SPEECH_ANALYSIS_PROMPT,
    FED_SPEECH_ANALYSIS_SCHEMA,
    format_fed_speech_prompt,
)
from llm.prompts.sec_filing_prompts import (
    SEC_SYSTEM_PROMPT,
    SEC_FILING_ANALYSIS_PROMPT,
    SEC_FILING_ANALYSIS_SCHEMA,
    format_sec_filing_prompt,
)
from llm.prompts.news_theme_prompts import (
    NEWS_SYSTEM_PROMPT,
    NEWS_THEME_ANALYSIS_PROMPT,
    NEWS_THEME_ANALYSIS_SCHEMA,
    format_news_theme_prompt,
)

__all__ = [
    # Earnings prompts
    "EARNINGS_SYSTEM_PROMPT",
    "EARNINGS_ANALYSIS_PROMPT",
    "EARNINGS_ANALYSIS_SCHEMA",
    "format_earnings_prompt",
    # Fed speech prompts
    "FED_SYSTEM_PROMPT",
    "FED_SPEECH_ANALYSIS_PROMPT",
    "FED_SPEECH_ANALYSIS_SCHEMA",
    "format_fed_speech_prompt",
    # SEC filing prompts
    "SEC_SYSTEM_PROMPT",
    "SEC_FILING_ANALYSIS_PROMPT",
    "SEC_FILING_ANALYSIS_SCHEMA",
    "format_sec_filing_prompt",
    # News theme prompts
    "NEWS_SYSTEM_PROMPT",
    "NEWS_THEME_ANALYSIS_PROMPT",
    "NEWS_THEME_ANALYSIS_SCHEMA",
    "format_news_theme_prompt",
]
