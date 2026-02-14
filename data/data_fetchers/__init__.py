"""
Data Fetchers for LLM Analysis

Provides data fetching infrastructure for various text sources:
- Earnings call transcripts
- Federal Reserve speeches
- SEC EDGAR filings
"""

from data.data_fetchers.earnings_fetcher import (
    EarningsTranscript,
    EarningsTranscriptFetcher,
)
from data.data_fetchers.fed_speech_fetcher import (
    FedSpeech,
    FedSpeechFetcher,
)
from data.data_fetchers.sec_edgar_fetcher import (
    SECEdgarFetcher,
    SECFiling,
)

__all__ = [
    # Earnings
    "EarningsTranscript",
    "EarningsTranscriptFetcher",
    # Fed
    "FedSpeech",
    "FedSpeechFetcher",
    # SEC
    "SECFiling",
    "SECEdgarFetcher",
]
