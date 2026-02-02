"""
Fed Speech Analyzer

LLM-powered analysis of Federal Reserve communications to extract macro trading signals.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from data.alt_data_types import AltDataSource, AlternativeSignal
from data.alternative_data_provider import AlternativeDataProvider
from data.data_fetchers import FedSpeech, FedSpeechFetcher
from llm import (
    LLMClientWithFallback,
    create_llm_client,
    FedSpeechAnalysis,
)
from llm.prompts.fed_speech_prompts import (
    FED_SYSTEM_PROMPT,
    format_fed_speech_prompt,
)

logger = logging.getLogger(__name__)


class FedSpeechAnalyzer(AlternativeDataProvider):
    """
    Analyzes Federal Reserve speeches using LLM to extract macro trading signals.

    This is a MARKET-WIDE signal provider, not symbol-specific.
    The signal applies to overall market direction.

    Extracts:
    - Rate path expectations (hawkish/dovish/neutral)
    - Key policy themes
    - Market implications for equities, bonds, dollar
    - Deviation from consensus

    Cache TTL: 24 hours
    """

    # Cache TTL: 24 hours in seconds
    CACHE_TTL_SECONDS = 24 * 60 * 60

    # Use "MARKET" as the symbol for market-wide signals
    MARKET_SYMBOL = "MARKET"

    def __init__(
        self,
        llm_client: Optional[LLMClientWithFallback] = None,
        speech_fetcher: Optional[FedSpeechFetcher] = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        """
        Initialize Fed speech analyzer.

        Args:
            llm_client: LLM client for analysis
            speech_fetcher: Fetcher for Fed speeches
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(
            source=AltDataSource.LLM_FED_SPEECH,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        self._llm_client = llm_client
        self._speech_fetcher = speech_fetcher or FedSpeechFetcher()

        # Cache the latest analysis for market-wide signal
        self._latest_analysis: Optional[FedSpeechAnalysis] = None
        self._latest_speech_date: Optional[datetime] = None

    async def initialize(self) -> bool:
        """Initialize the analyzer."""
        try:
            if self._llm_client is None:
                self._llm_client = create_llm_client()

            if self._llm_client is None:
                logger.warning("No LLM client available - analyzer will use mock responses")

            self._initialized = True
            logger.info("FedSpeechAnalyzer initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FedSpeechAnalyzer: {e}")
            return False

    async def fetch_signal(self, symbol: str) -> Optional[AlternativeSignal]:
        """
        Get market-wide Fed signal.

        Note: This is a market-wide signal. The symbol parameter is used
        to adjust the signal for specific assets (e.g., rate-sensitive stocks).

        Args:
            symbol: Stock ticker symbol (used for context adjustment)

        Returns:
            AlternativeSignal with Fed sentiment
        """
        try:
            # Fetch latest speeches
            speeches = await self._speech_fetcher.fetch_latest(limit=3)
            if not speeches:
                logger.debug("No Fed speeches available")
                return None

            # Use most recent speech
            latest_speech = speeches[0]

            # Check if we've already analyzed this speech
            if (
                self._latest_analysis is not None
                and self._latest_speech_date == latest_speech.date
            ):
                # Return cached analysis adjusted for symbol
                return self._create_signal(symbol, latest_speech, self._latest_analysis)

            # Analyze with LLM
            analysis = await self._analyze_speech(latest_speech)
            if not analysis:
                return None

            # Cache the analysis
            self._latest_analysis = analysis
            self._latest_speech_date = latest_speech.date

            # Create symbol-adjusted signal
            signal = self._create_signal(symbol, latest_speech, analysis)
            return signal

        except Exception as e:
            logger.error(f"Error analyzing Fed speech for {symbol}: {e}")
            self._error_count += 1
            return None

    async def get_market_signal(self) -> Optional[AlternativeSignal]:
        """
        Get pure market-wide Fed signal without symbol adjustment.

        Returns:
            AlternativeSignal for overall market direction
        """
        return await self.fetch_signal(self.MARKET_SYMBOL)

    async def _analyze_speech(
        self,
        speech: FedSpeech,
    ) -> Optional[FedSpeechAnalysis]:
        """Analyze speech using LLM."""
        try:
            if self._llm_client is None:
                return self._generate_mock_analysis(speech)

            # Format prompt
            prompt = format_fed_speech_prompt(
                speaker=speech.speaker,
                title=speech.title,
                date=speech.date.strftime("%Y-%m-%d"),
                event_type=speech.event_type,
                fomc_days=speech.days_to_fomc,
                speech_text=speech.content,
            )

            # Call LLM
            response = await self._llm_client.complete(
                prompt=prompt,
                system_prompt=FED_SYSTEM_PROMPT,
                max_tokens=1000,
                temperature=0.3,
            )

            if not response or not response.content:
                return None

            return self._parse_llm_response(response.content)

        except Exception as e:
            logger.error(f"LLM analysis error for Fed speech: {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[FedSpeechAnalysis]:
        """Parse LLM JSON response."""
        try:
            # Handle markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)

            # Map rate expectations
            rate_map = {
                "hawkish": "hawkish",
                "dovish": "dovish",
                "neutral": "neutral",
            }
            rate_expectations = rate_map.get(
                data.get("rate_expectations", "neutral"),
                "neutral"
            )

            # Parse market implications
            market_impl = data.get("market_implications", {})

            return FedSpeechAnalysis(
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                confidence=float(data.get("confidence", 0.5)),
                key_insights=data.get("key_insights", []),
                risks=data.get("risks", []),
                opportunities=data.get("opportunities", []),
                reasoning=data.get("reasoning", ""),
                rate_expectations=rate_expectations,
                key_themes=data.get("key_themes", []),
                policy_signals=data.get("policy_signals", []),
                market_implications={
                    "equities": market_impl.get("equities", "neutral"),
                    "bonds": market_impl.get("bonds", "neutral"),
                    "dollar": market_impl.get("dollar", "neutral"),
                },
                deviation_from_consensus=float(data.get("deviation_from_consensus", 0.0)),
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

    def _generate_mock_analysis(
        self,
        speech: FedSpeech,
    ) -> FedSpeechAnalysis:
        """Generate mock analysis for testing."""
        content = speech.content.lower()

        # Simple heuristics
        hawkish_words = ["inflation", "tightening", "restrictive", "higher", "concern"]
        dovish_words = ["growth", "employment", "support", "accommodative", "lower"]

        hawkish_count = sum(content.count(word) for word in hawkish_words)
        dovish_count = sum(content.count(word) for word in dovish_words)

        total = hawkish_count + dovish_count + 1
        sentiment = (dovish_count - hawkish_count) / total  # Dovish = positive for equities

        # Determine rate expectations
        if hawkish_count > dovish_count * 1.5:
            rate_expectations = "hawkish"
        elif dovish_count > hawkish_count * 1.5:
            rate_expectations = "dovish"
        else:
            rate_expectations = "neutral"

        return FedSpeechAnalysis(
            sentiment_score=max(-1, min(1, sentiment)),
            confidence=0.5,
            key_insights=[f"Speech by {speech.speaker} analyzed"],
            risks=[],
            opportunities=[],
            reasoning=f"Heuristic analysis. Hawkish words: {hawkish_count}, Dovish words: {dovish_count}",
            rate_expectations=rate_expectations,
            key_themes=["monetary policy"],
            policy_signals=[],
            market_implications={
                "equities": "bullish" if sentiment > 0.1 else "bearish" if sentiment < -0.1 else "neutral",
                "bonds": "bearish" if sentiment > 0.1 else "bullish" if sentiment < -0.1 else "neutral",
                "dollar": "bearish" if sentiment > 0.1 else "bullish" if sentiment < -0.1 else "neutral",
            },
            deviation_from_consensus=0.0,
        )

    def _create_signal(
        self,
        symbol: str,
        speech: FedSpeech,
        analysis: FedSpeechAnalysis,
    ) -> AlternativeSignal:
        """Create AlternativeSignal from analysis."""
        # Adjust signal based on rate sensitivity
        signal_adjustment = self._get_rate_sensitivity_adjustment(symbol, analysis)
        final_signal = analysis.sentiment_score + signal_adjustment
        final_signal = max(-1.0, min(1.0, final_signal))

        # Adjust confidence based on FOMC proximity
        confidence = analysis.confidence
        if abs(speech.days_to_fomc) <= 3:
            confidence = min(1.0, confidence * 1.2)  # Higher confidence near FOMC

        # Build metadata
        metadata: Dict[str, Any] = {
            "speaker": speech.speaker,
            "title": speech.title,
            "event_type": speech.event_type,
            "speech_date": speech.date.isoformat(),
            "days_to_fomc": speech.days_to_fomc,
            "rate_expectations": analysis.rate_expectations,
            "key_themes": analysis.key_themes,
            "market_implications": analysis.market_implications,
            "deviation_from_consensus": analysis.deviation_from_consensus,
            "reasoning": analysis.reasoning,
        }

        return AlternativeSignal(
            symbol=symbol,
            source=AltDataSource.LLM_FED_SPEECH,
            timestamp=datetime.now(),
            signal_value=final_signal,
            confidence=confidence,
            raw_data={"analysis": analysis.__dict__},
            metadata=metadata,
        )

    def _get_rate_sensitivity_adjustment(
        self,
        symbol: str,
        analysis: FedSpeechAnalysis,
    ) -> float:
        """
        Adjust signal based on symbol's rate sensitivity.

        Rate-sensitive sectors are more affected by Fed policy.
        """
        # Rate-sensitive symbols (simplified - in production use sector data)
        rate_sensitive = {
            "XLF", "KRE", "XLU", "XLB",  # Financials, utilities, materials
            "JPM", "BAC", "WFC", "GS",   # Big banks
            "NEE", "DUK", "SO",          # Utilities
        }

        growth_stocks = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        }

        if symbol == self.MARKET_SYMBOL:
            return 0.0

        # Growth stocks are more sensitive to hawkish Fed
        if symbol in growth_stocks:
            if analysis.rate_expectations == "hawkish":
                return -0.1  # Additional negative impact
            elif analysis.rate_expectations == "dovish":
                return 0.1  # Additional positive impact

        # Financials benefit from higher rates
        if symbol in rate_sensitive and "XLF" in symbol or any(
            bank in symbol for bank in ["JPM", "BAC", "WFC", "GS"]
        ):
            if analysis.rate_expectations == "hawkish":
                return 0.1  # Banks benefit from higher rates
            elif analysis.rate_expectations == "dovish":
                return -0.05

        return 0.0

    async def analyze_speech_series(
        self,
        limit: int = 5,
    ) -> List[AlternativeSignal]:
        """
        Analyze a series of recent Fed speeches.

        Useful for understanding Fed communication trends.

        Args:
            limit: Number of speeches to analyze

        Returns:
            List of signals from recent speeches
        """
        try:
            speeches = await self._speech_fetcher.fetch_latest(limit=limit)
            signals = []

            for speech in speeches:
                analysis = await self._analyze_speech(speech)
                if analysis:
                    signal = self._create_signal(self.MARKET_SYMBOL, speech, analysis)
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error analyzing speech series: {e}")
            return []
