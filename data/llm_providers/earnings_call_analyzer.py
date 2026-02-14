"""
Earnings Call Analyzer

LLM-powered analysis of earnings call transcripts to extract trading signals.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from data.alt_data_types import AltDataSource, AlternativeSignal
from data.alternative_data_provider import AlternativeDataProvider
from data.data_fetchers import EarningsTranscript, EarningsTranscriptFetcher
from llm import (
    EarningsAnalysis,
    LLMClientWithFallback,
    create_llm_client,
)
from llm.prompts.earnings_prompts import (
    EARNINGS_SYSTEM_PROMPT,
    format_earnings_prompt,
)

logger = logging.getLogger(__name__)


class EarningsCallAnalyzer(AlternativeDataProvider):
    """
    Analyzes earnings call transcripts using LLM to extract trading signals.

    Extracts:
    - Sentiment and tone from management
    - Guidance changes (raised, maintained, lowered)
    - Key business insights
    - Analyst sentiment from Q&A
    - Forward-looking statements

    Cache TTL: 7 days (transcripts don't change)
    """

    # Cache TTL: 7 days in seconds
    CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

    def __init__(
        self,
        llm_client: Optional[LLMClientWithFallback] = None,
        transcript_fetcher: Optional[EarningsTranscriptFetcher] = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        """
        Initialize earnings call analyzer.

        Args:
            llm_client: LLM client for analysis (creates default if None)
            transcript_fetcher: Fetcher for earnings transcripts
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(
            source=AltDataSource.LLM_EARNINGS,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        self._llm_client = llm_client
        self._transcript_fetcher = transcript_fetcher or EarningsTranscriptFetcher()

    async def initialize(self) -> bool:
        """Initialize the analyzer."""
        try:
            # Create LLM client if not provided
            if self._llm_client is None:
                self._llm_client = create_llm_client()

            # Test LLM connection
            if self._llm_client is None:
                logger.warning("No LLM client available - analyzer will use mock responses")

            self._initialized = True
            logger.info("EarningsCallAnalyzer initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize EarningsCallAnalyzer: {e}")
            return False

    async def fetch_signal(self, symbol: str) -> Optional[AlternativeSignal]:
        """
        Analyze earnings call and generate trading signal.

        Args:
            symbol: Stock ticker symbol

        Returns:
            AlternativeSignal with sentiment and insights
        """
        try:
            # Fetch latest transcript
            transcript = await self._transcript_fetcher.fetch_latest(symbol)
            if not transcript:
                logger.debug(f"No earnings transcript available for {symbol}")
                return None

            # Analyze with LLM
            analysis = await self._analyze_transcript(symbol, transcript)
            if not analysis:
                return None

            # Convert to AlternativeSignal
            signal = self._create_signal(symbol, transcript, analysis)
            return signal

        except Exception as e:
            logger.error(f"Error analyzing earnings for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _analyze_transcript(
        self,
        symbol: str,
        transcript: EarningsTranscript,
    ) -> Optional[EarningsAnalysis]:
        """Analyze transcript using LLM."""
        try:
            if self._llm_client is None:
                # Return mock analysis for testing
                return self._generate_mock_analysis(symbol, transcript)

            # Format prompt
            prompt = format_earnings_prompt(
                symbol=symbol,
                quarter=transcript.fiscal_quarter,
                year=transcript.fiscal_year,
                transcript_text=transcript.content,
                prepared_remarks=transcript.prepared_remarks,
                qa_section=transcript.qanda_section,
            )

            # Call LLM
            response = await self._llm_client.complete(
                prompt=prompt,
                system_prompt=EARNINGS_SYSTEM_PROMPT,
                max_tokens=1000,
                temperature=0.3,  # Low temperature for consistency
            )

            if not response or not response.content:
                return None

            # Parse response
            analysis = self._parse_llm_response(response.content)
            return analysis

        except Exception as e:
            logger.error(f"LLM analysis error for {symbol}: {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[EarningsAnalysis]:
        """Parse LLM JSON response into EarningsAnalysis."""
        try:
            # Try to extract JSON from response
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

            # Map guidance_change string to enum value
            guidance_map = {
                "raised": "raised",
                "maintained": "maintained",
                "lowered": "lowered",
                "not_mentioned": "not_mentioned",
            }
            guidance_change = guidance_map.get(
                data.get("guidance_change", "not_mentioned"),
                "not_mentioned"
            )

            # Map management_tone string
            tone_map = {
                "optimistic": "optimistic",
                "cautious": "cautious",
                "neutral": "neutral",
                "pessimistic": "pessimistic",
            }
            management_tone = tone_map.get(
                data.get("management_tone", "neutral"),
                "neutral"
            )

            return EarningsAnalysis(
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                confidence=float(data.get("confidence", 0.5)),
                key_insights=data.get("key_insights", []),
                risks=data.get("risks", []),
                opportunities=data.get("opportunities", []),
                reasoning=data.get("reasoning", ""),
                guidance_change=guidance_change,
                management_tone=management_tone,
                analyst_sentiment=float(data.get("analyst_sentiment", 0.0)),
                key_metrics_mentioned=data.get("key_metrics_mentioned", []),
                forward_looking_statements=data.get("forward_looking_statements", []),
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

    def _generate_mock_analysis(
        self,
        symbol: str,
        transcript: EarningsTranscript,
    ) -> EarningsAnalysis:
        """Generate mock analysis for testing."""
        # Simple heuristic based on word presence
        content = transcript.content.lower()

        # Count positive vs negative words
        positive_words = ["growth", "increase", "strong", "beat", "exceed", "raising", "optimistic"]
        negative_words = ["decline", "decrease", "weak", "miss", "lower", "concern", "challenging"]

        positive_count = sum(content.count(word) for word in positive_words)
        negative_count = sum(content.count(word) for word in negative_words)

        total = positive_count + negative_count + 1
        sentiment = (positive_count - negative_count) / total

        # Determine guidance change
        if "raising guidance" in content or "increase guidance" in content:
            guidance = "raised"
        elif "lowering guidance" in content or "reduce guidance" in content:
            guidance = "lowered"
        else:
            guidance = "maintained"

        return EarningsAnalysis(
            sentiment_score=max(-1, min(1, sentiment)),
            confidence=0.6,
            key_insights=[
                f"Transcript analysis for {symbol} {transcript.fiscal_quarter} {transcript.fiscal_year}",
            ],
            risks=[],
            opportunities=[],
            reasoning=f"Heuristic analysis based on word frequency. Positive: {positive_count}, Negative: {negative_count}.",
            guidance_change=guidance,
            management_tone="neutral",
            analyst_sentiment=sentiment * 0.8,
            key_metrics_mentioned=[],
            forward_looking_statements=[],
        )

    def _create_signal(
        self,
        symbol: str,
        transcript: EarningsTranscript,
        analysis: EarningsAnalysis,
    ) -> AlternativeSignal:
        """Create AlternativeSignal from analysis."""
        # Adjust signal based on guidance change
        signal_adjustment = 0.0
        if analysis.guidance_change == "raised":
            signal_adjustment = 0.2
        elif analysis.guidance_change == "lowered":
            signal_adjustment = -0.2

        # Calculate final signal
        final_signal = analysis.sentiment_score + signal_adjustment
        final_signal = max(-1.0, min(1.0, final_signal))

        # Build metadata
        metadata: Dict[str, Any] = {
            "fiscal_quarter": transcript.fiscal_quarter,
            "fiscal_year": transcript.fiscal_year,
            "transcript_date": transcript.date.isoformat(),
            "guidance_change": analysis.guidance_change,
            "management_tone": analysis.management_tone,
            "analyst_sentiment": analysis.analyst_sentiment,
            "key_insights": analysis.key_insights[:5],  # Top 5
            "reasoning": analysis.reasoning,
            "word_count": transcript.word_count,
            "source": transcript.source,
        }

        return AlternativeSignal(
            symbol=symbol,
            source=AltDataSource.LLM_EARNINGS,
            timestamp=datetime.now(),
            signal_value=final_signal,
            confidence=analysis.confidence,
            raw_data={"analysis": analysis.__dict__},
            metadata=metadata,
        )

    async def analyze_quarter(
        self,
        symbol: str,
        quarter: str,
        year: int,
    ) -> Optional[AlternativeSignal]:
        """
        Analyze a specific quarter's earnings call.

        Args:
            symbol: Stock ticker symbol
            quarter: Q1, Q2, Q3, or Q4
            year: Fiscal year

        Returns:
            AlternativeSignal for the specified quarter
        """
        try:
            transcript = await self._transcript_fetcher.fetch_by_quarter(
                symbol, quarter, year
            )
            if not transcript:
                return None

            analysis = await self._analyze_transcript(symbol, transcript)
            if not analysis:
                return None

            return self._create_signal(symbol, transcript, analysis)

        except Exception as e:
            logger.error(f"Error analyzing {symbol} {quarter} {year}: {e}")
            return None
