"""
SEC Filing Analyzer

LLM-powered analysis of SEC filings to extract trading signals.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from data.alt_data_types import AltDataSource, AlternativeSignal
from data.alternative_data_provider import AlternativeDataProvider
from data.data_fetchers import SECFiling, SECEdgarFetcher
from llm import (
    LLMClientWithFallback,
    create_llm_client,
    SECFilingAnalysis,
)
from llm.prompts.sec_filing_prompts import (
    SEC_SYSTEM_PROMPT,
    format_sec_filing_prompt,
)

logger = logging.getLogger(__name__)


class SECFilingAnalyzer(AlternativeDataProvider):
    """
    Analyzes SEC filings (10-K, 10-Q, 8-K) using LLM to extract trading signals.

    Extracts:
    - New or changed risk factors
    - Material changes in business
    - Litigation and legal matters
    - Going concern warnings
    - Management discussion tone

    Cache TTL: 30 days (filings don't change)
    """

    # Cache TTL: 30 days in seconds
    CACHE_TTL_SECONDS = 30 * 24 * 60 * 60

    # Filing types to analyze
    FILING_TYPES = ["10-K", "10-Q", "8-K"]

    def __init__(
        self,
        llm_client: Optional[LLMClientWithFallback] = None,
        filing_fetcher: Optional[SECEdgarFetcher] = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        """
        Initialize SEC filing analyzer.

        Args:
            llm_client: LLM client for analysis
            filing_fetcher: Fetcher for SEC filings
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(
            source=AltDataSource.LLM_SEC_FILING,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        self._llm_client = llm_client
        self._filing_fetcher = filing_fetcher or SECEdgarFetcher()

    async def initialize(self) -> bool:
        """Initialize the analyzer."""
        try:
            if self._llm_client is None:
                self._llm_client = create_llm_client()

            if self._llm_client is None:
                logger.warning("No LLM client available - analyzer will use mock responses")

            self._initialized = True
            logger.info("SECFilingAnalyzer initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SECFilingAnalyzer: {e}")
            return False

    async def fetch_signal(self, symbol: str) -> Optional[AlternativeSignal]:
        """
        Analyze most recent SEC filing and generate trading signal.

        Prioritizes 8-K (material events), then 10-Q/10-K.

        Args:
            symbol: Stock ticker symbol

        Returns:
            AlternativeSignal with filing insights
        """
        try:
            # Try to get recent 8-K first (most timely)
            filing = await self._filing_fetcher.fetch_latest(symbol, "8-K")

            # If no recent 8-K, get 10-Q or 10-K
            if not filing:
                filing = await self._filing_fetcher.fetch_latest(symbol, "10-Q")

            if not filing:
                filing = await self._filing_fetcher.fetch_latest(symbol, "10-K")

            if not filing:
                logger.debug(f"No SEC filings available for {symbol}")
                return None

            # Analyze with LLM
            analysis = await self._analyze_filing(symbol, filing)
            if not analysis:
                return None

            # Convert to AlternativeSignal
            signal = self._create_signal(symbol, filing, analysis)
            return signal

        except Exception as e:
            logger.error(f"Error analyzing SEC filings for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _analyze_filing(
        self,
        symbol: str,
        filing: SECFiling,
    ) -> Optional[SECFilingAnalysis]:
        """Analyze filing using LLM."""
        try:
            if self._llm_client is None:
                return self._generate_mock_analysis(symbol, filing)

            # Choose section to analyze based on filing type
            if filing.filing_type in ("10-K", "10-Q"):
                # Prefer risk factors, then MD&A
                content_to_analyze = filing.risk_factors or filing.mda_section or filing.content
                section = "Risk Factors" if filing.risk_factors else "MD&A"
            else:  # 8-K
                content_to_analyze = filing.content
                section = "Material Events"

            # Format prompt
            prompt = format_sec_filing_prompt(
                symbol=symbol,
                filing_type=filing.filing_type,
                filing_date=filing.filing_date.strftime("%Y-%m-%d"),
                filing_content=content_to_analyze,
                section=section,
            )

            # Call LLM
            response = await self._llm_client.complete(
                prompt=prompt,
                system_prompt=SEC_SYSTEM_PROMPT,
                max_tokens=1200,
                temperature=0.3,
            )

            if not response or not response.content:
                return None

            return self._parse_llm_response(response.content)

        except Exception as e:
            logger.error(f"LLM analysis error for {symbol} {filing.filing_type}: {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[SECFilingAnalysis]:
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

            return SECFilingAnalysis(
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                confidence=float(data.get("confidence", 0.5)),
                key_insights=data.get("key_insights", []),
                risks=data.get("risks", []),
                opportunities=data.get("opportunities", []),
                reasoning=data.get("reasoning", ""),
                material_changes=data.get("material_changes", []),
                new_risk_factors=data.get("new_risk_factors", []),
                removed_risk_factors=data.get("removed_risk_factors", []),
                litigation_mentions=data.get("litigation_mentions", []),
                going_concern_risk=bool(data.get("going_concern_risk", False)),
                related_party_concerns=data.get("related_party_concerns", []),
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
        filing: SECFiling,
    ) -> SECFilingAnalysis:
        """Generate mock analysis for testing."""
        content = filing.content.lower()

        # Simple heuristics
        negative_terms = [
            "risk", "litigation", "lawsuit", "concern", "uncertainty",
            "decline", "adverse", "material weakness", "going concern"
        ]
        positive_terms = [
            "growth", "improvement", "opportunity", "expansion",
            "innovation", "strong", "increased"
        ]

        negative_count = sum(content.count(term) for term in negative_terms)
        positive_count = sum(content.count(term) for term in positive_terms)

        total = positive_count + negative_count + 1
        sentiment = (positive_count - negative_count) / total

        # Check for going concern
        going_concern = "going concern" in content

        # Identify material changes for 8-K
        material_changes = []
        if filing.filing_type == "8-K":
            for item in filing.items_reported:
                material_changes.append(f"Item {item} reported")

        return SECFilingAnalysis(
            sentiment_score=max(-1, min(1, sentiment)),
            confidence=0.5,
            key_insights=[f"{filing.filing_type} analysis for {symbol}"],
            risks=[],
            opportunities=[],
            reasoning=f"Heuristic analysis. Positive terms: {positive_count}, Negative terms: {negative_count}",
            material_changes=material_changes,
            new_risk_factors=[],
            removed_risk_factors=[],
            litigation_mentions=[],
            going_concern_risk=going_concern,
            related_party_concerns=[],
        )

    def _create_signal(
        self,
        symbol: str,
        filing: SECFiling,
        analysis: SECFilingAnalysis,
    ) -> AlternativeSignal:
        """Create AlternativeSignal from analysis."""
        # Apply going concern penalty
        signal_adjustment = 0.0
        if analysis.going_concern_risk:
            signal_adjustment = -0.3  # Significant negative signal

        # Apply material changes adjustment for 8-K
        if filing.filing_type == "8-K":
            # Material negative events
            negative_items = {"1.02", "1.03", "2.05", "2.06", "4.02"}
            if any(item in negative_items for item in filing.items_reported):
                signal_adjustment -= 0.15

            # Potentially positive events (earnings, acquisition)
            positive_items = {"2.02", "5.02", "8.01"}
            if any(item in positive_items for item in filing.items_reported):
                # Analyze sentiment for context
                pass

        final_signal = analysis.sentiment_score + signal_adjustment
        final_signal = max(-1.0, min(1.0, final_signal))

        # Build metadata
        metadata: Dict[str, Any] = {
            "filing_type": filing.filing_type,
            "filing_date": filing.filing_date.isoformat(),
            "accession_number": filing.accession_number,
            "going_concern_risk": analysis.going_concern_risk,
            "material_changes": analysis.material_changes[:5],
            "new_risk_factors": analysis.new_risk_factors[:5],
            "litigation_mentions": analysis.litigation_mentions[:3],
            "reasoning": analysis.reasoning,
            "word_count": filing.word_count,
        }

        if filing.filing_type == "8-K":
            metadata["items_reported"] = filing.items_reported

        return AlternativeSignal(
            symbol=symbol,
            source=AltDataSource.LLM_SEC_FILING,
            timestamp=datetime.now(),
            signal_value=final_signal,
            confidence=analysis.confidence,
            raw_data={"analysis": analysis.__dict__},
            metadata=metadata,
        )

    async def analyze_filing_type(
        self,
        symbol: str,
        filing_type: str,
    ) -> Optional[AlternativeSignal]:
        """
        Analyze a specific filing type.

        Args:
            symbol: Stock ticker symbol
            filing_type: 10-K, 10-Q, or 8-K

        Returns:
            AlternativeSignal for the specified filing type
        """
        try:
            filing = await self._filing_fetcher.fetch_latest(symbol, filing_type)
            if not filing:
                return None

            analysis = await self._analyze_filing(symbol, filing)
            if not analysis:
                return None

            return self._create_signal(symbol, filing, analysis)

        except Exception as e:
            logger.error(f"Error analyzing {symbol} {filing_type}: {e}")
            return None

    async def get_material_events(
        self,
        symbol: str,
        days_back: int = 30,
    ) -> Optional[AlternativeSignal]:
        """
        Get signal from recent material events (8-K filings).

        Args:
            symbol: Stock ticker symbol
            days_back: Days to look back

        Returns:
            Aggregated signal from material events
        """
        try:
            filings = await self._filing_fetcher.fetch_8k_material_events(
                symbol, days_back
            )

            if not filings:
                return None

            # Analyze all and aggregate
            signals = []
            for filing in filings:
                analysis = await self._analyze_filing(symbol, filing)
                if analysis:
                    signals.append((filing, analysis))

            if not signals:
                return None

            # Aggregate signals (weighted by recency)
            total_weight = 0.0
            weighted_sentiment = 0.0
            all_changes = []
            all_risks = []

            for i, (filing, analysis) in enumerate(signals):
                # More recent filings get higher weight
                weight = 1.0 / (i + 1)
                weighted_sentiment += analysis.sentiment_score * weight
                total_weight += weight
                all_changes.extend(analysis.material_changes)
                all_risks.extend(analysis.new_risk_factors)

            avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
            avg_confidence = sum(a.confidence for _, a in signals) / len(signals)

            # Use most recent filing for metadata
            latest_filing, latest_analysis = signals[0]

            metadata: Dict[str, Any] = {
                "filing_type": "8-K (aggregated)",
                "num_filings": len(signals),
                "days_back": days_back,
                "all_material_changes": all_changes[:10],
                "all_risk_factors": all_risks[:10],
            }

            return AlternativeSignal(
                symbol=symbol,
                source=AltDataSource.LLM_SEC_FILING,
                timestamp=datetime.now(),
                signal_value=max(-1.0, min(1.0, avg_sentiment)),
                confidence=avg_confidence,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error getting material events for {symbol}: {e}")
            return None
