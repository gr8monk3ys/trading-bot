"""
News Theme Extractor

LLM-powered extraction of themes and catalysts from news articles.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from data.alt_data_types import AltDataSource, AlternativeSignal
from data.alternative_data_provider import AlternativeDataProvider
from llm import (
    LLMClientWithFallback,
    create_llm_client,
    NewsThemeAnalysis,
)
from llm.prompts.news_theme_prompts import (
    NEWS_SYSTEM_PROMPT,
    format_news_theme_prompt,
)

logger = logging.getLogger(__name__)


class NewsThemeExtractor(AlternativeDataProvider):
    """
    Extracts themes and trading signals from news articles using LLM.

    Uses existing Alpaca News API data and enhances it with LLM analysis
    to identify catalysts, themes, and time-sensitive opportunities.

    Extracts:
    - Primary and secondary themes
    - Catalysts (product launch, M&A, earnings, etc.)
    - Time sensitivity (immediate, short-term, long-term)
    - Market impact assessment

    Cache TTL: 4 hours (news changes frequently)
    """

    # Cache TTL: 4 hours in seconds
    CACHE_TTL_SECONDS = 4 * 60 * 60

    def __init__(
        self,
        llm_client: Optional[LLMClientWithFallback] = None,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        """
        Initialize news theme extractor.

        Args:
            llm_client: LLM client for analysis
            alpaca_api_key: Alpaca API key for news (or from env)
            alpaca_secret_key: Alpaca secret key (or from env)
            cache_ttl_seconds: Cache TTL in seconds
        """
        import os

        super().__init__(
            source=AltDataSource.LLM_NEWS_THEME,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        self._llm_client = llm_client
        self._alpaca_api_key = alpaca_api_key or os.getenv("ALPACA_API_KEY")
        self._alpaca_secret_key = alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY")

    async def initialize(self) -> bool:
        """Initialize the extractor."""
        try:
            if self._llm_client is None:
                self._llm_client = create_llm_client()

            if self._llm_client is None:
                logger.warning("No LLM client available - extractor will use mock responses")

            self._initialized = True
            logger.info("NewsThemeExtractor initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize NewsThemeExtractor: {e}")
            return False

    async def fetch_signal(self, symbol: str) -> Optional[AlternativeSignal]:
        """
        Extract themes from recent news and generate trading signal.

        Args:
            symbol: Stock ticker symbol

        Returns:
            AlternativeSignal with news themes
        """
        try:
            # Fetch recent news articles
            articles = await self._fetch_news(symbol)
            if not articles:
                logger.debug(f"No news articles available for {symbol}")
                return None

            # Analyze with LLM
            analysis = await self._analyze_news(symbol, articles)
            if not analysis:
                return None

            # Convert to AlternativeSignal
            signal = self._create_signal(symbol, articles, analysis)
            return signal

        except Exception as e:
            logger.error(f"Error extracting news themes for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _fetch_news(
        self,
        symbol: str,
        hours: int = 24,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetch news articles from Alpaca API."""
        import aiohttp

        try:
            if not self._alpaca_api_key or not self._alpaca_secret_key:
                # Return mock news for testing
                return self._generate_mock_news(symbol)

            # Alpaca News API endpoint
            url = "https://data.alpaca.markets/v1beta1/news"

            headers = {
                "APCA-API-KEY-ID": self._alpaca_api_key,
                "APCA-API-SECRET-KEY": self._alpaca_secret_key,
            }

            # Calculate start time
            start_time = datetime.utcnow() - timedelta(hours=hours)

            params = {
                "symbols": symbol,
                "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": limit,
                "sort": "desc",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=params, timeout=30
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Alpaca News API error: {response.status}")
                        return self._generate_mock_news(symbol)

                    data = await response.json()
                    return data.get("news", [])

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return self._generate_mock_news(symbol)

    def _generate_mock_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock news for testing."""
        now = datetime.utcnow()

        return [
            {
                "headline": f"{symbol} Reports Strong Quarterly Earnings",
                "summary": f"{symbol} beat analyst expectations with revenue growth of 15% year-over-year. Management raised full-year guidance citing strong demand.",
                "source": "Mock News",
                "created_at": (now - timedelta(hours=2)).isoformat() + "Z",
            },
            {
                "headline": f"{symbol} Announces New Product Launch",
                "summary": f"{symbol} unveiled its next-generation product line at an industry conference. Analysts expect significant market share gains.",
                "source": "Mock News",
                "created_at": (now - timedelta(hours=6)).isoformat() + "Z",
            },
            {
                "headline": f"Analyst Upgrades {symbol} to Buy",
                "summary": f"Leading Wall Street firm upgraded {symbol} from Hold to Buy, citing improved fundamentals and attractive valuation.",
                "source": "Mock News",
                "created_at": (now - timedelta(hours=12)).isoformat() + "Z",
            },
        ]

    async def _analyze_news(
        self,
        symbol: str,
        articles: List[Dict[str, Any]],
    ) -> Optional[NewsThemeAnalysis]:
        """Analyze news articles using LLM."""
        try:
            if self._llm_client is None:
                return self._generate_mock_analysis(symbol, articles)

            # Format prompt
            prompt = format_news_theme_prompt(
                symbol=symbol,
                articles=articles,
                hours=24,
            )

            # Call LLM
            response = await self._llm_client.complete(
                prompt=prompt,
                system_prompt=NEWS_SYSTEM_PROMPT,
                max_tokens=800,
                temperature=0.3,
            )

            if not response or not response.content:
                return None

            return self._parse_llm_response(response.content)

        except Exception as e:
            logger.error(f"LLM analysis error for {symbol} news: {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[NewsThemeAnalysis]:
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

            # Map time sensitivity
            sensitivity_map = {
                "immediate": "immediate",
                "short_term": "short_term",
                "long_term": "long_term",
            }
            time_sensitivity = sensitivity_map.get(
                data.get("time_sensitivity", "short_term"),
                "short_term"
            )

            # Map market impact
            impact_map = {
                "high": "high",
                "medium": "medium",
                "low": "low",
            }
            market_impact = impact_map.get(
                data.get("market_impact_estimate", "medium"),
                "medium"
            )

            return NewsThemeAnalysis(
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                confidence=float(data.get("confidence", 0.5)),
                key_insights=data.get("key_insights", []),
                risks=data.get("risks", []),
                opportunities=data.get("opportunities", []),
                reasoning=data.get("reasoning", ""),
                primary_theme=data.get("primary_theme", "general"),
                secondary_themes=data.get("secondary_themes", []),
                catalysts=data.get("catalysts_identified", []),
                time_sensitivity=time_sensitivity,
                market_impact=market_impact,
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
        articles: List[Dict[str, Any]],
    ) -> NewsThemeAnalysis:
        """Generate mock analysis for testing."""
        # Simple heuristics based on headlines
        all_text = " ".join(
            f"{a.get('headline', '')} {a.get('summary', '')}"
            for a in articles
        ).lower()

        # Count positive vs negative words
        positive_words = [
            "beat", "growth", "upgrade", "strong", "gain", "launch",
            "raised", "exceeded", "bullish", "opportunity"
        ]
        negative_words = [
            "miss", "decline", "downgrade", "weak", "loss", "cut",
            "lowered", "concern", "bearish", "risk"
        ]

        positive_count = sum(all_text.count(word) for word in positive_words)
        negative_count = sum(all_text.count(word) for word in negative_words)

        total = positive_count + negative_count + 1
        sentiment = (positive_count - negative_count) / total

        # Identify primary theme
        theme = "general"
        if "earnings" in all_text:
            theme = "earnings_reaction"
        elif "launch" in all_text or "product" in all_text:
            theme = "product_launch"
        elif "acquisition" in all_text or "merger" in all_text:
            theme = "acquisition"
        elif "upgrade" in all_text or "downgrade" in all_text:
            theme = "analyst_action"

        # Identify catalysts
        catalysts = []
        if "earnings" in all_text:
            catalysts.append("earnings_release")
        if "guidance" in all_text:
            catalysts.append("guidance_update")
        if "product" in all_text:
            catalysts.append("product_announcement")
        if "analyst" in all_text:
            catalysts.append("analyst_coverage")

        return NewsThemeAnalysis(
            sentiment_score=max(-1, min(1, sentiment)),
            confidence=0.5,
            key_insights=[f"Analyzed {len(articles)} articles for {symbol}"],
            risks=[],
            opportunities=[],
            reasoning=f"Heuristic analysis. Positive: {positive_count}, Negative: {negative_count}",
            primary_theme=theme,
            secondary_themes=[],
            catalysts=catalysts,
            time_sensitivity="short_term",
            market_impact="medium",
        )

    def _create_signal(
        self,
        symbol: str,
        articles: List[Dict[str, Any]],
        analysis: NewsThemeAnalysis,
    ) -> AlternativeSignal:
        """Create AlternativeSignal from analysis."""
        # Adjust signal based on time sensitivity
        signal_multiplier = 1.0
        if analysis.time_sensitivity == "immediate":
            signal_multiplier = 1.2  # Boost immediate signals
        elif analysis.time_sensitivity == "long_term":
            signal_multiplier = 0.8  # Discount long-term signals

        # Adjust based on market impact
        if analysis.market_impact == "high":
            signal_multiplier *= 1.1
        elif analysis.market_impact == "low":
            signal_multiplier *= 0.9

        final_signal = analysis.sentiment_score * signal_multiplier
        final_signal = max(-1.0, min(1.0, final_signal))

        # Build metadata
        metadata: Dict[str, Any] = {
            "article_count": len(articles),
            "primary_theme": analysis.primary_theme,
            "secondary_themes": analysis.secondary_themes,
            "catalysts": analysis.catalysts,
            "time_sensitivity": analysis.time_sensitivity,
            "market_impact": analysis.market_impact,
            "reasoning": analysis.reasoning,
        }

        # Include top article headlines
        if articles:
            metadata["top_headlines"] = [
                a.get("headline", "") for a in articles[:3]
            ]

        return AlternativeSignal(
            symbol=symbol,
            source=AltDataSource.LLM_NEWS_THEME,
            timestamp=datetime.now(),
            signal_value=final_signal,
            confidence=analysis.confidence,
            raw_data={"analysis": analysis.__dict__},
            metadata=metadata,
        )

    async def get_themes_for_symbols(
        self,
        symbols: List[str],
    ) -> Dict[str, NewsThemeAnalysis]:
        """
        Get theme analysis for multiple symbols.

        Useful for identifying cross-stock themes.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dict mapping symbol to NewsThemeAnalysis
        """
        results = {}

        for symbol in symbols:
            try:
                articles = await self._fetch_news(symbol)
                if articles:
                    analysis = await self._analyze_news(symbol, articles)
                    if analysis:
                        results[symbol] = analysis
            except Exception as e:
                logger.error(f"Error getting themes for {symbol}: {e}")

        return results

    async def find_sector_themes(
        self,
        symbols: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Identify common themes across a group of symbols.

        Args:
            symbols: List of stock ticker symbols (typically same sector)

        Returns:
            Dict with common themes and affected symbols
        """
        try:
            analyses = await self.get_themes_for_symbols(symbols)
            if not analyses:
                return None

            # Aggregate themes
            theme_counts: Dict[str, List[str]] = {}
            catalyst_counts: Dict[str, List[str]] = {}

            for symbol, analysis in analyses.items():
                # Count primary theme
                theme = analysis.primary_theme
                if theme not in theme_counts:
                    theme_counts[theme] = []
                theme_counts[theme].append(symbol)

                # Count catalysts
                for catalyst in analysis.catalysts:
                    if catalyst not in catalyst_counts:
                        catalyst_counts[catalyst] = []
                    catalyst_counts[catalyst].append(symbol)

            # Sort by frequency
            sorted_themes = sorted(
                theme_counts.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            sorted_catalysts = sorted(
                catalyst_counts.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )

            return {
                "themes": dict(sorted_themes),
                "catalysts": dict(sorted_catalysts),
                "num_symbols_analyzed": len(analyses),
            }

        except Exception as e:
            logger.error(f"Error finding sector themes: {e}")
            return None
