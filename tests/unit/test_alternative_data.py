"""
Tests for Alternative Data Framework.

These tests verify:
- Alt data types and dataclasses
- AlternativeDataProvider base class
- AltDataAggregator functionality
- Social sentiment providers
- Order flow analyzers
"""

from datetime import datetime

import pytest

from data.alt_data_types import (
    AggregatedSignal,
    AltDataSource,
    AlternativeSignal,
    OrderFlowSignal,
    SignalDirection,
    SignalStrength,
    SocialSentimentSignal,
)
from data.alternative_data_provider import (
    AltDataAggregator,
    AltDataCache,
    AlternativeDataProvider,
)

# ============================================================================
# ALT DATA TYPES TESTS
# ============================================================================


class TestAlternativeSignal:
    """Tests for AlternativeSignal dataclass."""

    def test_basic_creation(self):
        """Test basic signal creation."""
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
        )

        assert signal.symbol == "AAPL"
        assert signal.source == AltDataSource.REDDIT
        assert signal.signal_value == 0.5
        assert signal.confidence == 0.7

    def test_signal_value_clamping(self):
        """Test signal value is clamped to [-1, 1]."""
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=1.5,  # Too high
            confidence=0.7,
        )

        assert signal.signal_value == 1.0

        signal2 = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=-1.5,  # Too low
            confidence=0.7,
        )

        assert signal2.signal_value == -1.0

    def test_confidence_clamping(self):
        """Test confidence is clamped to [0, 1]."""
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=1.5,  # Too high
        )

        assert signal.confidence == 1.0

    def test_direction_derivation(self):
        """Test direction is derived from signal value."""
        bullish = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
        )
        assert bullish.direction == SignalDirection.BULLISH

        bearish = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.7,
        )
        assert bearish.direction == SignalDirection.BEARISH

        neutral = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.05,
            confidence=0.7,
        )
        assert neutral.direction == SignalDirection.NEUTRAL

    def test_strength_derivation(self):
        """Test strength is derived from confidence."""
        very_strong = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.9,
        )
        assert very_strong.strength == SignalStrength.VERY_STRONG

        weak = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.3,
        )
        assert weak.strength == SignalStrength.WEAK

    def test_weighted_signal(self):
        """Test weighted signal calculation."""
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.8,
            confidence=0.5,
        )

        assert signal.weighted_signal == 0.4  # 0.8 * 0.5

    def test_is_actionable(self):
        """Test actionable signal detection."""
        actionable = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.6,
        )
        assert actionable.is_actionable is True

        not_actionable_low_conf = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.2,
        )
        assert not_actionable_low_conf.is_actionable is False

        not_actionable_low_signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.1,
            confidence=0.6,
        )
        assert not_actionable_low_signal.is_actionable is False


class TestSocialSentimentSignal:
    """Tests for SocialSentimentSignal."""

    def test_meme_stock_risk_flag(self):
        """Test meme stock risk is flagged on high mention change."""
        normal = SocialSentimentSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.6,
            mention_change_pct=50,  # Normal
        )
        assert normal.meme_stock_risk is False

        meme = SocialSentimentSignal(
            symbol="GME",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.8,
            confidence=0.6,
            mention_change_pct=250,  # Extreme spike
        )
        assert meme.meme_stock_risk is True


class TestOrderFlowSignal:
    """Tests for OrderFlowSignal."""

    def test_order_flow_signal_creation(self):
        """Test creating order flow signal."""
        signal = OrderFlowSignal(
            symbol="AAPL",
            source=AltDataSource.OPTIONS_FLOW,
            timestamp=datetime.now(),
            signal_value=0.3,
            confidence=0.5,
            call_volume=50000,
            put_volume=30000,
            put_call_ratio=0.6,
            unusual_options_activity=True,
        )

        assert signal.call_volume == 50000
        assert signal.put_volume == 30000
        assert signal.put_call_ratio == 0.6
        assert signal.unusual_options_activity is True


class TestAggregatedSignal:
    """Tests for AggregatedSignal."""

    def test_aggregation_calculation(self):
        """Test composite signal calculation."""
        signals = [
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.REDDIT,
                timestamp=datetime.now(),
                signal_value=0.6,
                confidence=0.8,
            ),
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.OPTIONS_FLOW,
                timestamp=datetime.now(),
                signal_value=0.4,
                confidence=0.6,
            ),
        ]

        aggregated = AggregatedSignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            sources=[AltDataSource.REDDIT, AltDataSource.OPTIONS_FLOW],
            individual_signals=signals,
        )

        # Weighted average: (0.6*0.8 + 0.4*0.6) / (0.8 + 0.6) = 0.72/1.4 ≈ 0.514
        assert 0.5 < aggregated.composite_signal < 0.55
        assert aggregated.direction == SignalDirection.BULLISH

    def test_agreement_ratio(self):
        """Test agreement ratio calculation."""
        signals = [
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.REDDIT,
                timestamp=datetime.now(),
                signal_value=0.5,
                confidence=0.7,
            ),
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.TWITTER,
                timestamp=datetime.now(),
                signal_value=0.3,
                confidence=0.6,
            ),
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.OPTIONS_FLOW,
                timestamp=datetime.now(),
                signal_value=-0.4,
                confidence=0.5,
            ),
        ]

        aggregated = AggregatedSignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            sources=[s.source for s in signals],
            individual_signals=signals,
        )

        # 2 bullish, 1 bearish = 66.7% agreement
        assert 0.6 < aggregated.agreement_ratio < 0.7

    def test_high_conviction(self):
        """Test high conviction detection."""
        # High conviction: high confidence, high agreement, strong signal
        high_conv_signals = [
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.REDDIT,
                timestamp=datetime.now(),
                signal_value=0.7,
                confidence=0.8,
            ),
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.OPTIONS_FLOW,
                timestamp=datetime.now(),
                signal_value=0.6,
                confidence=0.7,
            ),
        ]

        high_conv = AggregatedSignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            sources=[s.source for s in high_conv_signals],
            individual_signals=high_conv_signals,
        )

        assert high_conv.is_high_conviction is True

        # Low conviction: disagreement
        low_conv_signals = [
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.REDDIT,
                timestamp=datetime.now(),
                signal_value=0.5,
                confidence=0.6,
            ),
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.OPTIONS_FLOW,
                timestamp=datetime.now(),
                signal_value=-0.5,
                confidence=0.6,
            ),
        ]

        low_conv = AggregatedSignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            sources=[s.source for s in low_conv_signals],
            individual_signals=low_conv_signals,
        )

        assert low_conv.is_high_conviction is False


# ============================================================================
# ALT DATA CACHE TESTS
# ============================================================================


class TestAltDataCache:
    """Tests for AltDataCache."""

    @pytest.fixture
    def cache(self):
        """Create a cache instance."""
        return AltDataCache(default_ttl_seconds=60)

    def test_cache_set_get(self, cache):
        """Test basic set and get."""
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
        )

        cache.set(AltDataSource.REDDIT, "AAPL", signal)
        retrieved = cache.get(AltDataSource.REDDIT, "AAPL")

        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.signal_value == 0.5

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get(AltDataSource.REDDIT, "AAPL")
        assert result is None

    def test_cache_expiry(self, cache):
        """Test cache entries expire."""
        import time

        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
        )

        # Set with very short TTL (1 second)
        cache.set(AltDataSource.REDDIT, "AAPL", signal, ttl_seconds=1)

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        result = cache.get(AltDataSource.REDDIT, "AAPL")
        assert result is None

    def test_cache_invalidate_single(self, cache):
        """Test invalidating a single entry."""
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
        )

        cache.set(AltDataSource.REDDIT, "AAPL", signal)
        cache.invalidate(AltDataSource.REDDIT, "AAPL")

        result = cache.get(AltDataSource.REDDIT, "AAPL")
        assert result is None

    def test_cache_hit_rate(self, cache):
        """Test hit rate calculation."""
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
        )

        cache.set(AltDataSource.REDDIT, "AAPL", signal)

        # 1 hit
        cache.get(AltDataSource.REDDIT, "AAPL")
        # 1 miss
        cache.get(AltDataSource.REDDIT, "MSFT")

        assert cache.hit_rate == 0.5


# ============================================================================
# ALT DATA AGGREGATOR TESTS
# ============================================================================


class TestAltDataAggregator:
    """Tests for AltDataAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create an aggregator instance."""
        return AltDataAggregator()

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""

        class MockProvider(AlternativeDataProvider):
            def __init__(self):
                super().__init__(source=AltDataSource.REDDIT)

            async def initialize(self) -> bool:
                self._initialized = True
                return True

            async def fetch_signal(self, symbol: str):
                return AlternativeSignal(
                    symbol=symbol,
                    source=AltDataSource.REDDIT,
                    timestamp=datetime.now(),
                    signal_value=0.5,
                    confidence=0.7,
                )

        return MockProvider()

    @pytest.mark.asyncio
    async def test_register_provider(self, aggregator, mock_provider):
        """Test registering a provider."""
        aggregator.register_provider(mock_provider)

        assert AltDataSource.REDDIT in aggregator._providers

    @pytest.mark.asyncio
    async def test_get_signals(self, aggregator, mock_provider):
        """Test getting signals from aggregator."""
        aggregator.register_provider(mock_provider)

        signals = await aggregator.get_signals(["AAPL", "MSFT"])

        assert "AAPL" in signals
        assert "MSFT" in signals
        assert signals["AAPL"].composite_signal == 0.5

    @pytest.mark.asyncio
    async def test_kill_switch(self, aggregator, mock_provider):
        """Test killing an underperforming source."""
        aggregator.register_provider(mock_provider)

        # Record poor performance
        for _ in range(60):
            aggregator.record_performance(AltDataSource.REDDIT, -0.05, 0.5)

        # Source should be killed (accuracy < 40%)
        assert AltDataSource.REDDIT in aggregator._killed_sources

        # Signals should not include killed source
        signals = await aggregator.get_signals(["AAPL"])
        assert "AAPL" not in signals  # No active providers

    @pytest.mark.asyncio
    async def test_revive_source(self, aggregator, mock_provider):
        """Test reviving a killed source."""
        aggregator.register_provider(mock_provider)
        aggregator.kill_source(AltDataSource.REDDIT, "Test kill")

        assert AltDataSource.REDDIT in aggregator._killed_sources

        aggregator.revive_source(AltDataSource.REDDIT)

        assert AltDataSource.REDDIT not in aggregator._killed_sources

    def test_update_source_weight(self, aggregator):
        """Test updating source weights."""
        aggregator.update_source_weight(AltDataSource.REDDIT, 0.25)

        assert aggregator._source_weights[AltDataSource.REDDIT] == 0.25

        # Test clamping
        aggregator.update_source_weight(AltDataSource.REDDIT, 1.5)
        assert aggregator._source_weights[AltDataSource.REDDIT] == 1.0


# ============================================================================
# SOCIAL SENTIMENT PROVIDER TESTS
# ============================================================================


class TestSocialSentimentProviders:
    """Tests for social sentiment providers."""

    @pytest.mark.asyncio
    async def test_reddit_provider_initialization(self):
        """Test Reddit provider initializes without credentials."""
        from data.social_sentiment_advanced import RedditSentimentProvider

        provider = RedditSentimentProvider()
        result = await provider.initialize()

        # Should succeed even without credentials (uses mock data)
        assert result is True
        assert provider._initialized is True

    @pytest.mark.asyncio
    async def test_reddit_provider_fetch_signal(self):
        """Test Reddit provider returns signals."""
        from data.social_sentiment_advanced import RedditSentimentProvider

        provider = RedditSentimentProvider()
        await provider.initialize()

        signal = await provider.fetch_signal("AAPL")

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.source == AltDataSource.REDDIT
        assert -1 <= signal.signal_value <= 1
        assert 0 <= signal.confidence <= 1

    @pytest.mark.asyncio
    async def test_ticker_extractor(self):
        """Test ticker extraction from text."""
        from data.social_sentiment_advanced import TickerExtractor

        extractor = TickerExtractor(valid_tickers={"AAPL", "TSLA", "GME"})

        # Test cashtag extraction
        text1 = "I'm bullish on $AAPL and $TSLA"
        tickers1 = extractor.extract(text1)
        assert "AAPL" in tickers1
        assert "TSLA" in tickers1

        # Test filtering out blacklisted words
        text2 = "CEO of AAPL said IPO was a success"
        tickers2 = extractor.extract(text2)
        assert "AAPL" in tickers2
        assert "CEO" not in tickers2
        assert "IPO" not in tickers2


# ============================================================================
# ORDER FLOW ANALYZER TESTS
# ============================================================================


class TestOrderFlowAnalyzer:
    """Tests for OrderFlowAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyzer_initialization(self):
        """Test order flow analyzer initializes."""
        from data.order_flow_analyzer import OrderFlowAnalyzer

        analyzer = OrderFlowAnalyzer()
        result = await analyzer.initialize()

        assert result is True
        assert analyzer._initialized is True

    @pytest.mark.asyncio
    async def test_analyzer_fetch_signal(self):
        """Test order flow analyzer returns signals."""
        from data.order_flow_analyzer import OrderFlowAnalyzer

        analyzer = OrderFlowAnalyzer()
        await analyzer.initialize()

        signal = await analyzer.fetch_signal("AAPL")

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.source == AltDataSource.OPTIONS_FLOW
        assert signal.call_volume >= 0
        assert signal.put_volume >= 0
        assert signal.put_call_ratio > 0

    @pytest.mark.asyncio
    async def test_dark_pool_provider(self):
        """Test dark pool provider."""
        from data.order_flow_analyzer import DarkPoolProvider

        provider = DarkPoolProvider()
        await provider.initialize()

        signal = await provider.fetch_signal("AAPL")

        assert signal is not None
        assert signal.source == AltDataSource.DARK_POOL


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestAltDataIntegration:
    """Integration tests for alternative data framework."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test full pipeline from providers to aggregated signal."""
        from data.order_flow_analyzer import OrderFlowAnalyzer
        from data.social_sentiment_advanced import RedditSentimentProvider

        # Create aggregator with multiple providers
        aggregator = AltDataAggregator()
        aggregator.register_provider(RedditSentimentProvider())
        aggregator.register_provider(OrderFlowAnalyzer())

        # Initialize all
        results = await aggregator.initialize_all()
        assert all(results.values())

        # Get signals
        signals = await aggregator.get_signals(["AAPL", "TSLA"])

        # Should have aggregated signals
        assert "AAPL" in signals
        assert len(signals["AAPL"].individual_signals) >= 1

    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test that caching prevents duplicate fetches."""
        from data.social_sentiment_advanced import RedditSentimentProvider

        aggregator = AltDataAggregator(cache_ttl_seconds=300)
        provider = RedditSentimentProvider()
        aggregator.register_provider(provider)

        # First fetch
        await aggregator.get_signals(["AAPL"])

        # Second fetch should use cache
        await aggregator.get_signals(["AAPL"])

        stats = aggregator.get_cache_stats()
        assert stats["hits"] >= 1  # Should have cache hit

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling when provider fails."""

        class FailingProvider(AlternativeDataProvider):
            def __init__(self):
                super().__init__(source=AltDataSource.TWITTER)

            async def initialize(self) -> bool:
                self._initialized = True
                return True

            async def fetch_signal(self, symbol: str):
                raise Exception("API error")

        aggregator = AltDataAggregator()
        aggregator.register_provider(FailingProvider())

        # Should not raise, just return empty
        signals = await aggregator.get_signals(["AAPL"])

        # May be empty due to error
        assert isinstance(signals, dict)
