#!/usr/bin/env python3
"""
Unit tests for Multi-Timeframe Analyzer.

Tests cover:
1. Timeframe analysis and trend detection
2. Signal aggregation across timeframes
3. Daily veto power
4. Confidence scoring
5. Edge cases
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.multi_timeframe_analyzer import MultiTimeframeAnalyzer


class MockBar:
    """Mock bar for testing."""

    def __init__(self, close):
        self.close = close


class MockBroker:
    """Mock broker for testing."""

    def __init__(self, bars_data=None):
        self.bars_data = bars_data or {}

    async def get_bars(self, symbol, timeframe, limit=50):
        """Return mock bars for testing."""
        if timeframe in self.bars_data:
            return self.bars_data[timeframe]

        # Default: generate bullish trend
        return [MockBar(100 + i * 0.5) for i in range(limit)]


class TestTimeframeAnalysis:
    """Test individual timeframe analysis."""

    @pytest.mark.asyncio
    async def test_bullish_trend_detection(self):
        """Test detection of bullish trend."""
        # Create bars with clear uptrend (short MA > long MA)
        bars = [MockBar(100 + i * 1.0) for i in range(50)]  # Steady uptrend
        broker = MockBroker({"5Min": bars, "15Min": bars, "1Hour": bars, "1Day": bars})

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer._analyze_timeframe("AAPL", "5Min", bars_needed=50)

        assert result["trend"] == "bullish"
        assert result["strength"] > 0

    @pytest.mark.asyncio
    async def test_bearish_trend_detection(self):
        """Test detection of bearish trend."""
        # Create bars with clear downtrend (short MA < long MA)
        bars = [MockBar(150 - i * 1.0) for i in range(50)]  # Steady downtrend
        broker = MockBroker({"5Min": bars, "15Min": bars, "1Hour": bars, "1Day": bars})

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer._analyze_timeframe("AAPL", "5Min", bars_needed=50)

        assert result["trend"] == "bearish"
        assert result["strength"] > 0

    @pytest.mark.asyncio
    async def test_neutral_trend_detection(self):
        """Test detection of neutral/sideways trend."""
        # Create flat bars (short MA ≈ long MA)
        bars = [MockBar(100.0) for _ in range(50)]  # Flat
        broker = MockBroker({"5Min": bars})

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer._analyze_timeframe("AAPL", "5Min", bars_needed=50)

        assert result["trend"] == "neutral"
        assert result["strength"] < 0.1

    @pytest.mark.asyncio
    async def test_insufficient_bars_raises_error(self):
        """Test that insufficient bars raises error."""
        bars = [MockBar(100) for _ in range(10)]  # Only 10 bars
        broker = MockBroker({"5Min": bars})

        analyzer = MultiTimeframeAnalyzer(broker)

        with pytest.raises(ValueError, match="Insufficient bars"):
            await analyzer._analyze_timeframe("AAPL", "5Min", bars_needed=50)


class TestSignalAggregation:
    """Test multi-timeframe signal aggregation."""

    @pytest.mark.asyncio
    async def test_all_bullish_gives_strong_buy(self):
        """Test that all bullish timeframes give strong buy signal."""
        # All timeframes bullish
        bullish_bars = [MockBar(100 + i * 2.0) for i in range(50)]  # Strong uptrend
        broker = MockBroker(
            {
                "5Min": bullish_bars,
                "15Min": bullish_bars,
                "1Hour": bullish_bars,
                "1Day": bullish_bars,
            }
        )

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL", min_confidence=0.5)

        assert result is not None
        assert result["signal"] == "buy"
        assert result["signal_strength"] in ["buy", "strong_buy"]
        assert result["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_all_bearish_gives_strong_sell(self):
        """Test that all bearish timeframes give strong sell signal."""
        # All timeframes bearish
        bearish_bars = [MockBar(200 - i * 2.0) for i in range(50)]  # Strong downtrend
        broker = MockBroker(
            {
                "5Min": bearish_bars,
                "15Min": bearish_bars,
                "1Hour": bearish_bars,
                "1Day": bearish_bars,
            }
        )

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL", min_confidence=0.5)

        assert result is not None
        assert result["signal"] == "sell"
        assert result["signal_strength"] in ["sell", "strong_sell"]
        assert result["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_mixed_signals_give_neutral(self):
        """Test that mixed signals give neutral result."""
        # Mixed: some bullish, some bearish
        bullish_bars = [MockBar(100 + i * 2.0) for i in range(50)]
        bearish_bars = [MockBar(200 - i * 2.0) for i in range(50)]

        broker = MockBroker(
            {
                "5Min": bullish_bars,
                "15Min": bearish_bars,
                "1Hour": bullish_bars,
                "1Day": bearish_bars,
            }
        )

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL", min_confidence=0.7)

        assert result is not None
        # Mixed signals should reduce confidence
        assert result["confidence"] < 0.7 or result["signal"] == "neutral"

    @pytest.mark.asyncio
    async def test_should_enter_respects_min_confidence(self):
        """Test that should_enter respects minimum confidence."""
        # Weak bullish (flat bars with slight uptrend)
        weak_bullish = [MockBar(100 + i * 0.1) for i in range(50)]
        broker = MockBroker(
            {
                "5Min": weak_bullish,
                "15Min": weak_bullish,
                "1Hour": weak_bullish,
                "1Day": weak_bullish,
            }
        )

        analyzer = MultiTimeframeAnalyzer(broker)

        # With high confidence requirement
        result = await analyzer.analyze("AAPL", min_confidence=0.95)

        # Weak trend shouldn't meet 95% confidence threshold
        assert result is not None
        if result["confidence"] < 0.95:
            assert not result["should_enter"]


class TestDailyVeto:
    """Test daily timeframe veto power."""

    @pytest.mark.asyncio
    async def test_daily_bearish_vetoes_buy_signal(self):
        """Test that bearish daily timeframe vetoes buy signals."""
        bullish_bars = [MockBar(100 + i * 2.0) for i in range(50)]
        bearish_bars = [MockBar(200 - i * 2.0) for i in range(50)]

        broker = MockBroker(
            {
                "5Min": bullish_bars,
                "15Min": bullish_bars,
                "1Hour": bullish_bars,
                "1Day": bearish_bars,  # Daily is bearish!
            }
        )

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL", min_confidence=0.5, require_daily_alignment=True)

        assert result is not None
        assert result["daily_conflicts"]
        assert not result["should_enter"]  # Should be vetoed

    @pytest.mark.asyncio
    async def test_daily_bullish_vetoes_sell_signal(self):
        """Test that bullish daily timeframe vetoes sell signals."""
        bullish_bars = [MockBar(100 + i * 2.0) for i in range(50)]
        bearish_bars = [MockBar(200 - i * 2.0) for i in range(50)]

        broker = MockBroker(
            {
                "5Min": bearish_bars,
                "15Min": bearish_bars,
                "1Hour": bearish_bars,
                "1Day": bullish_bars,  # Daily is bullish!
            }
        )

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL", min_confidence=0.5, require_daily_alignment=True)

        assert result is not None
        assert result["daily_conflicts"]
        assert not result["should_enter"]  # Should be vetoed

    @pytest.mark.asyncio
    async def test_daily_veto_can_be_disabled(self):
        """Test that daily veto can be disabled."""
        bullish_bars = [MockBar(100 + i * 2.0) for i in range(50)]
        bearish_bars = [MockBar(200 - i * 2.0) for i in range(50)]

        broker = MockBroker(
            {
                "5Min": bullish_bars,
                "15Min": bullish_bars,
                "1Hour": bullish_bars,
                "1Day": bearish_bars,  # Daily is bearish
            }
        )

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL", min_confidence=0.5, require_daily_alignment=False)

        assert result is not None
        assert not result["daily_conflicts"]  # Veto disabled


class TestTimeframeWeights:
    """Test timeframe weighting system."""

    def test_weights_sum_to_one(self):
        """Test that timeframe weights sum to 1.0."""
        total = sum(MultiTimeframeAnalyzer.WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_weights_are_valid(self):
        """Test that all weights are between 0 and 1."""
        for tf, weight in MultiTimeframeAnalyzer.WEIGHTS.items():
            assert 0 < weight < 1, f"Weight for {tf} should be between 0 and 1"

    def test_required_timeframes_present(self):
        """Test that all required timeframes have weights."""
        required = ["5Min", "15Min", "1Hour", "1Day"]
        for tf in required:
            assert tf in MultiTimeframeAnalyzer.WEIGHTS


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_broker_exception(self):
        """Test graceful handling of broker exceptions."""
        broker = MagicMock()
        broker.get_bars = AsyncMock(side_effect=Exception("API Error"))

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL")

        # Should return None gracefully, not raise
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_partial_data(self):
        """Test handling when some timeframes fail."""
        bullish_bars = [MockBar(100 + i * 2.0) for i in range(50)]

        # Only 3 timeframes available (minimum required)
        broker = MockBroker(
            {
                "5Min": bullish_bars,
                "15Min": bullish_bars,
                "1Hour": bullish_bars,
                # '1Day' missing
            }
        )

        # Mock get_bars to fail for '1Day'
        async def mock_get_bars(symbol, timeframe, limit=50):
            if timeframe == "1Day":
                raise Exception("Data not available")
            return bullish_bars

        broker.get_bars = mock_get_bars

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL", min_confidence=0.5)

        # Should still work with 3 timeframes
        assert result is not None

    @pytest.mark.asyncio
    async def test_insufficient_timeframes_returns_none(self):
        """Test that insufficient valid timeframes returns None."""
        bullish_bars = [MockBar(100 + i * 2.0) for i in range(50)]

        broker = MagicMock()

        # Only 2 timeframes available (below minimum)
        async def mock_get_bars(symbol, timeframe, limit=50):
            if timeframe in ["5Min", "15Min"]:
                return bullish_bars
            raise Exception("Data not available")

        broker.get_bars = AsyncMock(side_effect=mock_get_bars)

        analyzer = MultiTimeframeAnalyzer(broker)
        result = await analyzer.analyze("AAPL")

        # Should return None with only 2 valid timeframes
        assert result is None

    @pytest.mark.asyncio
    async def test_neutral_analysis_on_no_data(self):
        """Test neutral analysis is returned when no data available."""
        analyzer = MultiTimeframeAnalyzer(MagicMock())
        result = analyzer._neutral_analysis("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0
        assert not result["should_enter"]


class TestSummary:
    """Test summary generation."""

    def test_get_summary_returns_string(self):
        """Test that get_summary returns readable string."""
        analyzer = MultiTimeframeAnalyzer(MagicMock())

        analysis = {
            "symbol": "AAPL",
            "signal": "buy",
            "confidence": 0.85,
            "summary": "Test summary",
        }

        summary = analyzer.get_summary(analysis)
        assert isinstance(summary, str)
        assert "Test summary" in summary

    def test_get_summary_handles_none(self):
        """Test get_summary handles None gracefully."""
        analyzer = MultiTimeframeAnalyzer(MagicMock())
        summary = analyzer.get_summary(None)
        assert "No analysis available" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
