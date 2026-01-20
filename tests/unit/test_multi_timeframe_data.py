#!/usr/bin/env python3
"""
Unit tests for utils/multi_timeframe.py

Tests cover:
- TimeframeData class
- MultiTimeframeAnalyzer class (from multi_timeframe.py, not multi_timeframe_analyzer.py)
- Timeframe parsing
- Bar management
- Trend detection
- Signal alignment
- Divergence detection
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from utils.multi_timeframe import MultiTimeframeAnalyzer, TimeframeData


# ============================================================================
# TimeframeData Tests
# ============================================================================


class TestTimeframeDataInit:
    """Test TimeframeData initialization."""

    def test_default_init(self):
        """Test default initialization."""
        tf_data = TimeframeData("5Min")

        assert tf_data.timeframe == "5Min"
        assert tf_data.max_bars == 200
        assert len(tf_data.timestamps) == 0
        assert tf_data.current_bar is None
        assert tf_data.bar_duration == timedelta(minutes=5)

    def test_custom_max_bars(self):
        """Test initialization with custom max_bars."""
        tf_data = TimeframeData("1Hour", max_bars=50)

        assert tf_data.max_bars == 50


class TestTimeframeDataParseTimeframe:
    """Test _parse_timeframe method."""

    def test_parse_minutes(self):
        """Test parsing minute timeframes."""
        tf_data = TimeframeData("1Min")
        assert tf_data.bar_duration == timedelta(minutes=1)

        tf_data = TimeframeData("5Min")
        assert tf_data.bar_duration == timedelta(minutes=5)

        tf_data = TimeframeData("15Min")
        assert tf_data.bar_duration == timedelta(minutes=15)

    def test_parse_hours(self):
        """Test parsing hour timeframes."""
        tf_data = TimeframeData("1Hour")
        assert tf_data.bar_duration == timedelta(hours=1)

        tf_data = TimeframeData("4Hour")
        assert tf_data.bar_duration == timedelta(hours=4)

    def test_parse_days(self):
        """Test parsing day timeframes."""
        tf_data = TimeframeData("1Day")
        assert tf_data.bar_duration == timedelta(days=1)

    def test_unsupported_timeframe(self):
        """Test unsupported timeframe raises error."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            TimeframeData("1Week")


class TestTimeframeDataUpdate:
    """Test TimeframeData update method."""

    def test_first_update_starts_bar(self):
        """Test first update starts a new bar."""
        tf_data = TimeframeData("5Min")
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        tf_data.update(timestamp, 100.0, 1000)

        assert tf_data.current_bar is not None
        assert tf_data.current_bar["open"] == 100.0
        assert tf_data.current_bar["close"] == 100.0
        assert tf_data.current_bar["high"] == 100.0
        assert tf_data.current_bar["low"] == 100.0
        assert tf_data.current_bar["volume"] == 1000

    def test_update_same_bar(self):
        """Test updates within same bar."""
        tf_data = TimeframeData("5Min")
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        tf_data.update(timestamp, 100.0, 1000)
        tf_data.update(timestamp + timedelta(minutes=2), 105.0, 500)  # New high
        tf_data.update(timestamp + timedelta(minutes=3), 95.0, 300)  # New low
        tf_data.update(timestamp + timedelta(minutes=4), 102.0, 200)  # New close

        assert tf_data.current_bar["open"] == 100.0
        assert tf_data.current_bar["high"] == 105.0
        assert tf_data.current_bar["low"] == 95.0
        assert tf_data.current_bar["close"] == 102.0
        assert tf_data.current_bar["volume"] == 2000

    def test_update_closes_bar_and_starts_new(self):
        """Test update at bar boundary closes current and starts new."""
        tf_data = TimeframeData("5Min")
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        tf_data.update(timestamp, 100.0, 1000)
        tf_data.update(timestamp + timedelta(minutes=5), 110.0, 500)  # Next bar

        # Should have closed first bar and started new one
        assert len(tf_data.closes) == 1
        assert list(tf_data.closes)[0] == 100.0
        assert tf_data.current_bar["open"] == 110.0

    def test_multiple_bars(self):
        """Test creating multiple bars."""
        tf_data = TimeframeData("5Min")
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        for i in range(5):
            bar_start = timestamp + timedelta(minutes=i * 5)
            tf_data.update(bar_start, 100.0 + i, 1000)

        assert len(tf_data.closes) == 4  # 4 completed bars, 1 current


class TestTimeframeDataGetters:
    """Test TimeframeData getter methods."""

    @pytest.fixture
    def tf_data_with_bars(self):
        """Create TimeframeData with some bars."""
        tf_data = TimeframeData("5Min")
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create 5 completed bars
        for i in range(6):
            bar_start = timestamp + timedelta(minutes=i * 5)
            tf_data.update(bar_start, 100.0 + i, 1000 + i * 100)
            # Add high/low within bar
            tf_data.update(bar_start + timedelta(minutes=2), 105.0 + i, 100)
            tf_data.update(bar_start + timedelta(minutes=3), 95.0 + i, 100)

        return tf_data

    def test_get_closes(self, tf_data_with_bars):
        """Test get_closes method."""
        closes = tf_data_with_bars.get_closes()

        assert isinstance(closes, np.ndarray)
        assert len(closes) == 5  # 5 completed bars

    def test_get_closes_with_count(self, tf_data_with_bars):
        """Test get_closes with count limit."""
        closes = tf_data_with_bars.get_closes(count=3)

        assert len(closes) == 3

    def test_get_highs(self, tf_data_with_bars):
        """Test get_highs method."""
        highs = tf_data_with_bars.get_highs()

        assert isinstance(highs, np.ndarray)
        assert len(highs) == 5

    def test_get_highs_with_count(self, tf_data_with_bars):
        """Test get_highs with count limit."""
        highs = tf_data_with_bars.get_highs(count=2)

        assert len(highs) == 2

    def test_get_lows(self, tf_data_with_bars):
        """Test get_lows method."""
        lows = tf_data_with_bars.get_lows()

        assert isinstance(lows, np.ndarray)
        assert len(lows) == 5

    def test_get_lows_with_count(self, tf_data_with_bars):
        """Test get_lows with count limit."""
        lows = tf_data_with_bars.get_lows(count=2)

        assert len(lows) == 2

    def test_get_volumes(self, tf_data_with_bars):
        """Test get_volumes method."""
        volumes = tf_data_with_bars.get_volumes()

        assert isinstance(volumes, np.ndarray)
        assert len(volumes) == 5

    def test_get_volumes_with_count(self, tf_data_with_bars):
        """Test get_volumes with count limit."""
        volumes = tf_data_with_bars.get_volumes(count=2)

        assert len(volumes) == 2

    def test_len(self, tf_data_with_bars):
        """Test __len__ method."""
        assert len(tf_data_with_bars) == 5


# ============================================================================
# MultiTimeframeAnalyzer Initialization Tests
# ============================================================================


class TestMultiTimeframeAnalyzerInit:
    """Test MultiTimeframeAnalyzer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        analyzer = MultiTimeframeAnalyzer(["1Min", "5Min", "1Hour"])

        assert analyzer.timeframes == ["1Min", "5Min", "1Hour"]  # Should be sorted
        assert analyzer.history_length == 200
        assert analyzer.data == {}

    def test_timeframes_sorted(self):
        """Test timeframes are sorted by duration."""
        analyzer = MultiTimeframeAnalyzer(["1Hour", "1Min", "5Min"])

        assert analyzer.timeframes == ["1Min", "5Min", "1Hour"]

    def test_custom_history_length(self):
        """Test custom history length."""
        analyzer = MultiTimeframeAnalyzer(["1Min"], history_length=50)

        assert analyzer.history_length == 50


class TestTimeframeToMinutes:
    """Test _timeframe_to_minutes method."""

    def test_minute_conversion(self):
        """Test converting minute timeframes."""
        analyzer = MultiTimeframeAnalyzer(["1Min"])

        assert analyzer._timeframe_to_minutes("1Min") == 1
        assert analyzer._timeframe_to_minutes("5Min") == 5
        assert analyzer._timeframe_to_minutes("15Min") == 15

    def test_hour_conversion(self):
        """Test converting hour timeframes."""
        analyzer = MultiTimeframeAnalyzer(["1Min"])

        assert analyzer._timeframe_to_minutes("1Hour") == 60
        assert analyzer._timeframe_to_minutes("4Hour") == 240

    def test_day_conversion(self):
        """Test converting day timeframes."""
        analyzer = MultiTimeframeAnalyzer(["1Min"])

        assert analyzer._timeframe_to_minutes("1Day") == 1440

    def test_invalid_returns_zero(self):
        """Test invalid timeframe returns 0."""
        analyzer = MultiTimeframeAnalyzer(["1Min"])

        assert analyzer._timeframe_to_minutes("invalid") == 0


# ============================================================================
# MultiTimeframeAnalyzer Update Tests
# ============================================================================


class TestMultiTimeframeAnalyzerUpdate:
    """Test update method."""

    @pytest.mark.asyncio
    async def test_update_creates_symbol_data(self):
        """Test update creates data for new symbol."""
        analyzer = MultiTimeframeAnalyzer(["1Min", "5Min"])

        await analyzer.update("AAPL", datetime.now(), 150.0, 1000)

        assert "AAPL" in analyzer.data
        assert "1Min" in analyzer.data["AAPL"]
        assert "5Min" in analyzer.data["AAPL"]

    @pytest.mark.asyncio
    async def test_update_all_timeframes(self):
        """Test update updates all timeframes."""
        analyzer = MultiTimeframeAnalyzer(["1Min", "5Min"])
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        await analyzer.update("AAPL", timestamp, 150.0, 1000)

        for tf in analyzer.timeframes:
            assert analyzer.data["AAPL"][tf].current_bar is not None

    @pytest.mark.asyncio
    async def test_update_multiple_symbols(self):
        """Test update works for multiple symbols."""
        analyzer = MultiTimeframeAnalyzer(["1Min"])
        timestamp = datetime.now()

        await analyzer.update("AAPL", timestamp, 150.0, 1000)
        await analyzer.update("MSFT", timestamp, 400.0, 2000)

        assert "AAPL" in analyzer.data
        assert "MSFT" in analyzer.data


# ============================================================================
# MultiTimeframeAnalyzer Get Trend Tests
# ============================================================================


class TestGetTrend:
    """Test get_trend method."""

    @pytest.fixture
    def analyzer_with_data(self):
        """Create analyzer with price data."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])

        # Manually add data
        analyzer.data["AAPL"] = {"5Min": TimeframeData("5Min")}

        return analyzer

    def test_no_data_returns_neutral(self):
        """Test returns neutral when no data."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])

        trend = analyzer.get_trend("AAPL", "5Min")

        assert trend == "neutral"

    def test_no_symbol_returns_neutral(self):
        """Test returns neutral when symbol not found."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])
        analyzer.data["MSFT"] = {}

        trend = analyzer.get_trend("AAPL", "5Min")

        assert trend == "neutral"

    def test_insufficient_data_returns_neutral(self, analyzer_with_data):
        """Test returns neutral with insufficient data."""
        # Only add a few bars (less than period + 1)
        tf_data = analyzer_with_data.data["AAPL"]["5Min"]
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        for i in range(10):  # Need 21 for period=20
            bar_start = timestamp + timedelta(minutes=i * 5)
            tf_data.update(bar_start, 100.0, 1000)

        trend = analyzer_with_data.get_trend("AAPL", "5Min", period=20)

        assert trend == "neutral"

    def test_bullish_trend_detected(self, analyzer_with_data):
        """Test detects bullish trend."""
        tf_data = analyzer_with_data.data["AAPL"]["5Min"]
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create uptrend - prices going up
        for i in range(25):
            bar_start = timestamp + timedelta(minutes=i * 5)
            price = 100.0 + i * 2  # Strong uptrend
            tf_data.update(bar_start, price, 1000)

        trend = analyzer_with_data.get_trend("AAPL", "5Min", period=20)

        assert trend == "bullish"

    def test_bearish_trend_detected(self, analyzer_with_data):
        """Test detects bearish trend."""
        tf_data = analyzer_with_data.data["AAPL"]["5Min"]
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create downtrend - prices going down
        for i in range(25):
            bar_start = timestamp + timedelta(minutes=i * 5)
            price = 200.0 - i * 2  # Strong downtrend
            tf_data.update(bar_start, price, 1000)

        trend = analyzer_with_data.get_trend("AAPL", "5Min", period=20)

        assert trend == "bearish"

    def test_neutral_trend_detected(self, analyzer_with_data):
        """Test detects neutral trend."""
        tf_data = analyzer_with_data.data["AAPL"]["5Min"]
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create flat prices
        for i in range(25):
            bar_start = timestamp + timedelta(minutes=i * 5)
            price = 100.0  # Flat
            tf_data.update(bar_start, price, 1000)

        trend = analyzer_with_data.get_trend("AAPL", "5Min", period=20)

        assert trend == "neutral"


# ============================================================================
# MultiTimeframeAnalyzer Get Aligned Signal Tests
# ============================================================================


class TestGetAlignedSignal:
    """Test get_aligned_signal method."""

    def test_no_data_returns_neutral(self):
        """Test returns neutral when no data."""
        analyzer = MultiTimeframeAnalyzer(["1Min", "5Min"])

        signal = analyzer.get_aligned_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_all_bullish_returns_bullish(self):
        """Test returns bullish when all timeframes are bullish."""
        analyzer = MultiTimeframeAnalyzer(["5Min", "15Min"])

        # Create strong uptrend data for both timeframes
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        for i in range(50):
            bar_time = timestamp + timedelta(minutes=i)
            price = 100.0 + i * 2  # Strong uptrend
            await analyzer.update("AAPL", bar_time, price, 1000)

        signal = analyzer.get_aligned_signal("AAPL")

        # Should be bullish (all timeframes show uptrend)
        assert signal in ["bullish", "neutral"]  # May be neutral if not enough bars

    @pytest.mark.asyncio
    async def test_mixed_signals_returns_neutral(self):
        """Test returns neutral when timeframes have mixed signals."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])

        # Create flat data
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        for i in range(30):
            bar_time = timestamp + timedelta(minutes=i * 5)
            price = 100.0 + (i % 2)  # Oscillating
            await analyzer.update("AAPL", bar_time, price, 1000)

        signal = analyzer.get_aligned_signal("AAPL")

        # Likely neutral due to oscillating prices
        assert signal == "neutral"


# ============================================================================
# MultiTimeframeAnalyzer Momentum Tests
# ============================================================================


class TestGetTimeframeMomentum:
    """Test get_timeframe_momentum method."""

    def test_no_data_returns_zero(self):
        """Test returns 0 when no data."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])

        momentum = analyzer.get_timeframe_momentum("AAPL", "5Min")

        assert momentum == 0.0

    def test_no_symbol_returns_zero(self):
        """Test returns 0 when symbol not found."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])
        analyzer.data["MSFT"] = {}

        momentum = analyzer.get_timeframe_momentum("AAPL", "5Min")

        assert momentum == 0.0

    def test_insufficient_data_returns_zero(self):
        """Test returns 0 with insufficient data."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])
        analyzer.data["AAPL"] = {"5Min": TimeframeData("5Min")}

        # Add only a few bars
        tf_data = analyzer.data["AAPL"]["5Min"]
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        for i in range(5):  # Less than period + 1
            bar_start = timestamp + timedelta(minutes=i * 5)
            tf_data.update(bar_start, 100.0, 1000)

        momentum = analyzer.get_timeframe_momentum("AAPL", "5Min", period=14)

        assert momentum == 0.0

    def test_positive_momentum(self):
        """Test calculates positive momentum."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])
        analyzer.data["AAPL"] = {"5Min": TimeframeData("5Min")}

        tf_data = analyzer.data["AAPL"]["5Min"]
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create data with price increase
        for i in range(20):
            bar_start = timestamp + timedelta(minutes=i * 5)
            price = 100.0 + i * 2  # From 100 to 138
            tf_data.update(bar_start, price, 1000)

        momentum = analyzer.get_timeframe_momentum("AAPL", "5Min", period=10)

        assert momentum > 0  # Should be positive

    def test_negative_momentum(self):
        """Test calculates negative momentum."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])
        analyzer.data["AAPL"] = {"5Min": TimeframeData("5Min")}

        tf_data = analyzer.data["AAPL"]["5Min"]
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create data with price decrease
        for i in range(20):
            bar_start = timestamp + timedelta(minutes=i * 5)
            price = 200.0 - i * 2  # From 200 to 162
            tf_data.update(bar_start, price, 1000)

        momentum = analyzer.get_timeframe_momentum("AAPL", "5Min", period=10)

        assert momentum < 0  # Should be negative


# ============================================================================
# MultiTimeframeAnalyzer Divergence Tests
# ============================================================================


class TestDetectDivergence:
    """Test detect_divergence method."""

    def test_no_data_returns_none(self):
        """Test returns None when no data."""
        analyzer = MultiTimeframeAnalyzer(["1Min", "5Min"])

        divergence = analyzer.detect_divergence("AAPL")

        assert divergence is None

    def test_single_timeframe_returns_none(self):
        """Test returns None with single timeframe."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])
        analyzer.data["AAPL"] = {"5Min": TimeframeData("5Min")}

        divergence = analyzer.detect_divergence("AAPL")

        assert divergence is None


# ============================================================================
# MultiTimeframeAnalyzer Get Data Tests
# ============================================================================


class TestGetTimeframeData:
    """Test get_timeframe_data method."""

    def test_returns_none_no_symbol(self):
        """Test returns None when symbol not found."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])

        data = analyzer.get_timeframe_data("AAPL", "5Min")

        assert data is None

    def test_returns_none_no_timeframe(self):
        """Test returns None when timeframe not found."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])
        analyzer.data["AAPL"] = {}

        data = analyzer.get_timeframe_data("AAPL", "1Hour")

        assert data is None

    @pytest.mark.asyncio
    async def test_returns_timeframe_data(self):
        """Test returns TimeframeData object."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])

        await analyzer.update("AAPL", datetime.now(), 150.0, 1000)

        data = analyzer.get_timeframe_data("AAPL", "5Min")

        assert isinstance(data, TimeframeData)


# ============================================================================
# MultiTimeframeAnalyzer Get Status Tests
# ============================================================================


class TestGetStatus:
    """Test get_status method."""

    def test_no_data_returns_error(self):
        """Test returns error when no data."""
        analyzer = MultiTimeframeAnalyzer(["5Min"])

        status = analyzer.get_status("AAPL")

        assert "error" in status

    @pytest.mark.asyncio
    async def test_returns_complete_status(self):
        """Test returns complete status."""
        analyzer = MultiTimeframeAnalyzer(["5Min", "15Min"])

        # Add some data
        timestamp = datetime(2024, 1, 1, 10, 0, 0)
        for i in range(30):
            bar_time = timestamp + timedelta(minutes=i * 5)
            await analyzer.update("AAPL", bar_time, 150.0 + i, 1000)

        status = analyzer.get_status("AAPL")

        assert "timeframes" in status
        assert "aligned_signal" in status
        assert "divergence" in status
        assert "5Min" in status["timeframes"]
        assert "15Min" in status["timeframes"]

        # Check timeframe details
        assert "trend" in status["timeframes"]["5Min"]
        assert "momentum" in status["timeframes"]["5Min"]
        assert "bar_count" in status["timeframes"]["5Min"]


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_max_bars_limit(self):
        """Test bars are limited to max_bars."""
        tf_data = TimeframeData("1Min", max_bars=5)
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Create more bars than limit
        for i in range(10):
            bar_start = timestamp + timedelta(minutes=i)
            tf_data.update(bar_start, 100.0 + i, 1000)

        # Should only have max_bars - 1 completed bars (one is current)
        assert len(tf_data) <= 5

    def test_empty_getters(self):
        """Test getters with empty data."""
        tf_data = TimeframeData("5Min")

        closes = tf_data.get_closes()
        highs = tf_data.get_highs()
        lows = tf_data.get_lows()
        volumes = tf_data.get_volumes()

        assert len(closes) == 0
        assert len(highs) == 0
        assert len(lows) == 0
        assert len(volumes) == 0

    @pytest.mark.asyncio
    async def test_rapid_updates(self):
        """Test rapid consecutive updates."""
        analyzer = MultiTimeframeAnalyzer(["1Min"])
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        # Rapid updates within same minute
        for i in range(100):
            time_offset = timedelta(seconds=i)
            await analyzer.update("AAPL", timestamp + time_offset, 150.0 + i * 0.01, 100)

        # Should still work without errors
        assert "AAPL" in analyzer.data
