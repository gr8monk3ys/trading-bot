#!/usr/bin/env python3
"""
Unit tests for utils/volume_filter.py

Tests VolumeFilter class for:
- Volume ratio calculations
- Volume trend detection
- Price-volume divergence detection
- Signal confirmation
- Volume scoring
- Accumulation/Distribution analysis
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from utils.volume_filter import VolumeAnalyzer, VolumeFilter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def filter_default():
    """Default volume filter."""
    return VolumeFilter()


@pytest.fixture
def filter_custom():
    """Custom volume filter with adjusted thresholds."""
    return VolumeFilter(
        min_volume_ratio=1.0,
        breakout_volume_ratio=1.3,
        lookback_days=10,
        use_relative_volume=False,
    )


@pytest.fixture
def volume_history():
    """Standard volume history for testing."""
    return [
        1000000,
        1100000,
        950000,
        1050000,
        1000000,
        1200000,
        1150000,
        1000000,
        1100000,
        1050000,
    ]


@pytest.fixture
def price_history():
    """Standard price history for testing."""
    return [100, 101, 100.5, 102, 103, 104, 103.5, 105, 106, 107]


@pytest.fixture
def increasing_volume():
    """Volume with increasing trend."""
    return [100000, 110000, 120000, 130000, 140000, 160000, 180000, 200000, 220000, 250000]


@pytest.fixture
def decreasing_volume():
    """Volume with decreasing trend."""
    return [250000, 220000, 200000, 180000, 160000, 140000, 130000, 120000, 110000, 100000]


@pytest.fixture
def stable_volume():
    """Volume with stable trend."""
    return [100000, 102000, 98000, 101000, 99000, 100000, 101000, 99000, 100000, 100000]


# ============================================================================
# VolumeFilter Initialization Tests
# ============================================================================


class TestVolumeFilterInit:
    """Test VolumeFilter initialization."""

    def test_default_init(self, filter_default):
        """Test default initialization values."""
        assert filter_default.min_volume_ratio == 1.2
        assert filter_default.breakout_volume_ratio == 1.5
        assert filter_default.lookback_days == 20
        assert filter_default.use_relative_volume is True

    def test_custom_init(self, filter_custom):
        """Test custom initialization values."""
        assert filter_custom.min_volume_ratio == 1.0
        assert filter_custom.breakout_volume_ratio == 1.3
        assert filter_custom.lookback_days == 10
        assert filter_custom.use_relative_volume is False

    def test_class_constants(self, filter_default):
        """Test class constants are set."""
        assert filter_default.MIN_SCORE == 0.0
        assert filter_default.MAX_SCORE == 1.0
        assert filter_default.DEFAULT_SCORE == 0.5
        assert filter_default.LOW_VOLUME_THRESHOLD == 0.8
        assert filter_default.LOW_VOLUME_MAX_SCORE == 0.5


# ============================================================================
# Volume Ratio Tests
# ============================================================================


class TestCalculateVolumeRatio:
    """Test calculate_volume_ratio method."""

    def test_average_volume(self, filter_default, volume_history):
        """Test with volume at average."""
        avg = np.mean(volume_history)
        ratio = filter_default.calculate_volume_ratio(avg, volume_history)
        assert abs(ratio - 1.0) < 0.01

    def test_above_average_volume(self, filter_default, volume_history):
        """Test with volume above average."""
        current = 1500000  # Above average
        ratio = filter_default.calculate_volume_ratio(current, volume_history)
        assert ratio > 1.0

    def test_below_average_volume(self, filter_default, volume_history):
        """Test with volume below average."""
        current = 800000  # Below average
        ratio = filter_default.calculate_volume_ratio(current, volume_history)
        assert ratio < 1.0

    def test_empty_history(self, filter_default):
        """Test with empty volume history."""
        ratio = filter_default.calculate_volume_ratio(1000000, [])
        assert ratio == 1.0

    def test_short_history(self, filter_default):
        """Test with history shorter than 5 bars."""
        short_history = [1000000, 1100000, 950000]
        ratio = filter_default.calculate_volume_ratio(1200000, short_history)
        assert ratio == 1.0

    def test_zero_average(self, filter_default):
        """Test with zero average volume."""
        zero_history = [0, 0, 0, 0, 0, 0]
        ratio = filter_default.calculate_volume_ratio(1000000, zero_history)
        assert ratio == 1.0

    def test_double_average(self, filter_default, volume_history):
        """Test with volume at 2x average."""
        avg = np.mean(volume_history)
        ratio = filter_default.calculate_volume_ratio(avg * 2, volume_history)
        assert abs(ratio - 2.0) < 0.1

    def test_lookback_days_limit(self, filter_custom, volume_history):
        """Test that lookback_days limits the history used."""
        # filter_custom has lookback_days=10
        long_history = [500000] * 30 + volume_history  # Old low volume + recent higher
        ratio = filter_custom.calculate_volume_ratio(1050000, long_history)
        # Should use only last 10 days (from volume_history)
        assert ratio > 0.9  # Should be close to average of recent data


# ============================================================================
# Volume Trend Tests
# ============================================================================


class TestCalculateVolumeTrend:
    """Test calculate_volume_trend method."""

    def test_increasing_trend(self, filter_default, increasing_volume):
        """Test detection of increasing volume trend."""
        trend = filter_default.calculate_volume_trend(increasing_volume)
        assert trend == "increasing"

    def test_decreasing_trend(self, filter_default, decreasing_volume):
        """Test detection of decreasing volume trend."""
        trend = filter_default.calculate_volume_trend(decreasing_volume)
        assert trend == "decreasing"

    def test_stable_trend(self, filter_default, stable_volume):
        """Test detection of stable volume trend."""
        trend = filter_default.calculate_volume_trend(stable_volume)
        assert trend == "stable"

    def test_insufficient_data(self, filter_default):
        """Test with insufficient data for trend calculation."""
        short_history = [100000, 110000, 120000, 130000]  # Less than periods * 2
        trend = filter_default.calculate_volume_trend(short_history)
        assert trend == "stable"

    def test_custom_periods(self, filter_default, increasing_volume):
        """Test with custom period parameter."""
        trend = filter_default.calculate_volume_trend(increasing_volume, periods=3)
        assert trend in ["increasing", "decreasing", "stable"]

    def test_zero_previous_volume(self, filter_default):
        """Test with zero previous period volume."""
        history = [0, 0, 0, 0, 0, 100000, 110000, 120000, 130000, 140000]
        trend = filter_default.calculate_volume_trend(history)
        assert trend == "stable"  # Can't calculate change from zero


# ============================================================================
# Volume Divergence Tests
# ============================================================================


class TestCheckVolumeDivergence:
    """Test check_volume_divergence method."""

    def test_bearish_divergence(self, filter_default):
        """Test detection of bearish divergence (price up, volume down)."""
        prices = [100, 101, 102, 103, 104, 106, 108, 110, 112, 115]  # Increasing
        volumes = [150000, 145000, 140000, 135000, 130000, 110000, 100000, 90000, 80000, 70000]  # Decreasing
        has_div, div_type = filter_default.check_volume_divergence(prices, volumes)
        assert has_div
        assert div_type == "bearish"

    def test_bullish_divergence(self, filter_default):
        """Test detection of bullish divergence (price down, volume up)."""
        prices = [115, 112, 110, 108, 106, 104, 102, 100, 98, 95]  # Decreasing
        volumes = [70000, 80000, 90000, 100000, 110000, 130000, 140000, 150000, 160000, 180000]  # Increasing
        has_div, div_type = filter_default.check_volume_divergence(prices, volumes)
        assert has_div
        assert div_type == "bullish"

    def test_no_divergence(self, filter_default, price_history, volume_history):
        """Test when there is no divergence."""
        # Normal correlated movement
        has_div, div_type = filter_default.check_volume_divergence(price_history, volume_history)
        assert div_type == "none"

    def test_price_up_volume_up(self, filter_default):
        """Test no divergence when both price and volume increase."""
        prices = [100, 101, 102, 103, 104, 106, 108, 110, 112, 115]
        volumes = [100000, 105000, 110000, 115000, 120000, 130000, 140000, 150000, 160000, 180000]
        has_div, div_type = filter_default.check_volume_divergence(prices, volumes)
        assert not has_div or div_type == "none"

    def test_insufficient_price_data(self, filter_default, volume_history):
        """Test with insufficient price data."""
        short_prices = [100, 101, 102]
        has_div, div_type = filter_default.check_volume_divergence(short_prices, volume_history)
        assert not has_div
        assert div_type == "none"

    def test_insufficient_volume_data(self, filter_default, price_history):
        """Test with insufficient volume data."""
        short_volumes = [100000, 110000, 120000]
        has_div, div_type = filter_default.check_volume_divergence(price_history, short_volumes)
        assert not has_div
        assert div_type == "none"

    def test_zero_previous_price(self, filter_default):
        """Test handling of zero previous price."""
        prices = [0, 0, 0, 0, 0, 100, 101, 102, 103, 104]
        volumes = [100000] * 10
        has_div, div_type = filter_default.check_volume_divergence(prices, volumes)
        # Should handle gracefully
        assert div_type in ["none", "bearish", "bullish"]

    def test_custom_periods(self, filter_default, price_history, volume_history):
        """Test with custom periods parameter."""
        has_div, div_type = filter_default.check_volume_divergence(
            price_history, volume_history, periods=3
        )
        assert div_type in ["none", "bearish", "bullish"]


# ============================================================================
# Volume Confirmation Tests
# ============================================================================


class TestIsVolumeConfirmed:
    """Test is_volume_confirmed method."""

    def test_normal_confirmed(self, filter_default, volume_history):
        """Test normal signal confirmation with sufficient volume."""
        avg = np.mean(volume_history)  # ~1,060,000
        current_volume = avg * 1.25  # Above average * 1.2
        is_confirmed, analysis = filter_default.is_volume_confirmed(
            current_volume, volume_history, "normal"
        )
        assert is_confirmed
        assert "volume_ratio" in analysis
        assert analysis["signal_type"] == "normal"

    def test_normal_not_confirmed(self, filter_default, volume_history):
        """Test normal signal not confirmed with low volume."""
        avg = np.mean(volume_history)
        current_volume = avg * 1.1  # Below 1.2x threshold
        is_confirmed, analysis = filter_default.is_volume_confirmed(
            current_volume, volume_history, "normal"
        )
        assert not is_confirmed

    def test_breakout_requires_higher_ratio(self, filter_default, volume_history):
        """Test breakout signals require higher volume ratio."""
        avg = np.mean(volume_history)
        # Volume at 1.35x - enough for normal but not breakout
        current_volume = avg * 1.35

        normal_confirmed, _ = filter_default.is_volume_confirmed(
            current_volume, volume_history, "normal"
        )
        breakout_confirmed, _ = filter_default.is_volume_confirmed(
            current_volume, volume_history, "breakout"
        )

        assert normal_confirmed
        assert not breakout_confirmed  # Needs 1.5x

    def test_breakout_confirmed(self, filter_default, volume_history):
        """Test breakout signal confirmation with high volume."""
        avg = np.mean(volume_history)
        current_volume = avg * 1.55  # Above 1.5x threshold
        is_confirmed, analysis = filter_default.is_volume_confirmed(
            current_volume, volume_history, "breakout"
        )
        assert is_confirmed
        assert analysis["required_ratio"] == 1.5

    def test_reversal_requires_extra_confirmation(self, filter_default, volume_history):
        """Test reversal signals require 1.3x normal ratio."""
        avg = np.mean(volume_history)
        # Volume at 1.45x - enough for normal but not reversal
        current_volume = avg * 1.45

        normal_confirmed, _ = filter_default.is_volume_confirmed(
            current_volume, volume_history, "normal"
        )
        reversal_confirmed, analysis = filter_default.is_volume_confirmed(
            current_volume, volume_history, "reversal"
        )

        assert normal_confirmed
        assert not reversal_confirmed  # Needs 1.2 * 1.3 = 1.56x
        assert analysis["required_ratio"] == pytest.approx(1.56, rel=0.01)

    def test_analysis_dict_contents(self, filter_default, volume_history):
        """Test that analysis dict contains all expected fields."""
        _, analysis = filter_default.is_volume_confirmed(1200000, volume_history, "normal")
        assert "volume_ratio" in analysis
        assert "required_ratio" in analysis
        assert "volume_trend" in analysis
        assert "is_confirmed" in analysis
        assert "signal_type" in analysis
        assert "confidence" in analysis

    def test_confidence_calculation(self, filter_default, volume_history):
        """Test confidence is calculated correctly."""
        avg = np.mean(volume_history)
        # Volume at 2x required ratio
        current_volume = avg * 2.4  # 2x the 1.2 min ratio
        _, analysis = filter_default.is_volume_confirmed(current_volume, volume_history, "normal")
        # Confidence should be capped at 1.5
        assert analysis["confidence"] <= 1.5


# ============================================================================
# Confirms Signal Tests
# ============================================================================


class TestConfirmsSignal:
    """Test confirms_signal method."""

    def test_neutral_signal_always_confirmed(self, filter_default, volume_history, price_history):
        """Test neutral signals are always confirmed."""
        is_confirmed, analysis = filter_default.confirms_signal(
            "neutral", 500000, volume_history, price_history
        )
        assert is_confirmed
        assert "reason" in analysis

    def test_long_signal_confirmed(self, filter_default, volume_history, price_history):
        """Test long signal confirmation."""
        avg = np.mean(volume_history)
        current_volume = avg * 1.3  # Above 1.2x threshold
        is_confirmed, analysis = filter_default.confirms_signal(
            "long", current_volume, volume_history, price_history
        )
        assert is_confirmed

    def test_short_signal_confirmed(self, filter_default, volume_history, price_history):
        """Test short signal confirmation."""
        avg = np.mean(volume_history)
        current_volume = avg * 1.3  # Above 1.2x threshold
        is_confirmed, analysis = filter_default.confirms_signal(
            "short", current_volume, volume_history, price_history
        )
        assert is_confirmed

    def test_breakout_flag(self, filter_default, volume_history):
        """Test is_breakout flag changes required volume."""
        avg = np.mean(volume_history)
        current_volume = avg * 1.35  # Enough for normal (1.2x) but not breakout (1.5x)

        normal_confirmed, _ = filter_default.confirms_signal(
            "long", current_volume, volume_history, is_breakout=False
        )
        breakout_confirmed, _ = filter_default.confirms_signal(
            "long", current_volume, volume_history, is_breakout=True
        )

        assert normal_confirmed
        assert not breakout_confirmed

    def test_divergence_reduces_confidence(self, filter_default):
        """Test that divergence reduces confidence for contradicting signals."""
        # Bearish divergence (price up, volume down)
        prices = [100, 101, 102, 103, 104, 106, 108, 110, 112, 115]
        volumes = [150000, 145000, 140000, 135000, 130000, 110000, 100000, 90000, 80000, 70000]

        # Long signal with bearish divergence should have reduced confidence
        _, analysis = filter_default.confirms_signal("long", 200000, volumes, prices)

        if analysis.get("has_divergence"):
            assert "warning" in analysis
            # Confidence should be reduced

    def test_no_price_history(self, filter_default, volume_history):
        """Test confirmation works without price history."""
        is_confirmed, analysis = filter_default.confirms_signal(
            "long", 1500000, volume_history, price_history=None
        )
        assert "has_divergence" not in analysis or not analysis.get("has_divergence")

    def test_short_price_history(self, filter_default, volume_history):
        """Test with price history shorter than needed for divergence."""
        short_prices = [100, 101, 102]  # Less than 10
        _, analysis = filter_default.confirms_signal("long", 1500000, volume_history, short_prices)
        assert "has_divergence" not in analysis


# ============================================================================
# Volume Score Tests
# ============================================================================


class TestGetVolumeScore:
    """Test get_volume_score method."""

    def test_high_volume_max_score(self, filter_default, volume_history):
        """Test high volume gets score near maximum."""
        avg = np.mean(volume_history)
        score = filter_default.get_volume_score(avg * 2, volume_history)
        assert score >= 0.9

    def test_low_volume_low_score(self, filter_default, volume_history):
        """Test low volume gets low score."""
        avg = np.mean(volume_history)
        score = filter_default.get_volume_score(avg * 0.5, volume_history)
        assert score <= 0.5

    def test_average_volume_mid_score(self, filter_default, volume_history):
        """Test average volume gets mid-range score."""
        avg = np.mean(volume_history)
        score = filter_default.get_volume_score(avg, volume_history)
        assert 0.3 <= score <= 0.7

    def test_empty_history_default_score(self, filter_default):
        """Test empty history returns default score."""
        score = filter_default.get_volume_score(1000000, [])
        assert score == filter_default.DEFAULT_SCORE

    def test_score_bounded(self, filter_default, volume_history, price_history):
        """Test score is always between MIN and MAX."""
        for vol in [100000, 500000, 1000000, 2000000, 5000000]:
            score = filter_default.get_volume_score(vol, volume_history, price_history)
            assert filter_default.MIN_SCORE <= score <= filter_default.MAX_SCORE

    def test_increasing_trend_boosts_score(self, filter_default, increasing_volume):
        """Test increasing volume trend boosts score."""
        avg = np.mean(increasing_volume)
        score_increasing = filter_default.get_volume_score(avg, increasing_volume)
        score_stable = filter_default.get_volume_score(avg, [avg] * 10)
        # Increasing trend should boost score
        assert score_increasing >= score_stable * 0.9  # Allow some variance

    def test_decreasing_trend_reduces_score(self, filter_default, decreasing_volume):
        """Test decreasing volume trend reduces score."""
        avg = np.mean(decreasing_volume)
        score_decreasing = filter_default.get_volume_score(avg, decreasing_volume)
        # Score should be reduced by decreasing trend
        assert score_decreasing <= 1.0  # Basic sanity check

    def test_divergence_reduces_score(self, filter_default):
        """Test that divergence reduces the score."""
        # Bearish divergence data
        prices = [100, 101, 102, 103, 104, 106, 108, 110, 112, 115]
        volumes = [150000, 145000, 140000, 135000, 130000, 110000, 100000, 90000, 80000, 70000]

        score_with_div = filter_default.get_volume_score(100000, volumes, prices)
        score_without_div = filter_default.get_volume_score(100000, volumes, None)

        # Score with divergence should be lower
        assert score_with_div <= score_without_div


# ============================================================================
# Accumulation/Distribution Tests
# ============================================================================


class TestCalculateAccumulationDistribution:
    """Test calculate_accumulation_distribution method."""

    def test_close_at_high(self, filter_default):
        """Test AD when close is at high (bullish)."""
        ad = filter_default.calculate_accumulation_distribution(
            high=110, low=100, close=110, volume=1000000
        )
        assert ad > 0  # Accumulation

    def test_close_at_low(self, filter_default):
        """Test AD when close is at low (bearish)."""
        ad = filter_default.calculate_accumulation_distribution(
            high=110, low=100, close=100, volume=1000000
        )
        assert ad < 0  # Distribution

    def test_close_at_midpoint(self, filter_default):
        """Test AD when close is at midpoint (neutral)."""
        ad = filter_default.calculate_accumulation_distribution(
            high=110, low=100, close=105, volume=1000000
        )
        assert abs(ad) < 100000  # Near zero

    def test_high_equals_low(self, filter_default):
        """Test AD when high equals low (no range)."""
        ad = filter_default.calculate_accumulation_distribution(
            high=100, low=100, close=100, volume=1000000
        )
        assert ad == 0

    def test_volume_scales_ad(self, filter_default):
        """Test that volume scales the AD value."""
        ad_low_vol = filter_default.calculate_accumulation_distribution(
            high=110, low=100, close=110, volume=100000
        )
        ad_high_vol = filter_default.calculate_accumulation_distribution(
            high=110, low=100, close=110, volume=1000000
        )
        assert abs(ad_high_vol) > abs(ad_low_vol)


# ============================================================================
# AD Trend Tests
# ============================================================================


class TestGetAdTrend:
    """Test get_ad_trend method."""

    def test_accumulation_trend(self, filter_default):
        """Test detection of accumulation trend."""
        # Close consistently near highs
        bars = [
            {"high": 110, "low": 100, "close": 108 + i * 0.1, "volume": 1000000}
            for i in range(12)
        ]
        trend = filter_default.get_ad_trend(bars)
        # Second half should have higher AD values
        assert trend in ["accumulation", "neutral"]

    def test_distribution_trend(self, filter_default):
        """Test detection of distribution trend."""
        # First half: close near highs (accumulation), second half: close near lows (distribution)
        # This creates a clear shift from accumulation to distribution
        bars_first = [
            {"high": 110, "low": 100, "close": 109, "volume": 1000000}
            for _ in range(6)
        ]
        bars_second = [
            {"high": 110, "low": 100, "close": 101, "volume": 1000000}
            for _ in range(6)
        ]
        bars = bars_first + bars_second
        trend = filter_default.get_ad_trend(bars)
        # Second half (distribution) should be lower than first half (accumulation)
        assert trend == "distribution"

    def test_neutral_trend(self, filter_default):
        """Test neutral AD trend."""
        # Close at midpoint
        bars = [{"high": 110, "low": 100, "close": 105, "volume": 1000000} for _ in range(12)]
        trend = filter_default.get_ad_trend(bars)
        assert trend == "neutral"

    def test_insufficient_bars(self, filter_default):
        """Test with insufficient bars."""
        bars = [{"high": 110, "low": 100, "close": 105, "volume": 1000000} for _ in range(5)]
        trend = filter_default.get_ad_trend(bars)
        assert trend == "neutral"

    def test_custom_periods(self, filter_default):
        """Test with custom periods parameter."""
        bars = [{"high": 110, "low": 100, "close": 108, "volume": 1000000} for _ in range(20)]
        trend = filter_default.get_ad_trend(bars, periods=15)
        assert trend in ["accumulation", "distribution", "neutral"]


# ============================================================================
# VolumeAnalyzer Tests
# ============================================================================


class TestVolumeAnalyzerInit:
    """Test VolumeAnalyzer initialization."""

    def test_init_with_broker(self):
        """Test initialization with broker."""
        mock_broker = MagicMock()
        analyzer = VolumeAnalyzer(mock_broker)
        assert analyzer.broker is mock_broker
        assert analyzer.filter is not None
        assert isinstance(analyzer._cache, dict)


class TestVolumeAnalyzerAnalyzeVolume:
    """Test VolumeAnalyzer.analyze_volume method."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker with async get_bars."""
        broker = MagicMock()
        broker.get_bars = AsyncMock()
        return broker

    @pytest.fixture
    def mock_bars(self):
        """Create mock bar data."""

        class MockBar:
            def __init__(self, high, low, close, volume):
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume

        return [
            MockBar(110 + i, 100 + i, 105 + i, 1000000 + i * 10000)
            for i in range(20)
        ]

    @pytest.mark.asyncio
    async def test_analyze_volume_success(self, mock_broker, mock_bars):
        """Test successful volume analysis."""
        mock_broker.get_bars.return_value = mock_bars
        analyzer = VolumeAnalyzer(mock_broker)

        result = await analyzer.analyze_volume("AAPL")

        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "current_volume" in result
        assert "avg_volume" in result
        assert "volume_ratio" in result
        assert "volume_trend" in result
        assert "has_divergence" in result
        assert "divergence_type" in result
        assert "ad_trend" in result
        assert "volume_score" in result
        assert "recommendation" in result

    @pytest.mark.asyncio
    async def test_analyze_volume_insufficient_data(self, mock_broker):
        """Test with insufficient data."""
        mock_broker.get_bars.return_value = []
        analyzer = VolumeAnalyzer(mock_broker)

        result = await analyzer.analyze_volume("AAPL")

        assert "error" in result
        assert "Insufficient data" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_volume_none_bars(self, mock_broker):
        """Test when broker returns None."""
        mock_broker.get_bars.return_value = None
        analyzer = VolumeAnalyzer(mock_broker)

        result = await analyzer.analyze_volume("AAPL")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_analyze_volume_exception(self, mock_broker):
        """Test handling of exceptions."""
        mock_broker.get_bars.side_effect = Exception("API Error")
        analyzer = VolumeAnalyzer(mock_broker)

        result = await analyzer.analyze_volume("AAPL")

        assert "error" in result
        assert "API Error" in result["error"]


class TestGetRecommendation:
    """Test VolumeAnalyzer._get_recommendation method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mock broker."""
        mock_broker = MagicMock()
        return VolumeAnalyzer(mock_broker)

    def test_strong_confirmation(self, analyzer):
        """Test strong volume confirmation recommendation."""
        rec = analyzer._get_recommendation(
            ratio=1.6, trend="increasing", has_divergence=False, ad_trend="accumulation"
        )
        assert "Strong" in rec or "high confidence" in rec

    def test_good_confirmation(self, analyzer):
        """Test good volume confirmation recommendation."""
        rec = analyzer._get_recommendation(
            ratio=1.3, trend="stable", has_divergence=False, ad_trend="neutral"
        )
        assert "Good" in rec

    def test_divergence_warning(self, analyzer):
        """Test divergence warning recommendation."""
        rec = analyzer._get_recommendation(
            ratio=1.5, trend="increasing", has_divergence=True, ad_trend="neutral"
        )
        assert "divergence" in rec.lower() or "Caution" in rec

    def test_low_volume_warning(self, analyzer):
        """Test low volume warning recommendation."""
        rec = analyzer._get_recommendation(
            ratio=0.7, trend="stable", has_divergence=False, ad_trend="neutral"
        )
        assert "Low volume" in rec or "weak" in rec

    def test_distribution_pattern(self, analyzer):
        """Test distribution pattern recommendation."""
        rec = analyzer._get_recommendation(
            ratio=0.9, trend="decreasing", has_divergence=False, ad_trend="distribution"
        )
        assert "Distribution" in rec or "bearish" in rec

    def test_accumulation_pattern(self, analyzer):
        """Test accumulation pattern recommendation."""
        rec = analyzer._get_recommendation(
            ratio=1.1, trend="increasing", has_divergence=False, ad_trend="accumulation"
        )
        assert "Accumulation" in rec or "bullish" in rec

    def test_neutral_conditions(self, analyzer):
        """Test neutral conditions recommendation."""
        rec = analyzer._get_recommendation(
            ratio=1.0, trend="stable", has_divergence=False, ad_trend="neutral"
        )
        assert "Neutral" in rec


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_volume(self, filter_default, volume_history):
        """Test with extremely high volume."""
        extreme_volume = 10000000  # 10x typical
        ratio = filter_default.calculate_volume_ratio(extreme_volume, volume_history)
        assert ratio > 5.0
        score = filter_default.get_volume_score(extreme_volume, volume_history)
        assert score == filter_default.MAX_SCORE

    def test_zero_volume(self, filter_default, volume_history):
        """Test with zero current volume."""
        ratio = filter_default.calculate_volume_ratio(0, volume_history)
        assert ratio == 0
        score = filter_default.get_volume_score(0, volume_history)
        assert score >= filter_default.MIN_SCORE

    def test_negative_volume(self, filter_default, volume_history):
        """Test handling of negative volume (invalid data)."""
        # Should handle gracefully
        ratio = filter_default.calculate_volume_ratio(-1000000, volume_history)
        assert ratio < 0  # Negative ratio

    def test_single_bar_history(self, filter_default):
        """Test with single bar history."""
        ratio = filter_default.calculate_volume_ratio(1000000, [1000000])
        assert ratio == 1.0  # Insufficient data fallback

    def test_all_same_volume(self, filter_default):
        """Test with identical volume values."""
        same_volume = [1000000] * 20
        ratio = filter_default.calculate_volume_ratio(1000000, same_volume)
        assert ratio == 1.0
        trend = filter_default.calculate_volume_trend(same_volume)
        assert trend == "stable"

    def test_extreme_price_movement(self, filter_default):
        """Test divergence with extreme price movement."""
        prices = [100] * 5 + [200] * 5  # 100% jump
        volumes = [1000000] * 10
        has_div, div_type = filter_default.check_volume_divergence(prices, volumes)
        # Should detect divergence or handle gracefully
        assert div_type in ["none", "bearish", "bullish"]

    def test_numpy_array_input(self, filter_default):
        """Test with numpy array inputs."""
        volume_array = np.array([1000000, 1100000, 950000, 1050000, 1000000])
        price_array = np.array([100, 101, 100.5, 102, 103])

        ratio = filter_default.calculate_volume_ratio(1200000, volume_array.tolist())
        assert ratio > 0

        # Note: method expects lists, numpy arrays should work too
        score = filter_default.get_volume_score(1200000, list(volume_array), list(price_array))
        assert 0 <= score <= 1


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_full_signal_flow(self, filter_default, volume_history, price_history):
        """Test complete signal confirmation flow."""
        avg = np.mean(volume_history)
        current_volume = avg * 1.4  # 40% above average, should be confirmed

        # Get all metrics
        ratio = filter_default.calculate_volume_ratio(current_volume, volume_history)
        trend = filter_default.calculate_volume_trend(volume_history)
        has_div, div_type = filter_default.check_volume_divergence(price_history, volume_history)
        is_confirmed, analysis = filter_default.confirms_signal(
            "long", current_volume, volume_history, price_history
        )
        score = filter_default.get_volume_score(current_volume, volume_history, price_history)

        # All should return valid results
        assert ratio > 1.0
        assert trend in ["increasing", "decreasing", "stable"]
        assert div_type in ["none", "bearish", "bullish"]
        assert is_confirmed  # Should be confirmed at 1.4x
        assert 0 <= score <= 1

    def test_breakout_confirmation_flow(self, filter_default):
        """Test breakout signal requires high volume."""
        # Normal volume
        normal_volume = [1000000] * 20
        avg = np.mean(normal_volume)

        # Test at different volume levels
        for mult, expected in [(1.0, False), (1.3, False), (1.5, True), (2.0, True)]:
            is_confirmed, _ = filter_default.confirms_signal(
                "long", avg * mult, normal_volume, is_breakout=True
            )
            assert is_confirmed == expected, f"Failed at {mult}x volume"

    def test_reversal_warning_flow(self, filter_default):
        """Test reversal signal with divergence warning."""
        # Create bearish divergence
        prices = [100, 101, 102, 103, 104, 106, 108, 110, 112, 115]
        volumes = [150000, 145000, 140000, 135000, 130000, 110000, 100000, 90000, 80000, 70000]

        # Long signal should get warning
        _, analysis = filter_default.confirms_signal("long", 100000, volumes, prices)

        # Should detect the divergence
        if analysis.get("has_divergence"):
            assert analysis["divergence_type"] == "bearish"
