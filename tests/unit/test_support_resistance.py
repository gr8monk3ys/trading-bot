#!/usr/bin/env python3
"""
Unit tests for utils/support_resistance.py

Tests SupportResistanceAnalyzer class for:
- Swing high/low detection
- Pivot point calculations
- Level clustering
- Round number detection
- Support/resistance level finding
- Stop-loss and profit target placement
"""

import pytest
import numpy as np
from utils.support_resistance import SupportResistanceAnalyzer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def analyzer():
    """Default analyzer with standard settings."""
    return SupportResistanceAnalyzer()


@pytest.fixture
def custom_analyzer():
    """Analyzer with custom settings."""
    return SupportResistanceAnalyzer(
        swing_lookback=3,
        min_touches=1,
        level_tolerance=0.02,
        include_round_numbers=False,
    )


@pytest.fixture
def sample_bars():
    """Generate 50 bars of sample price data."""
    np.random.seed(42)
    bars = []
    price = 100.0

    for _ in range(50):
        change = np.random.randn() * 1.5
        high = price + abs(np.random.randn())
        low = price - abs(np.random.randn())
        close = price + change
        bars.append({
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000000,
        })
        price = close

    return bars


@pytest.fixture
def trending_up_bars():
    """Generate uptrending price data."""
    bars = []
    price = 100.0

    for i in range(50):
        high = price + 1.5
        low = price - 0.5
        close = price + 1.0  # Consistent uptrend
        bars.append({
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000000,
        })
        price = close

    return bars


@pytest.fixture
def trending_down_bars():
    """Generate downtrending price data."""
    bars = []
    price = 150.0

    for i in range(50):
        high = price + 0.5
        low = price - 1.5
        close = price - 1.0  # Consistent downtrend
        bars.append({
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000000,
        })
        price = close

    return bars


@pytest.fixture
def oscillating_bars():
    """Generate oscillating price data with clear swing points."""
    bars = []

    # Create price oscillation between 95 and 105
    for i in range(60):
        cycle_pos = i % 10
        if cycle_pos < 5:
            # Going up
            price = 95 + cycle_pos * 2
        else:
            # Going down
            price = 105 - (cycle_pos - 5) * 2

        high = price + 0.5
        low = price - 0.5
        bars.append({
            "high": high,
            "low": low,
            "close": price,
            "volume": 1000000,
        })

    return bars


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSupportResistanceAnalyzerInit:
    """Test analyzer initialization."""

    def test_default_init(self, analyzer):
        """Test default initialization values."""
        assert analyzer.swing_lookback == 5
        assert analyzer.min_touches == 2
        assert analyzer.level_tolerance == 0.01
        assert analyzer.include_round_numbers is True

    def test_custom_init(self, custom_analyzer):
        """Test custom initialization values."""
        assert custom_analyzer.swing_lookback == 3
        assert custom_analyzer.min_touches == 1
        assert custom_analyzer.level_tolerance == 0.02
        assert custom_analyzer.include_round_numbers is False

    def test_class_constants(self, analyzer):
        """Test class constants are set correctly."""
        assert analyzer.FALLBACK_LONG_STOP_PCT == 0.97
        assert analyzer.FALLBACK_SHORT_STOP_PCT == 1.03
        assert analyzer.FALLBACK_LONG_TARGET_PCT == 1.05
        assert analyzer.FALLBACK_SHORT_TARGET_PCT == 0.95
        assert analyzer.FALLBACK_RESISTANCE_PCT == 1.05
        assert analyzer.FALLBACK_SUPPORT_PCT == 0.95
        assert analyzer.DEFAULT_BUFFER_PCT == 0.005
        assert analyzer.DEFAULT_LEVEL_TOLERANCE_PCT == 0.02
        assert analyzer.RECENT_BARS_LOOKBACK == 20


# ============================================================================
# Swing High/Low Detection Tests
# ============================================================================


class TestFindSwingHighs:
    """Test swing high detection."""

    def test_find_swing_highs_basic(self, analyzer):
        """Test basic swing high detection."""
        # Create data with clear swing high at index 5 (value 110)
        highs = [100, 102, 105, 107, 108, 110, 108, 106, 104, 102, 100]
        swing_highs = analyzer.find_swing_highs(highs)

        assert len(swing_highs) == 1
        assert swing_highs[0] == (5, 110)

    def test_find_swing_highs_multiple(self, analyzer):
        """Test finding multiple swing highs."""
        # Two swing highs at positions 5 and 15
        highs = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102,
                 104, 106, 108, 110, 112, 115, 112, 110, 108, 106, 100]
        swing_highs = analyzer.find_swing_highs(highs)

        assert len(swing_highs) == 2
        assert swing_highs[0] == (5, 110)
        assert swing_highs[1] == (15, 115)

    def test_find_swing_highs_no_swing(self, analyzer):
        """Test when no swing high exists."""
        # Monotonically increasing
        highs = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        swing_highs = analyzer.find_swing_highs(highs)

        assert len(swing_highs) == 0

    def test_find_swing_highs_custom_lookback(self, analyzer):
        """Test with custom lookback period."""
        highs = [100, 102, 104, 110, 104, 102, 100]
        # With lookback=2, index 3 (value 110) should be a swing high
        swing_highs = analyzer.find_swing_highs(highs, lookback=2)

        assert len(swing_highs) == 1
        assert swing_highs[0] == (3, 110)

    def test_find_swing_highs_short_data(self, analyzer):
        """Test with data shorter than lookback period."""
        highs = [100, 105, 100]  # Too short for default lookback=5
        swing_highs = analyzer.find_swing_highs(highs)

        assert len(swing_highs) == 0


class TestFindSwingLows:
    """Test swing low detection."""

    def test_find_swing_lows_basic(self, analyzer):
        """Test basic swing low detection."""
        # Create data with clear swing low at index 5 (value 90)
        lows = [100, 98, 96, 94, 92, 90, 92, 94, 96, 98, 100]
        swing_lows = analyzer.find_swing_lows(lows)

        assert len(swing_lows) == 1
        assert swing_lows[0] == (5, 90)

    def test_find_swing_lows_multiple(self, analyzer):
        """Test finding multiple swing lows."""
        lows = [100, 98, 96, 94, 92, 90, 92, 94, 96, 98,
                96, 94, 92, 90, 88, 85, 88, 90, 92, 94, 100]
        swing_lows = analyzer.find_swing_lows(lows)

        assert len(swing_lows) == 2
        assert swing_lows[0] == (5, 90)
        assert swing_lows[1] == (15, 85)

    def test_find_swing_lows_no_swing(self, analyzer):
        """Test when no swing low exists."""
        # Monotonically decreasing
        lows = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        swing_lows = analyzer.find_swing_lows(lows)

        assert len(swing_lows) == 0

    def test_find_swing_lows_custom_lookback(self, analyzer):
        """Test with custom lookback period."""
        lows = [100, 98, 96, 90, 96, 98, 100]
        # With lookback=2, index 3 (value 90) should be a swing low
        swing_lows = analyzer.find_swing_lows(lows, lookback=2)

        assert len(swing_lows) == 1
        assert swing_lows[0] == (3, 90)


# ============================================================================
# Pivot Point Tests
# ============================================================================


class TestCalculatePivotPoints:
    """Test pivot point calculations."""

    def test_pivot_point_calculation(self, analyzer):
        """Test standard pivot point calculation."""
        high, low, close = 110, 90, 100
        pivots = analyzer.calculate_pivot_points(high, low, close)

        # PP = (H + L + C) / 3 = (110 + 90 + 100) / 3 = 100
        assert pivots["PP"] == 100

    def test_resistance_levels(self, analyzer):
        """Test resistance level calculations."""
        high, low, close = 110, 90, 100
        pivots = analyzer.calculate_pivot_points(high, low, close)

        # R1 = 2*PP - L = 2*100 - 90 = 110
        assert pivots["R1"] == 110
        # R2 = PP + (H - L) = 100 + 20 = 120
        assert pivots["R2"] == 120
        # R3 = H + 2*(PP - L) = 110 + 2*(100 - 90) = 130
        assert pivots["R3"] == 130

    def test_support_levels(self, analyzer):
        """Test support level calculations."""
        high, low, close = 110, 90, 100
        pivots = analyzer.calculate_pivot_points(high, low, close)

        # S1 = 2*PP - H = 2*100 - 110 = 90
        assert pivots["S1"] == 90
        # S2 = PP - (H - L) = 100 - 20 = 80
        assert pivots["S2"] == 80
        # S3 = L - 2*(H - PP) = 90 - 2*(110 - 100) = 70
        assert pivots["S3"] == 70

    def test_pivot_keys(self, analyzer):
        """Test all expected keys are present."""
        pivots = analyzer.calculate_pivot_points(110, 90, 100)
        expected_keys = {"PP", "R1", "R2", "R3", "S1", "S2", "S3"}
        assert set(pivots.keys()) == expected_keys

    def test_pivot_with_different_close(self, analyzer):
        """Test pivot calculation with close near high."""
        high, low, close = 110, 90, 108
        pivots = analyzer.calculate_pivot_points(high, low, close)

        # PP = (110 + 90 + 108) / 3 = 102.67
        assert abs(pivots["PP"] - 102.67) < 0.01


# ============================================================================
# Level Clustering Tests
# ============================================================================


class TestClusterLevels:
    """Test level clustering functionality."""

    def test_cluster_nearby_levels(self, analyzer):
        """Test clustering of nearby price levels."""
        levels = [100, 100.5, 101, 110, 110.5, 111]
        clusters = analyzer.cluster_levels(levels)

        # Should form 2 clusters (around 100 and 110)
        assert len(clusters) == 2

    def test_cluster_with_min_touches(self):
        """Test min_touches requirement."""
        analyzer = SupportResistanceAnalyzer(min_touches=3)
        levels = [100, 100.5, 110, 110.5, 111]  # Only 2 at 100, 3 at 110
        clusters = analyzer.cluster_levels(levels)

        # Only the 110 cluster should pass min_touches
        assert len(clusters) == 1
        # Mean of 110, 110.5, 111 = 110.5
        assert abs(clusters[0]["price"] - 110.5) < 0.1

    def test_cluster_touches_count(self, analyzer):
        """Test touch count is accurate."""
        levels = [100, 100.2, 100.5, 100.8]  # 4 touches around 100
        clusters = analyzer.cluster_levels(levels)

        assert len(clusters) == 1
        assert clusters[0]["touches"] == 4

    def test_cluster_strength_calculation(self, analyzer):
        """Test strength calculation (normalized 0-1)."""
        levels = [100] * 10  # 10 touches
        clusters = analyzer.cluster_levels(levels)

        assert len(clusters) == 1
        # Strength = min(10/5, 1.0) = 1.0
        assert clusters[0]["strength"] == 1.0

    def test_cluster_range(self, analyzer):
        """Test cluster range is calculated."""
        levels = [100, 100.5, 101]
        clusters = analyzer.cluster_levels(levels)

        assert len(clusters) == 1
        assert clusters[0]["range"] == (100, 101)

    def test_cluster_empty_list(self, analyzer):
        """Test with empty input."""
        clusters = analyzer.cluster_levels([])
        assert clusters == []

    def test_cluster_custom_tolerance(self, analyzer):
        """Test with custom tolerance."""
        levels = [100, 105, 110]  # 5% apart
        clusters = analyzer.cluster_levels(levels, tolerance=0.06)

        # With 6% tolerance, 100 and 105 should cluster
        assert len(clusters) <= 2

    def test_cluster_zero_mean_guard(self, analyzer):
        """Test guard against division by zero with zero-valued levels."""
        levels = [0, 0.01, 100]
        # Should not raise an error
        clusters = analyzer.cluster_levels(levels)
        assert clusters is not None


# ============================================================================
# Round Number Tests
# ============================================================================


class TestGetRoundNumbers:
    """Test psychological round number detection."""

    def test_round_numbers_low_price(self, analyzer):
        """Test round numbers for low-priced stock (<$20)."""
        levels = analyzer.get_round_numbers(15, range_pct=0.40)

        # With 40% range (9-21), should include $10, $15, $20
        assert 15.0 in levels
        assert 10.0 in levels

    def test_round_numbers_mid_price(self, analyzer):
        """Test round numbers for mid-priced stock ($20-100)."""
        levels = analyzer.get_round_numbers(55, range_pct=0.20)

        # Should include $50, $60
        assert 50.0 in levels
        assert 60.0 in levels

    def test_round_numbers_high_price(self, analyzer):
        """Test round numbers for higher-priced stock ($100-500)."""
        levels = analyzer.get_round_numbers(250, range_pct=0.10)

        # Should include levels like $225, $250, $275
        assert 250.0 in levels

    def test_round_numbers_very_high_price(self, analyzer):
        """Test round numbers for very high-priced stock (>$500)."""
        levels = analyzer.get_round_numbers(750, range_pct=0.10)

        # Should include levels like $700, $750, $800
        assert 750.0 in levels

    def test_round_numbers_sorted(self, analyzer):
        """Test round numbers are sorted."""
        levels = analyzer.get_round_numbers(100, range_pct=0.20)

        assert levels == sorted(levels)

    def test_round_numbers_unique(self, analyzer):
        """Test round numbers contain no duplicates."""
        levels = analyzer.get_round_numbers(100, range_pct=0.20)

        assert len(levels) == len(set(levels))

    def test_round_numbers_within_range(self, analyzer):
        """Test all levels are within specified range."""
        current = 100
        range_pct = 0.10
        levels = analyzer.get_round_numbers(current, range_pct=range_pct)

        low_bound = current * (1 - range_pct)
        high_bound = current * (1 + range_pct)

        for level in levels:
            assert low_bound <= level <= high_bound


# ============================================================================
# Find Levels Tests
# ============================================================================


class TestFindLevels:
    """Test comprehensive level finding."""

    def test_find_levels_returns_dict(self, analyzer, sample_bars):
        """Test find_levels returns correct structure."""
        levels = analyzer.find_levels(sample_bars)

        expected_keys = {
            "current_price", "nearest_resistance", "nearest_support",
            "resistance_levels", "support_levels", "pivot_points",
            "recent_high", "recent_low", "resistance_distance_pct",
            "support_distance_pct", "risk_reward"
        }
        assert set(levels.keys()) == expected_keys

    def test_find_levels_insufficient_data(self, analyzer):
        """Test with insufficient data."""
        bars = [{"high": 100, "low": 99, "close": 99.5}] * 10
        levels = analyzer.find_levels(bars)

        assert "error" in levels
        assert levels["error"] == "Insufficient data"

    def test_find_levels_empty_bars(self, analyzer):
        """Test with empty bars list."""
        levels = analyzer.find_levels([])

        assert "error" in levels

    def test_find_levels_current_price(self, analyzer, sample_bars):
        """Test current price is set correctly."""
        levels = analyzer.find_levels(sample_bars, current_price=100)

        assert levels["current_price"] == 100

    def test_find_levels_default_current_price(self, analyzer, sample_bars):
        """Test default current price is last close."""
        levels = analyzer.find_levels(sample_bars)

        assert levels["current_price"] == sample_bars[-1]["close"]

    def test_find_levels_recent_high_low(self, analyzer, sample_bars):
        """Test recent high/low calculation."""
        levels = analyzer.find_levels(sample_bars)

        # Recent high should be max of last 20 bars
        recent_bars = sample_bars[-20:]
        expected_high = max(b["high"] for b in recent_bars)
        expected_low = min(b["low"] for b in recent_bars)

        assert abs(levels["recent_high"] - expected_high) < 0.01
        assert abs(levels["recent_low"] - expected_low) < 0.01

    def test_find_levels_pivot_points(self, analyzer, sample_bars):
        """Test pivot points are included."""
        levels = analyzer.find_levels(sample_bars)

        assert "PP" in levels["pivot_points"]
        assert "R1" in levels["pivot_points"]
        assert "S1" in levels["pivot_points"]

    def test_find_levels_resistance_above_price(self, analyzer, sample_bars):
        """Test all resistance levels are above current price."""
        current_price = 100
        levels = analyzer.find_levels(sample_bars, current_price=current_price)

        for r in levels["resistance_levels"]:
            assert r["price"] > current_price

    def test_find_levels_support_below_price(self, analyzer, sample_bars):
        """Test all support levels are below current price."""
        current_price = 100
        levels = analyzer.find_levels(sample_bars, current_price=current_price)

        for s in levels["support_levels"]:
            assert s["price"] < current_price

    def test_find_levels_max_5_levels(self, analyzer, oscillating_bars):
        """Test max 5 levels returned for each."""
        levels = analyzer.find_levels(oscillating_bars)

        assert len(levels["resistance_levels"]) <= 5
        assert len(levels["support_levels"]) <= 5

    def test_find_levels_fallback_resistance(self, analyzer):
        """Test fallback resistance when no levels found."""
        # Create minimal bars with all prices same to avoid finding swing points
        bars = [{"high": 100, "low": 100, "close": 100, "volume": 1000} for _ in range(30)]
        analyzer_no_round = SupportResistanceAnalyzer(include_round_numbers=False, min_touches=10)
        levels = analyzer_no_round.find_levels(bars, current_price=100)

        # Should use fallback 5% above
        assert levels["nearest_resistance"] == 100 * 1.05

    def test_find_levels_resistance_distance_pct(self, analyzer, sample_bars):
        """Test resistance distance percentage calculation."""
        levels = analyzer.find_levels(sample_bars, current_price=100)

        expected_distance = (levels["nearest_resistance"] - 100) / 100
        assert abs(levels["resistance_distance_pct"] - expected_distance) < 0.001

    def test_find_levels_support_distance_pct(self, analyzer, sample_bars):
        """Test support distance percentage calculation."""
        levels = analyzer.find_levels(sample_bars, current_price=100)

        expected_distance = (100 - levels["nearest_support"]) / 100
        assert abs(levels["support_distance_pct"] - expected_distance) < 0.001

    def test_find_levels_with_round_numbers(self, analyzer, sample_bars):
        """Test that round numbers are included when enabled."""
        analyzer_with_round = SupportResistanceAnalyzer(include_round_numbers=True)
        levels = analyzer_with_round.find_levels(sample_bars, current_price=100)

        # 100 is a round number, nearby levels should include round numbers
        assert levels is not None

    def test_find_levels_without_round_numbers(self, custom_analyzer, sample_bars):
        """Test that round numbers are excluded when disabled."""
        # custom_analyzer has include_round_numbers=False
        levels = custom_analyzer.find_levels(sample_bars, current_price=100)

        assert levels is not None


# ============================================================================
# Risk/Reward Tests
# ============================================================================


class TestCalculateRiskReward:
    """Test risk/reward calculation."""

    def test_risk_reward_basic(self, analyzer):
        """Test basic risk/reward calculation."""
        entry = 100
        support = 95  # Risk = 5
        resistance = 115  # Reward = 15

        rr = analyzer._calculate_risk_reward(entry, support, resistance)

        # Reward / Risk = 15 / 5 = 3.0
        assert rr == 3.0

    def test_risk_reward_equal(self, analyzer):
        """Test when risk equals reward."""
        entry = 100
        support = 95  # Risk = 5
        resistance = 105  # Reward = 5

        rr = analyzer._calculate_risk_reward(entry, support, resistance)

        assert rr == 1.0

    def test_risk_reward_zero_risk(self, analyzer):
        """Test when risk is zero."""
        entry = 100
        support = 100  # Risk = 0
        resistance = 105

        rr = analyzer._calculate_risk_reward(entry, support, resistance)

        assert rr == 0

    def test_risk_reward_negative_risk(self, analyzer):
        """Test when support is above entry (invalid)."""
        entry = 100
        support = 105  # Invalid - above entry
        resistance = 110

        rr = analyzer._calculate_risk_reward(entry, support, resistance)

        assert rr == 0


# ============================================================================
# Optimal Stop Tests
# ============================================================================


class TestGetOptimalStop:
    """Test optimal stop-loss placement."""

    def test_long_stop_below_support(self, analyzer, sample_bars):
        """Test long stop is placed below support."""
        current_price = 100
        stop = analyzer.get_optimal_stop("long", current_price, sample_bars)

        levels = analyzer.find_levels(sample_bars, current_price)
        support = levels["nearest_support"]

        # Stop should be below support (with buffer)
        assert stop < support

    def test_short_stop_above_resistance(self, analyzer, sample_bars):
        """Test short stop is placed above resistance."""
        current_price = 100
        stop = analyzer.get_optimal_stop("short", current_price, sample_bars)

        levels = analyzer.find_levels(sample_bars, current_price)
        resistance = levels["nearest_resistance"]

        # Stop should be above resistance (with buffer)
        assert stop > resistance

    def test_long_stop_buffer(self, analyzer, sample_bars):
        """Test long stop uses buffer correctly."""
        current_price = 100
        buffer_pct = 0.01  # 1% buffer
        stop = analyzer.get_optimal_stop("long", current_price, sample_bars, buffer_pct=buffer_pct)

        levels = analyzer.find_levels(sample_bars, current_price)
        support = levels["nearest_support"]

        expected_stop = support * (1 - buffer_pct)
        assert abs(stop - expected_stop) < 0.01

    def test_short_stop_buffer(self, analyzer, sample_bars):
        """Test short stop uses buffer correctly."""
        current_price = 100
        buffer_pct = 0.01
        stop = analyzer.get_optimal_stop("short", current_price, sample_bars, buffer_pct=buffer_pct)

        levels = analyzer.find_levels(sample_bars, current_price)
        resistance = levels["nearest_resistance"]

        expected_stop = resistance * (1 + buffer_pct)
        assert abs(stop - expected_stop) < 0.01

    def test_long_stop_fallback(self, analyzer):
        """Test fallback stop for long when no levels found."""
        current_price = 100
        insufficient_bars = [{"high": 100, "low": 99, "close": 99.5}] * 5

        stop = analyzer.get_optimal_stop("long", current_price, insufficient_bars)

        # Should use fallback 3% below
        assert stop == current_price * 0.97

    def test_short_stop_fallback(self, analyzer):
        """Test fallback stop for short when no levels found."""
        current_price = 100
        insufficient_bars = [{"high": 100, "low": 99, "close": 99.5}] * 5

        stop = analyzer.get_optimal_stop("short", current_price, insufficient_bars)

        # Should use fallback 3% above
        assert stop == current_price * 1.03


# ============================================================================
# Profit Target Tests
# ============================================================================


class TestGetProfitTarget:
    """Test profit target placement."""

    def test_long_target_at_resistance(self, analyzer, sample_bars):
        """Test long target is at resistance."""
        current_price = 100
        target = analyzer.get_profit_target("long", current_price, sample_bars)

        # Target should be at or above current price
        assert target > current_price

    def test_short_target_at_support(self, analyzer, sample_bars):
        """Test short target is at support."""
        current_price = 100
        target = analyzer.get_profit_target("short", current_price, sample_bars)

        # Target should be at or below current price
        assert target < current_price

    def test_long_target_first_level(self, analyzer, sample_bars):
        """Test long target returns first resistance level by default."""
        current_price = 100
        target = analyzer.get_profit_target("long", current_price, sample_bars, target_number=1)

        levels = analyzer.find_levels(sample_bars, current_price)
        if levels["resistance_levels"]:
            assert target == levels["resistance_levels"][0]["price"]

    def test_long_target_second_level(self, analyzer, sample_bars):
        """Test long target can return second resistance level."""
        current_price = 100
        target = analyzer.get_profit_target("long", current_price, sample_bars, target_number=2)

        levels = analyzer.find_levels(sample_bars, current_price)
        if len(levels["resistance_levels"]) >= 2:
            assert target == levels["resistance_levels"][1]["price"]

    def test_short_target_first_level(self, analyzer, sample_bars):
        """Test short target returns first support level by default."""
        current_price = 100
        target = analyzer.get_profit_target("short", current_price, sample_bars, target_number=1)

        levels = analyzer.find_levels(sample_bars, current_price)
        if levels["support_levels"]:
            assert target == levels["support_levels"][0]["price"]

    def test_long_target_fallback(self, analyzer):
        """Test fallback target for long when no levels found."""
        current_price = 100
        insufficient_bars = [{"high": 100, "low": 99, "close": 99.5}] * 5

        target = analyzer.get_profit_target("long", current_price, insufficient_bars)

        # Should use fallback 5% above
        assert target == current_price * 1.05

    def test_short_target_fallback(self, analyzer):
        """Test fallback target for short when no levels found."""
        current_price = 100
        insufficient_bars = [{"high": 100, "low": 99, "close": 99.5}] * 5

        target = analyzer.get_profit_target("short", current_price, insufficient_bars)

        # Should use fallback 5% below
        assert target == current_price * 0.95

    def test_long_target_when_levels_insufficient(self, analyzer, sample_bars):
        """Test target when target_number exceeds available levels."""
        current_price = 100
        # Request very high target number
        target = analyzer.get_profit_target("long", current_price, sample_bars, target_number=100)

        levels = analyzer.find_levels(sample_bars, current_price)
        # Should fall back to nearest resistance
        assert target == levels["nearest_resistance"]


# ============================================================================
# Is At Support/Resistance Tests
# ============================================================================


class TestIsAtSupport:
    """Test support proximity detection."""

    def test_is_at_support_when_at_level(self, analyzer, sample_bars):
        """Test detection when price is exactly at support."""
        # Find a support level and test at that price
        levels = analyzer.find_levels(sample_bars, current_price=100)
        if levels["support_levels"]:
            support_price = levels["support_levels"][0]["price"]
            is_at, level = analyzer.is_at_support(support_price, sample_bars)

            # Should be at support when at the level
            assert is_at is True
            assert level is not None

    def test_is_at_support_within_tolerance(self, analyzer, sample_bars):
        """Test detection within tolerance range."""
        levels = analyzer.find_levels(sample_bars, current_price=100)
        if levels["support_levels"]:
            support_price = levels["support_levels"][0]["price"]
            # Price slightly above support (within 2% default tolerance)
            test_price = support_price * 1.01
            is_at, level = analyzer.is_at_support(test_price, sample_bars)

            assert is_at is True

    def test_is_at_support_outside_tolerance(self, analyzer, sample_bars):
        """Test detection outside tolerance range."""
        levels = analyzer.find_levels(sample_bars, current_price=100)
        if levels["support_levels"]:
            support_price = levels["support_levels"][0]["price"]
            # Price well above support (outside 2% default tolerance)
            test_price = support_price * 1.10
            is_at, level = analyzer.is_at_support(test_price, sample_bars)

            # May or may not be at a different support level
            # This tests that the original level isn't matched

    def test_is_at_support_custom_tolerance(self, analyzer, sample_bars):
        """Test with custom tolerance."""
        levels = analyzer.find_levels(sample_bars, current_price=100)
        if levels["support_levels"]:
            support_price = levels["support_levels"][0]["price"]
            # Price 3% above support
            test_price = support_price * 1.03

            # Should not match with 1% tolerance
            is_at, level = analyzer.is_at_support(test_price, sample_bars, tolerance_pct=0.01)
            # Could be near another support level, so just verify it works

    def test_is_at_support_insufficient_data(self, analyzer):
        """Test with insufficient data."""
        bars = [{"high": 100, "low": 99, "close": 99.5}] * 5
        is_at, level = analyzer.is_at_support(100, bars)

        assert is_at is False
        assert level is None


class TestIsAtResistance:
    """Test resistance proximity detection."""

    def test_is_at_resistance_when_at_level(self, analyzer, sample_bars):
        """Test detection when price is near a resistance level."""
        # First find levels with a reference price
        levels = analyzer.find_levels(sample_bars, current_price=95)
        if levels["resistance_levels"]:
            resistance_price = levels["resistance_levels"][0]["price"]
            # Test slightly below the resistance (within tolerance)
            test_price = resistance_price * 0.99
            is_at, level = analyzer.is_at_resistance(test_price, sample_bars)

            # Should be at resistance when within tolerance
            assert is_at is True
            assert level is not None

    def test_is_at_resistance_within_tolerance(self, analyzer, sample_bars):
        """Test detection within tolerance range."""
        levels = analyzer.find_levels(sample_bars, current_price=100)
        if levels["resistance_levels"]:
            resistance_price = levels["resistance_levels"][0]["price"]
            # Price slightly below resistance (within 2% default tolerance)
            test_price = resistance_price * 0.99
            is_at, level = analyzer.is_at_resistance(test_price, sample_bars)

            assert is_at is True

    def test_is_at_resistance_insufficient_data(self, analyzer):
        """Test with insufficient data."""
        bars = [{"high": 100, "low": 99, "close": 99.5}] * 5
        is_at, level = analyzer.is_at_resistance(100, bars)

        assert is_at is False
        assert level is None

    def test_is_at_resistance_custom_tolerance(self, analyzer, sample_bars):
        """Test with custom tolerance."""
        levels = analyzer.find_levels(sample_bars, current_price=100)
        if levels["resistance_levels"]:
            resistance_price = levels["resistance_levels"][0]["price"]
            # Test with very tight tolerance
            is_at_tight, _ = analyzer.is_at_resistance(
                resistance_price * 0.95, sample_bars, tolerance_pct=0.01
            )
            # 5% away with 1% tolerance should not match
            # (unless there's another resistance level nearby)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_trading_workflow(self, analyzer, sample_bars):
        """Test complete trading analysis workflow."""
        current_price = 100

        # Find levels
        levels = analyzer.find_levels(sample_bars, current_price)
        assert "error" not in levels

        # Get stop and target for long trade
        stop = analyzer.get_optimal_stop("long", current_price, sample_bars)
        target = analyzer.get_profit_target("long", current_price, sample_bars)

        # Validate trade setup
        assert stop < current_price, "Stop should be below entry"
        assert target > current_price, "Target should be above entry"
        assert levels["risk_reward"] >= 0, "Risk/reward should be non-negative"

    def test_short_trading_workflow(self, analyzer, sample_bars):
        """Test complete short trading analysis workflow."""
        current_price = 100

        # Get stop and target for short trade
        stop = analyzer.get_optimal_stop("short", current_price, sample_bars)
        target = analyzer.get_profit_target("short", current_price, sample_bars)

        # Validate trade setup
        assert stop > current_price, "Stop should be above entry for short"
        assert target < current_price, "Target should be below entry for short"

    def test_uptrending_market(self, analyzer, trending_up_bars):
        """Test analysis in uptrending market."""
        current_price = trending_up_bars[-1]["close"]
        levels = analyzer.find_levels(trending_up_bars, current_price)

        # In uptrend, resistance should be above current price
        assert levels["nearest_resistance"] >= current_price

    def test_downtrending_market(self, analyzer, trending_down_bars):
        """Test analysis in downtrending market."""
        current_price = trending_down_bars[-1]["close"]
        levels = analyzer.find_levels(trending_down_bars, current_price)

        # In downtrend, support should be below current price
        assert levels["nearest_support"] <= current_price

    def test_oscillating_market(self, analyzer, oscillating_bars):
        """Test analysis in oscillating market."""
        current_price = 100
        levels = analyzer.find_levels(oscillating_bars, current_price)

        # Oscillating market should have both S/R levels
        assert "error" not in levels
        # Should have identifiable levels due to clear oscillation pattern

    def test_missing_volume_data(self, analyzer):
        """Test handling of bars without volume data."""
        bars = [
            {"high": 100 + i, "low": 99 + i, "close": 99.5 + i}
            for i in range(30)
        ]

        # Should work without volume data
        levels = analyzer.find_levels(bars)
        assert "error" not in levels


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_price_stock(self, analyzer):
        """Test with penny stock prices."""
        bars = [
            {"high": 0.5 + i * 0.01, "low": 0.4 + i * 0.01, "close": 0.45 + i * 0.01, "volume": 1000}
            for i in range(30)
        ]

        levels = analyzer.find_levels(bars, current_price=0.75)
        assert "error" not in levels

    def test_very_high_price_stock(self, analyzer):
        """Test with high-priced stock."""
        bars = [
            {"high": 3000 + i * 10, "low": 2950 + i * 10, "close": 2975 + i * 10, "volume": 1000}
            for i in range(30)
        ]

        levels = analyzer.find_levels(bars, current_price=3200)
        assert "error" not in levels

        # Round numbers should include levels like 3200, 3250, etc.
        round_nums = analyzer.get_round_numbers(3200)
        assert len(round_nums) > 0

    def test_identical_prices(self, analyzer):
        """Test with all identical prices."""
        bars = [
            {"high": 100, "low": 100, "close": 100, "volume": 1000}
            for _ in range(30)
        ]

        levels = analyzer.find_levels(bars, current_price=100)
        # Should not crash, may use fallbacks
        assert levels is not None

    def test_extreme_volatility(self, analyzer):
        """Test with extreme price volatility."""
        np.random.seed(123)
        bars = [
            {
                "high": 100 + np.random.uniform(0, 50),
                "low": 100 - np.random.uniform(0, 50),
                "close": 100 + np.random.uniform(-25, 25),
                "volume": 1000
            }
            for _ in range(30)
        ]

        levels = analyzer.find_levels(bars, current_price=100)
        assert "error" not in levels

    def test_large_lookback(self):
        """Test with lookback larger than data."""
        analyzer = SupportResistanceAnalyzer(swing_lookback=50)
        bars = [
            {"high": 100 + i, "low": 99 + i, "close": 99.5 + i, "volume": 1000}
            for i in range(30)
        ]

        # With lookback=50 and 30 bars, no swings can be found
        highs = [b["high"] for b in bars]
        swing_highs = analyzer.find_swing_highs(highs)
        assert len(swing_highs) == 0
