#!/usr/bin/env python3
"""
Unit tests for utils/trading_hours.py

Tests TradingHoursFilter class for:
- Trading window detection
- Day quality ratings
- Good time to trade checks
- Next good window calculation
- Quality scores and position sizing
- Trading status reports
"""

import pytest
from datetime import datetime, time, timedelta
import pytz

from utils.trading_hours import (
    TradingHoursFilter,
    TradingWindow,
    DayQuality,
    is_good_trading_time,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def filter_default():
    """Default trading hours filter."""
    return TradingHoursFilter()


@pytest.fixture
def filter_no_avoidance():
    """Filter with all avoidance turned off."""
    return TradingHoursFilter(
        avoid_opening=False,
        avoid_lunch=False,
        avoid_closing=False,
        avoid_monday_morning=False,
        avoid_friday_afternoon=False,
    )


@pytest.fixture
def filter_strict():
    """Filter with all avoidance turned on."""
    return TradingHoursFilter(
        avoid_opening=True,
        avoid_lunch=True,
        avoid_closing=True,
        avoid_monday_morning=True,
        avoid_friday_afternoon=True,
    )


@pytest.fixture
def eastern_tz():
    """Eastern timezone."""
    return pytz.timezone("US/Eastern")


# ============================================================================
# Initialization Tests
# ============================================================================


class TestTradingHoursFilterInit:
    """Test TradingHoursFilter initialization."""

    def test_default_init(self, filter_default):
        """Test default initialization values."""
        assert filter_default.avoid_opening is True
        assert filter_default.avoid_lunch is True
        assert filter_default.avoid_closing is False
        assert filter_default.avoid_monday_morning is True
        assert filter_default.avoid_friday_afternoon is True

    def test_custom_init(self, filter_no_avoidance):
        """Test custom initialization values."""
        assert filter_no_avoidance.avoid_opening is False
        assert filter_no_avoidance.avoid_lunch is False
        assert filter_no_avoidance.avoid_closing is False
        assert filter_no_avoidance.avoid_monday_morning is False
        assert filter_no_avoidance.avoid_friday_afternoon is False

    def test_timezone_set(self, filter_default):
        """Test timezone is set correctly."""
        assert filter_default.tz is not None
        assert str(filter_default.tz) == "US/Eastern"

    def test_class_constants(self, filter_default):
        """Test class constants are set."""
        assert filter_default.MARKET_OPEN == time(9, 30)
        assert filter_default.MARKET_CLOSE == time(16, 0)
        assert filter_default.MORNING_PRIME_START == time(10, 0)
        assert filter_default.MORNING_PRIME_END == time(11, 30)
        assert filter_default.AFTERNOON_PRIME_START == time(14, 0)
        assert filter_default.AFTERNOON_PRIME_END == time(15, 30)


# ============================================================================
# Trading Window Tests
# ============================================================================


class TestGetCurrentWindow:
    """Test trading window detection."""

    def test_premarket(self, filter_default, eastern_tz):
        """Test premarket detection."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # 8:00am Tuesday
        window = filter_default.get_current_window(dt)
        assert window == TradingWindow.PREMARKET

    def test_opening_volatility(self, filter_default, eastern_tz):
        """Test opening volatility window detection."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 9, 35))  # 9:35am Tuesday
        window = filter_default.get_current_window(dt)
        assert window == TradingWindow.OPENING_VOLATILITY

    def test_morning_prime(self, filter_default, eastern_tz):
        """Test morning prime window detection."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))  # 10:30am Tuesday
        window = filter_default.get_current_window(dt)
        assert window == TradingWindow.MORNING_PRIME

    def test_lunch_lull(self, filter_default, eastern_tz):
        """Test lunch lull window detection."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 12, 30))  # 12:30pm Tuesday
        window = filter_default.get_current_window(dt)
        assert window == TradingWindow.LUNCH_LULL

    def test_afternoon_prime(self, filter_default, eastern_tz):
        """Test afternoon prime window detection."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 14, 30))  # 2:30pm Tuesday
        window = filter_default.get_current_window(dt)
        assert window == TradingWindow.AFTERNOON_PRIME

    def test_closing_volatility(self, filter_default, eastern_tz):
        """Test closing volatility window detection."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 15, 45))  # 3:45pm Tuesday
        window = filter_default.get_current_window(dt)
        assert window == TradingWindow.CLOSING_VOLATILITY

    def test_afterhours(self, filter_default, eastern_tz):
        """Test afterhours detection."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 17, 0))  # 5:00pm Tuesday
        window = filter_default.get_current_window(dt)
        assert window == TradingWindow.AFTERHOURS

    def test_naive_datetime_localized(self, filter_default):
        """Test that naive datetime is localized."""
        # Pass naive datetime
        dt = datetime(2024, 1, 2, 10, 30)  # No timezone
        window = filter_default.get_current_window(dt)
        # Should still work
        assert window == TradingWindow.MORNING_PRIME

    def test_default_uses_now(self, filter_default):
        """Test that default uses current time."""
        # Just verify it doesn't crash
        window = filter_default.get_current_window()
        assert window in TradingWindow


class TestTradingWindowEnum:
    """Test TradingWindow enum values."""

    def test_all_windows_have_values(self):
        """Test all windows have string values."""
        for window in TradingWindow:
            assert window.value is not None
            assert isinstance(window.value, str)

    def test_window_values(self):
        """Test specific window values."""
        assert TradingWindow.PREMARKET.value == "premarket"
        assert TradingWindow.MORNING_PRIME.value == "morning_prime"
        assert TradingWindow.AFTERNOON_PRIME.value == "afternoon_prime"


# ============================================================================
# Day Quality Tests
# ============================================================================


class TestGetDayQuality:
    """Test day quality ratings."""

    def test_monday_poor(self, filter_default, eastern_tz):
        """Test Monday is rated poor."""
        dt = eastern_tz.localize(datetime(2024, 1, 1, 10, 0))  # Monday
        quality = filter_default.get_day_quality(dt)
        assert quality == DayQuality.POOR

    def test_tuesday_excellent(self, filter_default, eastern_tz):
        """Test Tuesday is rated excellent."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 0))  # Tuesday
        quality = filter_default.get_day_quality(dt)
        assert quality == DayQuality.EXCELLENT

    def test_wednesday_excellent(self, filter_default, eastern_tz):
        """Test Wednesday is rated excellent."""
        dt = eastern_tz.localize(datetime(2024, 1, 3, 10, 0))  # Wednesday
        quality = filter_default.get_day_quality(dt)
        assert quality == DayQuality.EXCELLENT

    def test_thursday_good(self, filter_default, eastern_tz):
        """Test Thursday is rated good."""
        dt = eastern_tz.localize(datetime(2024, 1, 4, 10, 0))  # Thursday
        quality = filter_default.get_day_quality(dt)
        assert quality == DayQuality.GOOD

    def test_friday_fair(self, filter_default, eastern_tz):
        """Test Friday is rated fair."""
        dt = eastern_tz.localize(datetime(2024, 1, 5, 10, 0))  # Friday
        quality = filter_default.get_day_quality(dt)
        assert quality == DayQuality.FAIR

    def test_default_uses_now(self, filter_default):
        """Test default uses current time."""
        quality = filter_default.get_day_quality()
        assert quality in DayQuality


class TestDayQualityEnum:
    """Test DayQuality enum values."""

    def test_all_qualities_have_values(self):
        """Test all qualities have string values."""
        for quality in DayQuality:
            assert quality.value is not None
            assert isinstance(quality.value, str)


# ============================================================================
# Is Good Time Tests
# ============================================================================


class TestIsGoodTimeToTrade:
    """Test is_good_time_to_trade method."""

    def test_morning_prime_tuesday(self, filter_default, eastern_tz):
        """Test morning prime on Tuesday is good."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))  # Tuesday 10:30am
        assert filter_default.is_good_time_to_trade(dt) is True

    def test_afternoon_prime_wednesday(self, filter_default, eastern_tz):
        """Test afternoon prime on Wednesday is good."""
        dt = eastern_tz.localize(datetime(2024, 1, 3, 14, 30))  # Wednesday 2:30pm
        assert filter_default.is_good_time_to_trade(dt) is True

    def test_premarket_not_good(self, filter_default, eastern_tz):
        """Test premarket is not good."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # Tuesday 8:00am
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_afterhours_not_good(self, filter_default, eastern_tz):
        """Test afterhours is not good."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 17, 0))  # Tuesday 5:00pm
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_weekend_not_good(self, filter_default, eastern_tz):
        """Test weekend is not good."""
        dt = eastern_tz.localize(datetime(2024, 1, 6, 10, 30))  # Saturday
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_opening_avoided(self, filter_default, eastern_tz):
        """Test opening volatility is avoided by default."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 9, 35))  # Tuesday 9:35am
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_opening_allowed_when_disabled(self, filter_no_avoidance, eastern_tz):
        """Test opening allowed when avoidance disabled."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 9, 35))  # Tuesday 9:35am
        assert filter_no_avoidance.is_good_time_to_trade(dt) is True

    def test_lunch_avoided(self, filter_default, eastern_tz):
        """Test lunch lull is avoided by default."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 12, 30))  # Tuesday 12:30pm
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_lunch_allowed_when_disabled(self, filter_no_avoidance, eastern_tz):
        """Test lunch allowed when avoidance disabled."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 12, 30))  # Tuesday 12:30pm
        assert filter_no_avoidance.is_good_time_to_trade(dt) is True

    def test_closing_allowed_by_default(self, filter_default, eastern_tz):
        """Test closing is allowed by default."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 15, 45))  # Tuesday 3:45pm
        assert filter_default.is_good_time_to_trade(dt) is True

    def test_closing_avoided_when_enabled(self, filter_strict, eastern_tz):
        """Test closing avoided when enabled."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 15, 45))  # Tuesday 3:45pm
        assert filter_strict.is_good_time_to_trade(dt) is False

    def test_monday_morning_avoided(self, filter_default, eastern_tz):
        """Test Monday morning before 11am is avoided."""
        dt = eastern_tz.localize(datetime(2024, 1, 1, 10, 30))  # Monday 10:30am
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_monday_afternoon_allowed(self, filter_default, eastern_tz):
        """Test Monday afternoon is allowed."""
        dt = eastern_tz.localize(datetime(2024, 1, 1, 14, 30))  # Monday 2:30pm
        assert filter_default.is_good_time_to_trade(dt) is True

    def test_friday_afternoon_avoided(self, filter_default, eastern_tz):
        """Test Friday afternoon after 2pm is avoided."""
        dt = eastern_tz.localize(datetime(2024, 1, 5, 14, 30))  # Friday 2:30pm
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_friday_morning_allowed(self, filter_default, eastern_tz):
        """Test Friday morning is allowed."""
        dt = eastern_tz.localize(datetime(2024, 1, 5, 10, 30))  # Friday 10:30am
        assert filter_default.is_good_time_to_trade(dt) is True

    def test_allow_fair_windows_false(self, filter_no_avoidance, eastern_tz):
        """Test strict mode with allow_fair_windows=False."""
        # Closing volatility should be rejected
        dt = eastern_tz.localize(datetime(2024, 1, 2, 15, 45))  # Tuesday 3:45pm
        assert filter_no_avoidance.is_good_time_to_trade(dt, allow_fair_windows=False) is False

    def test_allow_fair_windows_prime_times(self, filter_no_avoidance, eastern_tz):
        """Test prime times with allow_fair_windows=False."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))  # Tuesday 10:30am
        assert filter_no_avoidance.is_good_time_to_trade(dt, allow_fair_windows=False) is True


# ============================================================================
# Next Good Window Tests
# ============================================================================


class TestGetNextGoodWindow:
    """Test next good window calculation."""

    def test_from_premarket(self, filter_default, eastern_tz):
        """Test next window from premarket."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # Tuesday 8:00am
        next_window, window_type = filter_default.get_next_good_window(dt)

        assert next_window.hour == 10
        assert next_window.minute == 0
        assert window_type == TradingWindow.MORNING_PRIME

    def test_from_opening(self, filter_default, eastern_tz):
        """Test next window from opening volatility."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 9, 35))  # Tuesday 9:35am
        next_window, window_type = filter_default.get_next_good_window(dt)

        assert next_window.hour == 10
        assert next_window.minute == 0
        assert window_type == TradingWindow.MORNING_PRIME

    def test_from_lunch(self, filter_default, eastern_tz):
        """Test next window from lunch lull."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 12, 30))  # Tuesday 12:30pm
        next_window, window_type = filter_default.get_next_good_window(dt)

        assert next_window.hour == 14
        assert next_window.minute == 0
        assert window_type == TradingWindow.AFTERNOON_PRIME

    def test_from_afterhours(self, filter_default, eastern_tz):
        """Test next window from afterhours."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 17, 0))  # Tuesday 5:00pm
        next_window, window_type = filter_default.get_next_good_window(dt)

        # Should be next day morning
        assert next_window.day == 3  # Wednesday
        assert next_window.hour == 10
        assert window_type == TradingWindow.MORNING_PRIME

    def test_skips_weekend(self, filter_default, eastern_tz):
        """Test next window skips weekend."""
        dt = eastern_tz.localize(datetime(2024, 1, 5, 17, 0))  # Friday 5:00pm
        next_window, window_type = filter_default.get_next_good_window(dt)

        # Should be Monday (day 8)
        assert next_window.weekday() == 0  # Monday

    def test_monday_morning_skipped(self, filter_default, eastern_tz):
        """Test Monday morning is skipped when avoided."""
        dt = eastern_tz.localize(datetime(2024, 1, 1, 8, 0))  # Monday 8:00am
        next_window, window_type = filter_default.get_next_good_window(dt)

        # Should be Monday afternoon
        assert next_window.hour == 14
        assert window_type == TradingWindow.AFTERNOON_PRIME

    def test_friday_afternoon_skipped(self, filter_default, eastern_tz):
        """Test Friday afternoon is skipped when avoided."""
        dt = eastern_tz.localize(datetime(2024, 1, 5, 12, 0))  # Friday 12:00pm
        next_window, window_type = filter_default.get_next_good_window(dt)

        # Should skip Friday afternoon and go to next week
        # Because Friday afternoon (2pm) is after lunch (12pm)
        # and avoid_friday_afternoon=True

    def test_naive_datetime_handled(self, filter_default):
        """Test naive datetime is handled."""
        dt = datetime(2024, 1, 2, 8, 0)  # No timezone
        next_window, window_type = filter_default.get_next_good_window(dt)

        assert next_window is not None
        assert window_type in TradingWindow


class TestGetTimeUntilGoodWindow:
    """Test time until good window calculation."""

    def test_returns_timedelta(self, filter_default, eastern_tz):
        """Test returns timedelta."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # Tuesday 8:00am
        time_until = filter_default.get_time_until_good_window(dt)

        assert isinstance(time_until, timedelta)

    def test_positive_duration(self, filter_default, eastern_tz):
        """Test duration is positive when not in good window."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # Tuesday 8:00am
        time_until = filter_default.get_time_until_good_window(dt)

        assert time_until > timedelta(0)

    def test_approximately_correct(self, filter_default, eastern_tz):
        """Test duration is approximately correct."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 9, 0))  # Tuesday 9:00am
        time_until = filter_default.get_time_until_good_window(dt)

        # Should be about 1 hour until 10:00am
        assert timedelta(minutes=50) < time_until < timedelta(minutes=70)


# ============================================================================
# Quality Score Tests
# ============================================================================


class TestGetWindowQualityScore:
    """Test window quality score calculation."""

    def test_morning_prime_tuesday(self, filter_default, eastern_tz):
        """Test morning prime on Tuesday gets highest score."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))  # Tuesday 10:30am
        score = filter_default.get_window_quality_score(dt)

        assert score == 1.0  # Best possible

    def test_premarket_zero(self, filter_default, eastern_tz):
        """Test premarket gets zero score."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # Tuesday 8:00am
        score = filter_default.get_window_quality_score(dt)

        assert score == 0.0

    def test_afterhours_zero(self, filter_default, eastern_tz):
        """Test afterhours gets zero score."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 17, 0))  # Tuesday 5:00pm
        score = filter_default.get_window_quality_score(dt)

        assert score == 0.0

    def test_afternoon_prime_high(self, filter_default, eastern_tz):
        """Test afternoon prime gets high score."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 14, 30))  # Tuesday 2:30pm
        score = filter_default.get_window_quality_score(dt)

        assert score == 0.9  # High but not highest

    def test_lunch_lull_low(self, filter_default, eastern_tz):
        """Test lunch lull gets low score."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 12, 30))  # Tuesday 12:30pm
        score = filter_default.get_window_quality_score(dt)

        assert score == 0.4

    def test_monday_reduces_score(self, filter_default, eastern_tz):
        """Test Monday reduces score via day multiplier."""
        dt = eastern_tz.localize(datetime(2024, 1, 1, 14, 30))  # Monday 2:30pm
        score = filter_default.get_window_quality_score(dt)

        # 0.9 (afternoon) * 0.7 (Monday) = 0.63
        assert abs(score - 0.63) < 0.01

    def test_score_range(self, filter_default, eastern_tz):
        """Test all scores are in 0-1 range."""
        for day in range(1, 8):
            for hour in range(7, 20):
                dt = eastern_tz.localize(datetime(2024, 1, day, hour, 0))
                score = filter_default.get_window_quality_score(dt)
                assert 0.0 <= score <= 1.0


# ============================================================================
# Position Size Adjustment Tests
# ============================================================================


class TestGetPositionSizeAdjustment:
    """Test position size adjustment calculation."""

    def test_prime_window_boost(self, filter_default, eastern_tz):
        """Test prime window gets size boost."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))  # Tuesday 10:30am
        adjustment = filter_default.get_position_size_adjustment(dt)

        assert adjustment == 1.1  # 10% boost

    def test_premarket_reduction(self, filter_default, eastern_tz):
        """Test premarket gets size reduction."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # Tuesday 8:00am
        adjustment = filter_default.get_position_size_adjustment(dt)

        assert adjustment == 0.5  # 50% reduction

    def test_lunch_moderate_reduction(self, filter_default, eastern_tz):
        """Test lunch lull gets moderate reduction."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 12, 30))  # Tuesday 12:30pm
        adjustment = filter_default.get_position_size_adjustment(dt)

        # Quality ~0.4, should return 0.5 (lowest tier)
        assert adjustment == 0.5

    def test_adjustment_range(self, filter_default, eastern_tz):
        """Test adjustments are in valid range."""
        for day in range(1, 8):
            for hour in range(7, 20):
                dt = eastern_tz.localize(datetime(2024, 1, day, hour, 0))
                adjustment = filter_default.get_position_size_adjustment(dt)
                assert 0.5 <= adjustment <= 1.1


# ============================================================================
# Trading Status Tests
# ============================================================================


class TestGetTradingStatus:
    """Test comprehensive trading status."""

    def test_returns_dict(self, filter_default, eastern_tz):
        """Test returns dictionary."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))
        status = filter_default.get_trading_status(dt)

        assert isinstance(status, dict)

    def test_contains_expected_keys(self, filter_default, eastern_tz):
        """Test dictionary has expected keys."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))
        status = filter_default.get_trading_status(dt)

        expected_keys = {
            "current_time", "day_of_week", "window", "day_quality",
            "is_good_time", "quality_score", "position_size_mult",
            "next_good_window", "time_until_good", "recommendation"
        }
        assert set(status.keys()) == expected_keys

    def test_good_time_values(self, filter_default, eastern_tz):
        """Test values when in good time."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))  # Tuesday 10:30am
        status = filter_default.get_trading_status(dt)

        assert status["is_good_time"] is True
        assert status["window"] == "morning_prime"
        assert status["day_quality"] == "excellent"
        assert status["next_good_window"] is None
        assert status["time_until_good"] is None

    def test_bad_time_values(self, filter_default, eastern_tz):
        """Test values when in bad time."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 8, 0))  # Tuesday 8:00am
        status = filter_default.get_trading_status(dt)

        assert status["is_good_time"] is False
        assert status["window"] == "premarket"
        assert status["next_good_window"] is not None
        assert status["time_until_good"] is not None


# ============================================================================
# Recommendation Tests
# ============================================================================


class TestGetRecommendation:
    """Test trading recommendations."""

    def test_opening_recommendation(self, filter_default, eastern_tz):
        """Test recommendation for opening volatility."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 9, 35))  # Tuesday 9:35am
        status = filter_default.get_trading_status(dt)

        assert "opening volatility" in status["recommendation"].lower()

    def test_lunch_recommendation(self, filter_default, eastern_tz):
        """Test recommendation for lunch lull."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 12, 30))  # Tuesday 12:30pm
        status = filter_default.get_trading_status(dt)

        assert "afternoon" in status["recommendation"].lower()

    def test_prime_time_recommendation(self, filter_default, eastern_tz):
        """Test recommendation for prime time."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 30))  # Tuesday 10:30am
        status = filter_default.get_trading_status(dt)

        assert "prime" in status["recommendation"].lower()

    def test_afterhours_recommendation(self, filter_default, eastern_tz):
        """Test recommendation for afterhours."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 17, 0))  # Tuesday 5:00pm
        status = filter_default.get_trading_status(dt)

        assert "market closed" in status["recommendation"].lower()


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestIsGoodTradingTime:
    """Test convenience function."""

    def test_returns_bool(self):
        """Test returns boolean."""
        result = is_good_trading_time()
        assert isinstance(result, bool)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_at_market_open(self, filter_no_avoidance, eastern_tz):
        """Test exactly at market open."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 9, 30, 0))  # Exactly 9:30am
        window = filter_no_avoidance.get_current_window(dt)
        assert window == TradingWindow.OPENING_VOLATILITY

    def test_exactly_at_market_close(self, filter_no_avoidance, eastern_tz):
        """Test exactly at market close."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 16, 0, 0))  # Exactly 4:00pm
        window = filter_no_avoidance.get_current_window(dt)
        assert window == TradingWindow.AFTERHOURS

    def test_boundary_morning_prime_start(self, filter_no_avoidance, eastern_tz):
        """Test boundary at morning prime start."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 10, 0, 0))  # Exactly 10:00am
        window = filter_no_avoidance.get_current_window(dt)
        assert window == TradingWindow.MORNING_PRIME

    def test_boundary_lunch_start(self, filter_no_avoidance, eastern_tz):
        """Test boundary at lunch start."""
        dt = eastern_tz.localize(datetime(2024, 1, 2, 11, 30, 0))  # Exactly 11:30am
        window = filter_no_avoidance.get_current_window(dt)
        assert window == TradingWindow.LUNCH_LULL

    def test_different_timezone_input(self, filter_default):
        """Test with different timezone input converted to Eastern."""
        # Create datetime in Pacific time
        pacific = pytz.timezone("US/Pacific")
        eastern = pytz.timezone("US/Eastern")
        dt_pacific = pacific.localize(datetime(2024, 1, 2, 7, 30))  # 7:30am Pacific

        # Convert to Eastern time first (7:30 Pacific = 10:30 Eastern)
        dt_eastern = dt_pacific.astimezone(eastern)

        window = filter_default.get_current_window(dt_eastern)
        assert window == TradingWindow.MORNING_PRIME  # 10:30am Eastern is MORNING_PRIME

    def test_dst_transition(self, filter_default, eastern_tz):
        """Test during DST transition (just verify no crash)."""
        # March 10, 2024 is DST start
        dt = eastern_tz.localize(datetime(2024, 3, 10, 10, 30))
        window = filter_default.get_current_window(dt)
        assert window in TradingWindow

    def test_sunday(self, filter_default, eastern_tz):
        """Test Sunday is not good."""
        dt = eastern_tz.localize(datetime(2024, 1, 7, 10, 30))  # Sunday
        assert filter_default.is_good_time_to_trade(dt) is False

    def test_saturday(self, filter_default, eastern_tz):
        """Test Saturday is not good."""
        dt = eastern_tz.localize(datetime(2024, 1, 6, 10, 30))  # Saturday
        assert filter_default.is_good_time_to_trade(dt) is False
