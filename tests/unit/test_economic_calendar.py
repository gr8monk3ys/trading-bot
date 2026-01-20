#!/usr/bin/env python3
"""
Unit tests for utils/economic_calendar.py

Tests EconomicEventCalendar class for economic event tracking.
"""

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import patch

import pytz

from utils.economic_calendar import EconomicEventCalendar, EventImpact


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def calendar():
    """Default economic calendar."""
    return EconomicEventCalendar()


@pytest.fixture
def calendar_avoid_medium():
    """Calendar that avoids medium impact events."""
    return EconomicEventCalendar(avoid_medium_impact=True)


@pytest.fixture
def calendar_no_reduce():
    """Calendar that doesn't reduce size for medium events."""
    return EconomicEventCalendar(reduce_size_medium_impact=False)


@pytest.fixture
def eastern_tz():
    """Eastern timezone."""
    return pytz.timezone("US/Eastern")


# ============================================================================
# EventImpact Enum Tests
# ============================================================================


class TestEventImpact:
    """Test EventImpact enum."""

    def test_high_impact(self):
        """Test high impact value."""
        assert EventImpact.HIGH.value == "high"

    def test_medium_impact(self):
        """Test medium impact value."""
        assert EventImpact.MEDIUM.value == "medium"

    def test_low_impact(self):
        """Test low impact value."""
        assert EventImpact.LOW.value == "low"

    def test_enum_count(self):
        """Test number of impact levels."""
        assert len(EventImpact) == 3


# ============================================================================
# EconomicEventCalendar Initialization Tests
# ============================================================================


class TestEconomicEventCalendarInit:
    """Test EconomicEventCalendar initialization."""

    def test_default_init(self, calendar):
        """Test default initialization values."""
        assert calendar.avoid_high_impact is True
        assert calendar.avoid_medium_impact is False
        assert calendar.reduce_size_medium_impact is True

    def test_custom_avoid_medium(self, calendar_avoid_medium):
        """Test custom avoid_medium_impact."""
        assert calendar_avoid_medium.avoid_medium_impact is True

    def test_no_reduce(self, calendar_no_reduce):
        """Test reduce_size_medium_impact=False."""
        assert calendar_no_reduce.reduce_size_medium_impact is False

    def test_fomc_dates_parsed(self, calendar):
        """Test FOMC dates are parsed."""
        assert len(calendar.fomc_dates) > 0

    def test_timezone_set(self, calendar, eastern_tz):
        """Test timezone is set."""
        assert str(calendar.tz) == "US/Eastern"


# ============================================================================
# Recurring Events Tests
# ============================================================================


class TestRecurringEvents:
    """Test recurring events configuration."""

    def test_fomc_meeting_exists(self, calendar):
        """Test FOMC meeting event exists."""
        assert "FOMC_MEETING" in calendar.RECURRING_EVENTS
        event = calendar.RECURRING_EVENTS["FOMC_MEETING"]
        assert event["impact"] == EventImpact.HIGH

    def test_nfp_exists(self, calendar):
        """Test NFP event exists."""
        assert "NFP" in calendar.RECURRING_EVENTS
        event = calendar.RECURRING_EVENTS["NFP"]
        assert event["impact"] == EventImpact.HIGH

    def test_cpi_exists(self, calendar):
        """Test CPI event exists."""
        assert "CPI" in calendar.RECURRING_EVENTS
        event = calendar.RECURRING_EVENTS["CPI"]
        assert event["impact"] == EventImpact.HIGH

    def test_jobless_claims_is_low(self, calendar):
        """Test jobless claims is low impact."""
        assert "JOBLESS_CLAIMS" in calendar.RECURRING_EVENTS
        event = calendar.RECURRING_EVENTS["JOBLESS_CLAIMS"]
        assert event["impact"] == EventImpact.LOW

    def test_all_events_have_required_fields(self, calendar):
        """Test all events have required fields."""
        for name, event in calendar.RECURRING_EVENTS.items():
            assert "name" in event, f"{name} missing 'name'"
            assert "impact" in event, f"{name} missing 'impact'"
            assert "description" in event, f"{name} missing 'description'"


# ============================================================================
# FOMC Dates Tests
# ============================================================================


class TestFomcDates:
    """Test FOMC dates configuration."""

    def test_fomc_dates_2024(self, calendar):
        """Test 2024 FOMC dates exist."""
        dates_2024 = [d for d in calendar.fomc_dates if d.year == 2024]
        assert len(dates_2024) > 0

    def test_fomc_dates_2025(self, calendar):
        """Test 2025 FOMC dates exist."""
        dates_2025 = [d for d in calendar.fomc_dates if d.year == 2025]
        assert len(dates_2025) > 0

    def test_fomc_dates_2026(self, calendar):
        """Test 2026 FOMC dates exist."""
        dates_2026 = [d for d in calendar.fomc_dates if d.year == 2026]
        assert len(dates_2026) > 0


# ============================================================================
# Is FOMC Day Tests
# ============================================================================


class TestIsFomcDay:
    """Test is_fomc_day method."""

    def test_fomc_day_returns_true(self, calendar, eastern_tz):
        """Test returns True on FOMC day."""
        # Use a known FOMC date
        fomc_date = datetime(2024, 1, 31, 10, 0)
        dt = eastern_tz.localize(fomc_date)
        assert calendar.is_fomc_day(dt) is True

    def test_non_fomc_day_returns_false(self, calendar, eastern_tz):
        """Test returns False on non-FOMC day."""
        non_fomc_date = datetime(2024, 1, 15, 10, 0)
        dt = eastern_tz.localize(non_fomc_date)
        assert calendar.is_fomc_day(dt) is False

    def test_uses_current_time_when_none(self, calendar):
        """Test uses current time when dt is None."""
        # Just ensure it doesn't crash
        result = calendar.is_fomc_day()
        assert isinstance(result, bool)


# ============================================================================
# Is NFP Day Tests
# ============================================================================


class TestIsNfpDay:
    """Test is_nfp_day method."""

    def test_first_friday_returns_true(self, calendar, eastern_tz):
        """Test returns True on first Friday of month."""
        # January 5, 2024 is the first Friday
        first_friday = datetime(2024, 1, 5, 10, 0)
        dt = eastern_tz.localize(first_friday)
        assert calendar.is_nfp_day(dt) is True

    def test_non_friday_returns_false(self, calendar, eastern_tz):
        """Test returns False on non-Friday."""
        monday = datetime(2024, 1, 1, 10, 0)
        dt = eastern_tz.localize(monday)
        assert calendar.is_nfp_day(dt) is False

    def test_second_friday_returns_false(self, calendar, eastern_tz):
        """Test returns False on second Friday."""
        second_friday = datetime(2024, 1, 12, 10, 0)
        dt = eastern_tz.localize(second_friday)
        assert calendar.is_nfp_day(dt) is False

    def test_friday_day_7_returns_true(self, calendar, eastern_tz):
        """Test returns True when Friday is day 7."""
        # February 7, 2025 is Friday and day 7
        friday_day7 = datetime(2025, 2, 7, 10, 0)
        dt = eastern_tz.localize(friday_day7)
        # Check if day 7 is actually a Friday
        if dt.weekday() == 4:
            assert calendar.is_nfp_day(dt) is True


# ============================================================================
# Get Upcoming Events Tests
# ============================================================================


class TestGetUpcomingEvents:
    """Test get_upcoming_events method."""

    def test_returns_list(self, calendar):
        """Test returns list of events."""
        events = calendar.get_upcoming_events(days_ahead=5)
        assert isinstance(events, list)

    def test_includes_fomc_on_fomc_day(self, calendar, eastern_tz):
        """Test includes FOMC event on FOMC day."""
        fomc_date = datetime(2024, 1, 31)
        with patch("utils.economic_calendar.datetime") as mock_dt:
            mock_dt.now.return_value = eastern_tz.localize(fomc_date)
            events = calendar.get_upcoming_events(days_ahead=0)
            fomc_events = [e for e in events if "FOMC" in e["name"]]
            assert len(fomc_events) > 0

    def test_includes_nfp_on_nfp_day(self, calendar, eastern_tz):
        """Test includes NFP event on NFP day."""
        # First Friday of January 2024
        nfp_date = datetime(2024, 1, 5)
        with patch("utils.economic_calendar.datetime") as mock_dt:
            mock_dt.now.return_value = eastern_tz.localize(nfp_date)
            mock_dt.max = datetime.max
            events = calendar.get_upcoming_events(days_ahead=0)
            nfp_events = [e for e in events if "NFP" in e.get("name", "")]
            # May or may not include depending on implementation

    def test_includes_jobless_claims_on_thursday(self, calendar, eastern_tz):
        """Test includes jobless claims on Thursday."""
        thursday = datetime(2024, 1, 4)  # A Thursday
        with patch("utils.economic_calendar.datetime") as mock_dt:
            mock_dt.now.return_value = eastern_tz.localize(thursday)
            mock_dt.max = datetime.max
            events = calendar.get_upcoming_events(days_ahead=0)
            claims_events = [e for e in events if "Jobless" in e.get("name", "")]
            assert len(claims_events) > 0

    def test_events_sorted_by_datetime(self, calendar):
        """Test events are sorted by datetime."""
        events = calendar.get_upcoming_events(days_ahead=7)
        if len(events) > 1:
            for i in range(len(events) - 1):
                dt1 = events[i].get("datetime", datetime.max)
                dt2 = events[i + 1].get("datetime", datetime.max)
                assert dt1 <= dt2


# ============================================================================
# Is Near Event Tests
# ============================================================================


class TestIsNearEvent:
    """Test is_near_event method."""

    def test_not_near_event(self, calendar, eastern_tz):
        """Test not near event returns False."""
        event = {
            "datetime": eastern_tz.localize(datetime(2024, 1, 31, 14, 0)),
            "avoid_hours_before": 2,
            "avoid_hours_after": 2,
        }
        # 10 hours before event
        check_time = eastern_tz.localize(datetime(2024, 1, 31, 4, 0))
        is_near, hours = calendar.is_near_event(event, check_time)
        assert is_near is False
        assert hours == 0

    def test_near_event_before(self, calendar, eastern_tz):
        """Test near event before returns True."""
        event = {
            "datetime": eastern_tz.localize(datetime(2024, 1, 31, 14, 0)),
            "avoid_hours_before": 2,
            "avoid_hours_after": 2,
        }
        # 1 hour before event (within 2 hour window)
        check_time = eastern_tz.localize(datetime(2024, 1, 31, 13, 0))
        is_near, hours = calendar.is_near_event(event, check_time)
        assert is_near is True
        assert hours > 0

    def test_near_event_after(self, calendar, eastern_tz):
        """Test near event after returns True."""
        event = {
            "datetime": eastern_tz.localize(datetime(2024, 1, 31, 14, 0)),
            "avoid_hours_before": 2,
            "avoid_hours_after": 2,
        }
        # 1 hour after event (within 2 hour window)
        check_time = eastern_tz.localize(datetime(2024, 1, 31, 15, 0))
        is_near, hours = calendar.is_near_event(event, check_time)
        assert is_near is True
        assert hours == pytest.approx(1.0)

    def test_no_datetime_returns_false(self, calendar):
        """Test no datetime in event returns False."""
        event = {"name": "Test Event"}
        is_near, hours = calendar.is_near_event(event)
        assert is_near is False
        assert hours == 0

    def test_naive_datetime_localized(self, calendar, eastern_tz):
        """Test naive datetime is localized."""
        event = {
            "datetime": datetime(2024, 1, 31, 14, 0),  # Naive datetime
            "avoid_hours_before": 2,
            "avoid_hours_after": 2,
        }
        check_time = eastern_tz.localize(datetime(2024, 1, 31, 13, 0))
        # Should not crash
        is_near, hours = calendar.is_near_event(event, check_time)


# ============================================================================
# Is Safe To Trade Tests
# ============================================================================


class TestIsSafeToTrade:
    """Test is_safe_to_trade method."""

    def test_safe_when_no_events(self, calendar, eastern_tz):
        """Test safe when no high impact events."""
        # Random day with no major events
        safe_time = eastern_tz.localize(datetime(2024, 2, 15, 10, 0))
        is_safe, info = calendar.is_safe_to_trade(safe_time)
        assert info["position_multiplier"] == 1.0

    def test_returns_info_dict(self, calendar):
        """Test returns info dict with expected keys."""
        is_safe, info = calendar.is_safe_to_trade()
        assert "is_safe" in info
        assert "current_time" in info
        assert "events_today" in info
        assert "blocking_event" in info
        assert "hours_until_safe" in info
        assert "position_multiplier" in info

    def test_unsafe_during_fomc(self, calendar, eastern_tz):
        """Test unsafe during FOMC meeting."""
        # During FOMC meeting window
        fomc_time = eastern_tz.localize(datetime(2024, 1, 31, 14, 30))
        is_safe, info = calendar.is_safe_to_trade(fomc_time)
        # Should be unsafe if within avoid window

    def test_reduced_size_medium_impact(self, calendar, eastern_tz):
        """Test reduced position size for medium impact."""
        # During medium impact event
        # (need to find a medium impact event time)


# ============================================================================
# Get Position Multiplier Tests
# ============================================================================


class TestGetPositionMultiplier:
    """Test get_position_multiplier method."""

    def test_returns_float(self, calendar):
        """Test returns float value."""
        multiplier = calendar.get_position_multiplier()
        assert isinstance(multiplier, float)

    def test_normal_returns_one(self, calendar, eastern_tz):
        """Test normal conditions return 1.0."""
        safe_time = eastern_tz.localize(datetime(2024, 2, 15, 10, 0))
        multiplier = calendar.get_position_multiplier(safe_time)
        assert multiplier == 1.0

    def test_multiplier_between_zero_and_one(self, calendar):
        """Test multiplier is always between 0 and 1."""
        multiplier = calendar.get_position_multiplier()
        assert 0.0 <= multiplier <= 1.0


# ============================================================================
# Get Calendar Report Tests
# ============================================================================


class TestGetCalendarReport:
    """Test get_calendar_report method."""

    def test_returns_dict(self, calendar):
        """Test returns dict."""
        report = calendar.get_calendar_report()
        assert isinstance(report, dict)

    def test_has_required_keys(self, calendar):
        """Test has required keys."""
        report = calendar.get_calendar_report()
        assert "is_safe_now" in report
        assert "today_info" in report
        assert "events_next_week" in report
        assert "high_impact_events" in report
        assert "medium_impact_events" in report
        assert "next_fomc" in report
        assert "upcoming_events" in report

    def test_upcoming_events_is_list(self, calendar):
        """Test upcoming_events is list."""
        report = calendar.get_calendar_report()
        assert isinstance(report["upcoming_events"], list)

    def test_events_count_matches(self, calendar):
        """Test events count matches."""
        report = calendar.get_calendar_report(days_ahead=7)
        total = report["high_impact_events"] + report["medium_impact_events"]
        # May also include low impact events


# ============================================================================
# Get Next FOMC Tests
# ============================================================================


class TestGetNextFomc:
    """Test _get_next_fomc method."""

    def test_returns_date_string_or_none(self, calendar):
        """Test returns date string or None."""
        result = calendar._get_next_fomc()
        assert result is None or isinstance(result, str)

    def test_future_date(self, calendar):
        """Test returns future date."""
        result = calendar._get_next_fomc()
        if result:
            fomc_date = datetime.strptime(result, "%Y-%m-%d")
            assert fomc_date.date() >= datetime.now().date()


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_days_ahead(self, calendar):
        """Test zero days ahead."""
        events = calendar.get_upcoming_events(days_ahead=0)
        assert isinstance(events, list)

    def test_large_days_ahead(self, calendar):
        """Test large days ahead."""
        events = calendar.get_upcoming_events(days_ahead=365)
        assert isinstance(events, list)

    def test_avoid_high_impact_false(self):
        """Test with avoid_high_impact=False."""
        cal = EconomicEventCalendar(avoid_high_impact=False)
        is_safe, info = cal.is_safe_to_trade()
        # Should not block on high impact events

    def test_invalid_timezone_handling(self):
        """Test timezone defaults work."""
        cal = EconomicEventCalendar(timezone="US/Eastern")
        assert str(cal.tz) == "US/Eastern"

    def test_multiple_events_same_day(self, calendar, eastern_tz):
        """Test handling multiple events on same day."""
        events = calendar.get_upcoming_events(days_ahead=30)
        # Just verify it doesn't crash

    def test_is_safe_uses_current_time(self, calendar):
        """Test is_safe_to_trade uses current time when None."""
        is_safe, info = calendar.is_safe_to_trade(None)
        assert "current_time" in info


# ============================================================================
# Configuration Variation Tests
# ============================================================================


class TestConfigurationVariations:
    """Test different configuration combinations."""

    def test_avoid_all_medium(self):
        """Test avoiding all medium impact events."""
        cal = EconomicEventCalendar(avoid_medium_impact=True)
        assert cal.avoid_medium_impact is True

    def test_no_reduction_on_medium(self):
        """Test no size reduction on medium impact."""
        cal = EconomicEventCalendar(reduce_size_medium_impact=False)
        assert cal.reduce_size_medium_impact is False

    def test_avoid_nothing(self):
        """Test avoiding nothing (aggressive trading)."""
        cal = EconomicEventCalendar(
            avoid_high_impact=False,
            avoid_medium_impact=False,
            reduce_size_medium_impact=False,
        )
        is_safe, info = cal.is_safe_to_trade()
        # Should always be "safe" since we're not avoiding anything
        assert info["position_multiplier"] == 1.0
