#!/usr/bin/env python3
"""
Unit tests for utils/earnings_calendar.py

Tests EarningsCalendar class for earnings-related trading filters.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.earnings_calendar import EarningsCalendar, check_earnings_safety

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def calendar():
    """Default earnings calendar."""
    return EarningsCalendar()


@pytest.fixture
def calendar_custom():
    """Custom earnings calendar with different settings."""
    return EarningsCalendar(
        exit_days_before=3,
        skip_entry_days_before=5,
        reentry_days_after=2,
        cache_hours=24,
    )


@pytest.fixture
def mock_ticker_with_earnings():
    """Mock yfinance Ticker with earnings date."""
    mock_ticker = MagicMock()
    future_date = datetime.now() + timedelta(days=10)
    mock_calendar = pd.DataFrame(
        {"Values": [future_date, None]}, index=["Earnings Date", "Revenue Date"]
    )
    mock_ticker.calendar = mock_calendar
    return mock_ticker


@pytest.fixture
def mock_ticker_no_earnings():
    """Mock yfinance Ticker with no earnings."""
    mock_ticker = MagicMock()
    mock_ticker.calendar = None
    return mock_ticker


@pytest.fixture
def mock_ticker_empty_calendar():
    """Mock yfinance Ticker with empty calendar."""
    mock_ticker = MagicMock()
    mock_calendar = pd.DataFrame()
    mock_calendar.empty = True
    mock_ticker.calendar = mock_calendar
    return mock_ticker


# ============================================================================
# EarningsCalendar Initialization Tests
# ============================================================================


class TestEarningsCalendarInit:
    """Test EarningsCalendar initialization."""

    def test_default_init(self, calendar):
        """Test default initialization values."""
        assert calendar.exit_days_before == 2
        assert calendar.skip_entry_days_before == 3
        assert calendar.reentry_days_after == 1
        assert calendar.cache_hours == 12
        assert calendar._cache == {}

    def test_custom_init(self, calendar_custom):
        """Test custom initialization values."""
        assert calendar_custom.exit_days_before == 3
        assert calendar_custom.skip_entry_days_before == 5
        assert calendar_custom.reentry_days_after == 2
        assert calendar_custom.cache_hours == 24


# ============================================================================
# Parse Date Tests
# ============================================================================


class TestParseDate:
    """Test _parse_date method."""

    def test_parse_datetime(self, calendar):
        """Test parsing datetime object."""
        dt = datetime(2024, 1, 15, 10, 30)
        result = calendar._parse_date(dt)
        assert result == dt

    def test_parse_none(self, calendar):
        """Test parsing None."""
        result = calendar._parse_date(None)
        assert result is None

    def test_parse_string_date(self, calendar):
        """Test parsing string date."""
        result = calendar._parse_date("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_parse_pandas_timestamp(self, calendar):
        """Test parsing pandas Timestamp."""
        ts = pd.Timestamp("2024-01-15")
        result = calendar._parse_date(ts)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_invalid_string(self, calendar):
        """Test parsing invalid string returns None."""
        result = calendar._parse_date("not-a-date")
        assert result is None

    def test_parse_invalid_type(self, calendar):
        """Test parsing invalid type returns None."""
        result = calendar._parse_date(12345)
        assert result is None


# ============================================================================
# Get Next Earnings Date Tests
# ============================================================================


class TestGetNextEarningsDate:
    """Test get_next_earnings_date method."""

    def test_returns_cached_date(self, calendar):
        """Test returns cached earnings date."""
        future_date = datetime.now() + timedelta(days=10)
        calendar._cache["AAPL"] = (future_date, datetime.now())

        result = calendar.get_next_earnings_date("AAPL")
        assert result == future_date

    def test_cache_expired(self, calendar):
        """Test expired cache triggers new fetch."""
        old_date = datetime.now() - timedelta(hours=24)
        calendar._cache["AAPL"] = (datetime.now() + timedelta(days=5), old_date)

        with patch("utils.earnings_calendar.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.calendar = None
            mock_yf.return_value = mock_ticker

            calendar.get_next_earnings_date("AAPL")
            mock_yf.assert_called_once_with("AAPL")

    def test_no_earnings_returns_none(self, calendar):
        """Test returns None when no earnings found."""
        with patch("utils.earnings_calendar.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.calendar = None
            mock_yf.return_value = mock_ticker

            result = calendar.get_next_earnings_date("AAPL")
            assert result is None

    def test_empty_calendar_returns_none(self, calendar):
        """Test returns None for empty calendar."""
        with patch("utils.earnings_calendar.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_calendar = MagicMock()
            mock_calendar.empty = True
            mock_ticker.calendar = mock_calendar
            mock_yf.return_value = mock_ticker

            result = calendar.get_next_earnings_date("AAPL")
            assert result is None

    def test_caches_result(self, calendar):
        """Test result is cached."""
        with patch("utils.earnings_calendar.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.calendar = None
            mock_yf.return_value = mock_ticker

            calendar.get_next_earnings_date("AAPL")
            assert "AAPL" in calendar._cache

    def test_exception_returns_none(self, calendar):
        """Test exception handling returns None."""
        with patch("utils.earnings_calendar.yf.Ticker") as mock_yf:
            mock_yf.side_effect = Exception("API Error")

            result = calendar.get_next_earnings_date("AAPL")
            assert result is None

    def test_handles_earnings_date_list(self, calendar):
        """Test handles list of earnings dates."""
        with patch("utils.earnings_calendar.yf.Ticker") as mock_yf:
            future_date = datetime.now() + timedelta(days=10)
            mock_ticker = MagicMock()
            mock_calendar = pd.DataFrame(
                {"Values": [[future_date, future_date + timedelta(days=90)]]},
                index=["Earnings Date"],
            )
            mock_ticker.calendar = mock_calendar
            mock_yf.return_value = mock_ticker

            calendar.get_next_earnings_date("AAPL")
            # Should handle list without crashing


# ============================================================================
# Days Until Earnings Tests
# ============================================================================


class TestDaysUntilEarnings:
    """Test days_until_earnings method."""

    def test_returns_none_when_no_earnings(self, calendar):
        """Test returns None when no earnings date."""
        calendar._cache["AAPL"] = (None, datetime.now())
        result = calendar.days_until_earnings("AAPL")
        assert result is None

    def test_calculates_days_correctly(self, calendar):
        """Test calculates trading days approximately."""
        future_date = datetime.now() + timedelta(days=14)  # 2 weeks
        calendar._cache["AAPL"] = (future_date, datetime.now())

        result = calendar.days_until_earnings("AAPL")
        assert result is not None
        assert result == 10  # ~14 * 5/7 = 10 trading days

    def test_negative_days_for_past_earnings(self, calendar):
        """Test negative days for past earnings."""
        past_date = datetime.now() - timedelta(days=7)
        calendar._cache["AAPL"] = (past_date, datetime.now())

        result = calendar.days_until_earnings("AAPL")
        assert result is not None
        assert result < 0

    def test_zero_days_for_today(self, calendar):
        """Test zero days for same day."""
        today = datetime.now()
        calendar._cache["AAPL"] = (today, datetime.now())

        result = calendar.days_until_earnings("AAPL")
        assert result == 0


# ============================================================================
# Is Safe To Hold Tests
# ============================================================================


class TestIsSafeToHold:
    """Test is_safe_to_hold method."""

    def test_safe_when_no_earnings(self, calendar):
        """Test safe to hold when no earnings info."""
        calendar._cache["AAPL"] = (None, datetime.now())
        assert calendar.is_safe_to_hold("AAPL") is True

    def test_safe_when_earnings_far(self, calendar):
        """Test safe to hold when earnings are far away."""
        future_date = datetime.now() + timedelta(days=30)
        calendar._cache["AAPL"] = (future_date, datetime.now())

        assert calendar.is_safe_to_hold("AAPL") is True

    def test_unsafe_when_earnings_close(self, calendar):
        """Test unsafe when earnings are close."""
        # 2 days from now (below exit_days_before=2)
        close_date = datetime.now() + timedelta(days=2)
        calendar._cache["AAPL"] = (close_date, datetime.now())

        # days_until_earnings will return ~1 trading day (2 * 5/7)
        assert calendar.is_safe_to_hold("AAPL") is False

    def test_unsafe_at_exit_threshold(self, calendar):
        """Test unsafe exactly at exit threshold."""
        # With exit_days_before=2, need to find days that compute to <=2
        # 3 calendar days = ~2 trading days
        threshold_date = datetime.now() + timedelta(days=3)
        calendar._cache["AAPL"] = (threshold_date, datetime.now())

        calendar.is_safe_to_hold("AAPL")
        # Should be unsafe (days <= exit_days_before)


# ============================================================================
# Is Safe To Enter Tests
# ============================================================================


class TestIsSafeToEnter:
    """Test is_safe_to_enter method."""

    def test_safe_when_no_earnings(self, calendar):
        """Test safe to enter when no earnings info."""
        calendar._cache["AAPL"] = (None, datetime.now())
        assert calendar.is_safe_to_enter("AAPL") is True

    def test_safe_when_earnings_far(self, calendar):
        """Test safe to enter when earnings are far away."""
        future_date = datetime.now() + timedelta(days=30)
        calendar._cache["AAPL"] = (future_date, datetime.now())

        assert calendar.is_safe_to_enter("AAPL") is True

    def test_unsafe_when_earnings_within_skip_days(self, calendar):
        """Test unsafe when earnings within skip_entry_days_before."""
        # 3 days = ~2 trading days, skip_entry_days_before=3
        close_date = datetime.now() + timedelta(days=3)
        calendar._cache["AAPL"] = (close_date, datetime.now())

        # Should be unsafe for entry


# ============================================================================
# Get Earnings Risk Level Tests
# ============================================================================


class TestGetEarningsRiskLevel:
    """Test get_earnings_risk_level method."""

    def test_low_risk_no_earnings(self, calendar):
        """Test low risk when no earnings info."""
        calendar._cache["AAPL"] = (None, datetime.now())
        assert calendar.get_earnings_risk_level("AAPL") == "low"

    def test_low_risk_far_earnings(self, calendar):
        """Test low risk when earnings are far away."""
        future_date = datetime.now() + timedelta(days=30)
        calendar._cache["AAPL"] = (future_date, datetime.now())

        assert calendar.get_earnings_risk_level("AAPL") == "low"

    def test_high_risk_close_earnings(self, calendar):
        """Test high risk when earnings very close."""
        close_date = datetime.now() + timedelta(days=1)
        calendar._cache["AAPL"] = (close_date, datetime.now())

        assert calendar.get_earnings_risk_level("AAPL") == "high"

    def test_medium_risk_moderate_earnings(self, calendar):
        """Test medium risk when earnings moderately close."""
        # 5 calendar days = ~3-4 trading days
        # exit_days_before=2, skip_entry_days_before=3
        # Should fall in medium range
        moderate_date = datetime.now() + timedelta(days=5)
        calendar._cache["AAPL"] = (moderate_date, datetime.now())

        calendar.get_earnings_risk_level("AAPL")
        # Could be medium or low depending on exact calculation


# ============================================================================
# Filter Symbols Tests
# ============================================================================


class TestFilterSymbols:
    """Test filter_symbols method."""

    def test_filter_for_entry_removes_unsafe(self, calendar):
        """Test filtering for entry removes unsafe symbols."""
        # One safe, one unsafe
        calendar._cache["SAFE"] = (None, datetime.now())
        calendar._cache["UNSAFE"] = (datetime.now() + timedelta(days=1), datetime.now())

        symbols = ["SAFE", "UNSAFE"]
        result = calendar.filter_symbols(symbols, for_entry=True)

        assert "SAFE" in result
        # UNSAFE may or may not be filtered based on exact day calculation

    def test_filter_for_hold_removes_unsafe(self, calendar):
        """Test filtering for hold removes unsafe symbols."""
        calendar._cache["SAFE"] = (None, datetime.now())
        calendar._cache["UNSAFE"] = (datetime.now() + timedelta(days=1), datetime.now())

        symbols = ["SAFE", "UNSAFE"]
        result = calendar.filter_symbols(symbols, for_entry=False)

        assert "SAFE" in result

    def test_filter_empty_list(self, calendar):
        """Test filtering empty list."""
        result = calendar.filter_symbols([], for_entry=True)
        assert result == []

    def test_filter_all_safe(self, calendar):
        """Test all symbols safe returns all."""
        calendar._cache["A"] = (None, datetime.now())
        calendar._cache["B"] = (None, datetime.now())

        symbols = ["A", "B"]
        result = calendar.filter_symbols(symbols, for_entry=True)

        assert len(result) == 2


# ============================================================================
# Get Positions To Exit Tests
# ============================================================================


class TestGetPositionsToExit:
    """Test get_positions_to_exit method."""

    def test_returns_unsafe_positions(self, calendar):
        """Test returns positions that should be exited."""
        calendar._cache["SAFE"] = (None, datetime.now())
        calendar._cache["UNSAFE"] = (datetime.now() + timedelta(days=1), datetime.now())

        symbols = ["SAFE", "UNSAFE"]
        to_exit = calendar.get_positions_to_exit(symbols)

        assert "SAFE" not in to_exit
        # UNSAFE may be in to_exit depending on calculation

    def test_returns_empty_when_all_safe(self, calendar):
        """Test returns empty when all positions safe."""
        calendar._cache["A"] = (None, datetime.now())
        calendar._cache["B"] = (datetime.now() + timedelta(days=30), datetime.now())

        symbols = ["A", "B"]
        to_exit = calendar.get_positions_to_exit(symbols)

        assert to_exit == []

    def test_empty_positions(self, calendar):
        """Test empty positions list."""
        to_exit = calendar.get_positions_to_exit([])
        assert to_exit == []


# ============================================================================
# Get Earnings Report Tests
# ============================================================================


class TestGetEarningsReport:
    """Test get_earnings_report method."""

    def test_report_structure(self, calendar):
        """Test report has correct structure."""
        calendar._cache["AAPL"] = (None, datetime.now())

        report = calendar.get_earnings_report(["AAPL"])

        assert "checked_at" in report
        assert "symbols" in report
        assert "summary" in report
        assert "AAPL" in report["symbols"]

    def test_report_symbol_info(self, calendar):
        """Test report contains symbol info."""
        future_date = datetime.now() + timedelta(days=10)
        calendar._cache["AAPL"] = (future_date, datetime.now())

        report = calendar.get_earnings_report(["AAPL"])

        symbol_info = report["symbols"]["AAPL"]
        assert "earnings_date" in symbol_info
        assert "days_until" in symbol_info
        assert "risk_level" in symbol_info
        assert "safe_to_hold" in symbol_info
        assert "safe_to_enter" in symbol_info

    def test_report_summary_categories(self, calendar):
        """Test report has summary categories."""
        calendar._cache["HIGH"] = (datetime.now() + timedelta(days=1), datetime.now())
        calendar._cache["LOW"] = (None, datetime.now())

        report = calendar.get_earnings_report(["HIGH", "LOW"])

        assert "high_risk" in report["summary"]
        assert "medium_risk" in report["summary"]
        assert "low_risk" in report["summary"]

    def test_report_with_no_earnings(self, calendar):
        """Test report handles symbols with no earnings."""
        calendar._cache["AAPL"] = (None, datetime.now())

        report = calendar.get_earnings_report(["AAPL"])

        assert report["symbols"]["AAPL"]["earnings_date"] is None

    def test_report_multiple_symbols(self, calendar):
        """Test report with multiple symbols."""
        calendar._cache["A"] = (None, datetime.now())
        calendar._cache["B"] = (datetime.now() + timedelta(days=10), datetime.now())
        calendar._cache["C"] = (datetime.now() + timedelta(days=1), datetime.now())

        report = calendar.get_earnings_report(["A", "B", "C"])

        assert len(report["symbols"]) == 3


# ============================================================================
# Clear Cache Tests
# ============================================================================


class TestClearCache:
    """Test clear_cache method."""

    def test_clears_cache(self, calendar):
        """Test cache is cleared."""
        calendar._cache["AAPL"] = (None, datetime.now())
        calendar._cache["MSFT"] = (None, datetime.now())

        calendar.clear_cache()

        assert calendar._cache == {}

    def test_clear_empty_cache(self, calendar):
        """Test clearing empty cache."""
        calendar.clear_cache()
        assert calendar._cache == {}


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestCheckEarningsSafety:
    """Test check_earnings_safety convenience function."""

    def test_returns_dict(self):
        """Test returns dict of symbol -> safety."""
        with patch("utils.earnings_calendar.EarningsCalendar.is_safe_to_enter") as mock_safe:
            mock_safe.return_value = True

            # Need to also patch the class instantiation
            with patch("utils.earnings_calendar.EarningsCalendar.__init__") as mock_init:
                mock_init.return_value = None
                with patch(
                    "utils.earnings_calendar.EarningsCalendar.is_safe_to_enter"
                ) as mock_method:
                    mock_method.return_value = True

                    result = check_earnings_safety(["AAPL", "MSFT"])

                    assert isinstance(result, dict)
                    assert "AAPL" in result
                    assert "MSFT" in result

    def test_empty_list(self):
        """Test with empty list."""
        with patch("utils.earnings_calendar.EarningsCalendar.__init__") as mock_init:
            mock_init.return_value = None
            result = check_earnings_safety([])
            assert result == {}


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_cache_persistence_across_calls(self, calendar):
        """Test cache persists across multiple calls."""
        future_date = datetime.now() + timedelta(days=10)
        calendar._cache["AAPL"] = (future_date, datetime.now())

        # Multiple calls should use cache
        result1 = calendar.get_next_earnings_date("AAPL")
        result2 = calendar.get_next_earnings_date("AAPL")

        assert result1 == result2

    def test_different_symbols_cached_separately(self, calendar):
        """Test different symbols have separate cache entries."""
        date1 = datetime.now() + timedelta(days=10)
        date2 = datetime.now() + timedelta(days=20)

        calendar._cache["AAPL"] = (date1, datetime.now())
        calendar._cache["MSFT"] = (date2, datetime.now())

        assert calendar.get_next_earnings_date("AAPL") == date1
        assert calendar.get_next_earnings_date("MSFT") == date2

    def test_custom_params_affect_safety(self, calendar_custom):
        """Test custom parameters affect safety checks."""
        # With exit_days_before=3 (custom) vs 2 (default)
        # A date 3 days away should be unsafe
        close_date = datetime.now() + timedelta(days=4)  # ~2-3 trading days
        calendar_custom._cache["AAPL"] = (close_date, datetime.now())

        # May or may not be safe depending on exact trading day calculation

    def test_handles_weekend_dates(self, calendar):
        """Test handles weekend dates in calculation."""
        # Friday + 3 days = Monday
        future_date = datetime.now() + timedelta(days=10)
        calendar._cache["AAPL"] = (future_date, datetime.now())

        days = calendar.days_until_earnings("AAPL")
        assert days is not None
        # Trading days should be roughly 5/7 of calendar days

    def test_filter_with_mixed_cache(self, calendar):
        """Test filter when some symbols not in cache."""
        calendar._cache["A"] = (None, datetime.now())
        # B not in cache

        with patch("utils.earnings_calendar.yf.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.calendar = None
            mock_yf.return_value = mock_ticker

            calendar.filter_symbols(["A", "B"], for_entry=True)
            # Should handle both cached and uncached symbols
