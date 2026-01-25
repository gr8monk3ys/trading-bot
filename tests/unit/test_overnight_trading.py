#!/usr/bin/env python3
"""
Unit tests for overnight trading functionality (Blue Ocean ATS).

Tests for:
- TradingSession enum
- Overnight session detection
- Weekend/holiday handling
- Symbol overnight tradability checks
- Position size adjustments
"""

import pytest
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytz

from utils.extended_hours import (
    ExtendedHoursManager,
    TradingSession,
    format_session_info,
    ET,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_broker():
    """Mock broker for testing."""
    broker = MagicMock()
    broker.submit_order_advanced = AsyncMock(return_value={"order_id": "test123"})
    broker.get_asset = AsyncMock(return_value={
        "symbol": "AAPL",
        "overnight_tradeable": True,
        "overnight_halted": False,
    })
    return broker


@pytest.fixture
def manager(mock_broker):
    """Default extended hours manager with overnight enabled."""
    return ExtendedHoursManager(mock_broker, enable_overnight=True)


@pytest.fixture
def manager_overnight_disabled(mock_broker):
    """Manager with overnight trading disabled."""
    return ExtendedHoursManager(
        mock_broker,
        enable_overnight=False,
        enable_pre_market=True,
        enable_after_hours=True,
    )


@pytest.fixture
def manager_no_broker():
    """Manager without broker (for non-API tests)."""
    return ExtendedHoursManager(enable_overnight=True)


# ============================================================================
# TradingSession Enum Tests
# ============================================================================


class TestTradingSessionEnum:
    """Test TradingSession enum values."""

    def test_enum_values(self):
        """Test that all session types have correct string values."""
        assert TradingSession.CLOSED.value == "closed"
        assert TradingSession.PRE_MARKET.value == "pre_market"
        assert TradingSession.REGULAR.value == "regular"
        assert TradingSession.AFTER_HOURS.value == "after_hours"
        assert TradingSession.OVERNIGHT.value == "overnight"

    def test_enum_unique(self):
        """Test that all enum values are unique."""
        values = [session.value for session in TradingSession]
        assert len(values) == len(set(values))


# ============================================================================
# Overnight Session Detection Tests
# ============================================================================


class TestOvernightSessionDetection:
    """Test overnight session detection."""

    def test_overnight_session_late_night(self, manager_no_broker):
        """Test overnight detection at 10 PM."""
        # Tuesday 10 PM ET - overnight session
        dt = ET.localize(datetime(2024, 1, 9, 22, 0))  # Tuesday 10 PM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.OVERNIGHT

    def test_overnight_session_early_morning(self, manager_no_broker):
        """Test overnight detection at 2 AM."""
        # Wednesday 2 AM ET - overnight session
        dt = ET.localize(datetime(2024, 1, 10, 2, 0))  # Wednesday 2 AM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.OVERNIGHT

    def test_overnight_session_boundary_start(self, manager_no_broker):
        """Test overnight session starts at 8 PM."""
        # Tuesday 8:00 PM ET - overnight starts
        dt = ET.localize(datetime(2024, 1, 9, 20, 0))
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.OVERNIGHT

    def test_overnight_session_boundary_end(self, manager_no_broker):
        """Test overnight session ends at 4 AM."""
        # Wednesday 3:59 AM ET - still overnight
        dt = ET.localize(datetime(2024, 1, 10, 3, 59))
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.OVERNIGHT

        # Wednesday 4:00 AM ET - pre-market starts
        dt = ET.localize(datetime(2024, 1, 10, 4, 0))
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.PRE_MARKET

    def test_overnight_disabled_returns_closed(self, manager_overnight_disabled):
        """Test overnight returns CLOSED when disabled."""
        dt = ET.localize(datetime(2024, 1, 9, 22, 0))  # Tuesday 10 PM
        session = manager_overnight_disabled.get_current_session(dt)
        assert session == TradingSession.CLOSED


# ============================================================================
# Weekend Handling Tests
# ============================================================================


class TestWeekendHandling:
    """Test weekend and edge cases for overnight trading."""

    def test_saturday_closed(self, manager_no_broker):
        """Test Saturday is fully closed."""
        dt = ET.localize(datetime(2024, 1, 13, 12, 0))  # Saturday noon
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.CLOSED

    def test_saturday_night_closed(self, manager_no_broker):
        """Test Saturday night is closed."""
        dt = ET.localize(datetime(2024, 1, 13, 22, 0))  # Saturday 10 PM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.CLOSED

    def test_sunday_before_8pm_closed(self, manager_no_broker):
        """Test Sunday before 8 PM is closed."""
        dt = ET.localize(datetime(2024, 1, 14, 19, 59))  # Sunday 7:59 PM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.CLOSED

    def test_sunday_8pm_overnight_starts(self, manager_no_broker):
        """Test Sunday 8 PM starts overnight session."""
        dt = ET.localize(datetime(2024, 1, 14, 20, 0))  # Sunday 8 PM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.OVERNIGHT

    def test_friday_after_hours_end(self, manager_no_broker):
        """Test Friday after-hours ends normally."""
        # Friday 7:30 PM - still after-hours
        dt = ET.localize(datetime(2024, 1, 12, 19, 30))
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.AFTER_HOURS

    def test_friday_no_overnight(self, manager_no_broker):
        """Test Friday night has no overnight (market closes for weekend)."""
        # Friday 8:00 PM - should be closed (no overnight into Saturday)
        dt = ET.localize(datetime(2024, 1, 12, 20, 0))
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.CLOSED


# ============================================================================
# Regular Session Detection Tests (backward compatibility)
# ============================================================================


class TestRegularSessionDetection:
    """Test regular session detection still works."""

    def test_pre_market(self, manager_no_broker):
        """Test pre-market detection."""
        dt = ET.localize(datetime(2024, 1, 9, 6, 0))  # Tuesday 6 AM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.PRE_MARKET

    def test_regular_hours(self, manager_no_broker):
        """Test regular hours detection."""
        dt = ET.localize(datetime(2024, 1, 9, 11, 0))  # Tuesday 11 AM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.REGULAR

    def test_after_hours(self, manager_no_broker):
        """Test after-hours detection."""
        dt = ET.localize(datetime(2024, 1, 9, 17, 0))  # Tuesday 5 PM
        session = manager_no_broker.get_current_session(dt)
        assert session == TradingSession.AFTER_HOURS


# ============================================================================
# is_market_open Tests
# ============================================================================


class TestIsMarketOpen:
    """Test is_market_open method with overnight support."""

    def test_overnight_open_when_enabled(self, manager_no_broker):
        """Test overnight is open when enabled."""
        dt = ET.localize(datetime(2024, 1, 9, 22, 0))
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            assert manager_no_broker.is_market_open() is True

    def test_overnight_closed_when_disabled(self, manager_overnight_disabled):
        """Test overnight is closed when disabled."""
        # Even if we're in overnight time, disabled manager returns closed
        dt = ET.localize(datetime(2024, 1, 9, 22, 0))
        assert manager_overnight_disabled.is_market_open() is False

    def test_regular_hours_always_open(self, manager_no_broker):
        """Test regular hours is always open."""
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.REGULAR):
            assert manager_no_broker.is_market_open() is True
            assert manager_no_broker.is_market_open(include_extended=False) is True
            assert manager_no_broker.is_market_open(include_overnight=False) is True


# ============================================================================
# is_overnight Tests
# ============================================================================


class TestIsOvernight:
    """Test is_overnight method."""

    def test_is_overnight_true(self, manager_no_broker):
        """Test is_overnight returns True during overnight."""
        dt = ET.localize(datetime(2024, 1, 9, 22, 0))
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            assert manager_no_broker.is_overnight() is True

    def test_is_overnight_false_during_day(self, manager_no_broker):
        """Test is_overnight returns False during day."""
        dt = ET.localize(datetime(2024, 1, 9, 11, 0))
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.REGULAR):
            assert manager_no_broker.is_overnight() is False


# ============================================================================
# Overnight Tradability Tests
# ============================================================================


class TestOvernightTradability:
    """Test overnight tradability checks."""

    @pytest.mark.asyncio
    async def test_can_trade_overnight_success(self, manager):
        """Test successful overnight trade eligibility."""
        # Set session to overnight
        with patch.object(manager, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            can_trade, reason = await manager.can_trade_overnight("AAPL")
            assert can_trade is True
            assert "overnight" in reason.lower()

    @pytest.mark.asyncio
    async def test_cannot_trade_overnight_wrong_session(self, manager):
        """Test cannot trade overnight outside overnight session."""
        with patch.object(manager, 'get_current_session', return_value=TradingSession.REGULAR):
            can_trade, reason = await manager.can_trade_overnight("AAPL")
            assert can_trade is False
            assert "Not in overnight session" in reason

    @pytest.mark.asyncio
    async def test_cannot_trade_overnight_disabled(self, manager_overnight_disabled):
        """Test cannot trade overnight when disabled."""
        with patch.object(manager_overnight_disabled, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            can_trade, reason = await manager_overnight_disabled.can_trade_overnight("AAPL")
            assert can_trade is False
            assert "disabled" in reason.lower()

    @pytest.mark.asyncio
    async def test_cannot_trade_symbol_not_overnight_enabled(self, manager):
        """Test cannot trade symbol that doesn't support overnight."""
        # Mock broker returns symbol not overnight tradeable
        manager.broker.get_asset = AsyncMock(return_value={
            "symbol": "XYZ",
            "overnight_tradeable": False,
            "overnight_halted": False,
        })

        with patch.object(manager, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            can_trade, reason = await manager.can_trade_overnight("XYZ")
            assert can_trade is False
            assert "not available" in reason.lower()

    @pytest.mark.asyncio
    async def test_check_overnight_tradeable(self, manager):
        """Test check_overnight_tradeable method."""
        result = await manager.check_overnight_tradeable(manager.broker, "AAPL")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_overnight_tradeable_halted(self, manager):
        """Test check_overnight_tradeable when halted."""
        manager.broker.get_asset = AsyncMock(return_value={
            "symbol": "AAPL",
            "overnight_tradeable": True,
            "overnight_halted": True,  # Halted
        })
        result = await manager.check_overnight_tradeable(manager.broker, "AAPL")
        assert result is False


# ============================================================================
# Position Size Adjustment Tests
# ============================================================================


class TestPositionSizeAdjustments:
    """Test position size adjustments for overnight trading."""

    def test_adjust_position_size_overnight(self, manager):
        """Test overnight position size adjustment (30% default)."""
        adjusted = manager.adjust_position_size_for_overnight(10000.0)
        assert adjusted == 3000.0

    def test_adjust_position_size_overnight_zero(self, manager):
        """Test overnight adjustment with zero position."""
        adjusted = manager.adjust_position_size_for_overnight(0.0)
        assert adjusted == 0.0

    def test_get_position_size_multiplier_overnight(self, manager_no_broker):
        """Test position size multiplier during overnight."""
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            multiplier = manager_no_broker.get_position_size_multiplier()
            assert multiplier == 0.3

    def test_get_position_size_multiplier_regular(self, manager_no_broker):
        """Test position size multiplier during regular hours."""
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.REGULAR):
            multiplier = manager_no_broker.get_position_size_multiplier()
            assert multiplier == 1.0

    def test_get_position_size_multiplier_extended(self, manager_no_broker):
        """Test position size multiplier during extended hours."""
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.PRE_MARKET):
            multiplier = manager_no_broker.get_position_size_multiplier()
            assert multiplier == 0.5

    def test_get_position_size_multiplier_closed(self, manager_no_broker):
        """Test position size multiplier when closed."""
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.CLOSED):
            multiplier = manager_no_broker.get_position_size_multiplier()
            assert multiplier == 0.0


# ============================================================================
# Session Info Tests
# ============================================================================


class TestSessionInfo:
    """Test session info for overnight."""

    def test_overnight_session_info(self, manager_no_broker):
        """Test get_session_info returns overnight details."""
        dt = ET.localize(datetime(2024, 1, 9, 22, 0))
        with patch.object(manager_no_broker, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            info = manager_no_broker.get_session_info(dt)

        assert info["session"] == "overnight"
        assert info["is_overnight"] is True
        assert info["is_extended"] is False
        assert info["is_regular_hours"] is False
        assert "Blue Ocean ATS" in info["session_name"]
        assert "notes" in info

    def test_format_overnight_session_info(self):
        """Test format_session_info for overnight session."""
        info = {
            "session": "overnight",
            "session_name": "Overnight (Blue Ocean ATS)",
            "current_time": "2024-01-09 22:00:00 EST",
            "start_time": "8:00 PM ET",
            "end_time": "4:00 AM ET",
            "is_overnight": True,
            "is_extended": False,
            "is_regular_hours": False,
            "can_trade": True,
            "liquidity": "Very Low",
            "volatility": "Low-Medium",
            "position_size_adj": "30% of regular",
            "recommended_strategy": "Low-risk positions",
            "notes": ["Trades execute via Blue Ocean ATS"],
        }
        output = format_session_info(info)
        assert "OVERNIGHT TRADING" in output
        assert "Blue Ocean ATS" in output
        assert "Very Low" in output


# ============================================================================
# Extended Hours Strategies Tests
# ============================================================================


class TestOvernightStrategies:
    """Test strategy recommendations for overnight."""

    def test_overnight_strategies(self, manager_no_broker):
        """Test overnight strategy recommendations."""
        strategies = manager_no_broker.get_extended_hours_strategies("overnight")
        assert strategies["primary"] == "low_risk_positioning"
        assert "Blue Ocean ATS" in strategies["venue"]
        assert "tips" in strategies
        assert len(strategies["tips"]) > 0
        assert any("limit" in tip.lower() for tip in strategies["tips"])

    def test_overnight_strategies_enum(self, manager_no_broker):
        """Test overnight strategies with enum input."""
        strategies = manager_no_broker.get_extended_hours_strategies(TradingSession.OVERNIGHT)
        assert strategies["primary"] == "low_risk_positioning"


# ============================================================================
# Execute Overnight Trade Tests
# ============================================================================


class TestExecuteOvernightTrade:
    """Test execute_extended_hours_trade during overnight."""

    @pytest.mark.asyncio
    async def test_execute_overnight_trade_checks_tradability(self, manager):
        """Test overnight trade execution checks tradability."""
        with patch.object(manager, 'get_current_session', return_value=TradingSession.OVERNIGHT):
            with patch.object(manager, 'can_trade_overnight', return_value=(True, "OK")) as mock_check:
                with patch.object(manager, 'get_extended_hours_quote', return_value=None):
                    result = await manager.execute_extended_hours_trade("AAPL", "buy", 10)
                    mock_check.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_execute_trade_rejected_outside_sessions(self, manager):
        """Test trade rejected when not in valid session."""
        with patch.object(manager, 'get_current_session', return_value=TradingSession.CLOSED):
            result = await manager.execute_extended_hours_trade("AAPL", "buy", 10)
            assert result is None


# ============================================================================
# Timezone Handling Tests
# ============================================================================


class TestTimezoneHandling:
    """Test timezone handling for overnight sessions."""

    def test_naive_datetime_localized(self, manager_no_broker):
        """Test naive datetime is localized to ET."""
        naive_dt = datetime(2024, 1, 9, 22, 0)  # No timezone
        session = manager_no_broker.get_current_session(naive_dt)
        assert session == TradingSession.OVERNIGHT

    def test_utc_datetime_converted(self, manager_no_broker):
        """Test UTC datetime is converted to ET."""
        utc = pytz.UTC
        utc_dt = utc.localize(datetime(2024, 1, 10, 3, 0))  # 3 AM UTC = 10 PM ET
        session = manager_no_broker.get_current_session(utc_dt)
        assert session == TradingSession.OVERNIGHT

    def test_pacific_timezone_converted(self, manager_no_broker):
        """Test Pacific time is converted to ET."""
        pacific = pytz.timezone("US/Pacific")
        pacific_dt = pacific.localize(datetime(2024, 1, 9, 19, 0))  # 7 PM PT = 10 PM ET
        session = manager_no_broker.get_current_session(pacific_dt)
        assert session == TradingSession.OVERNIGHT


# ============================================================================
# Manager Initialization Tests
# ============================================================================


class TestManagerInitialization:
    """Test ExtendedHoursManager initialization with overnight."""

    def test_default_overnight_enabled(self, mock_broker):
        """Test overnight is enabled by default."""
        manager = ExtendedHoursManager(mock_broker)
        assert manager.enable_overnight is True

    def test_overnight_can_be_disabled(self, mock_broker):
        """Test overnight can be disabled."""
        manager = ExtendedHoursManager(mock_broker, enable_overnight=False)
        assert manager.enable_overnight is False

    def test_overnight_params_exist(self, manager):
        """Test overnight params are initialized."""
        assert "position_size_multiplier" in manager.overnight_params
        assert "max_spread_pct" in manager.overnight_params
        assert manager.overnight_params["position_size_multiplier"] == 0.3

    def test_no_broker_initialization(self):
        """Test manager works without broker."""
        manager = ExtendedHoursManager(enable_overnight=True)
        assert manager.broker is None
        assert manager.enable_overnight is True


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for overnight trading."""

    def test_all_sessions_disabled(self, mock_broker):
        """Test behavior when all sessions disabled."""
        manager = ExtendedHoursManager(
            mock_broker,
            enable_pre_market=False,
            enable_after_hours=False,
            enable_overnight=False,
        )

        # Any time outside regular hours should be CLOSED
        dt = ET.localize(datetime(2024, 1, 9, 22, 0))
        session = manager.get_current_session(dt)
        assert session == TradingSession.CLOSED

    def test_market_open_all_sessions_disabled(self, mock_broker):
        """Test is_market_open when all extended sessions disabled."""
        manager = ExtendedHoursManager(
            mock_broker,
            enable_pre_market=False,
            enable_after_hours=False,
            enable_overnight=False,
        )

        # During overnight time, market should not be open
        with patch.object(manager, 'get_current_session', return_value=TradingSession.CLOSED):
            assert manager.is_market_open() is False

    def test_overnight_params_custom_multiplier(self, mock_broker):
        """Test custom overnight position size multiplier."""
        manager = ExtendedHoursManager(mock_broker)
        manager.overnight_params["position_size_multiplier"] = 0.5
        adjusted = manager.adjust_position_size_for_overnight(10000.0)
        assert adjusted == 5000.0
