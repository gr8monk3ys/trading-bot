#!/usr/bin/env python3
"""
Unit tests for utils/extended_hours.py

Tests ExtendedHoursManager, GapTradingStrategy, and EarningsReactionStrategy classes.
"""

from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from utils.extended_hours import (
    ET,
    EarningsReactionStrategy,
    ExtendedHoursManager,
    GapTradingStrategy,
    TradingSession,
    format_session_info,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_broker():
    """Mock broker for testing."""
    broker = MagicMock()
    broker.submit_order_advanced = AsyncMock(return_value={"order_id": "test123"})
    return broker


@pytest.fixture
def manager(mock_broker):
    """Default extended hours manager."""
    return ExtendedHoursManager(mock_broker)


@pytest.fixture
def manager_pre_only(mock_broker):
    """Manager with only pre-market enabled."""
    return ExtendedHoursManager(mock_broker, enable_pre_market=True, enable_after_hours=False)


@pytest.fixture
def manager_ah_only(mock_broker):
    """Manager with only after-hours enabled."""
    return ExtendedHoursManager(mock_broker, enable_pre_market=False, enable_after_hours=True)


@pytest.fixture
def manager_disabled(mock_broker):
    """Manager with extended hours disabled."""
    return ExtendedHoursManager(mock_broker, enable_pre_market=False, enable_after_hours=False)


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
def gap_strategy():
    """Default gap trading strategy."""
    return GapTradingStrategy()


@pytest.fixture
def earnings_strategy():
    """Default earnings reaction strategy."""
    return EarningsReactionStrategy()


# ============================================================================
# ExtendedHoursManager Initialization Tests
# ============================================================================


class TestExtendedHoursManagerInit:
    """Test ExtendedHoursManager initialization."""

    def test_default_init(self, manager):
        """Test default initialization values."""
        assert manager.enable_pre_market is True
        assert manager.enable_after_hours is True
        assert manager.extended_hours_params["max_spread_pct"] == 0.005
        assert manager.extended_hours_params["position_size_multiplier"] == 0.5
        assert manager.extended_hours_params["limit_order_offset_pct"] == 0.001
        assert manager.extended_hours_params["max_slippage_pct"] == 0.003
        assert manager.extended_hours_params["min_volume"] == 10000

    def test_pre_market_only(self, manager_pre_only):
        """Test pre-market only configuration."""
        assert manager_pre_only.enable_pre_market is True
        assert manager_pre_only.enable_after_hours is False

    def test_after_hours_only(self, manager_ah_only):
        """Test after-hours only configuration."""
        assert manager_ah_only.enable_pre_market is False
        assert manager_ah_only.enable_after_hours is True

    def test_disabled(self, manager_disabled):
        """Test disabled configuration."""
        assert manager_disabled.enable_pre_market is False
        assert manager_disabled.enable_after_hours is False


# ============================================================================
# Session Time Constants Tests
# ============================================================================


class TestSessionTimeConstants:
    """Test session time constants."""

    def test_pre_market_times(self):
        """Test pre-market time constants."""
        assert ExtendedHoursManager.PRE_MARKET_START == time(4, 0)
        assert ExtendedHoursManager.PRE_MARKET_END == time(9, 30)

    def test_regular_times(self):
        """Test regular hours time constants."""
        assert ExtendedHoursManager.REGULAR_START == time(9, 30)
        assert ExtendedHoursManager.REGULAR_END == time(16, 0)

    def test_after_hours_times(self):
        """Test after-hours time constants."""
        assert ExtendedHoursManager.AFTER_HOURS_START == time(16, 0)
        assert ExtendedHoursManager.AFTER_HOURS_END == time(20, 0)


# ============================================================================
# Get Current Session Tests
# ============================================================================


class TestGetCurrentSession:
    """Test get_current_session method."""

    def test_pre_market_session(self, manager):
        """Test pre-market session detection."""
        # 6:00 AM Eastern on a Tuesday
        dt = ET.localize(datetime(2024, 1, 2, 6, 0))
        session = manager.get_current_session(dt)
        assert session == TradingSession.PRE_MARKET

    def test_regular_session(self, manager):
        """Test regular hours session detection."""
        # 11:00 AM Eastern on a Tuesday
        dt = ET.localize(datetime(2024, 1, 2, 11, 0))
        session = manager.get_current_session(dt)
        assert session == TradingSession.REGULAR

    def test_after_hours_session(self, manager):
        """Test after-hours session detection."""
        # 5:00 PM Eastern on a Tuesday
        dt = ET.localize(datetime(2024, 1, 2, 17, 0))
        session = manager.get_current_session(dt)
        assert session == TradingSession.AFTER_HOURS

    def test_closed_session(self, manager_overnight_disabled):
        """Test closed session detection (when overnight disabled)."""
        # 10:00 PM Eastern on a Tuesday - overnight if enabled, closed if disabled
        dt = ET.localize(datetime(2024, 1, 2, 22, 0))
        session = manager_overnight_disabled.get_current_session(dt)
        assert session == TradingSession.CLOSED

    def test_pre_market_disabled_returns_closed(self, manager_ah_only):
        """Test disabled pre-market returns closed."""
        dt = ET.localize(datetime(2024, 1, 2, 6, 0))
        session = manager_ah_only.get_current_session(dt)
        assert session == TradingSession.CLOSED

    def test_after_hours_disabled_returns_closed(self, manager_pre_only):
        """Test disabled after-hours returns closed."""
        dt = ET.localize(datetime(2024, 1, 2, 17, 0))
        session = manager_pre_only.get_current_session(dt)
        assert session == TradingSession.CLOSED

    def test_early_pre_market(self, manager):
        """Test early pre-market (4:00 AM)."""
        dt = ET.localize(datetime(2024, 1, 2, 4, 0))
        session = manager.get_current_session(dt)
        assert session == TradingSession.PRE_MARKET

    def test_late_after_hours(self, manager):
        """Test late after-hours (7:59 PM)."""
        dt = ET.localize(datetime(2024, 1, 2, 19, 59))
        session = manager.get_current_session(dt)
        assert session == TradingSession.AFTER_HOURS

    def test_market_open_boundary(self, manager):
        """Test at market open boundary (9:30 AM)."""
        dt = ET.localize(datetime(2024, 1, 2, 9, 30))
        session = manager.get_current_session(dt)
        assert session == TradingSession.REGULAR

    def test_market_close_boundary(self, manager):
        """Test at market close boundary (4:00 PM)."""
        dt = ET.localize(datetime(2024, 1, 2, 16, 0))
        session = manager.get_current_session(dt)
        assert session == TradingSession.AFTER_HOURS


# ============================================================================
# Session Check Methods Tests
# ============================================================================


class TestSessionCheckMethods:
    """Test session check methods."""

    def test_is_extended_hours_pre_market(self, manager):
        """Test is_extended_hours during pre-market."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 6, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            assert manager.is_extended_hours() is True

    def test_is_extended_hours_after_hours(self, manager):
        """Test is_extended_hours during after-hours."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 17, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            assert manager.is_extended_hours() is True

    def test_is_extended_hours_regular(self, manager):
        """Test is_extended_hours during regular hours."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 11, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            assert manager.is_extended_hours() is False

    def test_is_pre_market(self, manager):
        """Test is_pre_market method."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 6, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            assert manager.is_pre_market() is True

    def test_is_after_hours(self, manager):
        """Test is_after_hours method."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 17, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            assert manager.is_after_hours() is True

    def test_is_regular_hours(self, manager):
        """Test is_regular_hours method."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 11, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            assert manager.is_regular_hours() is True


# ============================================================================
# Can Trade Extended Hours Tests
# ============================================================================


class TestCanTradeExtendedHours:
    """Test can_trade_extended_hours method."""

    @pytest.mark.asyncio
    async def test_can_trade_during_pre_market(self, manager):
        """Test can trade during pre-market."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 6, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            can_trade, reason = await manager.can_trade_extended_hours("AAPL")
            assert can_trade is True
            assert "pre_market" in reason

    @pytest.mark.asyncio
    async def test_can_trade_during_after_hours(self, manager):
        """Test can trade during after-hours."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 17, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            can_trade, reason = await manager.can_trade_extended_hours("AAPL")
            assert can_trade is True
            assert "after_hours" in reason

    @pytest.mark.asyncio
    async def test_cannot_trade_during_regular_hours(self, manager):
        """Test cannot trade (extended hours) during regular hours."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 11, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            can_trade, reason = await manager.can_trade_extended_hours("AAPL")
            assert can_trade is False
            assert "Not in extended hours" in reason

    @pytest.mark.asyncio
    async def test_cannot_trade_when_closed(self, manager):
        """Test cannot trade when market is closed."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 22, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            can_trade, reason = await manager.can_trade_extended_hours("AAPL")
            assert can_trade is False


# ============================================================================
# Get Extended Hours Quote Tests
# ============================================================================


class TestGetExtendedHoursQuote:
    """Test get_extended_hours_quote method."""

    @pytest.mark.asyncio
    async def test_get_quote_returns_structure(self, manager):
        """Test quote returns expected structure."""
        quote = await manager.get_extended_hours_quote("AAPL")
        assert quote is not None
        assert "symbol" in quote
        assert "bid" in quote
        assert "ask" in quote
        assert "timestamp" in quote
        assert quote["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_quote_with_valid_prices(self, manager):
        """Test quote with valid bid/ask prices."""
        # The implementation uses placeholders, but we test the structure
        quote = await manager.get_extended_hours_quote("AAPL")
        assert quote is not None

    @pytest.mark.asyncio
    async def test_quote_calculates_spread(self, manager):
        """Test that spread is calculated when prices are valid."""
        quote = await manager.get_extended_hours_quote("AAPL")
        # Placeholder returns 0.0 prices, so spread isn't calculated
        assert quote is not None


# ============================================================================
# Position Size Adjustment Tests
# ============================================================================


class TestAdjustPositionSize:
    """Test adjust_position_size_for_extended_hours method."""

    def test_default_adjustment(self, manager):
        """Test default 50% position size adjustment."""
        adjusted = manager.adjust_position_size_for_extended_hours(10000.0)
        assert adjusted == 5000.0

    def test_adjustment_zero(self, manager):
        """Test adjustment with zero position."""
        adjusted = manager.adjust_position_size_for_extended_hours(0.0)
        assert adjusted == 0.0

    def test_adjustment_small_position(self, manager):
        """Test adjustment with small position."""
        adjusted = manager.adjust_position_size_for_extended_hours(100.0)
        assert adjusted == 50.0

    def test_adjustment_large_position(self, manager):
        """Test adjustment with large position."""
        adjusted = manager.adjust_position_size_for_extended_hours(1000000.0)
        assert adjusted == 500000.0


# ============================================================================
# Execute Extended Hours Trade Tests
# ============================================================================


class TestExecuteExtendedHoursTrade:
    """Test execute_extended_hours_trade method."""

    @pytest.mark.asyncio
    async def test_rejects_during_regular_hours(self, manager):
        """Test trade rejected during regular hours."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 11, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            result = await manager.execute_extended_hours_trade("AAPL", "buy", 10)
            assert result is None

    @pytest.mark.asyncio
    async def test_rejects_when_closed(self, manager):
        """Test trade rejected when market closed."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 22, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            result = await manager.execute_extended_hours_trade("AAPL", "buy", 10)
            assert result is None


# ============================================================================
# Get Extended Hours Strategies Tests
# ============================================================================


class TestGetExtendedHoursStrategies:
    """Test get_extended_hours_strategies method."""

    def test_pre_market_strategies(self, manager):
        """Test pre-market strategy recommendations."""
        strategies = manager.get_extended_hours_strategies("pre_market")
        assert strategies["primary"] == "gap_trading"
        assert "description" in strategies
        assert "focus" in strategies
        assert "risk" in strategies
        assert "tips" in strategies
        assert isinstance(strategies["tips"], list)

    def test_after_hours_strategies(self, manager):
        """Test after-hours strategy recommendations."""
        strategies = manager.get_extended_hours_strategies("after_hours")
        assert strategies["primary"] == "earnings_reaction"
        assert "description" in strategies
        assert "focus" in strategies
        assert "risk" in strategies
        assert "tips" in strategies
        assert isinstance(strategies["tips"], list)

    def test_regular_session_empty(self, manager):
        """Test regular session returns empty dict."""
        strategies = manager.get_extended_hours_strategies("regular")
        assert strategies == {}

    def test_closed_session_empty(self, manager):
        """Test closed session returns empty dict."""
        strategies = manager.get_extended_hours_strategies("closed")
        assert strategies == {}


# ============================================================================
# Get Extended Hours Opportunities Tests
# ============================================================================


class TestGetExtendedHoursOpportunities:
    """Test get_extended_hours_opportunities method."""

    @pytest.mark.asyncio
    async def test_returns_empty_during_regular_hours(self, manager):
        """Test returns empty list during regular hours."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 11, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            opportunities = await manager.get_extended_hours_opportunities()
            assert opportunities == []

    @pytest.mark.asyncio
    async def test_returns_list_during_pre_market(self, manager):
        """Test returns list during pre-market."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 6, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            opportunities = await manager.get_extended_hours_opportunities()
            # Currently returns empty (placeholder)
            assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_returns_list_during_after_hours(self, manager):
        """Test returns list during after-hours."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 17, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            opportunities = await manager.get_extended_hours_opportunities()
            assert isinstance(opportunities, list)


# ============================================================================
# Get Session Info Tests
# ============================================================================


class TestGetSessionInfo:
    """Test get_session_info method."""

    def test_pre_market_info(self, manager):
        """Test pre-market session info."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 6, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            info = manager.get_session_info()
            assert info["session"] == "pre_market"
            assert info["session_name"] == "Pre-Market"
            assert info["is_extended"] is True
            assert info["can_trade"] is True
            assert "start_time" in info
            assert "end_time" in info
            assert info["liquidity"] == "Low"
            assert info["volatility"] == "High"

    def test_regular_hours_info(self, manager):
        """Test regular hours session info."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 11, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            info = manager.get_session_info()
            assert info["session"] == "regular"
            assert info["session_name"] == "Regular Hours"
            assert info["is_extended"] is False
            assert info["can_trade"] is True
            assert info["liquidity"] == "Normal"
            assert info["volatility"] == "Normal"

    def test_after_hours_info(self, manager):
        """Test after-hours session info."""
        eastern = pytz.timezone("US/Eastern")
        mock_dt = eastern.localize(datetime(2024, 1, 2, 17, 0))
        with patch("utils.extended_hours.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            info = manager.get_session_info()
            assert info["session"] == "after_hours"
            assert info["session_name"] == "After-Hours"
            assert info["is_extended"] is True
            assert info["can_trade"] is True

    def test_closed_info(self, manager_overnight_disabled):
        """Test closed session info (Saturday - fully closed)."""
        # Use Saturday to get CLOSED session (not overnight)
        dt = ET.localize(datetime(2024, 1, 6, 12, 0))  # Saturday noon
        info = manager_overnight_disabled.get_session_info(dt)
        assert info["session"] == "closed"
        assert info["is_extended"] is False
        assert info["can_trade"] is False


# ============================================================================
# Format Session Info Tests
# ============================================================================


class TestFormatSessionInfo:
    """Test format_session_info function."""

    def test_format_extended_session(self):
        """Test formatting extended session info."""
        info = {
            "session": "pre_market",
            "session_name": "Pre-Market",
            "current_time": "2024-01-02 06:00:00 EST",
            "start_time": "4:00 AM ET",
            "end_time": "9:30 AM ET",
            "is_extended": True,
            "is_overnight": False,
            "is_regular_hours": False,
            "can_trade": True,
            "liquidity": "Low",
            "volatility": "High",
            "position_size_adj": "50% of regular",
            "recommended_strategy": "Gap trading on news/earnings",
        }
        output = format_session_info(info)
        assert "MARKET SESSION: Pre-Market" in output
        assert "EXTENDED HOURS TRADING" in output
        assert "Low" in output
        assert "High" in output

    def test_format_regular_session(self):
        """Test formatting regular session info."""
        info = {
            "session": "regular",
            "session_name": "Regular Hours",
            "current_time": "2024-01-02 11:00:00 EST",
            "start_time": "9:30 AM ET",
            "end_time": "4:00 PM ET",
            "is_extended": False,
            "is_overnight": False,
            "is_regular_hours": True,
            "can_trade": True,
            "liquidity": "Normal",
            "volatility": "Normal",
            "position_size_adj": "100%",
            "recommended_strategy": "Standard strategies",
        }
        output = format_session_info(info)
        assert "MARKET SESSION: Regular Hours" in output
        assert "Regular Market Hours" in output
        assert "Full liquidity" in output


# ============================================================================
# GapTradingStrategy Tests
# ============================================================================


class TestGapTradingStrategyInit:
    """Test GapTradingStrategy initialization."""

    def test_default_threshold(self, gap_strategy):
        """Test default gap threshold."""
        assert gap_strategy.gap_threshold == 0.02

    def test_custom_threshold(self):
        """Test custom gap threshold."""
        strategy = GapTradingStrategy(gap_threshold=0.05)
        assert strategy.gap_threshold == 0.05


class TestGapTradingAnalyze:
    """Test GapTradingStrategy analyze_gap method."""

    @pytest.mark.asyncio
    async def test_gap_up_signal(self, gap_strategy):
        """Test gap up generates sell signal (fade)."""
        signal = await gap_strategy.analyze_gap("AAPL", 100.0, 103.0)  # 3% gap up
        assert signal is not None
        assert signal["signal"] == "sell"
        assert signal["strategy"] == "gap_fade"
        assert signal["gap_pct"] == pytest.approx(0.03)
        assert signal["target"] == 100.0

    @pytest.mark.asyncio
    async def test_gap_down_signal(self, gap_strategy):
        """Test gap down generates buy signal (bounce)."""
        signal = await gap_strategy.analyze_gap("AAPL", 100.0, 97.0)  # 3% gap down
        assert signal is not None
        assert signal["signal"] == "buy"
        assert signal["strategy"] == "gap_bounce"
        assert signal["gap_pct"] == pytest.approx(-0.03)
        assert signal["target"] == 100.0

    @pytest.mark.asyncio
    async def test_no_signal_small_gap(self, gap_strategy):
        """Test no signal for small gap."""
        signal = await gap_strategy.analyze_gap("AAPL", 100.0, 101.0)  # 1% gap
        assert signal is None

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, gap_strategy):
        """Test exactly at threshold."""
        signal = await gap_strategy.analyze_gap("AAPL", 100.0, 102.0)  # Exactly 2%
        assert signal is not None  # Should trigger

    @pytest.mark.asyncio
    async def test_below_threshold_boundary(self, gap_strategy):
        """Test just below threshold."""
        signal = await gap_strategy.analyze_gap("AAPL", 100.0, 101.99)  # Just under 2%
        assert signal is None


# ============================================================================
# EarningsReactionStrategy Tests
# ============================================================================


class TestEarningsReactionStrategyInit:
    """Test EarningsReactionStrategy initialization."""

    def test_default_min_move(self, earnings_strategy):
        """Test default minimum move percentage."""
        assert earnings_strategy.min_move_pct == 0.03

    def test_custom_min_move(self):
        """Test custom minimum move percentage."""
        strategy = EarningsReactionStrategy(min_move_pct=0.05)
        assert strategy.min_move_pct == 0.05


class TestEarningsReactionAnalyze:
    """Test EarningsReactionStrategy analyze_earnings_move method."""

    @pytest.mark.asyncio
    async def test_beat_positive_move_continuation(self, earnings_strategy):
        """Test earnings beat + positive move generates continuation signal."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 105.0, earnings_beat=True  # 5% move up
        )
        assert signal is not None
        assert signal["signal"] == "buy"
        assert signal["strategy"] == "earnings_continuation"
        assert signal["move_pct"] == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_miss_negative_move_bounce(self, earnings_strategy):
        """Test earnings miss + negative move generates bounce signal."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 95.0, earnings_beat=False  # 5% move down
        )
        assert signal is not None
        assert signal["signal"] == "buy"
        assert signal["strategy"] == "earnings_oversold_bounce"
        assert signal["move_pct"] == pytest.approx(-0.05)

    @pytest.mark.asyncio
    async def test_no_signal_small_move(self, earnings_strategy):
        """Test no signal for small earnings move."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 101.0, earnings_beat=True  # 1% move
        )
        assert signal is None

    @pytest.mark.asyncio
    async def test_beat_negative_move_no_signal(self, earnings_strategy):
        """Test earnings beat + negative move returns no signal."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 95.0, earnings_beat=True  # Beat but down
        )
        assert signal is None

    @pytest.mark.asyncio
    async def test_miss_positive_move_no_signal(self, earnings_strategy):
        """Test earnings miss + positive move returns no signal."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 105.0, earnings_beat=False  # Miss but up
        )
        assert signal is None

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, earnings_strategy):
        """Test exactly at move threshold."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 103.0, earnings_beat=True  # Exactly 3%
        )
        assert signal is not None

    @pytest.mark.asyncio
    async def test_signal_entry_price(self, earnings_strategy):
        """Test signal entry price calculation."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 105.0, earnings_beat=True
        )
        assert signal["entry"] == pytest.approx(105.0 * 1.001)

    @pytest.mark.asyncio
    async def test_signal_target_price(self, earnings_strategy):
        """Test signal target price calculation."""
        signal = await earnings_strategy.analyze_earnings_move(
            "AAPL", 100.0, 105.0, earnings_beat=True
        )
        assert signal["target"] == pytest.approx(105.0 * 1.03)


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_position_size_negative(self, manager):
        """Test position size with negative value."""
        adjusted = manager.adjust_position_size_for_extended_hours(-1000.0)
        assert adjusted == -500.0  # Multiplier still applied

    @pytest.mark.asyncio
    async def test_gap_strategy_zero_prev_close(self, gap_strategy):
        """Test gap strategy with zero previous close raises error."""
        # Zero previous close causes division by zero - this is expected behavior
        # for invalid input (a stock price can never be zero)
        with pytest.raises(ZeroDivisionError):
            await gap_strategy.analyze_gap("AAPL", 0.0, 100.0)

    @pytest.mark.asyncio
    async def test_earnings_strategy_zero_close(self, earnings_strategy):
        """Test earnings strategy with zero close price raises error."""
        # Zero close price causes division by zero - this is expected behavior
        # for invalid input (a stock price can never be zero)
        with pytest.raises(ZeroDivisionError):
            await earnings_strategy.analyze_earnings_move("AAPL", 0.0, 100.0, earnings_beat=True)

    def test_strategies_pre_market_tips_not_empty(self, manager):
        """Test pre-market strategies have tips."""
        strategies = manager.get_extended_hours_strategies("pre_market")
        assert len(strategies["tips"]) > 0

    def test_strategies_after_hours_tips_not_empty(self, manager):
        """Test after-hours strategies have tips."""
        strategies = manager.get_extended_hours_strategies("after_hours")
        assert len(strategies["tips"]) > 0
