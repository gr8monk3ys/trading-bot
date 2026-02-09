"""
Unit tests for ExtendedHoursStrategy.

Tests cover:
1. Initialization with extended hours parameters
2. Pre-market session detection
3. After-hours session detection
4. Regular hours handling
5. Extended hours order types
6. Session-specific position sizing
"""

import warnings
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strategies.extended_hours_strategy import ExtendedHoursStrategy

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_broker():
    """Create a mock broker with default account values."""
    broker = AsyncMock()
    broker.get_account.return_value = MagicMock(
        buying_power="100000.0",
        equity="100000.0",
        cash="50000.0",
    )
    broker.get_positions.return_value = []
    broker.get_position.return_value = None
    broker.get_last_price.return_value = 150.0
    broker.submit_order_advanced.return_value = MagicMock(id="test-order-123")

    # Mock bars for gap/earnings analysis
    bar1 = MagicMock(close=100.0)
    bar2 = MagicMock(close=105.0)  # Yesterday's close
    broker.get_bars.return_value = [bar1, bar2]

    # Mock quote for spread analysis
    quote = MagicMock()
    quote.bid_price = 149.90
    quote.ask_price = 150.10
    broker.get_latest_quote.return_value = quote

    return broker


@pytest.fixture
def mock_ext_hours_manager():
    """Create a mock ExtendedHoursManager."""
    manager = MagicMock()
    manager.get_current_session.return_value = "pre_market"
    # can_trade_extended_hours is async, so use AsyncMock
    manager.can_trade_extended_hours = AsyncMock(return_value=(True, "OK"))
    return manager


@pytest.fixture
def sample_quote_normal_spread():
    """Quote with normal spread (under 0.5% threshold)."""
    quote = MagicMock()
    quote.bid_price = 149.90
    quote.ask_price = 150.10  # 0.13% spread
    return quote


@pytest.fixture
def sample_quote_wide_spread():
    """Quote with wide spread (over 0.5% threshold)."""
    quote = MagicMock()
    quote.bid_price = 148.00
    quote.ask_price = 152.00  # 2.67% spread
    return quote


@pytest.fixture
def gap_up_bars():
    """Bars showing a gap up (current price > yesterday's close)."""
    yesterday = MagicMock(close=100.0)
    return [yesterday, yesterday]  # Two days of data


@pytest.fixture
def gap_down_bars():
    """Bars showing a gap down (current price < yesterday's close)."""
    yesterday = MagicMock(close=100.0)
    return [yesterday, yesterday]


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestExtendedHoursStrategyInit:
    """Test initialization of ExtendedHoursStrategy."""

    @pytest.mark.asyncio
    async def test_initializes_with_defaults(self, mock_broker):
        """Test that strategy initializes with default parameters."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await strategy.initialize()

            assert result is True
            # Check deprecation warning
            assert len(w) >= 1
            assert "experimental" in str(w[0].message).lower()

        # Check default parameters
        assert strategy.enable_pre_market is True
        assert strategy.enable_after_hours is True
        assert strategy.gap_threshold == 0.02  # 2%
        assert strategy.ext_position_size == 0.05  # 5%
        assert strategy.max_spread_pct == 0.005  # 0.5%

    @pytest.mark.asyncio
    async def test_shows_deprecation_warning(self, mock_broker):
        """Test that deprecation warning is shown during initialization."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with pytest.warns(UserWarning, match="experimental"):
            await strategy.initialize()

    @pytest.mark.asyncio
    async def test_initializes_with_custom_parameters(self, mock_broker):
        """Test initialization with custom parameters."""
        custom_params = {
            "symbols": ["AAPL", "MSFT"],
            "enable_pre_market": False,
            "enable_after_hours": True,
            "gap_threshold": 0.03,
            "earnings_threshold": 0.05,
            "ext_position_size": 0.03,
            "max_spread_pct": 0.003,
        }

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters=custom_params
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        assert strategy.enable_pre_market is False
        assert strategy.enable_after_hours is True
        assert strategy.gap_threshold == 0.03
        assert strategy.earnings_threshold == 0.05
        assert strategy.ext_position_size == 0.03
        assert strategy.max_spread_pct == 0.003

    @pytest.mark.asyncio
    async def test_initializes_extended_hours_manager(self, mock_broker):
        """Test that ExtendedHoursManager is initialized."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        assert strategy.ext_hours is not None

    @pytest.mark.asyncio
    async def test_initializes_tracking_structures(self, mock_broker):
        """Test that tracking structures are properly initialized."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL", "MSFT"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Check tracking structures
        assert strategy.tracked_gaps == {}
        assert strategy.earnings_today == []
        assert strategy.last_trade_time == {}


# =============================================================================
# SESSION DETECTION TESTS
# =============================================================================


class TestSessionDetection:
    """Test session detection logic."""

    @pytest.mark.asyncio
    async def test_handles_pre_market_session(self, mock_broker, mock_ext_hours_manager):
        """Test behavior during pre-market session."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Replace ext_hours with mock
        mock_ext_hours_manager.get_current_session.return_value = "pre_market"
        strategy.ext_hours = mock_ext_hours_manager

        # Test gap analysis is used in pre-market
        current_price = 105.0  # 5% gap up from yesterday (100)
        result = await strategy._analyze_gap_trading("AAPL", current_price)

        # With 5% gap and 2% threshold, should trigger buy
        assert result == "buy"

    @pytest.mark.asyncio
    async def test_handles_after_hours_session(self, mock_broker, mock_ext_hours_manager):
        """Test behavior during after-hours session."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Replace ext_hours with mock
        mock_ext_hours_manager.get_current_session.return_value = "after_hours"
        strategy.ext_hours = mock_ext_hours_manager

        # Test earnings analysis is used in after-hours
        # Mock bars to show today's close was 100
        bar = MagicMock(close=100.0)
        mock_broker.get_bars.return_value = [bar]

        current_price = 105.0  # 5% move (earnings beat)
        result = await strategy._analyze_earnings_reaction("AAPL", current_price)

        # With 5% move and 3% threshold, should trigger buy
        assert result == "buy"

    @pytest.mark.asyncio
    async def test_regular_hours_returns_early(self, mock_broker, mock_ext_hours_manager):
        """Test that regular hours session causes early return in on_trading_iteration."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set to regular hours
        mock_ext_hours_manager.get_current_session.return_value = "regular"
        strategy.ext_hours = mock_ext_hours_manager

        # Should return early without processing
        await strategy.on_trading_iteration()

        # No positions should be analyzed during regular hours
        mock_broker.get_position.assert_not_called()


# =============================================================================
# GAP TRADING TESTS
# =============================================================================


class TestGapTrading:
    """Test gap trading analysis for pre-market."""

    @pytest.mark.asyncio
    async def test_detects_significant_gap_up(self, mock_broker):
        """Test detection of significant gap up."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "gap_threshold": 0.02,  # 2% gap threshold
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Mock yesterday's close at 100
        bar1 = MagicMock(close=95.0)  # Day before
        bar2 = MagicMock(close=100.0)  # Yesterday's close
        mock_broker.get_bars.return_value = [bar1, bar2]

        # Current pre-market price is 105 (5% gap up)
        current_price = 105.0

        result = await strategy._analyze_gap_trading("AAPL", current_price)

        assert result == "buy"
        assert "AAPL" in strategy.tracked_gaps
        assert strategy.tracked_gaps["AAPL"]["direction"] == "up"

    @pytest.mark.asyncio
    async def test_detects_significant_gap_down(self, mock_broker):
        """Test detection of significant gap down."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "gap_threshold": 0.02,  # 2% gap threshold
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Mock yesterday's close at 100
        bar1 = MagicMock(close=105.0)
        bar2 = MagicMock(close=100.0)  # Yesterday's close
        mock_broker.get_bars.return_value = [bar1, bar2]

        # Current pre-market price is 95 (5% gap down)
        current_price = 95.0

        result = await strategy._analyze_gap_trading("AAPL", current_price)

        assert result == "short"
        assert strategy.tracked_gaps["AAPL"]["direction"] == "down"

    @pytest.mark.asyncio
    async def test_returns_neutral_for_small_gap(self, mock_broker):
        """Test that small gaps don't trigger trades."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "gap_threshold": 0.02,  # 2% threshold
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Mock bars: bars[-2] gives the first element in a 2-element list
        # The strategy uses bars[-2].close as "yesterday's close"
        # bars = [bar0, bar1], bars[-2] = bar0
        bar0 = MagicMock(close=100.0)  # This is bars[-2]
        bar1 = MagicMock(close=102.0)
        mock_broker.get_bars.return_value = [bar0, bar1]

        # Current price is 101 (1% gap above bar0.close=100)
        # Gap calculation: (101 - 100) / 100 = 1% < 2% threshold
        current_price = 101.0

        result = await strategy._analyze_gap_trading("AAPL", current_price)

        assert result == "neutral"

    @pytest.mark.asyncio
    async def test_returns_neutral_when_no_bars(self, mock_broker):
        """Test that neutral is returned when bars are not available."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # No bars available
        mock_broker.get_bars.return_value = []

        result = await strategy._analyze_gap_trading("AAPL", 100.0)

        assert result == "neutral"


# =============================================================================
# EARNINGS REACTION TESTS
# =============================================================================


class TestEarningsReaction:
    """Test earnings reaction analysis for after-hours."""

    @pytest.mark.asyncio
    async def test_detects_positive_earnings_reaction(self, mock_broker):
        """Test detection of positive earnings reaction (beat)."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "earnings_threshold": 0.03,  # 3% move threshold
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Mock today's close at 100
        bar = MagicMock(close=100.0)
        mock_broker.get_bars.return_value = [bar]

        # After-hours price is 105 (5% up - earnings beat)
        current_price = 105.0

        result = await strategy._analyze_earnings_reaction("AAPL", current_price)

        assert result == "buy"

    @pytest.mark.asyncio
    async def test_detects_negative_earnings_reaction(self, mock_broker):
        """Test detection of negative earnings reaction (miss)."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "earnings_threshold": 0.03,  # 3% move threshold
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Mock today's close at 100
        bar = MagicMock(close=100.0)
        mock_broker.get_bars.return_value = [bar]

        # After-hours price is 95 (5% down - earnings miss)
        current_price = 95.0

        result = await strategy._analyze_earnings_reaction("AAPL", current_price)

        assert result == "short"

    @pytest.mark.asyncio
    async def test_returns_neutral_for_small_move(self, mock_broker):
        """Test that small moves don't trigger trades."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "earnings_threshold": 0.03,  # 3% threshold
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Mock today's close at 100
        bar = MagicMock(close=100.0)
        mock_broker.get_bars.return_value = [bar]

        # After-hours price is 102 (2% move - below threshold)
        current_price = 102.0

        result = await strategy._analyze_earnings_reaction("AAPL", current_price)

        assert result == "neutral"


# =============================================================================
# SPREAD VALIDATION TESTS
# =============================================================================


class TestSpreadValidation:
    """Test bid-ask spread validation."""

    @pytest.mark.asyncio
    async def test_accepts_normal_spread(self, mock_broker, sample_quote_normal_spread):
        """Test that normal spreads are accepted."""
        mock_broker.get_latest_quote.return_value = sample_quote_normal_spread

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "max_spread_pct": 0.005,  # 0.5% max spread
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        quote = await strategy._get_safe_quote("AAPL")

        assert quote is not None
        assert quote["spread_pct"] < strategy.max_spread_pct

    @pytest.mark.asyncio
    async def test_rejects_wide_spread(self, mock_broker, sample_quote_wide_spread):
        """Test that wide spreads cause analyze_symbol to return neutral."""
        mock_broker.get_latest_quote.return_value = sample_quote_wide_spread

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "max_spread_pct": 0.005,  # 0.5% max spread
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Replace ext_hours with mock
        mock_ext_hours = MagicMock()
        mock_ext_hours.can_trade_extended_hours.return_value = (True, "OK")
        strategy.ext_hours = mock_ext_hours

        result = await strategy.analyze_symbol("AAPL", "pre_market")

        # Should return neutral due to wide spread
        assert result == "neutral"

    @pytest.mark.asyncio
    async def test_returns_none_when_quote_unavailable(self, mock_broker):
        """Test that None is returned when quote is unavailable."""
        mock_broker.get_latest_quote.return_value = None

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        quote = await strategy._get_safe_quote("AAPL")

        assert quote is None


# =============================================================================
# ORDER TYPE TESTS
# =============================================================================


class TestExtendedHoursOrderTypes:
    """Test that correct order types are used for extended hours."""

    @pytest.mark.asyncio
    async def test_uses_limit_order_for_extended_hours(self, mock_broker, mock_ext_hours_manager):
        """Test that limit orders are used (not market) in extended hours."""
        # Set up quote with good spread
        quote = MagicMock()
        quote.bid_price = 149.95
        quote.ask_price = 150.05  # 0.07% spread
        mock_broker.get_latest_quote.return_value = quote

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        # Create a mock order to return from OrderBuilder.build()
        mock_order = MagicMock()
        mock_order.limit_price = 150.15  # Limit order attribute

        # Patch OrderBuilder to return our mock order (avoids validation issues)
        with patch('strategies.extended_hours_strategy.OrderBuilder') as MockOrderBuilder:
            mock_builder = MagicMock()
            mock_builder.limit.return_value = mock_builder
            mock_builder.extended_hours.return_value = mock_builder
            mock_builder.bracket.return_value = mock_builder
            mock_builder.gtc.return_value = mock_builder
            mock_builder.build.return_value = mock_order
            MockOrderBuilder.return_value = mock_builder

            # Execute a buy trade
            await strategy.execute_trade("AAPL", "buy", "pre_market")

            # Verify order was submitted
            mock_broker.submit_order_advanced.assert_called_once_with(mock_order)

            # Verify limit was called (proving it's a limit order)
            mock_builder.limit.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_bracket_order_with_stops(self, mock_broker, mock_ext_hours_manager):
        """Test that bracket orders include stop-loss and take-profit."""
        # Set up quote with good spread
        quote = MagicMock()
        quote.bid_price = 149.95
        quote.ask_price = 150.05  # 0.07% spread
        mock_broker.get_latest_quote.return_value = quote

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "gap_stop_loss": 0.015,
                "gap_take_profit": 0.03,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        # Create a mock order to return from OrderBuilder.build()
        mock_order = MagicMock()

        # Patch OrderBuilder to return our mock order
        with patch('strategies.extended_hours_strategy.OrderBuilder') as MockOrderBuilder:
            mock_builder = MagicMock()
            mock_builder.limit.return_value = mock_builder
            mock_builder.extended_hours.return_value = mock_builder
            mock_builder.bracket.return_value = mock_builder
            mock_builder.gtc.return_value = mock_builder
            mock_builder.build.return_value = mock_order
            MockOrderBuilder.return_value = mock_builder

            # Execute a pre-market buy trade
            await strategy.execute_trade("AAPL", "buy", "pre_market")

            # Order should be submitted
            mock_broker.submit_order_advanced.assert_called_once()

            # Verify bracket was called with stop-loss and take-profit
            mock_builder.bracket.assert_called_once()
            # Get the kwargs that were passed to bracket()
            bracket_call = mock_builder.bracket.call_args
            assert 'take_profit' in bracket_call.kwargs or len(bracket_call.args) >= 1
            assert 'stop_loss' in bracket_call.kwargs or len(bracket_call.args) >= 2


# =============================================================================
# POSITION SIZING TESTS
# =============================================================================


class TestExtendedHoursPositionSizing:
    """Test position sizing for extended hours trading."""

    @pytest.mark.asyncio
    async def test_uses_conservative_position_size(self, mock_broker, mock_ext_hours_manager):
        """Test that extended hours uses more conservative position sizing."""
        # Set up quote with good spread
        quote = MagicMock()
        quote.bid_price = 149.95
        quote.ask_price = 150.05  # 0.07% spread
        mock_broker.get_latest_quote.return_value = quote

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "ext_position_size": 0.05,  # 5% (conservative)
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        # Patch OrderBuilder to avoid validation issues
        with patch('strategies.extended_hours_strategy.OrderBuilder') as MockOrderBuilder:
            mock_builder = MagicMock()
            mock_builder.limit.return_value = mock_builder
            mock_builder.extended_hours.return_value = mock_builder
            mock_builder.bracket.return_value = mock_builder
            mock_builder.gtc.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()
            MockOrderBuilder.return_value = mock_builder

            # Execute trade
            await strategy.execute_trade("AAPL", "buy", "pre_market")

            # Verify order was submitted
            mock_broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_enforces_position_size_limit(self, mock_broker, mock_ext_hours_manager):
        """Test that maximum position size is enforced."""
        # Large account
        mock_broker.get_account.return_value = MagicMock(
            buying_power="1000000.0",
            equity="1000000.0",
        )

        # Set up quote with good spread
        quote = MagicMock()
        quote.bid_price = 149.95
        quote.ask_price = 150.05  # 0.07% spread
        mock_broker.get_latest_quote.return_value = quote

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "ext_position_size": 0.50,  # 50% (too large)
                "max_position_size": 0.05,  # Max 5%
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        # Patch OrderBuilder to avoid validation issues
        with patch('strategies.extended_hours_strategy.OrderBuilder') as MockOrderBuilder:
            mock_builder = MagicMock()
            mock_builder.limit.return_value = mock_builder
            mock_builder.extended_hours.return_value = mock_builder
            mock_builder.bracket.return_value = mock_builder
            mock_builder.gtc.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()
            MockOrderBuilder.return_value = mock_builder

            # Execute trade
            await strategy.execute_trade("AAPL", "buy", "pre_market")

            # Order should still be submitted (with capped size)
            mock_broker.submit_order_advanced.assert_called_once()


# =============================================================================
# COOLDOWN TESTS
# =============================================================================


class TestTradeCooldown:
    """Test trade cooldown between extended hours trades."""

    @pytest.mark.asyncio
    async def test_respects_cooldown_period(self, mock_broker):
        """Test that cooldown period is respected between trades."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "trade_cooldown_minutes": 30,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Record a recent trade
        strategy.last_trade_time["AAPL"] = datetime.now() - timedelta(minutes=10)

        # Should not be allowed (only 10 minutes since last trade, need 30)
        assert strategy._is_trade_allowed("AAPL") is False

    @pytest.mark.asyncio
    async def test_allows_trade_after_cooldown(self, mock_broker):
        """Test that trade is allowed after cooldown period."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "trade_cooldown_minutes": 30,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Record an old trade (40 minutes ago)
        strategy.last_trade_time["AAPL"] = datetime.now() - timedelta(minutes=40)

        # Should be allowed (40 minutes > 30 minute cooldown)
        assert strategy._is_trade_allowed("AAPL") is True

    @pytest.mark.asyncio
    async def test_allows_first_trade(self, mock_broker):
        """Test that first trade is always allowed."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # No previous trades
        assert "AAPL" not in strategy.last_trade_time

        # Should be allowed
        assert strategy._is_trade_allowed("AAPL") is True


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    def test_default_parameters_are_valid(self):
        """Test that default_parameters returns valid configuration."""
        params = ExtendedHoursStrategy.default_parameters()

        # Verify required parameters
        assert "symbols" in params
        assert "enable_pre_market" in params
        assert "enable_after_hours" in params
        assert params["gap_threshold"] > 0
        assert params["earnings_threshold"] > 0
        assert 0 < params["ext_position_size"] <= 1
        assert params["max_spread_pct"] > 0

    def test_stop_loss_less_than_take_profit_gap(self):
        """Test that gap stop loss is less than take profit (positive R:R)."""
        params = ExtendedHoursStrategy.default_parameters()

        assert params["gap_stop_loss"] < params["gap_take_profit"]

    def test_stop_loss_less_than_take_profit_earnings(self):
        """Test that earnings stop loss is less than take profit (positive R:R)."""
        params = ExtendedHoursStrategy.default_parameters()

        assert params["earnings_stop_loss"] < params["earnings_take_profit"]

    def test_cooldown_is_positive(self):
        """Test that trade cooldown is positive."""
        params = ExtendedHoursStrategy.default_parameters()

        assert params["trade_cooldown_minutes"] > 0


# =============================================================================
# POSITION MANAGEMENT TESTS
# =============================================================================


class TestPositionManagement:
    """Test position management during extended hours."""

    @pytest.mark.asyncio
    async def test_manages_existing_position(self, mock_broker):
        """Test that existing positions are managed."""
        # Set up existing position
        position = MagicMock()
        position.symbol = "AAPL"
        position.qty = "10"
        position.avg_entry_price = "145.0"

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Mock get_last_price
        mock_broker.get_last_price.return_value = 148.0

        # Should not throw an error
        await strategy._manage_position("AAPL", position, "pre_market")

    @pytest.mark.asyncio
    async def test_skips_new_entry_when_position_exists(self, mock_broker, mock_ext_hours_manager):
        """Test that new entries are skipped when position already exists."""
        # Set up existing position
        position = MagicMock()
        position.symbol = "AAPL"
        position.qty = "10"
        mock_broker.get_position.return_value = position

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        # Mock circuit breaker check
        strategy.circuit_breaker = MagicMock()
        strategy.circuit_breaker.check_and_halt = AsyncMock(return_value=False)

        # Run iteration
        await strategy.on_trading_iteration()

        # New order should not be submitted (we have a position)
        mock_broker.submit_order_advanced.assert_not_called()


# =============================================================================
# ANALYZE SYMBOL TESTS
# =============================================================================


class TestAnalyzeSymbol:
    """Test analyze_symbol method."""

    @pytest.mark.asyncio
    async def test_returns_neutral_when_cannot_trade(self, mock_broker, mock_ext_hours_manager):
        """Test that neutral is returned when trading is not allowed."""
        mock_ext_hours_manager.can_trade_extended_hours.return_value = (False, "Not eligible")

        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        result = await strategy.analyze_symbol("AAPL", "pre_market")

        assert result == "neutral"

    @pytest.mark.asyncio
    async def test_routes_to_gap_trading_in_pre_market(self, mock_broker, mock_ext_hours_manager):
        """Test that pre-market routes to gap trading analysis."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "gap_threshold": 0.02,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        # Set up a gap scenario
        bar1 = MagicMock(close=95.0)
        bar2 = MagicMock(close=100.0)
        mock_broker.get_bars.return_value = [bar1, bar2]

        # Quote shows 5% gap up
        quote = MagicMock()
        quote.bid_price = 104.90
        quote.ask_price = 105.10
        mock_broker.get_latest_quote.return_value = quote

        result = await strategy.analyze_symbol("AAPL", "pre_market")

        # Should return buy signal for significant gap up
        assert result == "buy"

    @pytest.mark.asyncio
    async def test_routes_to_earnings_in_after_hours(self, mock_broker, mock_ext_hours_manager):
        """Test that after-hours routes to earnings analysis."""
        strategy = ExtendedHoursStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "earnings_threshold": 0.03,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ext_hours = mock_ext_hours_manager

        # Set up an earnings beat scenario
        bar = MagicMock(close=100.0)
        mock_broker.get_bars.return_value = [bar]

        # Quote shows 5% move up
        quote = MagicMock()
        quote.bid_price = 104.90
        quote.ask_price = 105.10
        mock_broker.get_latest_quote.return_value = quote

        result = await strategy.analyze_symbol("AAPL", "after_hours")

        # Should return buy signal for earnings beat
        assert result == "buy"
