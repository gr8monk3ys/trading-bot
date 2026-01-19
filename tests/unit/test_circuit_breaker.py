"""
Comprehensive tests for utils/circuit_breaker.py

Tests cover:
- CircuitBreaker initialization
- Initialize with broker
- Check and halt functionality
- Daily loss limit detection
- Rapid drawdown detection
- Trigger halt process
- Emergency position closing
- Daily reset
- Manual reset with safety controls
- Status reporting
- Edge cases and error handling

DRY Principles Applied:
- Module-level imports (no repeated imports in tests)
- Named constants for all magic numbers
- Module-level fixtures shared across test classes
- Assertion messages on key assertions
"""

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unit.conftest import (
    DEFAULT_STARTING_CASH,
    DEFAULT_STARTING_EQUITY,
    create_mock_account,
    create_mock_position,
)

# Module-level import - avoid repeated imports in each test
from utils.circuit_breaker import CircuitBreaker

# =============================================================================
# CONSTANTS - No magic numbers in tests
# Note: DEFAULT_STARTING_EQUITY, DEFAULT_STARTING_CASH imported from conftest
# =============================================================================

# Circuit breaker specific values
DEFAULT_MAX_DAILY_LOSS = 0.03  # 3%
CUSTOM_MAX_DAILY_LOSS = 0.05  # 5%
SMALL_MAX_DAILY_LOSS = 0.01  # 1%

# Equity scenarios
PROFIT_EQUITY = 105000.0  # 5% profit
SLIGHT_PROFIT_EQUITY = 102000.0  # 2% profit
WITHIN_LIMIT_EQUITY = 98000.0  # 2% loss (within 3% limit)
UNDER_BOTH_THRESHOLDS_EQUITY = 98100.0  # 1.9% loss (under rapid drawdown and daily limit)
EXACT_LIMIT_EQUITY = 97000.0  # 3% loss (exact limit)
OVER_LIMIT_EQUITY = 96900.0  # 3.1% loss (exceeds limit)
SLIGHT_DECLINE_EQUITY = 99000.0  # 1% decline

# Rapid drawdown scenario
PEAK_EQUITY_FOR_DRAWDOWN = 110000.0
RAPID_DRAWDOWN_EQUITY = 107700.0  # 2.09% drop from 110000

# Test positions
TEST_POSITION_QTY_1 = "100"
TEST_POSITION_QTY_2 = "50"
TEST_SYMBOL_1 = "AAPL"
TEST_SYMBOL_2 = "MSFT"

# Manual reset
CONFIRM_RESET_TOKEN = "CONFIRM_RESET"
WRONG_TOKEN = "WRONG_TOKEN"

# Test datetime
TEST_HALT_DATETIME = datetime(2024, 1, 15, 10, 30, 0)
TEST_DATE = date(2024, 1, 15)


# =============================================================================
# MODULE-LEVEL FIXTURES
# =============================================================================


@pytest.fixture
def circuit_breaker():
    """Create CircuitBreaker with default parameters."""
    return CircuitBreaker()


@pytest.fixture
def circuit_breaker_custom():
    """Create CircuitBreaker with custom max_daily_loss."""
    return CircuitBreaker(max_daily_loss=CUSTOM_MAX_DAILY_LOSS, auto_close_positions=False)


@pytest.fixture
def mock_broker():
    """Create mock broker with default account values.

    Note: Uses create_mock_account from conftest for consistency.
    """
    broker = AsyncMock()
    broker.get_account.return_value = create_mock_account()
    return broker


@pytest.fixture
def initialized_cb(mock_broker):
    """Create initialized CircuitBreaker ready for testing."""
    cb = CircuitBreaker(max_daily_loss=DEFAULT_MAX_DAILY_LOSS)
    cb.broker = mock_broker
    cb.starting_equity = DEFAULT_STARTING_EQUITY
    cb.peak_equity_today = DEFAULT_STARTING_EQUITY
    cb.last_reset_date = datetime.now().date()
    cb.trading_halted = False
    return cb


@pytest.fixture
def halted_cb(mock_broker):
    """Create halted CircuitBreaker for manual reset tests."""
    cb = CircuitBreaker()
    cb.broker = mock_broker
    cb.trading_halted = True
    cb.halt_triggered_at = datetime.now()
    cb.starting_equity = DEFAULT_STARTING_EQUITY

    # Mock get_account for reset
    account = MagicMock()
    account.equity = str(EXACT_LIMIT_EQUITY)
    cb.broker.get_account.return_value = account

    return cb


def create_account_with_equity(equity: float) -> MagicMock:
    """Helper to create mock account with specific equity only.

    Note: For full account mocking, use create_mock_account from conftest.
    This helper is for tests that only need equity set.
    """
    account = MagicMock()
    account.equity = str(equity)
    return account


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization."""

    def test_default_initialization(self, circuit_breaker):
        """Test CircuitBreaker with default parameters."""
        assert (
            circuit_breaker.max_daily_loss == DEFAULT_MAX_DAILY_LOSS
        ), f"Expected default max_daily_loss {DEFAULT_MAX_DAILY_LOSS}"
        assert circuit_breaker.auto_close_positions is True
        assert circuit_breaker.trading_halted is False
        assert circuit_breaker.starting_balance is None
        assert circuit_breaker.starting_equity is None
        assert circuit_breaker.peak_equity_today is None
        assert circuit_breaker.halt_triggered_at is None
        assert circuit_breaker.last_reset_date is None
        assert circuit_breaker.broker is None

    def test_custom_initialization(self, circuit_breaker_custom):
        """Test CircuitBreaker with custom parameters."""
        assert circuit_breaker_custom.max_daily_loss == CUSTOM_MAX_DAILY_LOSS
        assert circuit_breaker_custom.auto_close_positions is False

    def test_initialization_with_small_loss_limit(self):
        """Test CircuitBreaker with small loss limit."""
        cb = CircuitBreaker(max_daily_loss=SMALL_MAX_DAILY_LOSS)
        assert cb.max_daily_loss == SMALL_MAX_DAILY_LOSS


class TestInitializeWithBroker:
    """Tests for initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_sets_starting_values(self, circuit_breaker, mock_broker):
        """Test that initialize sets starting balance and equity."""
        await circuit_breaker.initialize(mock_broker)

        assert circuit_breaker.broker == mock_broker
        assert (
            circuit_breaker.starting_balance == DEFAULT_STARTING_CASH
        ), f"Expected starting_balance {DEFAULT_STARTING_CASH}"
        assert (
            circuit_breaker.starting_equity == DEFAULT_STARTING_EQUITY
        ), f"Expected starting_equity {DEFAULT_STARTING_EQUITY}"
        assert circuit_breaker.peak_equity_today == DEFAULT_STARTING_EQUITY
        assert circuit_breaker.last_reset_date == datetime.now().date()

    @pytest.mark.asyncio
    async def test_initialize_with_broker_error(self, circuit_breaker, mock_broker):
        """Test initialize raises on broker error."""
        mock_broker.get_account.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await circuit_breaker.initialize(mock_broker)


class TestCheckAndHalt:
    """Tests for check_and_halt method."""

    @pytest.mark.asyncio
    async def test_check_without_initialization_raises(self, circuit_breaker):
        """Test check_and_halt raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await circuit_breaker.check_and_halt()

    @pytest.mark.asyncio
    async def test_check_no_halt_when_profit(self, initialized_cb):
        """Test no halt when making profit."""
        initialized_cb.broker.get_account.return_value = create_account_with_equity(PROFIT_EQUITY)

        result = await initialized_cb.check_and_halt()

        assert result is False, "Should not halt when in profit"
        assert initialized_cb.trading_halted is False

    @pytest.mark.asyncio
    async def test_check_updates_peak_equity(self, initialized_cb):
        """Test that peak equity is updated when equity increases."""
        initialized_cb.broker.get_account.return_value = create_account_with_equity(PROFIT_EQUITY)

        await initialized_cb.check_and_halt()

        assert (
            initialized_cb.peak_equity_today == PROFIT_EQUITY
        ), f"Peak equity should be updated to {PROFIT_EQUITY}"

    @pytest.mark.asyncio
    async def test_check_no_halt_within_limit(self, initialized_cb):
        """Test no halt when loss is within limit."""
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            WITHIN_LIMIT_EQUITY
        )

        result = await initialized_cb.check_and_halt()

        assert result is False, "Should not halt when within loss limit"
        assert initialized_cb.trading_halted is False

    @pytest.mark.asyncio
    async def test_check_halts_at_daily_loss_limit(self, initialized_cb):
        """Test halt when daily loss limit reached."""
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            OVER_LIMIT_EQUITY
        )
        initialized_cb._trigger_halt = AsyncMock()
        initialized_cb.auto_close_positions = False

        result = await initialized_cb.check_and_halt()

        assert result is True, "Should halt when loss exceeds limit"
        initialized_cb._trigger_halt.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_halts_on_rapid_drawdown(self, initialized_cb):
        """Test halt on rapid drawdown from peak."""
        # First, make profit to set peak higher
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            PEAK_EQUITY_FOR_DRAWDOWN
        )
        await initialized_cb.check_and_halt()

        assert (
            initialized_cb.peak_equity_today == PEAK_EQUITY_FOR_DRAWDOWN
        ), "Peak equity should be updated after profit"

        # Now simulate rapid drawdown (2.09% from peak when rapid threshold is 2%)
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            RAPID_DRAWDOWN_EQUITY
        )
        initialized_cb._trigger_halt = AsyncMock()
        initialized_cb.auto_close_positions = False

        result = await initialized_cb.check_and_halt()

        assert result is True, "Should halt on rapid drawdown"
        initialized_cb._trigger_halt.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_returns_true_when_already_halted(self, initialized_cb):
        """Test check returns True when already halted."""
        initialized_cb.trading_halted = True

        result = await initialized_cb.check_and_halt()

        assert result is True, "Should return True when already halted"
        initialized_cb.broker.get_account.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_halts_on_error(self, initialized_cb):
        """Test trading halted on error (fail safe)."""
        initialized_cb.broker.get_account.side_effect = Exception("Network error")

        result = await initialized_cb.check_and_halt()

        assert result is True, "Should halt on error (fail safe)"
        assert initialized_cb.trading_halted is True

    @pytest.mark.asyncio
    async def test_check_resets_on_new_day(self, initialized_cb):
        """Test circuit breaker resets at start of new day."""
        initialized_cb.last_reset_date = datetime.now().date() - timedelta(days=1)
        initialized_cb._reset_for_new_day = AsyncMock()
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            DEFAULT_STARTING_EQUITY
        )

        await initialized_cb.check_and_halt()

        initialized_cb._reset_for_new_day.assert_called_once()


class TestTriggerHalt:
    """Tests for _trigger_halt method."""

    @pytest.fixture
    def cb_for_trigger(self, mock_broker):
        """Create CircuitBreaker for trigger tests."""
        cb = CircuitBreaker(max_daily_loss=DEFAULT_MAX_DAILY_LOSS, auto_close_positions=True)
        cb.broker = mock_broker
        cb.starting_equity = DEFAULT_STARTING_EQUITY
        cb.trading_halted = False
        return cb

    @pytest.mark.asyncio
    async def test_trigger_halt_sets_state(self, cb_for_trigger):
        """Test trigger halt sets state correctly."""
        cb_for_trigger._emergency_close_positions = AsyncMock()

        await cb_for_trigger._trigger_halt(
            EXACT_LIMIT_EQUITY, DEFAULT_MAX_DAILY_LOSS, "daily_loss_limit"
        )

        assert cb_for_trigger.trading_halted is True, "trading_halted should be True after trigger"
        assert cb_for_trigger.halt_triggered_at is not None, "halt_triggered_at should be set"

    @pytest.mark.asyncio
    async def test_trigger_halt_calls_emergency_close(self, cb_for_trigger):
        """Test trigger halt calls emergency close when configured."""
        cb_for_trigger._emergency_close_positions = AsyncMock()

        await cb_for_trigger._trigger_halt(
            EXACT_LIMIT_EQUITY, DEFAULT_MAX_DAILY_LOSS, "daily_loss_limit"
        )

        cb_for_trigger._emergency_close_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_halt_skips_close_when_disabled(self, cb_for_trigger):
        """Test trigger halt skips close when auto_close disabled."""
        cb_for_trigger.auto_close_positions = False
        cb_for_trigger._emergency_close_positions = AsyncMock()

        await cb_for_trigger._trigger_halt(
            EXACT_LIMIT_EQUITY, DEFAULT_MAX_DAILY_LOSS, "daily_loss_limit"
        )

        cb_for_trigger._emergency_close_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_halt_with_rapid_drawdown_reason(self, cb_for_trigger):
        """Test trigger halt with rapid drawdown reason."""
        cb_for_trigger._emergency_close_positions = AsyncMock()

        await cb_for_trigger._trigger_halt(WITHIN_LIMIT_EQUITY, 0.02, "rapid_drawdown")

        assert cb_for_trigger.trading_halted is True


class TestEmergencyClosePositions:
    """Tests for _emergency_close_positions method."""

    @pytest.fixture
    def cb_for_close(self, mock_broker):
        """Create CircuitBreaker for emergency close tests."""
        cb = CircuitBreaker()
        cb.broker = mock_broker
        return cb

    @pytest.mark.asyncio
    async def test_close_positions_with_no_positions(self, cb_for_close):
        """Test close positions when no positions exist."""
        cb_for_close.broker.get_positions.return_value = []

        await cb_for_close._emergency_close_positions()

        cb_for_close.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_positions_submits_sell_orders(self, cb_for_close):
        """Test close positions submits sell orders for each position."""
        pos1 = create_mock_position(TEST_SYMBOL_1, TEST_POSITION_QTY_1)
        pos2 = create_mock_position(TEST_SYMBOL_2, TEST_POSITION_QTY_2)

        cb_for_close.broker.get_positions.return_value = [pos1, pos2]

        order_result = MagicMock()
        order_result.id = "order123"
        cb_for_close.broker.submit_order_advanced.return_value = order_result

        with patch("brokers.order_builder.OrderBuilder") as MockOrderBuilder:
            mock_builder = MagicMock()
            mock_builder.market.return_value = mock_builder
            mock_builder.day.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()
            MockOrderBuilder.return_value = mock_builder

            await cb_for_close._emergency_close_positions()

        assert (
            cb_for_close.broker.submit_order_advanced.call_count == 2
        ), "Should submit orders for both positions"

    @pytest.mark.asyncio
    async def test_close_positions_handles_order_error(self, cb_for_close):
        """Test close positions handles error for individual position."""
        pos1 = create_mock_position(TEST_SYMBOL_1, TEST_POSITION_QTY_1)

        cb_for_close.broker.get_positions.return_value = [pos1]
        cb_for_close.broker.submit_order_advanced.side_effect = Exception("Order failed")

        with patch("brokers.order_builder.OrderBuilder") as MockOrderBuilder:
            mock_builder = MagicMock()
            mock_builder.market.return_value = mock_builder
            mock_builder.day.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()
            MockOrderBuilder.return_value = mock_builder

            # Should not raise
            await cb_for_close._emergency_close_positions()

    @pytest.mark.asyncio
    async def test_close_positions_handles_get_positions_error(self, cb_for_close):
        """Test close positions handles error getting positions."""
        cb_for_close.broker.get_positions.side_effect = Exception("API error")

        # Should not raise
        await cb_for_close._emergency_close_positions()


class TestResetForNewDay:
    """Tests for _reset_for_new_day method."""

    @pytest.fixture
    def cb_for_reset(self, mock_broker):
        """Create CircuitBreaker with halted state for reset tests."""
        cb = CircuitBreaker()
        cb.broker = mock_broker
        cb.starting_equity = DEFAULT_STARTING_EQUITY
        cb.peak_equity_today = PROFIT_EQUITY
        cb.trading_halted = True
        cb.halt_triggered_at = datetime.now()
        cb.last_reset_date = datetime.now().date() - timedelta(days=1)
        return cb

    @pytest.mark.asyncio
    async def test_reset_updates_equity(self, cb_for_reset):
        """Test reset updates equity values."""
        cb_for_reset.broker.get_account.return_value = create_account_with_equity(
            SLIGHT_PROFIT_EQUITY
        )

        await cb_for_reset._reset_for_new_day()

        assert (
            cb_for_reset.starting_equity == SLIGHT_PROFIT_EQUITY
        ), f"Starting equity should be updated to {SLIGHT_PROFIT_EQUITY}"
        assert cb_for_reset.peak_equity_today == SLIGHT_PROFIT_EQUITY

    @pytest.mark.asyncio
    async def test_reset_clears_halt_state(self, cb_for_reset):
        """Test reset clears halt state."""
        cb_for_reset.broker.get_account.return_value = create_account_with_equity(
            DEFAULT_STARTING_EQUITY
        )

        await cb_for_reset._reset_for_new_day()

        assert cb_for_reset.trading_halted is False, "trading_halted should be False after reset"
        assert cb_for_reset.halt_triggered_at is None

    @pytest.mark.asyncio
    async def test_reset_updates_last_reset_date(self, cb_for_reset):
        """Test reset updates last reset date."""
        cb_for_reset.broker.get_account.return_value = create_account_with_equity(
            DEFAULT_STARTING_EQUITY
        )

        await cb_for_reset._reset_for_new_day()

        assert cb_for_reset.last_reset_date == datetime.now().date()

    @pytest.mark.asyncio
    async def test_reset_handles_broker_error(self, cb_for_reset):
        """Test reset handles broker error."""
        cb_for_reset.broker.get_account.side_effect = Exception("API error")

        # Should not raise
        await cb_for_reset._reset_for_new_day()


class TestIsHalted:
    """Tests for is_halted method."""

    def test_is_halted_returns_true_when_halted(self, circuit_breaker):
        """Test is_halted returns True when halted."""
        circuit_breaker.trading_halted = True
        assert circuit_breaker.is_halted() is True, "is_halted should return True when halted"

    def test_is_halted_returns_false_when_not_halted(self, circuit_breaker):
        """Test is_halted returns False when not halted."""
        circuit_breaker.trading_halted = False
        assert circuit_breaker.is_halted() is False, "is_halted should return False when not halted"


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_returns_all_fields(self):
        """Test get_status returns all required fields."""
        cb = CircuitBreaker(max_daily_loss=CUSTOM_MAX_DAILY_LOSS)
        cb.starting_equity = DEFAULT_STARTING_EQUITY
        cb.peak_equity_today = PROFIT_EQUITY
        cb.trading_halted = False
        cb.halt_triggered_at = None
        cb.last_reset_date = datetime.now().date()

        status = cb.get_status()

        assert status["halted"] is False
        assert status["max_daily_loss"] == CUSTOM_MAX_DAILY_LOSS
        assert status["starting_equity"] == DEFAULT_STARTING_EQUITY
        assert status["peak_equity_today"] == PROFIT_EQUITY
        assert status["halt_triggered_at"] is None
        assert status["last_reset_date"] is not None

    def test_get_status_with_halt_triggered(self):
        """Test get_status when halt was triggered."""
        cb = CircuitBreaker()
        cb.trading_halted = True
        cb.halt_triggered_at = TEST_HALT_DATETIME
        cb.starting_equity = DEFAULT_STARTING_EQUITY
        cb.peak_equity_today = DEFAULT_STARTING_EQUITY
        cb.last_reset_date = TEST_DATE

        status = cb.get_status()

        assert status["halted"] is True
        assert (
            status["halt_triggered_at"] == "2024-01-15T10:30:00"
        ), "halt_triggered_at should be ISO format"

    def test_get_status_with_none_values(self, circuit_breaker):
        """Test get_status with None values."""
        status = circuit_breaker.get_status()

        assert status["starting_equity"] is None
        assert status["peak_equity_today"] is None
        assert status["halt_triggered_at"] is None
        assert status["last_reset_date"] is None


class TestManualReset:
    """Tests for manual_reset method."""

    @pytest.mark.asyncio
    async def test_manual_reset_requires_confirmation_token(self, halted_cb):
        """Test manual reset requires confirmation token."""
        with pytest.raises(ValueError, match="confirmation_token='CONFIRM_RESET'"):
            await halted_cb.manual_reset()

    @pytest.mark.asyncio
    async def test_manual_reset_rejects_wrong_token(self, halted_cb):
        """Test manual reset rejects wrong token."""
        with pytest.raises(ValueError, match="confirmation_token='CONFIRM_RESET'"):
            await halted_cb.manual_reset(confirmation_token=WRONG_TOKEN)

    @pytest.mark.asyncio
    async def test_manual_reset_succeeds_with_correct_token(self, halted_cb):
        """Test manual reset succeeds with correct token."""
        result = await halted_cb.manual_reset(confirmation_token=CONFIRM_RESET_TOKEN)

        assert result is True, "Manual reset should succeed with correct token"
        assert halted_cb.trading_halted is False

    @pytest.mark.asyncio
    async def test_manual_reset_enforces_cooldown(self, halted_cb):
        """Test manual reset enforces cooldown period."""
        # First reset
        result1 = await halted_cb.manual_reset(confirmation_token=CONFIRM_RESET_TOKEN)
        assert result1 is True

        # Set halted again
        halted_cb.trading_halted = True

        # Second reset should be rejected due to cooldown
        result2 = await halted_cb.manual_reset(confirmation_token=CONFIRM_RESET_TOKEN)
        assert result2 is False, "Second reset should be rejected due to cooldown"

    @pytest.mark.asyncio
    async def test_manual_reset_force_bypasses_cooldown(self, halted_cb):
        """Test manual reset with force bypasses cooldown."""
        # First reset
        await halted_cb.manual_reset(confirmation_token=CONFIRM_RESET_TOKEN)

        # Set halted again
        halted_cb.trading_halted = True

        # Force reset should succeed
        result = await halted_cb.manual_reset(confirmation_token=CONFIRM_RESET_TOKEN, force=True)
        assert result is True, "Force reset should bypass cooldown"

    @pytest.mark.asyncio
    async def test_manual_reset_tracks_last_reset_time(self, halted_cb):
        """Test manual reset tracks last reset time."""
        assert not hasattr(halted_cb, "_last_manual_reset") or halted_cb._last_manual_reset is None

        await halted_cb.manual_reset(confirmation_token=CONFIRM_RESET_TOKEN)

        assert hasattr(halted_cb, "_last_manual_reset")
        assert halted_cb._last_manual_reset is not None, "Should track last manual reset time"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_exact_loss_limit_triggers_halt(self, initialized_cb):
        """Test that exact loss limit triggers halt."""
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            EXACT_LIMIT_EQUITY
        )
        initialized_cb._trigger_halt = AsyncMock()
        initialized_cb.auto_close_positions = False

        result = await initialized_cb.check_and_halt()

        assert result is True, "Exact loss limit should trigger halt"
        initialized_cb._trigger_halt.assert_called_once()

    @pytest.mark.asyncio
    async def test_just_under_limit_does_not_halt(self, initialized_cb):
        """Test loss just under limit does not halt."""
        # Need to stay under both daily loss limit (3%) AND rapid drawdown (2%)
        # 98100 is 1.9% loss - under both thresholds
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            UNDER_BOTH_THRESHOLDS_EQUITY
        )

        result = await initialized_cb.check_and_halt()

        assert result is False, "1.9% loss should not trigger halt (under 2% rapid drawdown)"
        assert initialized_cb.trading_halted is False

    @pytest.mark.asyncio
    async def test_peak_not_updated_on_decline(self, initialized_cb):
        """Test peak equity not updated when equity declines."""
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            SLIGHT_DECLINE_EQUITY
        )

        await initialized_cb.check_and_halt()

        assert (
            initialized_cb.peak_equity_today == DEFAULT_STARTING_EQUITY
        ), "Peak equity should remain unchanged on decline"

    def test_initialization_with_zero_loss_limit(self):
        """Test initialization with zero loss limit (will trigger immediately)."""
        cb = CircuitBreaker(max_daily_loss=0.0)
        assert cb.max_daily_loss == 0.0

    @pytest.mark.asyncio
    async def test_multiple_checks_within_limit(self, initialized_cb):
        """Test multiple checks while within limit."""
        initialized_cb.broker.get_account.return_value = create_account_with_equity(
            SLIGHT_DECLINE_EQUITY
        )

        for i in range(5):
            result = await initialized_cb.check_and_halt()
            assert result is False, f"Check {i+1} should not halt"

        assert initialized_cb.trading_halted is False

    @pytest.mark.asyncio
    async def test_halt_persists_across_checks(self, initialized_cb):
        """Test that halt persists across multiple checks."""
        initialized_cb.trading_halted = True

        for i in range(3):
            result = await initialized_cb.check_and_halt()
            assert result is True, f"Check {i+1} should return True when halted"

        assert initialized_cb.trading_halted is True
