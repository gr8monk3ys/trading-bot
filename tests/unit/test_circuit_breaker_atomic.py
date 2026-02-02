"""
Tests for Phase 1.2: Circuit Breaker Atomicity.

These tests verify that:
1. TradingHaltedException is raised instead of returning bool
2. True peak equity tracking works correctly (doesn't reset on recovery)
3. OrderGateway properly catches TradingHaltedException
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestTradingHaltedException:
    """Tests for TradingHaltedException."""

    def test_exception_exists(self):
        """TradingHaltedException should be importable."""
        from utils.circuit_breaker import TradingHaltedException
        assert TradingHaltedException is not None

    def test_exception_attributes(self):
        """TradingHaltedException should have reason and loss_pct attributes."""
        from utils.circuit_breaker import TradingHaltedException

        error = TradingHaltedException(
            "Trading halted",
            reason="daily_loss_limit",
            loss_pct=0.05,
        )

        assert error.reason == "daily_loss_limit"
        assert error.loss_pct == 0.05
        assert "Trading halted" in str(error)

    def test_exception_with_event(self):
        """TradingHaltedException should support economic event info."""
        from utils.circuit_breaker import TradingHaltedException

        error = TradingHaltedException(
            "Event blocking",
            reason="economic_event",
            event_name="FOMC",
        )

        assert error.reason == "economic_event"
        assert error.event_name == "FOMC"


class TestEnforceBeforeOrder:
    """Tests for enforce_before_order atomic method."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        from utils.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(max_daily_loss=0.03)
        return cb

    async def test_raises_when_already_halted(self, circuit_breaker):
        """enforce_before_order should raise when already halted."""
        from utils.circuit_breaker import TradingHaltedException

        # Mock broker
        mock_broker = MagicMock()
        mock_broker.get_account = AsyncMock()

        circuit_breaker.broker = mock_broker
        circuit_breaker.trading_halted = True
        circuit_breaker._halt_reason = "daily_loss_limit"
        circuit_breaker._halt_loss_pct = 0.035

        with pytest.raises(TradingHaltedException) as exc_info:
            await circuit_breaker.enforce_before_order()

        assert exc_info.value.reason == "daily_loss_limit"
        assert exc_info.value.loss_pct == 0.035

    async def test_raises_on_economic_event(self, circuit_breaker):
        """enforce_before_order should raise for high-impact economic events."""
        from utils.circuit_breaker import TradingHaltedException

        # Mock broker
        mock_broker = MagicMock()
        circuit_breaker.broker = mock_broker
        circuit_breaker.last_reset_date = datetime.now().date()

        # Mock is_blocked_by_event to return True
        circuit_breaker.is_blocked_by_event = MagicMock(
            return_value=(True, "FOMC", 2.5)
        )

        with pytest.raises(TradingHaltedException) as exc_info:
            await circuit_breaker.enforce_before_order(is_exit_order=False)

        assert exc_info.value.reason == "economic_event"
        assert exc_info.value.event_name == "FOMC"

    async def test_allows_exit_during_economic_event(self, circuit_breaker):
        """enforce_before_order should allow exits during economic events."""
        # Mock broker
        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        circuit_breaker.broker = mock_broker
        circuit_breaker.last_reset_date = datetime.now().date()
        circuit_breaker.starting_equity = 100000
        circuit_breaker.peak_equity_today = 100000
        circuit_breaker._true_peak_equity = 100000

        # Mock is_blocked_by_event to return True (but exit should bypass)
        circuit_breaker.is_blocked_by_event = MagicMock(
            return_value=(True, "FOMC", 2.5)
        )

        # Should NOT raise for exit orders
        await circuit_breaker.enforce_before_order(is_exit_order=True)

    async def test_raises_on_daily_loss_exceeded(self, circuit_breaker):
        """enforce_before_order should raise when daily loss exceeded."""
        from utils.circuit_breaker import TradingHaltedException

        # Mock broker with loss
        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.equity = "96000"  # 4% loss from 100000
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.get_positions = AsyncMock(return_value=[])

        circuit_breaker.broker = mock_broker
        circuit_breaker.last_reset_date = datetime.now().date()
        circuit_breaker.starting_equity = 100000
        circuit_breaker.peak_equity_today = 100000
        circuit_breaker._true_peak_equity = 100000
        circuit_breaker.is_blocked_by_event = MagicMock(return_value=(False, None, None))

        with pytest.raises(TradingHaltedException) as exc_info:
            await circuit_breaker.enforce_before_order()

        assert exc_info.value.reason == "daily_loss_limit"
        assert exc_info.value.loss_pct >= 0.03  # 3% threshold


class TestTruePeakEquityTracking:
    """Tests for true peak equity tracking (doesn't reset on recovery)."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        from utils.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(max_daily_loss=0.05)  # 5% for easier testing
        return cb

    async def test_true_peak_tracks_maximum(self, circuit_breaker):
        """_true_peak_equity should track the true maximum, not reset on recovery."""
        mock_broker = MagicMock()
        mock_broker.get_positions = AsyncMock(return_value=[])

        circuit_breaker.broker = mock_broker
        circuit_breaker.last_reset_date = datetime.now().date()
        circuit_breaker.starting_equity = 100000
        circuit_breaker.peak_equity_today = 100000
        circuit_breaker._true_peak_equity = 100000
        circuit_breaker.is_blocked_by_event = MagicMock(return_value=(False, None, None))

        # Simulate: equity goes up to 101000 (small increase)
        mock_account = MagicMock()
        mock_account.equity = "101000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        await circuit_breaker.enforce_before_order()
        assert circuit_breaker._true_peak_equity == 101000

        # Simulate: equity drops slightly to 100500 (0.5% drawdown from peak - well under threshold)
        mock_account.equity = "100500"

        # Clear cache to force fresh check
        circuit_breaker._order_check_cache = None
        circuit_breaker._order_check_time = None

        await circuit_breaker.enforce_before_order()

        # True peak should NOT reset - it should still be 101000
        assert circuit_breaker._true_peak_equity == 101000

        # Old peak_equity_today might have reset (backward compat), but true peak stays
        assert circuit_breaker._true_peak_equity >= circuit_breaker.peak_equity_today

    async def test_true_peak_catches_cumulative_drawdown(self, circuit_breaker):
        """True peak should catch drawdowns that partial recovery would mask."""
        from utils.circuit_breaker import TradingHaltedException

        mock_broker = MagicMock()
        mock_broker.get_positions = AsyncMock(return_value=[])

        # Scenario: Start at 100K, peak at 105K, drop to 100K, partial recover to 102K, drop to 98K
        # Old logic: Peak resets to 102K, drawdown appears as 3.9% (102K->98K)
        # New logic: True peak stays at 105K, drawdown is 6.67% (105K->98K)

        circuit_breaker.broker = mock_broker
        circuit_breaker.last_reset_date = datetime.now().date()
        circuit_breaker.starting_equity = 100000
        circuit_breaker.peak_equity_today = 100000
        circuit_breaker._true_peak_equity = None  # Will be set on first check
        circuit_breaker.is_blocked_by_event = MagicMock(return_value=(False, None, None))

        # Step 1: Peak at 105K
        mock_account = MagicMock()
        mock_account.equity = "105000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        await circuit_breaker.enforce_before_order()
        assert circuit_breaker._true_peak_equity == 105000

        # Step 2: Drop to 100K, recover to 102K
        circuit_breaker._order_check_cache = None
        mock_account.equity = "102000"
        # DON'T reset true peak - it should stay at 105K
        await circuit_breaker.enforce_before_order()
        assert circuit_breaker._true_peak_equity == 105000  # Still 105K

        # Step 3: Drop to 98K (6.67% from true peak - exceeds 5% * 0.67 = 3.35% rapid drawdown)
        circuit_breaker._order_check_cache = None
        mock_account.equity = "98000"

        # This should trigger halt due to rapid drawdown from TRUE peak
        # Old logic would see 102K -> 98K = 3.9% (barely triggers at 3.35%)
        # New logic sees 105K -> 98K = 6.67% (definitely triggers)
        with pytest.raises(TradingHaltedException) as exc_info:
            await circuit_breaker.enforce_before_order()

        assert exc_info.value.reason == "rapid_drawdown"


class TestBackwardCompatibility:
    """Tests for backward compatibility with check_before_order."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        from utils.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(max_daily_loss=0.03)
        return cb

    async def test_check_before_order_returns_bool(self, circuit_breaker):
        """check_before_order should still return bool for backward compat."""
        mock_broker = MagicMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        circuit_breaker.broker = mock_broker
        circuit_breaker.last_reset_date = datetime.now().date()
        circuit_breaker.starting_equity = 100000
        circuit_breaker.peak_equity_today = 100000
        circuit_breaker._true_peak_equity = 100000
        circuit_breaker.is_blocked_by_event = MagicMock(return_value=(False, None, None))

        # Should return False (not halted)
        result = await circuit_breaker.check_before_order()
        assert result is False

    async def test_check_before_order_returns_true_on_halt(self, circuit_breaker):
        """check_before_order should return True when halted."""
        mock_broker = MagicMock()
        circuit_breaker.broker = mock_broker
        circuit_breaker.trading_halted = True

        # Should return True (halted)
        result = await circuit_breaker.check_before_order()
        assert result is True


class TestOrderGatewayCircuitBreakerIntegration:
    """Tests for OrderGateway using atomic circuit breaker."""

    async def test_gateway_catches_trading_halted_exception(self):
        """OrderGateway should catch TradingHaltedException and reject order."""
        from utils.order_gateway import OrderGateway, OrderResult
        from utils.circuit_breaker import TradingHaltedException

        # Create mock broker and circuit breaker
        mock_broker = MagicMock()
        mock_broker.get_positions = AsyncMock(return_value=[])

        mock_circuit_breaker = MagicMock()
        mock_circuit_breaker.enforce_before_order = AsyncMock(
            side_effect=TradingHaltedException(
                "Trading halted",
                reason="daily_loss_limit",
                loss_pct=0.035,
            )
        )

        # Create gateway
        gateway = OrderGateway(
            broker=mock_broker,
            circuit_breaker=mock_circuit_breaker,
            enforce_gateway=False,  # Don't enable broker enforcement for this test
        )

        # Create mock order
        mock_order = MagicMock()
        mock_order.symbol = "AAPL"
        mock_order.qty = 100
        mock_order.side = "buy"

        # Submit order - should be rejected
        result = await gateway.submit_order(
            order_request=mock_order,
            strategy_name="TestStrategy",
        )

        assert result.success is False
        assert "Circuit breaker" in result.rejection_reason
        assert "daily_loss_limit" in result.rejection_reason
