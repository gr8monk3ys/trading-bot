"""
Tests for Phase 1.1: Atomic Order Enforcement via OrderGateway.

These tests verify that:
1. Direct broker access is blocked when gateway enforcement is enabled
2. OrderGateway can submit orders with valid token
3. Invalid tokens are rejected
4. BaseStrategy routes orders through gateway
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


# Test GatewayBypassError exception
class TestGatewayBypassError:
    """Tests for GatewayBypassError exception."""

    def test_exception_exists(self):
        """GatewayBypassError should be importable."""
        from brokers.alpaca_broker import GatewayBypassError
        assert GatewayBypassError is not None

    def test_exception_is_broker_error(self):
        """GatewayBypassError should inherit from BrokerError."""
        from brokers.alpaca_broker import BrokerError, GatewayBypassError
        assert issubclass(GatewayBypassError, BrokerError)

    def test_exception_message(self):
        """GatewayBypassError should have informative message."""
        from brokers.alpaca_broker import GatewayBypassError
        error = GatewayBypassError("Test message")
        assert "Test message" in str(error)


# Test AlpacaBroker gateway enforcement
class TestAlpacaBrokerGatewayEnforcement:
    """Tests for AlpacaBroker gateway enforcement methods."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker with gateway enforcement attributes."""
        # We can't easily instantiate AlpacaBroker without credentials,
        # so we'll test the logic by mocking
        from brokers.alpaca_broker import AlpacaBroker

        # Create a mock that has the same methods
        broker = MagicMock(spec=AlpacaBroker)
        broker._gateway_required = False
        broker._gateway_caller_token = None

        return broker

    def test_enable_gateway_requirement_sets_flag(self):
        """enable_gateway_requirement should set _gateway_required flag."""
        from brokers.alpaca_broker import AlpacaBroker

        # Create minimal mock
        broker = MagicMock()
        broker._gateway_required = False
        broker._gateway_caller_token = None

        # Call the actual method
        AlpacaBroker.enable_gateway_requirement(broker)

        assert broker._gateway_required is True
        assert broker._gateway_caller_token is not None
        assert len(broker._gateway_caller_token) == 32  # 16 bytes hex = 32 chars

    def test_enable_gateway_requirement_returns_token(self):
        """enable_gateway_requirement should return authorization token."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = MagicMock()
        broker._gateway_required = False
        broker._gateway_caller_token = None

        token = AlpacaBroker.enable_gateway_requirement(broker)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) == 32

    def test_disable_gateway_requirement_clears_flag(self):
        """disable_gateway_requirement should clear enforcement."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = MagicMock()
        broker._gateway_required = True
        broker._gateway_caller_token = "some_token"

        AlpacaBroker.disable_gateway_requirement(broker)

        assert broker._gateway_required is False
        assert broker._gateway_caller_token is None


# Test submit_order_advanced enforcement
class TestSubmitOrderAdvancedEnforcement:
    """Tests for submit_order_advanced gateway enforcement."""

    @pytest.fixture
    def mock_order_request(self):
        """Create a mock order request."""
        order = MagicMock()
        order.symbol = "AAPL"
        order.qty = 100
        order.side = "buy"
        return order

    async def test_submit_order_blocked_when_gateway_required(self, mock_order_request):
        """submit_order_advanced should raise GatewayBypassError when enforcement enabled."""
        from brokers.alpaca_broker import GatewayBypassError

        # Create mock broker
        broker = MagicMock()
        broker._gateway_required = True

        # Import the actual method
        from brokers.alpaca_broker import AlpacaBroker

        # Test that exception is raised
        with pytest.raises(GatewayBypassError) as exc_info:
            await AlpacaBroker.submit_order_advanced(broker, mock_order_request)

        assert "Direct order submission is disabled" in str(exc_info.value)
        assert "OrderGateway" in str(exc_info.value)


# Test _internal_submit_order
class TestInternalSubmitOrder:
    """Tests for _internal_submit_order method."""

    @pytest.fixture
    def mock_order_request(self):
        """Create a mock order request."""
        order = MagicMock()
        order.symbol = "AAPL"
        order.qty = 100
        order.side = MagicMock()
        order.side.value = "buy"
        return order

    async def test_internal_submit_rejects_invalid_token(self, mock_order_request):
        """_internal_submit_order should reject invalid tokens."""
        from brokers.alpaca_broker import AlpacaBroker, GatewayBypassError

        broker = MagicMock()
        broker._gateway_required = True
        broker._gateway_caller_token = "valid_token_12345678901234"

        with pytest.raises(GatewayBypassError) as exc_info:
            await AlpacaBroker._internal_submit_order(
                broker, mock_order_request, gateway_token="wrong_token"
            )

        assert "Invalid gateway authorization token" in str(exc_info.value)

    async def test_internal_submit_accepts_valid_token(self, mock_order_request):
        """_internal_submit_order should accept valid tokens."""
        from brokers.alpaca_broker import AlpacaBroker

        valid_token = "valid_token_12345678901234"

        # Create mock broker with all needed attributes
        broker = MagicMock()
        broker._gateway_required = True
        broker._gateway_caller_token = valid_token
        broker._calculate_market_impact = AsyncMock(return_value={
            "participation_rate": 0.001,
            "expected_slippage_pct": 0.001,
            "safe_to_trade": True,
        })
        broker.ORDER_API_TIMEOUT = 15.0
        broker.MAX_PARTICIPATION_RATE = 0.05

        # Mock the trading client
        mock_result = MagicMock()
        mock_result.id = "order123"
        mock_result.symbol = "AAPL"
        mock_result.qty = 100
        mock_result.type = "market"
        mock_result.order_class = "simple"
        mock_result.notional = None

        broker._async_call_with_timeout = AsyncMock(return_value=mock_result)

        # Should not raise - use the actual method
        result = await AlpacaBroker._internal_submit_order(
            broker, mock_order_request, gateway_token=valid_token
        )

        assert result is not None
        assert result.id == "order123"


# Test OrderGateway integration
class TestOrderGatewayIntegration:
    """Tests for OrderGateway integration with broker enforcement."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = MagicMock()
        broker._gateway_required = False
        broker._gateway_caller_token = None
        broker.enable_gateway_requirement = MagicMock(
            return_value="test_token_1234567890123456"
        )
        broker.get_positions = AsyncMock(return_value=[])
        return broker

    def test_gateway_enables_enforcement_on_init(self, mock_broker):
        """OrderGateway should enable enforcement on initialization."""
        from utils.order_gateway import OrderGateway

        gateway = OrderGateway(mock_broker, enforce_gateway=True)

        mock_broker.enable_gateway_requirement.assert_called_once()
        assert gateway._gateway_token == "test_token_1234567890123456"

    def test_gateway_skips_enforcement_when_disabled(self, mock_broker):
        """OrderGateway should skip enforcement when enforce_gateway=False."""
        from utils.order_gateway import OrderGateway

        gateway = OrderGateway(mock_broker, enforce_gateway=False)

        mock_broker.enable_gateway_requirement.assert_not_called()
        assert gateway._gateway_token is None


# Test BaseStrategy gateway usage
class TestBaseStrategyGatewayUsage:
    """Tests for BaseStrategy using OrderGateway."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = MagicMock()
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_account = AsyncMock(return_value=MagicMock(equity="100000"))
        return broker

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock OrderGateway."""
        from utils.order_gateway import OrderResult

        gateway = MagicMock()
        gateway.submit_order = AsyncMock(return_value=OrderResult(
            success=True,
            order_id="order123",
            symbol="AAPL",
            side="buy",
            quantity=100,
        ))
        gateway.submit_exit_order = AsyncMock(return_value=OrderResult(
            success=True,
            order_id="exit123",
            symbol="AAPL",
            side="sell",
            quantity=100,
        ))
        return gateway

    def test_strategy_accepts_order_gateway(self, mock_broker, mock_gateway):
        """BaseStrategy should accept order_gateway parameter."""
        from strategies.base_strategy import BaseStrategy

        # Can't instantiate abstract class directly, so test the __init__ signature
        # by checking the attribute is set
        class TestStrategy(BaseStrategy):
            async def analyze_symbol(self, symbol):
                return {"action": "hold"}
            async def execute_trade(self, symbol, signal):
                pass

        strategy = TestStrategy(
            broker=mock_broker,
            order_gateway=mock_gateway,
        )

        assert strategy.order_gateway is mock_gateway

    async def test_submit_entry_order_uses_gateway(self, mock_broker, mock_gateway):
        """submit_entry_order should route through OrderGateway."""
        from strategies.base_strategy import BaseStrategy

        class TestStrategy(BaseStrategy):
            async def analyze_symbol(self, symbol):
                return {"action": "hold"}
            async def execute_trade(self, symbol, signal):
                pass

        strategy = TestStrategy(
            broker=mock_broker,
            order_gateway=mock_gateway,
        )

        mock_order = MagicMock()
        mock_order.symbol = "AAPL"

        result = await strategy.submit_entry_order(mock_order)

        mock_gateway.submit_order.assert_called_once()
        assert result.success is True

    async def test_submit_exit_order_uses_gateway(self, mock_broker, mock_gateway):
        """submit_exit_order should route through OrderGateway."""
        from strategies.base_strategy import BaseStrategy

        class TestStrategy(BaseStrategy):
            async def analyze_symbol(self, symbol):
                return {"action": "hold"}
            async def execute_trade(self, symbol, signal):
                pass

        # Add a position to exit
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        strategy = TestStrategy(
            broker=mock_broker,
            order_gateway=mock_gateway,
        )

        await strategy.submit_exit_order("AAPL", 100)

        mock_gateway.submit_exit_order.assert_called_once()


# Integration test
class TestFullGatewayEnforcementFlow:
    """Integration tests for complete gateway enforcement flow."""

    async def test_direct_access_blocked_gateway_allowed(self):
        """
        Verify that:
        1. Direct broker access is blocked
        2. Gateway access works with valid token
        """
        from brokers.alpaca_broker import AlpacaBroker, GatewayBypassError

        # Create minimal mock broker
        broker = MagicMock()
        broker._gateway_required = False
        broker._gateway_caller_token = None
        broker.get_positions = AsyncMock(return_value=[])

        # Step 1: Enable enforcement
        token = AlpacaBroker.enable_gateway_requirement(broker)
        assert broker._gateway_required is True

        # Step 2: Direct access should fail
        mock_order = MagicMock()
        mock_order.symbol = "AAPL"
        mock_order.qty = 100

        with pytest.raises(GatewayBypassError):
            await AlpacaBroker.submit_order_advanced(broker, mock_order)

        # Step 3: Gateway with token should work
        broker._async_call_with_timeout = AsyncMock(return_value=MagicMock(
            id="order123",
            symbol="AAPL",
            qty=100,
            type="market",
            order_class="simple",
            notional=None,
        ))
        broker._calculate_market_impact = AsyncMock(return_value={
            "participation_rate": 0.001,
            "expected_slippage_pct": 0.001,
            "safe_to_trade": True,
        })
        broker.ORDER_API_TIMEOUT = 15.0
        broker.MAX_PARTICIPATION_RATE = 0.05

        result = await AlpacaBroker._internal_submit_order(
            broker, mock_order, gateway_token=token
        )
        assert result.id == "order123"
