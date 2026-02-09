"""
Tests for Multi-Broker Failover

Tests:
- Broker interface conformance
- Multi-broker manager failover
- Health monitoring
- Automatic failback
- Operation routing
"""

from datetime import datetime

import pytest

from brokers.broker_interface import (
    AccountInfo,
    Bar,
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerStatus,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from brokers.multi_broker import (
    BrokerHealth,
    FailoverEvent,
    FailoverLog,
    MultiBrokerManager,
    print_broker_status,
)


class MockBroker(BrokerInterface):
    """Mock broker for testing."""

    def __init__(self, name: str = "MockBroker", is_paper: bool = True):
        self._name = name
        self._is_paper = is_paper
        self._connected = True
        self._should_fail = False
        self._fail_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    def set_should_fail(self, should_fail: bool):
        """Set whether operations should fail."""
        self._should_fail = should_fail

    async def connect(self) -> bool:
        if self._should_fail:
            raise BrokerConnectionError("Connection failed")
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_status(self) -> BrokerStatus:
        if self._should_fail:
            return BrokerStatus.DISCONNECTED
        return BrokerStatus.CONNECTED if self._connected else BrokerStatus.DISCONNECTED

    async def health_check(self) -> bool:
        if self._should_fail:
            self._fail_count += 1
            return False
        return self._connected

    async def get_account(self) -> AccountInfo:
        if self._should_fail:
            raise BrokerError("Get account failed")
        return AccountInfo(
            broker_name=self.name,
            account_id="TEST123",
            equity=100000,
            cash=50000,
            buying_power=150000,
            portfolio_value=100000,
        )

    async def get_positions(self) -> list:
        if self._should_fail:
            raise BrokerError("Get positions failed")
        return [
            Position(
                symbol="AAPL",
                quantity=100,
                avg_entry_price=150.0,
                market_value=15000,
                unrealized_pnl=500,
                unrealized_pnl_pct=0.033,
                broker_name=self.name,
            )
        ]

    async def get_position(self, symbol: str):
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def submit_order(self, request: OrderRequest) -> Order:
        if self._should_fail:
            raise BrokerError("Order submission failed")
        return Order(
            order_id="ORD123",
            client_order_id="CLIENT123",
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            status=OrderStatus.ACCEPTED,
            broker_name=self.name,
        )

    async def cancel_order(self, order_id: str) -> bool:
        if self._should_fail:
            return False
        return True

    async def get_order(self, order_id: str):
        if self._should_fail:
            return None
        return Order(
            order_id=order_id,
            client_order_id="CLIENT123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.FILLED,
        )

    async def get_orders(self, status=None, symbols=None, limit=100):
        if self._should_fail:
            return []
        return []

    async def cancel_all_orders(self) -> int:
        if self._should_fail:
            return 0
        return 0

    async def get_bars(self, symbol, timeframe, start, end=None, limit=1000):
        if self._should_fail:
            raise BrokerError("Get bars failed")
        return [
            Bar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=1000000,
            )
        ]

    async def get_latest_quote(self, symbol: str):
        if self._should_fail:
            raise BrokerError("Get quote failed")
        return {"bid": 152.0, "ask": 153.0, "last": 152.5, "volume": 50000}

    async def get_clock(self):
        if self._should_fail:
            raise BrokerError("Get clock failed")
        return {"is_open": True, "next_open": "09:30", "next_close": "16:00"}

    async def close_position(self, symbol: str):
        if self._should_fail:
            return None
        return Order(
            order_id="ORD456",
            client_order_id="CLIENT456",
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.FILLED,
        )

    async def close_all_positions(self):
        if self._should_fail:
            return []
        return []


class TestBrokerInterface:
    """Tests for broker interface compliance."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        return MockBroker()

    @pytest.mark.asyncio
    async def test_connect(self, mock_broker):
        """Test connect method."""
        result = await mock_broker.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_account(self, mock_broker):
        """Test get_account returns AccountInfo."""
        account = await mock_broker.get_account()
        assert isinstance(account, AccountInfo)
        assert account.equity > 0

    @pytest.mark.asyncio
    async def test_get_positions(self, mock_broker):
        """Test get_positions returns list of Position."""
        positions = await mock_broker.get_positions()
        assert isinstance(positions, list)
        if positions:
            assert isinstance(positions[0], Position)

    @pytest.mark.asyncio
    async def test_submit_order(self, mock_broker):
        """Test submit_order returns Order."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        order = await mock_broker.submit_order(request)
        assert isinstance(order, Order)
        assert order.status == OrderStatus.ACCEPTED

    @pytest.mark.asyncio
    async def test_health_check(self, mock_broker):
        """Test health_check returns bool."""
        result = await mock_broker.health_check()
        assert isinstance(result, bool)


class TestMultiBrokerManager:
    """Tests for MultiBrokerManager class."""

    @pytest.fixture
    def primary_broker(self):
        """Create primary broker."""
        return MockBroker("Primary")

    @pytest.fixture
    def backup_broker(self):
        """Create backup broker."""
        return MockBroker("Backup")

    @pytest.fixture
    def manager(self, primary_broker, backup_broker):
        """Create multi-broker manager."""
        return MultiBrokerManager(
            primary=primary_broker,
            backups=[backup_broker],
            health_check_interval=1,
            failure_threshold=2,
            recovery_threshold=2,
            auto_start_monitoring=False,
        )

    def test_initial_state(self, manager, primary_broker):
        """Test initial state uses primary broker."""
        assert manager.active_broker == primary_broker
        assert manager.is_failed_over is False

    def test_name_property(self, manager):
        """Test name includes active broker."""
        assert "Primary" in manager.name

    def test_is_paper_property(self, manager):
        """Test is_paper forwards to active broker."""
        assert manager.is_paper is True

    @pytest.mark.asyncio
    async def test_get_account_through_manager(self, manager):
        """Test get_account routes to active broker."""
        account = await manager.get_account()
        assert isinstance(account, AccountInfo)

    @pytest.mark.asyncio
    async def test_get_positions_through_manager(self, manager):
        """Test get_positions routes to active broker."""
        positions = await manager.get_positions()
        assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_submit_order_through_manager(self, manager):
        """Test submit_order routes to active broker."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order = await manager.submit_order(request)
        assert isinstance(order, Order)

    @pytest.mark.asyncio
    async def test_failover_on_primary_failure(self, manager, primary_broker, backup_broker):
        """Test failover when primary fails."""
        # Simulate primary failure
        primary_broker.set_should_fail(True)

        # Force check all brokers
        await manager._check_all_brokers()

        # Check multiple times to exceed threshold
        for _ in range(3):
            await manager._check_broker_health(primary_broker)

        # Evaluate failover
        await manager._evaluate_failover()

        # Should have failed over to backup
        if manager._broker_health[primary_broker.name].consecutive_failures >= manager.failure_threshold:
            assert manager.is_failed_over

    @pytest.mark.asyncio
    async def test_operation_succeeds_on_backup_after_failover(self, manager, primary_broker, backup_broker):
        """Test operations work after failover."""
        primary_broker.set_should_fail(True)

        # Operation should still succeed by falling through to backup
        account = await manager.get_account()
        assert isinstance(account, AccountInfo)

    @pytest.mark.asyncio
    async def test_connect_connects_all_brokers(self, manager):
        """Test connect connects to all brokers."""
        result = await manager.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_disconnects_all(self, manager):
        """Test disconnect disconnects all brokers."""
        await manager.disconnect()
        # Should not raise

    def test_get_broker_health(self, manager):
        """Test get_broker_health returns health dict."""
        health = manager.get_broker_health()
        assert "Primary" in health
        assert isinstance(health["Primary"], BrokerHealth)

    def test_get_failover_log(self, manager):
        """Test get_failover_log returns list."""
        log = manager.get_failover_log()
        assert isinstance(log, list)

    def test_get_status_summary(self, manager):
        """Test get_status_summary returns dict."""
        summary = manager.get_status_summary()
        assert "active_broker" in summary
        assert "is_failed_over" in summary
        assert "primary_health" in summary


class TestBrokerHealth:
    """Tests for BrokerHealth dataclass."""

    def test_is_healthy_when_connected(self):
        """Test is_healthy returns True when connected with no failures."""
        health = BrokerHealth(
            broker_name="Test",
            status=BrokerStatus.CONNECTED,
            last_check=datetime.now(),
            last_success=datetime.now(),
            consecutive_failures=0,
            response_time_ms=50.0,
        )

        assert health.is_healthy is True

    def test_is_healthy_false_with_failures(self):
        """Test is_healthy returns False with failures."""
        health = BrokerHealth(
            broker_name="Test",
            status=BrokerStatus.CONNECTED,
            last_check=datetime.now(),
            last_success=datetime.now(),
            consecutive_failures=1,
            response_time_ms=50.0,
        )

        assert health.is_healthy is False

    def test_is_healthy_false_when_disconnected(self):
        """Test is_healthy returns False when disconnected."""
        health = BrokerHealth(
            broker_name="Test",
            status=BrokerStatus.DISCONNECTED,
            last_check=datetime.now(),
            last_success=None,
            consecutive_failures=0,
            response_time_ms=None,
        )

        assert health.is_healthy is False


class TestFailoverLog:
    """Tests for FailoverLog dataclass."""

    def test_failover_log_creation(self):
        """Test creating a failover log entry."""
        log = FailoverLog(
            timestamp=datetime.now(),
            event=FailoverEvent.FAILOVER_TO_BACKUP,
            from_broker="Primary",
            to_broker="Backup",
            reason="Primary failed health check",
        )

        assert log.event == FailoverEvent.FAILOVER_TO_BACKUP
        assert log.from_broker == "Primary"


class TestFailoverEvent:
    """Tests for FailoverEvent enum."""

    def test_all_events_exist(self):
        """Test all expected events exist."""
        expected = [
            "PRIMARY_FAILED",
            "FAILOVER_TO_BACKUP",
            "FAILBACK_TO_PRIMARY",
            "BACKUP_FAILED",
            "ALL_BROKERS_FAILED",
            "HEALTH_CHECK_PASSED",
            "HEALTH_CHECK_FAILED",
        ]

        for name in expected:
            assert hasattr(FailoverEvent, name)


class TestOrderRequest:
    """Tests for OrderRequest dataclass."""

    def test_market_order_request(self):
        """Test creating a market order request."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        assert request.symbol == "AAPL"
        assert request.side == OrderSide.BUY
        assert request.quantity == 100

    def test_limit_order_request(self):
        """Test creating a limit order request."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )

        assert request.limit_price == 150.0

    def test_bracket_order_request(self):
        """Test creating a bracket order request."""
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            take_profit_price=170.0,
            stop_loss_price=140.0,
        )

        assert request.take_profit_price == 170.0
        assert request.stop_loss_price == 140.0


class TestAccountInfo:
    """Tests for AccountInfo dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        account = AccountInfo(
            broker_name="Test",
            account_id="ABC123",
            equity=100000,
            cash=50000,
            buying_power=150000,
            portfolio_value=100000,
        )

        d = account.to_dict()
        assert d["broker_name"] == "Test"
        assert d["equity"] == 100000


class TestPosition:
    """Tests for Position dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
            market_value=15500,
            unrealized_pnl=500,
            unrealized_pnl_pct=0.033,
        )

        d = position.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["quantity"] == 100


class TestOrder:
    """Tests for Order dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        order = Order(
            order_id="ORD123",
            client_order_id="CLIENT123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.FILLED,
        )

        d = order.to_dict()
        assert d["order_id"] == "ORD123"
        assert d["side"] == "buy"
        assert d["status"] == "filled"


class TestPrintBrokerStatus:
    """Tests for print_broker_status function."""

    def test_print_status_no_error(self, capsys):
        """Test print_broker_status doesn't raise."""
        primary = MockBroker("Primary")
        backup = MockBroker("Backup")
        manager = MultiBrokerManager(
            primary=primary,
            backups=[backup],
            auto_start_monitoring=False,
        )

        print_broker_status(manager)

        captured = capsys.readouterr()
        assert "MULTI-BROKER STATUS" in captured.out
        assert "Primary" in captured.out
