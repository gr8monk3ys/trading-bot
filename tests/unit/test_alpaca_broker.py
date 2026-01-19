#!/usr/bin/env python3
"""
Comprehensive unit tests for brokers/alpaca_broker.py

Tests cover:
- Custom exceptions (BrokerError, BrokerConnectionError, OrderError)
- retry_with_backoff decorator
- AlpacaBroker class initialization
- Symbol validation
- Subscriber management
- Account and position methods
- Order methods (submit, cancel, replace, get)
- Market data methods (get_last_price, get_bars, get_news)
- WebSocket handlers
"""


# Mock the config module before importing the broker
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.modules["config"] = Mock(
    ALPACA_CREDS={"API_KEY": "test_api_key", "API_SECRET": "test_api_secret"},
    SYMBOLS=["AAPL", "MSFT", "GOOGL"],
)

from alpaca.trading.enums import QueryOrderStatus

# ============================================================================
# Test Custom Exceptions
# ============================================================================


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_broker_error_is_exception(self):
        """BrokerError should be an Exception subclass."""
        from brokers.alpaca_broker import BrokerError

        assert issubclass(BrokerError, Exception)

    def test_broker_error_with_message(self):
        """BrokerError should store message correctly."""
        from brokers.alpaca_broker import BrokerError

        error = BrokerError("Test error message")
        assert str(error) == "Test error message"

    def test_broker_connection_error_is_broker_error(self):
        """BrokerConnectionError should be a BrokerError subclass."""
        from brokers.alpaca_broker import BrokerConnectionError, BrokerError

        assert issubclass(BrokerConnectionError, BrokerError)

    def test_order_error_is_broker_error(self):
        """OrderError should be a BrokerError subclass."""
        from brokers.alpaca_broker import BrokerError, OrderError

        assert issubclass(OrderError, BrokerError)

    def test_can_catch_broker_connection_error_as_broker_error(self):
        """BrokerConnectionError should be catchable as BrokerError."""
        from brokers.alpaca_broker import BrokerConnectionError, BrokerError

        try:
            raise BrokerConnectionError("Connection failed")
        except BrokerError as e:
            assert str(e) == "Connection failed"

    def test_can_catch_order_error_as_broker_error(self):
        """OrderError should be catchable as BrokerError."""
        from brokers.alpaca_broker import BrokerError, OrderError

        try:
            raise OrderError("Order failed")
        except BrokerError as e:
            assert str(e) == "Order failed"


# ============================================================================
# Test retry_with_backoff Decorator
# ============================================================================


class TestRetryWithBackoff:
    """Test the retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_successful_first_try(self):
        """Function should return immediately on success."""
        from brokers.alpaca_broker import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Should retry on ConnectionError."""
        from brokers.alpaca_broker import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self):
        """Should retry on TimeoutError."""
        from brokers.alpaca_broker import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timeout")
            return "success"

        result = await timeout_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_transient_error_message(self):
        """Should retry on errors with transient keywords."""
        from brokers.alpaca_broker import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def rate_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 rate limit exceeded")
            return "success"

        result = await rate_limited_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient_error(self):
        """Should NOT retry on non-transient errors."""
        from brokers.alpaca_broker import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def non_transient_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid value")

        with pytest.raises(ValueError):
            await non_transient_func()

        # Should only be called once since error is not transient
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Should raise last exception when all retries exhausted."""
        from brokers.alpaca_broker import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await always_fails()

        assert call_count == 3


# ============================================================================
# Test AlpacaBroker Initialization
# ============================================================================


class TestAlpacaBrokerInit:
    """Test AlpacaBroker initialization."""

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_init_with_paper_true(self, mock_trading, mock_data, mock_stream):
        """Should initialize with paper=True."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        assert broker.paper is True
        assert broker.NAME == "alpaca"
        assert broker.IS_BACKTESTING_BROKER is False
        mock_trading.assert_called_once()
        mock_data.assert_called_once()
        mock_stream.assert_called_once()

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_init_with_paper_false(self, mock_trading, mock_data, mock_stream):
        """Should initialize with paper=False."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=False)

        assert broker.paper is False

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_init_with_paper_string_true(self, mock_trading, mock_data, mock_stream):
        """Should handle paper='true' string."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper="true")

        assert broker.paper is True

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_init_with_paper_string_false(self, mock_trading, mock_data, mock_stream):
        """Should handle paper='false' string."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper="false")

        assert broker.paper is False

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_init_initializes_tracking_attributes(self, mock_trading, mock_data, mock_stream):
        """Should initialize all tracking attributes."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        assert broker._filled_positions == []
        assert broker._subscribers == set()
        assert broker._ws_task is None
        assert broker._connected is False
        assert broker._reconnect_attempts == 0
        assert broker._subscribed_symbols == set()

    @patch("brokers.alpaca_broker.ALPACA_CREDS", {"API_KEY": "", "API_SECRET": ""})
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_init_raises_on_missing_credentials(self, mock_trading, mock_data, mock_stream):
        """Should raise ValueError when credentials are missing."""
        from brokers.alpaca_broker import AlpacaBroker

        with pytest.raises(ValueError, match="Alpaca API credentials not found"):
            AlpacaBroker(paper=True)


# ============================================================================
# Test Symbol Validation
# ============================================================================


class TestSymbolValidation:
    """Test the _validate_symbol static method."""

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_uppercase(self, mock_trading, mock_data, mock_stream):
        """Should convert symbol to uppercase."""
        from brokers.alpaca_broker import AlpacaBroker

        result = AlpacaBroker._validate_symbol("aapl")
        assert result == "AAPL"

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_strip_whitespace(self, mock_trading, mock_data, mock_stream):
        """Should strip whitespace from symbol."""
        from brokers.alpaca_broker import AlpacaBroker

        result = AlpacaBroker._validate_symbol("  AAPL  ")
        assert result == "AAPL"

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_empty_raises(self, mock_trading, mock_data, mock_stream):
        """Should raise ValueError for empty symbol."""
        from brokers.alpaca_broker import AlpacaBroker

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            AlpacaBroker._validate_symbol("")

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_none_raises(self, mock_trading, mock_data, mock_stream):
        """Should raise ValueError for None symbol."""
        from brokers.alpaca_broker import AlpacaBroker

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            AlpacaBroker._validate_symbol(None)

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_non_string_raises(self, mock_trading, mock_data, mock_stream):
        """Should raise ValueError for non-string symbol."""
        from brokers.alpaca_broker import AlpacaBroker

        with pytest.raises(ValueError, match="Symbol must be a string"):
            AlpacaBroker._validate_symbol(123)

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_too_long_raises(self, mock_trading, mock_data, mock_stream):
        """Should raise ValueError for symbol > 10 chars."""
        from brokers.alpaca_broker import AlpacaBroker

        with pytest.raises(ValueError, match="Symbol too long"):
            AlpacaBroker._validate_symbol("VERYLONGSYMBOL")

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_invalid_chars_raises(self, mock_trading, mock_data, mock_stream):
        """Should raise ValueError for invalid characters."""
        from brokers.alpaca_broker import AlpacaBroker

        with pytest.raises(ValueError, match="Invalid symbol format"):
            AlpacaBroker._validate_symbol("AAPL$")

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_with_dot(self, mock_trading, mock_data, mock_stream):
        """Should accept symbols with dots (e.g., BRK.B)."""
        from brokers.alpaca_broker import AlpacaBroker

        result = AlpacaBroker._validate_symbol("BRK.B")
        assert result == "BRK.B"

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_validate_symbol_with_hyphen(self, mock_trading, mock_data, mock_stream):
        """Should accept symbols with hyphens."""
        from brokers.alpaca_broker import AlpacaBroker

        result = AlpacaBroker._validate_symbol("SPY-USD")
        assert result == "SPY-USD"


# ============================================================================
# Test Subscriber Management
# ============================================================================


class TestSubscriberManagement:
    """Test subscriber management methods."""

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_add_subscriber(self, mock_trading, mock_data, mock_stream):
        """Should add subscriber to set."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        subscriber = Mock()

        broker._add_subscriber(subscriber)

        assert subscriber in broker._subscribers

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_add_subscriber_no_duplicates(self, mock_trading, mock_data, mock_stream):
        """Should not add duplicate subscribers."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        subscriber = Mock()

        broker._add_subscriber(subscriber)
        broker._add_subscriber(subscriber)

        assert len(broker._subscribers) == 1

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_remove_subscriber(self, mock_trading, mock_data, mock_stream):
        """Should remove subscriber from set."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        subscriber = Mock()

        broker._add_subscriber(subscriber)
        broker._remove_subscriber(subscriber)

        assert subscriber not in broker._subscribers

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_remove_nonexistent_subscriber(self, mock_trading, mock_data, mock_stream):
        """Should handle removing non-existent subscriber gracefully."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        subscriber = Mock()

        # Should not raise
        broker._remove_subscriber(subscriber)

        assert subscriber not in broker._subscribers


# ============================================================================
# Test Account Methods
# ============================================================================


class TestAccountMethods:
    """Test account-related methods."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_account_success(self, mock_trading, mock_data, mock_stream):
        """Should return account info."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_account = Mock()
        mock_account.id = "test-account-id"
        mock_account.cash = "100000.00"
        mock_account.buying_power = "400000.00"
        mock_trading.return_value.get_account.return_value = mock_account

        broker = AlpacaBroker(paper=True)
        account = await broker.get_account()

        assert account.id == "test-account-id"
        mock_trading.return_value.get_account.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_account_raises_on_error(self, mock_trading, mock_data, mock_stream):
        """Should raise on API error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_account.side_effect = Exception("API error")

        broker = AlpacaBroker(paper=True)

        with pytest.raises(Exception):
            await broker.get_account()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_market_status_success(self, mock_trading, mock_data, mock_stream):
        """Should return market status."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_clock = Mock()
        mock_clock.is_open = True
        mock_clock.next_open = datetime.now() + timedelta(days=1)
        mock_clock.next_close = datetime.now() + timedelta(hours=4)
        mock_clock.timestamp = datetime.now()
        mock_trading.return_value.get_clock.return_value = mock_clock

        broker = AlpacaBroker(paper=True)
        status = await broker.get_market_status()

        assert status["is_open"] is True
        assert "next_open" in status
        assert "next_close" in status

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_market_status_returns_default_on_error(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return safe default on error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_clock.side_effect = Exception("API error")

        broker = AlpacaBroker(paper=True)
        status = await broker.get_market_status()

        assert status == {"is_open": False}


# ============================================================================
# Test Position Methods
# ============================================================================


class TestPositionMethods:
    """Test position-related methods."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_positions_success(self, mock_trading, mock_data, mock_stream):
        """Should return all positions."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_trading.return_value.get_all_positions.return_value = [mock_position]

        broker = AlpacaBroker(paper=True)
        positions = await broker.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_position_success(self, mock_trading, mock_data, mock_stream):
        """Should return specific position."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_trading.return_value.get_position.return_value = mock_position

        broker = AlpacaBroker(paper=True)
        position = await broker.get_position("AAPL")

        assert position.symbol == "AAPL"
        mock_trading.return_value.get_position.assert_called_with("AAPL")

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_position_returns_none_on_not_found(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return None when position not found."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_position.side_effect = Exception("Position not found")

        broker = AlpacaBroker(paper=True)
        position = await broker.get_position("AAPL")

        assert position is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_position_validates_symbol(self, mock_trading, mock_data, mock_stream):
        """Should validate symbol before API call."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        position = await broker.get_position("")

        assert position is None
        mock_trading.return_value.get_position.assert_not_called()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_tracked_positions(self, mock_trading, mock_data, mock_stream):
        """Should return positions for strategy."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_trading.return_value.get_all_positions.return_value = [mock_position]

        broker = AlpacaBroker(paper=True)
        positions = await broker.get_tracked_positions("test_strategy")

        assert len(positions) == 1


# ============================================================================
# Test Order Methods
# ============================================================================


class TestOrderMethods:
    """Test order-related methods."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_submit_market_order(self, mock_trading, mock_data, mock_stream):
        """Should submit market order."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_result = Mock()
        mock_result.id = "order-123"
        mock_result.symbol = "AAPL"
        mock_result.qty = "100"
        mock_trading.return_value.submit_order.return_value = mock_result

        broker = AlpacaBroker(paper=True)
        order = {"symbol": "AAPL", "side": "buy", "quantity": 100, "type": "market"}
        result = await broker.submit_order(order)

        assert result.id == "order-123"
        mock_trading.return_value.submit_order.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_submit_limit_order(self, mock_trading, mock_data, mock_stream):
        """Should submit limit order."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_result = Mock()
        mock_result.id = "order-123"
        mock_result.symbol = "AAPL"
        mock_result.qty = "100"
        mock_trading.return_value.submit_order.return_value = mock_result

        broker = AlpacaBroker(paper=True)
        order = {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 100,
            "type": "limit",
            "limit_price": 150.00,
        }
        result = await broker.submit_order(order)

        assert result.id == "order-123"

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_submit_order_unsupported_type(self, mock_trading, mock_data, mock_stream):
        """Should raise on unsupported order type."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        order = {"symbol": "AAPL", "side": "buy", "quantity": 100, "type": "unsupported"}

        with pytest.raises(ValueError, match="Unsupported order type"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_submit_order_advanced_with_order_builder(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should submit order from OrderBuilder."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_result = Mock()
        mock_result.id = "order-123"
        mock_result.symbol = "AAPL"
        mock_result.qty = "100"
        mock_result.type = "market"
        mock_result.order_class = "simple"
        mock_trading.return_value.submit_order.return_value = mock_result

        # Create a mock OrderBuilder
        from brokers.order_builder import OrderBuilder

        order_builder = OrderBuilder("AAPL", "buy", 100).market().day()

        broker = AlpacaBroker(paper=True)
        result = await broker.submit_order_advanced(order_builder)

        assert result.id == "order-123"

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_cancel_order_success(self, mock_trading, mock_data, mock_stream):
        """Should cancel order by ID."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        result = await broker.cancel_order("order-123")

        assert result is True
        mock_trading.return_value.cancel_order_by_id.assert_called_with("order-123")

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_cancel_order_failure(self, mock_trading, mock_data, mock_stream):
        """Should return False on cancel failure."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.cancel_order_by_id.side_effect = Exception("Order not found")

        broker = AlpacaBroker(paper=True)
        result = await broker.cancel_order("order-123")

        assert result is False

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_cancel_all_orders(self, mock_trading, mock_data, mock_stream):
        """Should cancel all orders."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.cancel_orders.return_value = [Mock(), Mock()]

        broker = AlpacaBroker(paper=True)
        result = await broker.cancel_all_orders()

        assert len(result) == 2

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_replace_order(self, mock_trading, mock_data, mock_stream):
        """Should replace order with new parameters."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_result = Mock()
        mock_result.id = "order-456"
        mock_trading.return_value.replace_order_by_id.return_value = mock_result

        broker = AlpacaBroker(paper=True)
        result = await broker.replace_order("order-123", qty=150, limit_price=155.00)

        assert result.id == "order-456"
        mock_trading.return_value.replace_order_by_id.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_order_by_id(self, mock_trading, mock_data, mock_stream):
        """Should get order by ID."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_order = Mock()
        mock_order.id = "order-123"
        mock_trading.return_value.get_order_by_id.return_value = mock_order

        broker = AlpacaBroker(paper=True)
        order = await broker.get_order_by_id("order-123")

        assert order.id == "order-123"

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_order_by_id_not_found(self, mock_trading, mock_data, mock_stream):
        """Should return None when order not found."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_order_by_id.side_effect = Exception("Not found")

        broker = AlpacaBroker(paper=True)
        order = await broker.get_order_by_id("order-123")

        assert order is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_order_by_client_id(self, mock_trading, mock_data, mock_stream):
        """Should get order by client order ID."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_order = Mock()
        mock_order.client_order_id = "client-123"
        mock_trading.return_value.get_order_by_client_id.return_value = mock_order

        broker = AlpacaBroker(paper=True)
        order = await broker.get_order_by_client_id("client-123")

        assert order.client_order_id == "client-123"

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_orders(self, mock_trading, mock_data, mock_stream):
        """Should get orders with status filter."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_orders = [Mock(), Mock()]
        mock_trading.return_value.get_orders.return_value = mock_orders

        broker = AlpacaBroker(paper=True)
        orders = await broker.get_orders(status=QueryOrderStatus.OPEN, limit=50)

        assert len(orders) == 2
        mock_trading.return_value.get_orders.assert_called_once()


# ============================================================================
# Test Market Data Methods
# ============================================================================


class TestMarketDataMethods:
    """Test market data methods."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_last_price_success(self, mock_trading, mock_data, mock_stream):
        """Should return last trade price."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trade = Mock()
        mock_trade.price = 150.25
        mock_data.return_value.get_stock_latest_trade.return_value = {"AAPL": mock_trade}

        broker = AlpacaBroker(paper=True)
        price = await broker.get_last_price("AAPL")

        assert price == 150.25

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_last_price_not_found(self, mock_trading, mock_data, mock_stream):
        """Should return None when price not found."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_data.return_value.get_stock_latest_trade.return_value = {}

        broker = AlpacaBroker(paper=True)
        price = await broker.get_last_price("AAPL")

        assert price is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_last_price_on_error(self, mock_trading, mock_data, mock_stream):
        """Should return None on API error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_data.return_value.get_stock_latest_trade.side_effect = Exception("API error")

        broker = AlpacaBroker(paper=True)
        price = await broker.get_last_price("AAPL")

        assert price is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_bars_success(self, mock_trading, mock_data, mock_stream):
        """Should return historical bars."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_bar = Mock()
        mock_bar.open = 150.0
        mock_bar.high = 155.0
        mock_bar.low = 149.0
        mock_bar.close = 152.0
        mock_bar.volume = 1000000

        mock_response = Mock()
        mock_response.data = {"AAPL": [mock_bar]}
        mock_data.return_value.get_stock_bars.return_value = mock_response

        broker = AlpacaBroker(paper=True)
        bars = await broker.get_bars("AAPL", limit=5)

        assert len(bars) == 1
        assert bars[0].close == 152.0

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_bars_with_string_timeframe(self, mock_trading, mock_data, mock_stream):
        """Should convert string timeframe."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_response = Mock()
        mock_response.data = {"AAPL": []}
        mock_data.return_value.get_stock_bars.return_value = mock_response

        broker = AlpacaBroker(paper=True)
        await broker.get_bars("AAPL", timeframe="1Min", limit=5)

        mock_data.return_value.get_stock_bars.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_bars_returns_empty_on_not_found(self, mock_trading, mock_data, mock_stream):
        """Should return empty list when bars not found."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_response = Mock()
        mock_response.data = {}
        mock_data.return_value.get_stock_bars.return_value = mock_response

        broker = AlpacaBroker(paper=True)
        bars = await broker.get_bars("AAPL")

        assert bars == []

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_bars_returns_empty_on_error(self, mock_trading, mock_data, mock_stream):
        """Should return empty list on API error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_data.return_value.get_stock_bars.side_effect = Exception("API error")

        broker = AlpacaBroker(paper=True)
        bars = await broker.get_bars("AAPL")

        assert bars == []

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_news_returns_empty(self, mock_trading, mock_data, mock_stream):
        """get_news should return empty list (not implemented)."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        news = await broker.get_news("AAPL", datetime.now() - timedelta(days=7), datetime.now())

        assert news == []


# ============================================================================
# Test WebSocket Handlers
# ============================================================================


class TestWebSocketHandlers:
    """Test WebSocket handler methods."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_handle_trade_updates_fill(self, mock_trading, mock_data, mock_stream):
        """Should handle fill event and notify subscribers."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        # Add mock subscriber
        subscriber = Mock()
        subscriber.on_trade_update = AsyncMock()
        broker._add_subscriber(subscriber)

        # Simulate trade update
        data = {"event": "fill", "order": {"id": "order-123"}}

        await broker._handle_trade_updates(data)

        subscriber.on_trade_update.assert_called_once_with(data)

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_handle_trade_updates_partial_fill(self, mock_trading, mock_data, mock_stream):
        """Should handle partial_fill event."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        subscriber = Mock()
        subscriber.on_trade_update = AsyncMock()
        broker._add_subscriber(subscriber)

        data = {"event": "partial_fill", "order": {"id": "order-123"}}

        await broker._handle_trade_updates(data)

        subscriber.on_trade_update.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_handle_bars(self, mock_trading, mock_data, mock_stream):
        """Should handle bar data and notify subscribers."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        subscriber = Mock()
        subscriber.on_bar = AsyncMock()
        broker._add_subscriber(subscriber)

        data = {
            "S": "AAPL",
            "o": 150.0,
            "h": 155.0,
            "l": 149.0,
            "c": 152.0,
            "v": 1000000,
            "t": datetime.now().timestamp() * 1000,
        }

        await broker._handle_bars(data)

        subscriber.on_bar.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_handle_quotes(self, mock_trading, mock_data, mock_stream):
        """Should handle quote data and notify subscribers."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        subscriber = Mock()
        subscriber.on_quote = AsyncMock()
        broker._add_subscriber(subscriber)

        data = {
            "S": "AAPL",
            "bp": 150.0,
            "ap": 150.05,
            "bs": 100,
            "as": 200,
            "t": datetime.now().timestamp() * 1000,
        }

        await broker._handle_quotes(data)

        subscriber.on_quote.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_handle_trades(self, mock_trading, mock_data, mock_stream):
        """Should handle trade data and notify subscribers."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        subscriber = Mock()
        subscriber.on_trade = AsyncMock()
        broker._add_subscriber(subscriber)

        data = {"S": "AAPL", "p": 150.25, "s": 100, "t": datetime.now().timestamp() * 1000}

        await broker._handle_trades(data)

        subscriber.on_trade.assert_called_once()


# ============================================================================
# Test WebSocket Connection Management
# ============================================================================


class TestWebSocketConnection:
    """Test WebSocket connection management."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_is_connected_initially_false(self, mock_trading, mock_data, mock_stream):
        """Should be disconnected initially."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        is_connected = await broker.is_connected()

        assert is_connected is False

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_start_websocket_creates_task(self, mock_trading, mock_data, mock_stream):
        """Should create websocket handler task."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        await broker.start_websocket()

        assert broker._ws_task is not None

        # Clean up
        await broker.stop_websocket()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_start_websocket_no_duplicate(self, mock_trading, mock_data, mock_stream):
        """Should not create duplicate websocket tasks."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        await broker.start_websocket()
        task1 = broker._ws_task

        await broker.start_websocket()
        task2 = broker._ws_task

        assert task1 is task2

        # Clean up
        await broker.stop_websocket()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_stop_websocket_cancels_task(self, mock_trading, mock_data, mock_stream):
        """Should cancel websocket task."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        await broker.start_websocket()
        await broker.stop_websocket()

        assert broker._ws_task is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_stop_websocket_when_not_running(self, mock_trading, mock_data, mock_stream):
        """Should handle stopping when not running."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        # Should not raise
        await broker.stop_websocket()

        assert broker._ws_task is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_subscribe_to_symbols_when_not_connected(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return False when not connected."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        result = await broker._subscribe_to_symbols(["AAPL", "MSFT"])

        assert result is False


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_handle_bars_with_no_subscribers(self, mock_trading, mock_data, mock_stream):
        """Should handle bars with no subscribers gracefully."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        data = {
            "S": "AAPL",
            "o": 150.0,
            "h": 155.0,
            "l": 149.0,
            "c": 152.0,
            "v": 1000000,
            "t": datetime.now().timestamp() * 1000,
        }

        # Should not raise
        await broker._handle_bars(data)

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_handle_bars_with_missing_data(self, mock_trading, mock_data, mock_stream):
        """Should handle bars with missing fields."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        # Missing all fields
        data = {}

        # Should not raise
        await broker._handle_bars(data)

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_submit_order_with_default_type(self, mock_trading, mock_data, mock_stream):
        """Should default to market order."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_result = Mock()
        mock_result.id = "order-123"
        mock_result.symbol = "AAPL"
        mock_result.qty = "100"
        mock_trading.return_value.submit_order.return_value = mock_result

        broker = AlpacaBroker(paper=True)
        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            # No 'type' specified
        }

        result = await broker.submit_order(order)

        assert result.id == "order-123"

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_tracked_positions_returns_empty_on_error(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return empty list on error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_all_positions.side_effect = Exception("API error")

        broker = AlpacaBroker(paper=True)
        positions = await broker.get_tracked_positions("strategy")

        assert positions == []


# ============================================================================
# Test Timeframe Conversion
# ============================================================================


class TestTimeframeConversion:
    """Test timeframe string to object conversion."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_timeframe_1min(self, mock_trading, mock_data, mock_stream):
        """Should convert '1Min' string."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_response = Mock()
        mock_response.data = {"AAPL": []}
        mock_data.return_value.get_stock_bars.return_value = mock_response

        broker = AlpacaBroker(paper=True)
        await broker.get_bars("AAPL", timeframe="1Min")

        # Verify call was made (timeframe conversion worked)
        mock_data.return_value.get_stock_bars.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_timeframe_1hour(self, mock_trading, mock_data, mock_stream):
        """Should convert '1Hour' string."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_response = Mock()
        mock_response.data = {"AAPL": []}
        mock_data.return_value.get_stock_bars.return_value = mock_response

        broker = AlpacaBroker(paper=True)
        await broker.get_bars("AAPL", timeframe="1Hour")

        mock_data.return_value.get_stock_bars.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_timeframe_day(self, mock_trading, mock_data, mock_stream):
        """Should convert 'Day' string."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_response = Mock()
        mock_response.data = {"AAPL": []}
        mock_data.return_value.get_stock_bars.return_value = mock_response

        broker = AlpacaBroker(paper=True)
        await broker.get_bars("AAPL", timeframe="Day")

        mock_data.return_value.get_stock_bars.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
