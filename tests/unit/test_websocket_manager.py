"""
Unit tests for WebSocketManager.

Tests cover:
- Initialization and configuration
- Subscription management
- Handler registration
- Connection state management
- Error handling
"""

# Mock environment variables before importing
import os
from datetime import datetime
from unittest.mock import MagicMock

import pytest

os.environ["TESTING"] = "true"
os.environ["ALPACA_API_KEY"] = "test_key"
os.environ["ALPACA_SECRET_KEY"] = "test_secret"

from utils.websocket_manager import WebSocketManager


class TestWebSocketManagerInit:
    """Tests for WebSocketManager initialization."""

    def test_init_with_valid_credentials(self):
        """Test initialization with valid credentials."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret",
            feed="iex"
        )

        assert manager._api_key == "test_key"
        assert manager._secret_key == "test_secret"
        assert manager._feed == "iex"
        assert not manager.is_running
        assert not manager.is_connected

    def test_init_with_sip_feed(self):
        """Test initialization with SIP feed."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret",
            feed="sip"
        )

        assert manager._feed == "sip"

    def test_init_with_empty_credentials_raises(self):
        """Test that empty credentials raise ValueError."""
        with pytest.raises(ValueError, match="API key and secret key are required"):
            WebSocketManager(api_key="", secret_key="test_secret")

        with pytest.raises(ValueError, match="API key and secret key are required"):
            WebSocketManager(api_key="test_key", secret_key="")

    def test_init_with_none_credentials_raises(self):
        """Test that None credentials raise ValueError."""
        with pytest.raises(ValueError):
            WebSocketManager(api_key=None, secret_key="test_secret")


class TestWebSocketManagerSubscriptions:
    """Tests for subscription management."""

    def test_subscribe_bars_single_symbol(self):
        """Test subscribing to bars for a single symbol."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager.subscribe_bars(["AAPL"])

        assert "AAPL" in manager._subscribed_bars
        assert len(manager._subscribed_bars) == 1

    def test_subscribe_bars_multiple_symbols(self):
        """Test subscribing to bars for multiple symbols."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager.subscribe_bars(["AAPL", "MSFT", "GOOGL"])

        assert manager._subscribed_bars == {"AAPL", "MSFT", "GOOGL"}

    def test_subscribe_bars_normalizes_case(self):
        """Test that symbol case is normalized to uppercase."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager.subscribe_bars(["aapl", "Msft", "GOOGL"])

        assert manager._subscribed_bars == {"AAPL", "MSFT", "GOOGL"}

    def test_subscribe_bars_with_handler(self):
        """Test subscribing with a handler function."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        async def mock_handler(bar):
            pass

        manager.subscribe_bars(["AAPL"], mock_handler)

        assert "AAPL" in manager._subscribed_bars
        assert mock_handler in manager._bar_handlers["AAPL"]

    def test_subscribe_quotes(self):
        """Test subscribing to quotes."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager.subscribe_quotes(["AAPL", "MSFT"])

        assert manager._subscribed_quotes == {"AAPL", "MSFT"}

    def test_subscribe_trades(self):
        """Test subscribing to trades."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager.subscribe_trades(["AAPL"])

        assert "AAPL" in manager._subscribed_trades

    def test_unsubscribe_bars(self):
        """Test unsubscribing from bars."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager.subscribe_bars(["AAPL", "MSFT"])
        manager.unsubscribe_bars(["AAPL"])

        assert "AAPL" not in manager._subscribed_bars
        assert "MSFT" in manager._subscribed_bars

    def test_unsubscribe_quotes(self):
        """Test unsubscribing from quotes."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager.subscribe_quotes(["AAPL", "MSFT"])
        manager.unsubscribe_quotes(["MSFT"])

        assert "MSFT" not in manager._subscribed_quotes

    def test_unsubscribe_removes_handlers(self):
        """Test that unsubscribing removes associated handlers."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        async def mock_handler(bar):
            pass

        manager.subscribe_bars(["AAPL"], mock_handler)
        manager.unsubscribe_bars(["AAPL"])

        assert "AAPL" not in manager._bar_handlers


class TestWebSocketManagerHandlers:
    """Tests for handler registration and routing."""

    def test_add_global_bar_handler(self):
        """Test adding a global bar handler."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        async def global_handler(bar):
            pass

        manager.add_global_bar_handler(global_handler)

        assert global_handler in manager._global_bar_handlers

    def test_add_global_quote_handler(self):
        """Test adding a global quote handler."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        async def global_handler(quote):
            pass

        manager.add_global_quote_handler(global_handler)

        assert global_handler in manager._global_quote_handlers

    def test_add_global_trade_handler(self):
        """Test adding a global trade handler."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        async def global_handler(trade):
            pass

        manager.add_global_trade_handler(global_handler)

        assert global_handler in manager._global_trade_handlers

    def test_multiple_handlers_per_symbol(self):
        """Test that multiple handlers can be registered for the same symbol."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        async def handler1(bar):
            pass

        async def handler2(bar):
            pass

        manager.subscribe_bars(["AAPL"], handler1)
        manager.subscribe_bars(["AAPL"], handler2)

        assert handler1 in manager._bar_handlers["AAPL"]
        assert handler2 in manager._bar_handlers["AAPL"]


class TestWebSocketManagerBarHandling:
    """Tests for bar data handling."""

    @pytest.mark.asyncio
    async def test_handle_bar_calls_symbol_handlers(self):
        """Test that bar handler is called for symbol-specific subscriptions."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        received_bars = []

        async def handler(bar):
            received_bars.append(bar)

        manager.subscribe_bars(["AAPL"], handler)

        # Create mock bar
        mock_bar = MagicMock()
        mock_bar.symbol = "AAPL"
        mock_bar.close = 150.0

        await manager._handle_bar(mock_bar)

        assert len(received_bars) == 1
        assert received_bars[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_handle_bar_calls_global_handlers(self):
        """Test that global handlers are called for all bars."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        received_bars = []

        async def global_handler(bar):
            received_bars.append(bar)

        manager.add_global_bar_handler(global_handler)
        manager.subscribe_bars(["AAPL", "MSFT"])

        # Create mock bars
        mock_bar_aapl = MagicMock()
        mock_bar_aapl.symbol = "AAPL"

        mock_bar_msft = MagicMock()
        mock_bar_msft.symbol = "MSFT"

        await manager._handle_bar(mock_bar_aapl)
        await manager._handle_bar(mock_bar_msft)

        assert len(received_bars) == 2

    @pytest.mark.asyncio
    async def test_handle_bar_updates_message_count(self):
        """Test that message count is updated on bar receipt."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        mock_bar = MagicMock()
        mock_bar.symbol = "AAPL"

        initial_count = manager.message_count

        await manager._handle_bar(mock_bar)

        assert manager.message_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_handle_bar_updates_last_message_time(self):
        """Test that last message time is updated on bar receipt."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        mock_bar = MagicMock()
        mock_bar.symbol = "AAPL"

        before = datetime.now()
        await manager._handle_bar(mock_bar)
        after = datetime.now()

        assert manager.last_message_time is not None
        assert before <= manager.last_message_time <= after

    @pytest.mark.asyncio
    async def test_handle_bar_with_sync_handler(self):
        """Test that synchronous handlers also work."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        received_bars = []

        def sync_handler(bar):
            received_bars.append(bar)

        manager.subscribe_bars(["AAPL"], sync_handler)

        mock_bar = MagicMock()
        mock_bar.symbol = "AAPL"

        await manager._handle_bar(mock_bar)

        assert len(received_bars) == 1

    @pytest.mark.asyncio
    async def test_handle_bar_continues_on_handler_error(self):
        """Test that one handler error doesn't stop other handlers."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        received_bars = []

        async def failing_handler(bar):
            raise Exception("Handler error")

        async def working_handler(bar):
            received_bars.append(bar)

        manager.subscribe_bars(["AAPL"], failing_handler)
        manager.subscribe_bars(["AAPL"], working_handler)

        mock_bar = MagicMock()
        mock_bar.symbol = "AAPL"

        # Should not raise, and working handler should still be called
        await manager._handle_bar(mock_bar)

        assert len(received_bars) == 1


class TestWebSocketManagerStats:
    """Tests for statistics and info methods."""

    def test_get_subscription_info(self):
        """Test getting subscription information."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        async def handler(data):
            pass

        manager.subscribe_bars(["AAPL", "MSFT"], handler)
        manager.subscribe_quotes(["AAPL"])
        manager.add_global_bar_handler(handler)

        info = manager.get_subscription_info()

        assert set(info["bars"]) == {"AAPL", "MSFT"}
        assert set(info["quotes"]) == {"AAPL"}
        assert info["bar_handlers"]["AAPL"] == 1
        assert info["global_bar_handlers"] == 1

    def test_get_connection_stats(self):
        """Test getting connection statistics."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        stats = manager.get_connection_stats()

        assert "is_running" in stats
        assert "is_connected" in stats
        assert "message_count" in stats
        assert "feed" in stats
        assert stats["feed"] == "iex"

    def test_message_count_property(self):
        """Test message count property."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        assert manager.message_count == 0

    def test_last_message_time_property(self):
        """Test last message time property."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        assert manager.last_message_time is None


class TestWebSocketManagerConnectionState:
    """Tests for connection state management."""

    def test_is_running_initially_false(self):
        """Test that is_running is False initially."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        assert not manager.is_running

    def test_is_connected_initially_false(self):
        """Test that is_connected is False initially."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        assert not manager.is_connected

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """Test that start() is idempotent when already running."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        manager._running = True

        # Should return without error
        await manager.start()

        # Still running
        assert manager._running

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Test that stop() handles not-running state gracefully."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        # Should not raise
        await manager.stop()

        assert not manager.is_running


class TestWebSocketManagerQuoteHandling:
    """Tests for quote data handling."""

    @pytest.mark.asyncio
    async def test_handle_quote_calls_handlers(self):
        """Test that quote handlers are called."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        received_quotes = []

        async def handler(quote):
            received_quotes.append(quote)

        manager.subscribe_quotes(["AAPL"], handler)

        mock_quote = MagicMock()
        mock_quote.symbol = "AAPL"
        mock_quote.bid_price = 149.50
        mock_quote.ask_price = 150.00

        await manager._handle_quote(mock_quote)

        assert len(received_quotes) == 1


class TestWebSocketManagerTradeHandling:
    """Tests for trade data handling."""

    @pytest.mark.asyncio
    async def test_handle_trade_calls_handlers(self):
        """Test that trade handlers are called."""
        manager = WebSocketManager(
            api_key="test_key",
            secret_key="test_secret"
        )

        received_trades = []

        async def handler(trade):
            received_trades.append(trade)

        manager.subscribe_trades(["AAPL"], handler)

        mock_trade = MagicMock()
        mock_trade.symbol = "AAPL"
        mock_trade.price = 150.00
        mock_trade.size = 100

        await manager._handle_trade(mock_trade)

        assert len(received_trades) == 1
