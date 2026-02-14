"""
Comprehensive error handling tests for AlpacaBroker.

Tests cover:
1. TestRetryWithBackoff - Retry decorator behavior with exponential backoff
2. TestOrderSubmissionErrors - Order submission error handling
3. TestPriceCaching - Price caching optimization
4. TestSymbolValidation - Symbol validation and sanitization
5. TestConnectionErrors - Connection and network error handling

These tests ensure the broker handles errors gracefully and provides
reliable behavior under adverse network conditions.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from brokers.alpaca_broker import (
    AlpacaBroker,
    BrokerConnectionError,
    retry_with_backoff,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_alpaca_creds():
    """Mock Alpaca credentials for testing."""
    return {
        "API_KEY": "test_api_key",
        "API_SECRET": "test_api_secret",
    }


@pytest.fixture
def mock_trading_client():
    """Create a mock trading client."""
    client = MagicMock()
    # Setup default return values
    account = MagicMock()
    account.equity = "100000.00"
    account.cash = "50000.00"
    account.buying_power = "200000.00"
    client.get_account.return_value = account
    client.get_all_positions.return_value = []
    return client


@pytest.fixture
def mock_data_client():
    """Create a mock data client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_stream():
    """Create a mock stream client."""
    stream = MagicMock()
    return stream


# =============================================================================
# TEST CLASS: TestRetryWithBackoff
# =============================================================================


class TestRetryWithBackoff:
    """Test the retry_with_backoff decorator behavior."""

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        """Should retry on ConnectionError up to max_retries."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return "success"

        result = await failing_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_timeout_error(self):
        """Should retry on TimeoutError."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def timeout_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Request timed out")
            return "success"

        result = await timeout_function()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_os_error(self):
        """Should retry on OSError (network-related)."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def os_error_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("Network unreachable")
            return "success"

        result = await os_error_function()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_invalid_symbol(self):
        """Should not retry on APIError for invalid symbol (non-transient)."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def invalid_symbol_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid symbol: XXXX")

        with pytest.raises(ValueError, match="Invalid symbol"):
            await invalid_symbol_function()

        # Should fail immediately without retry
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_authentication_error(self):
        """Should not retry on authentication errors."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def auth_error_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Unauthorized: Invalid API key")

        with pytest.raises(Exception, match="Unauthorized"):
            await auth_error_function()

        # Should fail immediately without retry
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Should increase delay exponentially between retries."""
        delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            delays.append(duration)
            await original_sleep(0.001)  # Actually sleep a tiny bit

        call_count = 0

        @retry_with_backoff(max_retries=4, initial_delay=0.1, max_delay=10, jitter=0)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection refused")

        with patch("asyncio.sleep", mock_sleep):
            with pytest.raises(ConnectionError):
                await failing_function()

        # Should have 3 delays (retries - 1)
        assert len(delays) == 3

        # Check exponential backoff: 0.1, 0.2, 0.4 (with some tolerance)
        assert 0.09 <= delays[0] <= 0.11  # ~0.1
        assert 0.18 <= delays[1] <= 0.22  # ~0.2
        assert 0.36 <= delays[2] <= 0.44  # ~0.4

    @pytest.mark.asyncio
    async def test_stops_after_max_retries(self):
        """Should stop retrying after max_retries reached."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection refused")

        with pytest.raises(ConnectionError):
            await always_failing_function()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        """Should return result when call succeeds."""

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        async def successful_function():
            return {"status": "ok", "data": [1, 2, 3]}

        result = await successful_function()

        assert result == {"status": "ok", "data": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_error(self):
        """Should retry on rate limit (429) errors."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def rate_limited_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return "success"

        result = await rate_limited_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_503_error(self):
        """Should retry on service unavailable (503) errors."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, max_delay=0.1)
        async def service_unavailable_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("503 Service Unavailable")
            return "success"

        result = await service_unavailable_function()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_jitter_adds_randomness(self):
        """Should add jitter to delay to prevent thundering herd."""
        delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            delays.append(duration)
            await original_sleep(0.001)

        call_count = 0

        @retry_with_backoff(max_retries=10, initial_delay=1.0, max_delay=100, jitter=0.5)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection refused")

        with patch("asyncio.sleep", mock_sleep):
            with pytest.raises(ConnectionError):
                await failing_function()

        # With jitter=0.5, delays should vary within +/- 50% of base
        # Check that not all delays are exactly the same (jitter applied)
        assert len(delays) >= 2
        # First delay base is 1.0, range should be [0.5, 1.5]
        assert 0.4 <= delays[0] <= 1.6

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Should cap delay at max_delay."""
        delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            delays.append(duration)
            await original_sleep(0.001)

        call_count = 0

        @retry_with_backoff(max_retries=6, initial_delay=1.0, max_delay=5.0, jitter=0)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection refused")

        with patch("asyncio.sleep", mock_sleep):
            with pytest.raises(ConnectionError):
                await failing_function()

        # Last delays should be capped at max_delay (5.0)
        # Exponential: 1, 2, 4, 8->5, 16->5
        assert delays[-1] <= 5.1  # With tolerance


# =============================================================================
# TEST CLASS: TestOrderSubmissionErrors
# =============================================================================


class TestOrderSubmissionErrors:
    """Test order submission error handling."""

    @pytest.fixture
    def broker_with_mocks(self, mock_alpaca_creds):
        """Create broker with mocked clients."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper=True)
            broker.trading_client = MagicMock()
            broker.data_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_handles_insufficient_buying_power(self, broker_with_mocks):
        """Should handle insufficient funds gracefully."""
        broker = broker_with_mocks

        # Simulate insufficient buying power error
        broker.trading_client.submit_order.side_effect = Exception("insufficient buying power")

        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1000000,
            "type": "market",
        }

        with pytest.raises(Exception, match="insufficient buying power"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    async def test_handles_market_closed(self, broker_with_mocks):
        """Should handle market closed error."""
        broker = broker_with_mocks

        # Simulate market closed error
        broker.trading_client.submit_order.side_effect = Exception("market is closed")

        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "type": "market",
        }

        with pytest.raises(Exception, match="market is closed"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    async def test_handles_invalid_quantity(self, broker_with_mocks):
        """Should handle invalid quantity error."""
        broker = broker_with_mocks

        # Simulate invalid quantity error
        broker.trading_client.submit_order.side_effect = Exception("qty must be greater than 0")

        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": -10,
            "type": "market",
        }

        with pytest.raises(Exception, match="qty must be greater than 0"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    async def test_handles_symbol_not_found(self, broker_with_mocks):
        """Should handle unknown symbol error."""
        broker = broker_with_mocks

        # Simulate symbol not found error
        broker.trading_client.submit_order.side_effect = Exception("asset not found")

        order = {
            "symbol": "INVALID123",
            "side": "buy",
            "quantity": 10,
            "type": "market",
        }

        with pytest.raises(Exception, match="asset not found"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    async def test_handles_order_rejected(self, broker_with_mocks):
        """Should handle order rejected error."""
        broker = broker_with_mocks

        broker.trading_client.submit_order.side_effect = Exception(
            "order rejected: pattern day trader restriction"
        )

        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "type": "market",
        }

        with pytest.raises(Exception, match="order rejected"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    async def test_handles_unsupported_order_type(self, broker_with_mocks):
        """Should raise ValueError for unsupported order types."""
        broker = broker_with_mocks

        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "type": "stop_limit",  # Not directly supported in simple submit_order
        }

        with pytest.raises(ValueError, match="Unsupported order type"):
            await broker.submit_order(order)

    @pytest.mark.asyncio
    async def test_submit_order_success(self, broker_with_mocks):
        """Should return order result on success."""
        broker = broker_with_mocks

        mock_result = MagicMock()
        mock_result.id = "test-order-123"
        mock_result.symbol = "AAPL"
        mock_result.qty = "10"
        broker.trading_client.submit_order.return_value = mock_result

        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "type": "market",
        }

        result = await broker.submit_order(order)

        assert result.id == "test-order-123"
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_limit_order_submission(self, broker_with_mocks):
        """Should correctly submit limit orders."""
        broker = broker_with_mocks

        mock_result = MagicMock()
        mock_result.id = "test-order-456"
        mock_result.symbol = "AAPL"
        mock_result.qty = "10"
        broker.trading_client.submit_order.return_value = mock_result

        order = {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 10,
            "type": "limit",
            "limit_price": 150.00,
        }

        result = await broker.submit_order(order)

        assert result.id == "test-order-456"
        broker.trading_client.submit_order.assert_called_once()


# =============================================================================
# TEST CLASS: TestPriceCaching
# =============================================================================


class TestPriceCaching:
    """Test the price caching optimization."""

    @pytest.fixture
    def broker_with_mocks(self, mock_alpaca_creds):
        """Create broker with mocked clients."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper=True)
            broker.data_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_returns_cached_price_within_ttl(self, broker_with_mocks):
        """Should return cached price if within TTL."""
        broker = broker_with_mocks

        # Setup mock response
        mock_trade = MagicMock()
        mock_trade.price = 150.50
        broker.data_client.get_stock_latest_trade.return_value = {"AAPL": mock_trade}

        # First call - should hit API
        price1 = await broker.get_last_price("AAPL")
        assert price1 == 150.50
        assert broker.data_client.get_stock_latest_trade.call_count == 1

        # Second call - should return cached
        price2 = await broker.get_last_price("AAPL")
        assert price2 == 150.50
        # Should still be 1 (cached)
        assert broker.data_client.get_stock_latest_trade.call_count == 1

    @pytest.mark.asyncio
    async def test_fetches_new_price_after_ttl(self, broker_with_mocks):
        """Should fetch new price after TTL expires."""
        broker = broker_with_mocks

        # Setup mock response
        mock_trade = MagicMock()
        mock_trade.price = 150.50
        broker.data_client.get_stock_latest_trade.return_value = {"AAPL": mock_trade}

        # First call
        price1 = await broker.get_last_price("AAPL")
        assert price1 == 150.50

        # Manually expire the cache
        broker._price_cache["AAPL"] = (
            150.50,
            datetime.now() - timedelta(seconds=10),
        )

        # Update mock to return new price
        mock_trade.price = 151.00
        broker.data_client.get_stock_latest_trade.return_value = {"AAPL": mock_trade}

        # Second call - should hit API (cache expired)
        price2 = await broker.get_last_price("AAPL")
        assert price2 == 151.00
        assert broker.data_client.get_stock_latest_trade.call_count == 2

    @pytest.mark.asyncio
    async def test_caches_price_per_symbol(self, broker_with_mocks):
        """Should maintain separate cache per symbol."""
        broker = broker_with_mocks

        # Setup mock responses for different symbols
        mock_aapl = MagicMock()
        mock_aapl.price = 150.50

        mock_msft = MagicMock()
        mock_msft.price = 350.25

        def get_trade_response(request):
            symbol = request.symbol_or_symbols[0]
            if symbol == "AAPL":
                return {"AAPL": mock_aapl}
            elif symbol == "MSFT":
                return {"MSFT": mock_msft}
            return {}

        broker.data_client.get_stock_latest_trade.side_effect = get_trade_response

        # Fetch AAPL price
        price_aapl = await broker.get_last_price("AAPL")
        assert price_aapl == 150.50

        # Fetch MSFT price
        price_msft = await broker.get_last_price("MSFT")
        assert price_msft == 350.25

        # Both should be in cache
        assert "AAPL" in broker._price_cache
        assert "MSFT" in broker._price_cache

        # Verify cached values
        assert broker._price_cache["AAPL"][0] == 150.50
        assert broker._price_cache["MSFT"][0] == 350.25

    @pytest.mark.asyncio
    async def test_cache_handles_no_data(self, broker_with_mocks):
        """Should handle case when no price data is returned."""
        broker = broker_with_mocks

        # Setup mock to return empty response
        broker.data_client.get_stock_latest_trade.return_value = {}

        price = await broker.get_last_price("UNKNOWN")
        assert price is None

    @pytest.mark.asyncio
    async def test_cache_handles_api_error(self, broker_with_mocks):
        """Should return None on API error."""
        broker = broker_with_mocks

        broker.data_client.get_stock_latest_trade.side_effect = Exception("API error")

        price = await broker.get_last_price("AAPL")
        assert price is None


# =============================================================================
# TEST CLASS: TestSymbolValidation
# =============================================================================


class TestSymbolValidation:
    """Test symbol validation and sanitization."""

    def test_validates_empty_symbol(self):
        """Should raise ValueError for empty symbol."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            AlpacaBroker._validate_symbol("")

    def test_validates_none_symbol(self):
        """Should raise ValueError for None symbol."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            AlpacaBroker._validate_symbol(None)

    def test_validates_non_string_symbol(self):
        """Should raise ValueError for non-string symbol."""
        with pytest.raises(ValueError, match="Symbol must be a string"):
            AlpacaBroker._validate_symbol(123)

    def test_validates_too_long_symbol(self):
        """Should raise ValueError for symbols too long."""
        with pytest.raises(ValueError, match="Symbol too long"):
            AlpacaBroker._validate_symbol("ABCDEFGHIJK")  # 11 chars

    def test_validates_invalid_characters(self):
        """Should raise ValueError for invalid characters."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            AlpacaBroker._validate_symbol("AAPL$")

    def test_sanitizes_lowercase(self):
        """Should convert to uppercase."""
        result = AlpacaBroker._validate_symbol("aapl")
        assert result == "AAPL"

    def test_sanitizes_whitespace(self):
        """Should strip whitespace."""
        result = AlpacaBroker._validate_symbol("  AAPL  ")
        assert result == "AAPL"

    def test_allows_valid_symbol(self):
        """Should accept valid symbols."""
        assert AlpacaBroker._validate_symbol("AAPL") == "AAPL"
        assert AlpacaBroker._validate_symbol("MSFT") == "MSFT"
        assert AlpacaBroker._validate_symbol("BRK.A") == "BRK.A"
        assert AlpacaBroker._validate_symbol("BRK-B") == "BRK-B"

    def test_allows_etf_symbols(self):
        """Should accept ETF symbols with numbers."""
        assert AlpacaBroker._validate_symbol("SPY") == "SPY"
        assert AlpacaBroker._validate_symbol("QQQ") == "QQQ"
        assert AlpacaBroker._validate_symbol("VTI") == "VTI"


# =============================================================================
# TEST CLASS: TestConnectionErrors
# =============================================================================


class TestConnectionErrors:
    """Test connection and network error handling."""

    @pytest.fixture
    def broker_with_mocks(self, mock_alpaca_creds):
        """Create broker with mocked clients."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper=True)
            broker.trading_client = MagicMock()
            broker.data_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_get_account_retries_on_connection_error(self, broker_with_mocks):
        """Should retry get_account on connection errors."""
        broker = broker_with_mocks
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            account = MagicMock()
            account.equity = "100000"
            return account

        broker.trading_client.get_account.side_effect = side_effect

        # Patch sleep to speed up test
        with patch("asyncio.sleep", return_value=None):
            result = await broker.get_account()

        assert result.equity == "100000"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_get_positions_raises_on_timeout(self, broker_with_mocks):
        """Should raise BrokerConnectionError on timeout (fail fast for unreachable broker)."""
        broker = broker_with_mocks

        # Mock the _async_call_with_timeout to simulate timeout
        async def timeout_side_effect(*args, **kwargs):
            raise asyncio.TimeoutError("Request timed out")

        with patch.object(broker, "_async_call_with_timeout", side_effect=timeout_side_effect):
            with pytest.raises(BrokerConnectionError) as exc_info:
                await broker.get_positions()

        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_position_returns_none_on_not_found(self, broker_with_mocks):
        """Should return None when position not found."""
        broker = broker_with_mocks

        broker.trading_client.get_position.side_effect = Exception("Position not found")

        with patch("asyncio.sleep", return_value=None):
            result = await broker.get_position("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_order_returns_false_on_error(self, broker_with_mocks):
        """Should return False when cancel fails."""
        broker = broker_with_mocks

        broker.trading_client.cancel_order_by_id.side_effect = Exception("Order not found")

        with patch("asyncio.sleep", return_value=None):
            result = await broker.cancel_order("nonexistent-order-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_returns_true_on_success(self, broker_with_mocks):
        """Should return True when cancel succeeds."""
        broker = broker_with_mocks

        broker.trading_client.cancel_order_by_id.return_value = None

        result = await broker.cancel_order("valid-order-id")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_bars_returns_empty_on_error(self, broker_with_mocks):
        """Should return empty list on get_bars error."""
        broker = broker_with_mocks

        broker.data_client.get_stock_bars.side_effect = Exception("API error")

        with patch("asyncio.sleep", return_value=None):
            result = await broker.get_bars("AAPL")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_market_status_returns_default_on_error(self, broker_with_mocks):
        """Should return safe default when market status fails."""
        broker = broker_with_mocks

        broker.trading_client.get_clock.side_effect = Exception("API error")

        result = await broker.get_market_status()

        assert result == {"is_open": False}


# =============================================================================
# TEST CLASS: TestCancelAllOrders
# =============================================================================


class TestCancelAllOrders:
    """Test cancel all orders functionality."""

    @pytest.fixture
    def broker_with_mocks(self, mock_alpaca_creds):
        """Create broker with mocked clients."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper=True)
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_cancel_all_orders_success(self, broker_with_mocks):
        """Should cancel all orders successfully."""
        broker = broker_with_mocks

        mock_result = [MagicMock(), MagicMock()]
        broker.trading_client.cancel_orders.return_value = mock_result

        result = await broker.cancel_all_orders()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cancel_all_orders_returns_empty_on_error(self, broker_with_mocks):
        """Should return empty list on error."""
        broker = broker_with_mocks

        broker.trading_client.cancel_orders.side_effect = Exception("API error")

        with patch("asyncio.sleep", return_value=None):
            result = await broker.cancel_all_orders()

        assert result == []


# =============================================================================
# TEST CLASS: TestReplaceOrder
# =============================================================================


class TestReplaceOrder:
    """Test order replacement functionality."""

    @pytest.fixture
    def broker_with_mocks(self, mock_alpaca_creds):
        """Create broker with mocked clients."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper=True)
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_replace_order_success(self, broker_with_mocks):
        """Should replace order successfully."""
        broker = broker_with_mocks

        mock_order = MagicMock()
        mock_order.id = "replaced-order-123"
        broker.trading_client.replace_order_by_id.return_value = mock_order

        result = await broker.replace_order("original-order-id", qty=20, limit_price=155.00)

        assert result.id == "replaced-order-123"

    @pytest.mark.asyncio
    async def test_replace_order_raises_on_error(self, broker_with_mocks):
        """Should raise exception on replace error."""
        broker = broker_with_mocks

        broker.trading_client.replace_order_by_id.side_effect = Exception(
            "Order cannot be replaced"
        )

        with pytest.raises(Exception, match="Order cannot be replaced"):
            await broker.replace_order("order-id", qty=20)


# =============================================================================
# TEST CLASS: TestGetOrderMethods
# =============================================================================


class TestGetOrderMethods:
    """Test order retrieval methods."""

    @pytest.fixture
    def broker_with_mocks(self, mock_alpaca_creds):
        """Create broker with mocked clients."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper=True)
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_get_order_by_id_success(self, broker_with_mocks):
        """Should return order by ID."""
        broker = broker_with_mocks

        mock_order = MagicMock()
        mock_order.id = "order-123"
        broker.trading_client.get_order_by_id.return_value = mock_order

        result = await broker.get_order_by_id("order-123")

        assert result.id == "order-123"

    @pytest.mark.asyncio
    async def test_get_order_by_id_returns_none_on_error(self, broker_with_mocks):
        """Should return None when order not found."""
        broker = broker_with_mocks

        broker.trading_client.get_order_by_id.side_effect = Exception("Order not found")

        with patch("asyncio.sleep", return_value=None):
            result = await broker.get_order_by_id("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_order_by_client_id_success(self, broker_with_mocks):
        """Should return order by client ID."""
        broker = broker_with_mocks

        mock_order = MagicMock()
        mock_order.client_order_id = "my-client-id"
        broker.trading_client.get_order_by_client_id.return_value = mock_order

        result = await broker.get_order_by_client_id("my-client-id")

        assert result.client_order_id == "my-client-id"

    @pytest.mark.asyncio
    async def test_get_orders_success(self, broker_with_mocks):
        """Should return list of orders."""
        broker = broker_with_mocks

        mock_orders = [MagicMock(), MagicMock()]
        broker.trading_client.get_orders.return_value = mock_orders

        result = await broker.get_orders()

        assert len(result) == 2


# =============================================================================
# TEST CLASS: TestBrokerInitialization
# =============================================================================


class TestBrokerInitialization:
    """Test broker initialization and configuration."""

    def test_raises_on_missing_credentials(self):
        """Should raise when credentials are missing."""
        empty_creds = {"API_KEY": "", "API_SECRET": ""}

        with patch("brokers.alpaca_broker.ALPACA_CREDS", empty_creds):
            with pytest.raises(ValueError, match="credentials not found"):
                AlpacaBroker(paper=True)

    def test_handles_string_paper_mode(self, mock_alpaca_creds):
        """Should handle string 'true' for paper mode."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper="true")
            assert broker.paper is True

            broker2 = AlpacaBroker(paper="false")
            assert broker2.paper is False

    def test_initializes_price_cache(self, mock_alpaca_creds):
        """Should initialize empty price cache."""
        with (
            patch("brokers.alpaca_broker.ALPACA_CREDS", mock_alpaca_creds),
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            broker = AlpacaBroker(paper=True)

            assert broker._price_cache == {}
            assert broker._price_cache_ttl == timedelta(seconds=5)
