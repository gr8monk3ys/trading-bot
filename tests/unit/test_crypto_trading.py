"""
Unit tests for cryptocurrency trading functionality.

Tests cover:
- Crypto symbol detection and normalization
- Crypto order building (market, limit, notional)
- Crypto data fetching (bars, quotes, prices)
- Crypto order submission
- 24/7 trading support

Note: Most broker tests use mocks to avoid actual API calls.
"""

# Set up test environment before importing broker modules
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from alpaca.trading.enums import OrderSide, TimeInForce

os.environ["TESTING"] = "True"
os.environ["ALPACA_API_KEY"] = "test_key"
os.environ["ALPACA_SECRET_KEY"] = "test_secret"


# =============================================================================
# ORDER BUILDER CRYPTO TESTS
# =============================================================================


class TestOrderBuilderCryptoDetection:
    """Test crypto symbol detection in OrderBuilder."""

    def test_detects_crypto_with_slash(self):
        """Symbols with slash are detected as crypto."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("BTC/USD", "buy", 0.5)
        assert builder.is_crypto is True
        assert builder.symbol == "BTC/USD"

    def test_detects_crypto_without_slash(self):
        """Crypto symbols without slash are detected and normalized."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("BTCUSD", "buy", 0.5)
        assert builder.is_crypto is True
        assert builder.symbol == "BTC/USD"

    def test_detects_crypto_with_dash(self):
        """Crypto symbols with dash are detected and normalized."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("BTC-USD", "buy", 0.5)
        assert builder.is_crypto is True
        assert builder.symbol == "BTC/USD"

    def test_detects_stock_symbols(self):
        """Stock symbols are not detected as crypto."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("AAPL", "buy", 100)
        assert builder.is_crypto is False
        assert builder.symbol == "AAPL"

    def test_all_supported_crypto_pairs(self):
        """All supported crypto pairs are detected."""
        from brokers.order_builder import OrderBuilder

        pairs = [
            "BTC/USD",
            "ETH/USD",
            "SOL/USD",
            "AVAX/USD",
            "DOGE/USD",
            "SHIB/USD",
            "LTC/USD",
            "BCH/USD",
            "LINK/USD",
            "UNI/USD",
        ]

        for pair in pairs:
            builder = OrderBuilder(pair, "buy", 1.0)
            assert builder.is_crypto is True, f"{pair} should be detected as crypto"

    def test_lowercase_crypto_symbols(self):
        """Lowercase crypto symbols are handled correctly."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("btc/usd", "buy", 0.5)
        assert builder.is_crypto is True
        assert builder.symbol == "BTC/USD"


class TestOrderBuilderCryptoDefaults:
    """Test default values for crypto orders."""

    def test_crypto_defaults_to_gtc(self):
        """Crypto orders default to GTC time-in-force."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("BTC/USD", "buy", 0.5)
        assert builder._time_in_force == TimeInForce.GTC

    def test_stock_defaults_to_day(self):
        """Stock orders default to DAY time-in-force."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("AAPL", "buy", 100)
        assert builder._time_in_force == TimeInForce.DAY

    def test_crypto_can_override_tif(self):
        """Crypto time-in-force can be overridden."""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("BTC/USD", "buy", 0.5).ioc()
        assert builder._time_in_force == TimeInForce.IOC


class TestOrderBuilderCryptoOrders:
    """Test building crypto orders."""

    def test_crypto_market_order_with_qty(self):
        """Build crypto market order with quantity."""
        from brokers.order_builder import OrderBuilder

        order = OrderBuilder("BTC/USD", "buy", 0.5).market().build()

        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.BUY
        assert order.qty == 0.5
        assert order.time_in_force == TimeInForce.GTC

    def test_crypto_market_order_with_notional(self):
        """Build crypto market order with notional amount."""
        from brokers.order_builder import OrderBuilder

        order = OrderBuilder("ETH/USD", "buy").notional(1000.00).market().build()

        assert order.symbol == "ETH/USD"
        assert order.side == OrderSide.BUY
        assert order.notional == 1000.00
        assert order.time_in_force == TimeInForce.GTC

    def test_crypto_limit_order(self):
        """Build crypto limit order."""
        from brokers.order_builder import OrderBuilder

        order = OrderBuilder("BTC/USD", "buy", 0.5).limit(40000.00).build()

        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.BUY
        assert order.qty == 0.5
        assert order.limit_price == 40000.00

    def test_crypto_sell_order(self):
        """Build crypto sell order."""
        from brokers.order_builder import OrderBuilder

        order = OrderBuilder("BTC/USD", "sell", 0.25).market().build()

        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.SELL
        assert order.qty == 0.25


class TestCryptoConvenienceFunctions:
    """Test crypto convenience functions."""

    def test_crypto_market_order_function_with_qty(self):
        """crypto_market_order() with quantity."""
        from brokers.order_builder import crypto_market_order

        order = crypto_market_order("BTC/USD", "buy", qty=0.5)

        assert order.symbol == "BTC/USD"
        assert order.qty == 0.5
        assert order.time_in_force == TimeInForce.GTC

    def test_crypto_market_order_function_with_notional(self):
        """crypto_market_order() with notional amount."""
        from brokers.order_builder import crypto_market_order

        order = crypto_market_order("ETHUSD", "buy", notional=1000.00)

        assert order.symbol == "ETH/USD"
        assert order.notional == 1000.00

    def test_crypto_market_order_requires_qty_or_notional(self):
        """crypto_market_order() requires qty or notional."""
        from brokers.order_builder import crypto_market_order

        with pytest.raises(ValueError, match="Either qty or notional"):
            crypto_market_order("BTC/USD", "buy")

    def test_crypto_market_order_rejects_both_qty_and_notional(self):
        """crypto_market_order() rejects both qty and notional."""
        from brokers.order_builder import crypto_market_order

        with pytest.raises(ValueError, match="either qty or notional"):
            crypto_market_order("BTC/USD", "buy", qty=0.5, notional=1000.00)

    def test_crypto_limit_order_function(self):
        """crypto_limit_order() function."""
        from brokers.order_builder import crypto_limit_order

        order = crypto_limit_order("BTC/USD", "buy", 0.5, 40000.00)

        assert order.symbol == "BTC/USD"
        assert order.qty == 0.5
        assert order.limit_price == 40000.00

    def test_is_crypto_symbol_function(self):
        """is_crypto_symbol() helper function."""
        from brokers.order_builder import is_crypto_symbol

        assert is_crypto_symbol("BTC/USD") is True
        assert is_crypto_symbol("BTCUSD") is True
        assert is_crypto_symbol("btc-usd") is True
        assert is_crypto_symbol("AAPL") is False
        assert is_crypto_symbol("MSFT") is False


# =============================================================================
# BROKER CRYPTO TESTS
# =============================================================================


class TestAlpacaBrokerCryptoSymbols:
    """Test crypto symbol handling in AlpacaBroker."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        with (
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            from brokers.alpaca_broker import AlpacaBroker

            return AlpacaBroker(paper=True)

    def test_is_crypto_with_slash(self, mock_broker):
        """is_crypto() detects slash-format symbols."""
        assert mock_broker.is_crypto("BTC/USD") is True
        assert mock_broker.is_crypto("ETH/USD") is True

    def test_is_crypto_without_slash(self, mock_broker):
        """is_crypto() detects no-slash format symbols."""
        assert mock_broker.is_crypto("BTCUSD") is True
        assert mock_broker.is_crypto("ETHUSD") is True

    def test_is_crypto_stock_symbols(self, mock_broker):
        """is_crypto() returns False for stock symbols."""
        assert mock_broker.is_crypto("AAPL") is False
        assert mock_broker.is_crypto("MSFT") is False
        assert mock_broker.is_crypto("GOOGL") is False

    def test_normalize_crypto_symbol_with_slash(self, mock_broker):
        """normalize_crypto_symbol() handles slash format."""
        assert mock_broker.normalize_crypto_symbol("BTC/USD") == "BTC/USD"

    def test_normalize_crypto_symbol_without_slash(self, mock_broker):
        """normalize_crypto_symbol() normalizes no-slash format."""
        assert mock_broker.normalize_crypto_symbol("BTCUSD") == "BTC/USD"
        assert mock_broker.normalize_crypto_symbol("ETHUSD") == "ETH/USD"

    def test_normalize_crypto_symbol_with_dash(self, mock_broker):
        """normalize_crypto_symbol() handles dash format."""
        assert mock_broker.normalize_crypto_symbol("BTC-USD") == "BTC/USD"

    def test_normalize_crypto_symbol_lowercase(self, mock_broker):
        """normalize_crypto_symbol() handles lowercase."""
        assert mock_broker.normalize_crypto_symbol("btc/usd") == "BTC/USD"
        assert mock_broker.normalize_crypto_symbol("btcusd") == "BTC/USD"

    def test_normalize_crypto_symbol_invalid(self, mock_broker):
        """normalize_crypto_symbol() raises for invalid symbols."""
        with pytest.raises(ValueError, match="Unrecognized crypto symbol"):
            mock_broker.normalize_crypto_symbol("INVALID")


class TestAlpacaBrokerCryptoData:
    """Test crypto data fetching methods."""

    @pytest.fixture
    def mock_broker_with_crypto(self):
        """Create a mock broker with crypto client mocked."""
        with (
            patch("brokers.alpaca_broker.TradingClient"),
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
            patch("brokers.alpaca_broker.CryptoHistoricalDataClient") as mock_crypto,
        ):
            from brokers.alpaca_broker import AlpacaBroker

            broker = AlpacaBroker(paper=True)
            broker._mock_crypto_client = mock_crypto
            return broker

    @pytest.mark.asyncio
    async def test_get_crypto_last_price(self, mock_broker_with_crypto):
        """get_crypto_last_price() returns price from cache or API."""
        broker = mock_broker_with_crypto

        # Mock the crypto client
        mock_client = MagicMock()
        mock_trade = MagicMock()
        mock_trade.price = 45000.00
        mock_client.get_crypto_latest_trade.return_value = {"BTC/USD": mock_trade}
        broker._crypto_data_client = mock_client

        with patch("asyncio.to_thread", return_value={"BTC/USD": mock_trade}):
            price = await broker.get_crypto_last_price("BTC/USD")
            assert price == 45000.00

    @pytest.mark.asyncio
    async def test_get_crypto_last_price_caching(self, mock_broker_with_crypto):
        """get_crypto_last_price() uses cache."""
        broker = mock_broker_with_crypto

        # Pre-populate cache
        from datetime import datetime

        broker._price_cache["crypto:BTC/USD"] = (45000.00, datetime.now())

        # Should return cached value without API call
        price = await broker.get_crypto_last_price("BTC/USD")
        assert price == 45000.00

    @pytest.mark.asyncio
    async def test_is_crypto_tradeable(self, mock_broker_with_crypto):
        """is_crypto_tradeable() validates symbols."""
        broker = mock_broker_with_crypto

        assert await broker.is_crypto_tradeable("BTC/USD") is True
        assert await broker.is_crypto_tradeable("BTCUSD") is True
        assert await broker.is_crypto_tradeable("INVALID") is False


class TestAlpacaBrokerCryptoOrders:
    """Test crypto order submission."""

    @pytest.fixture
    def mock_broker_for_orders(self):
        """Create a mock broker for order testing."""
        with (
            patch("brokers.alpaca_broker.TradingClient") as mock_trading,
            patch("brokers.alpaca_broker.StockHistoricalDataClient"),
            patch("brokers.alpaca_broker.StockDataStream"),
        ):
            from brokers.alpaca_broker import AlpacaBroker

            broker = AlpacaBroker(paper=True)
            broker._mock_trading = mock_trading
            return broker

    @pytest.mark.asyncio
    async def test_submit_crypto_order_market_qty(self, mock_broker_for_orders):
        """submit_crypto_order() with market order and quantity."""
        broker = mock_broker_for_orders

        # Mock order response
        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "BTC/USD"
        mock_order.side.value = "buy"
        mock_order.qty = "0.5"
        mock_order.notional = None
        mock_order.type.value = "market"
        mock_order.status.value = "new"
        mock_order.created_at = datetime.now()

        with patch("asyncio.to_thread", return_value=mock_order):
            result = await broker.submit_crypto_order(symbol="BTC/USD", side="buy", qty=0.5)

            assert result["id"] == "order123"
            assert result["symbol"] == "BTC/USD"
            assert result["side"] == "buy"
            assert result["qty"] == "0.5"

    @pytest.mark.asyncio
    async def test_submit_crypto_order_market_notional(self, mock_broker_for_orders):
        """submit_crypto_order() with market order and notional amount."""
        broker = mock_broker_for_orders

        mock_order = MagicMock()
        mock_order.id = "order456"
        mock_order.symbol = "ETH/USD"
        mock_order.side.value = "buy"
        mock_order.qty = None
        mock_order.notional = "1000.00"
        mock_order.type.value = "market"
        mock_order.status.value = "new"
        mock_order.created_at = datetime.now()

        with patch("asyncio.to_thread", return_value=mock_order):
            result = await broker.submit_crypto_order(
                symbol="ETH/USD", side="buy", notional=1000.00
            )

            assert result["notional"] == "1000.00"
            assert result["qty"] is None

    @pytest.mark.asyncio
    async def test_submit_crypto_order_limit(self, mock_broker_for_orders):
        """submit_crypto_order() with limit order."""
        broker = mock_broker_for_orders

        mock_order = MagicMock()
        mock_order.id = "order789"
        mock_order.symbol = "BTC/USD"
        mock_order.side.value = "buy"
        mock_order.qty = "0.5"
        mock_order.notional = None
        mock_order.type.value = "limit"
        mock_order.status.value = "new"
        mock_order.created_at = datetime.now()

        with patch("asyncio.to_thread", return_value=mock_order):
            result = await broker.submit_crypto_order(
                symbol="BTC/USD", side="buy", qty=0.5, order_type="limit", limit_price=40000.00
            )

            assert result["type"] == "limit"

    @pytest.mark.asyncio
    async def test_submit_crypto_order_validation_no_qty_or_notional(self, mock_broker_for_orders):
        """submit_crypto_order() requires qty or notional."""
        broker = mock_broker_for_orders

        with pytest.raises(ValueError, match="Either qty or notional"):
            await broker.submit_crypto_order(symbol="BTC/USD", side="buy")

    @pytest.mark.asyncio
    async def test_submit_crypto_order_validation_both_qty_and_notional(
        self, mock_broker_for_orders
    ):
        """submit_crypto_order() rejects both qty and notional."""
        broker = mock_broker_for_orders

        with pytest.raises(ValueError, match="either qty or notional"):
            await broker.submit_crypto_order(
                symbol="BTC/USD", side="buy", qty=0.5, notional=1000.00
            )

    @pytest.mark.asyncio
    async def test_submit_crypto_order_limit_requires_price(self, mock_broker_for_orders):
        """submit_crypto_order() limit order requires limit_price."""
        broker = mock_broker_for_orders

        with pytest.raises(ValueError, match="limit_price required"):
            await broker.submit_crypto_order(
                symbol="BTC/USD", side="buy", qty=0.5, order_type="limit"
            )

    @pytest.mark.asyncio
    async def test_submit_crypto_order_normalizes_symbol(self, mock_broker_for_orders):
        """submit_crypto_order() normalizes symbol format."""
        broker = mock_broker_for_orders

        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "BTC/USD"
        mock_order.side.value = "buy"
        mock_order.qty = "0.5"
        mock_order.notional = None
        mock_order.type.value = "market"
        mock_order.status.value = "new"
        mock_order.created_at = datetime.now()

        with patch("asyncio.to_thread", return_value=mock_order):
            # Use BTCUSD format (without slash)
            result = await broker.submit_crypto_order(symbol="BTCUSD", side="buy", qty=0.5)

            # Should be normalized to BTC/USD
            assert result["symbol"] == "BTC/USD"


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestCryptoConfig:
    """Test crypto configuration."""

    def test_crypto_symbols_defined(self):
        """CRYPTO_SYMBOLS is defined in config."""
        from config import CRYPTO_SYMBOLS

        assert isinstance(CRYPTO_SYMBOLS, list)
        assert len(CRYPTO_SYMBOLS) > 0
        assert "BTC/USD" in CRYPTO_SYMBOLS
        assert "ETH/USD" in CRYPTO_SYMBOLS

    def test_crypto_params_defined(self):
        """CRYPTO_PARAMS is defined in config."""
        from config import CRYPTO_PARAMS

        assert isinstance(CRYPTO_PARAMS, dict)
        assert "ENABLED" in CRYPTO_PARAMS
        assert "POSITION_SIZE" in CRYPTO_PARAMS
        assert "STOP_LOSS" in CRYPTO_PARAMS
        assert "TAKE_PROFIT" in CRYPTO_PARAMS

    def test_crypto_params_have_higher_volatility_settings(self):
        """Crypto params should account for higher volatility."""
        from config import CRYPTO_PARAMS, TRADING_PARAMS

        # Crypto position size should be lower (more conservative)
        assert CRYPTO_PARAMS["POSITION_SIZE"] <= TRADING_PARAMS["POSITION_SIZE"]

        # Crypto stop loss should be higher (wider to account for volatility)
        assert CRYPTO_PARAMS["STOP_LOSS"] >= TRADING_PARAMS["STOP_LOSS"]
