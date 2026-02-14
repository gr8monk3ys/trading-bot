"""
Unit tests for SimpleSymbolSelector.

Tests the symbol selection logic including:
- Initialization and configuration
- Getting tradable stocks
- Getting most active stocks
- Filtering by criteria
- Selecting top symbols
"""

from unittest.mock import patch


class MockAsset:
    """Mock Alpaca asset object."""

    def __init__(
        self, symbol: str, tradable: bool = True, fractionable: bool = True, marginable: bool = True
    ):
        self.symbol = symbol
        self.tradable = tradable
        self.fractionable = fractionable
        self.marginable = marginable


class MockTrade:
    """Mock Alpaca trade object."""

    def __init__(self, price: float):
        self.price = price


class MockBar:
    """Mock Alpaca bar object."""

    def __init__(self, open_: float, high: float, low: float, close: float, volume: int):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class MockBarsResponse:
    """Mock Alpaca bars response."""

    def __init__(self, data: dict):
        self.data = data


class TestSimpleSymbolSelectorInit:
    """Test SimpleSymbolSelector initialization."""

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_init_with_credentials(self, mock_trading_client, mock_data_client):
        """Test initialization with API credentials."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector(api_key="test_key", secret_key="test_secret", paper=True)

        mock_trading_client.assert_called_once_with("test_key", "test_secret", paper=True)
        mock_data_client.assert_called_once_with("test_key", "test_secret")

        assert selector.min_price == 10.0
        assert selector.max_price == 500.0
        assert selector.min_volume == 1_000_000

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_init_with_live_trading(self, mock_trading_client, mock_data_client):
        """Test initialization with live trading (paper=False)."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        SimpleSymbolSelector(api_key="live_key", secret_key="live_secret", paper=False)

        mock_trading_client.assert_called_once_with("live_key", "live_secret", paper=False)


class TestGetAllTradableStocks:
    """Test getting all tradable stocks."""

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_get_all_tradable_stocks_success(self, mock_trading_client, mock_data_client):
        """Test successful retrieval of tradable stocks."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        # Create mock assets
        mock_assets = [
            MockAsset("AAPL", tradable=True, fractionable=True, marginable=True),
            MockAsset("MSFT", tradable=True, fractionable=True, marginable=True),
            MockAsset("UNTRADABLE", tradable=False, fractionable=True, marginable=True),
            MockAsset("NO_FRAC", tradable=True, fractionable=False, marginable=True),
        ]

        mock_trading_client.return_value.get_all_assets.return_value = mock_assets

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.get_all_tradable_stocks()

        assert "AAPL" in result
        assert "MSFT" in result
        assert "UNTRADABLE" not in result  # Not tradable
        assert "NO_FRAC" not in result  # Not fractionable

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_get_all_tradable_stocks_empty(self, mock_trading_client, mock_data_client):
        """Test when no stocks are tradable."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_trading_client.return_value.get_all_assets.return_value = []

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.get_all_tradable_stocks()

        assert result == []

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_get_all_tradable_stocks_error(self, mock_trading_client, mock_data_client):
        """Test error handling when API call fails."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_trading_client.return_value.get_all_assets.side_effect = Exception("API Error")

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.get_all_tradable_stocks()

        assert result == []


class TestGetMostActiveStocks:
    """Test getting most active stocks from curated list."""

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_get_most_active_stocks_default(self, mock_trading_client, mock_data_client):
        """Test getting most active stocks with default top_n."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.get_most_active_stocks()

        # Default is top 50
        assert len(result) <= 50
        # Should contain some expected symbols
        assert any(s in result for s in ["AAPL", "MSFT", "GOOGL", "AMZN"])

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_get_most_active_stocks_limited(self, mock_trading_client, mock_data_client):
        """Test getting most active stocks with limited top_n."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.get_most_active_stocks(top_n=10)

        assert len(result) == 10

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_get_most_active_stocks_no_limit(self, mock_trading_client, mock_data_client):
        """Test getting all active stocks without limit."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.get_most_active_stocks(top_n=0)

        # Should include S&P 100 + growth stocks + ETFs (minus duplicates)
        assert len(result) > 100

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_get_most_active_stocks_contains_etfs(self, mock_trading_client, mock_data_client):
        """Test that result includes ETFs."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.get_most_active_stocks(top_n=0)

        etfs = ["SPY", "QQQ", "IWM"]
        for etf in etfs:
            assert etf in result


class TestFilterByCriteria:
    """Test filtering symbols by trading criteria."""

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_success(self, mock_trading_client, mock_data_client):
        """Test successful filtering with valid symbols."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        # Mock latest trade response
        mock_data_client.return_value.get_stock_latest_trade.return_value = {
            "AAPL": MockTrade(price=175.0)
        }

        # Mock bars response with volume and price data
        bars = [
            MockBar(170.0, 180.0, 165.0, 175.0, 2_000_000),
            MockBar(175.0, 185.0, 170.0, 180.0, 2_500_000),
        ]
        mock_data_client.return_value.get_stock_bars.return_value = MockBarsResponse({"AAPL": bars})

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["AAPL"])

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["price"] == 175.0
        assert "avg_volume" in result[0]
        assert "momentum_5d" in result[0]

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_price_too_low(self, mock_trading_client, mock_data_client):
        """Test filtering removes stocks with price below minimum."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_data_client.return_value.get_stock_latest_trade.return_value = {
            "CHEAP": MockTrade(price=5.0)  # Below $10 minimum
        }

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["CHEAP"])

        assert len(result) == 0

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_price_too_high(self, mock_trading_client, mock_data_client):
        """Test filtering removes stocks with price above maximum."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_data_client.return_value.get_stock_latest_trade.return_value = {
            "EXPENSIVE": MockTrade(price=600.0)  # Above $500 maximum
        }

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["EXPENSIVE"])

        assert len(result) == 0

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_low_volume(self, mock_trading_client, mock_data_client):
        """Test filtering removes stocks with volume below minimum."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_data_client.return_value.get_stock_latest_trade.return_value = {
            "LOWVOL": MockTrade(price=50.0)
        }

        # Low volume bars
        bars = [
            MockBar(50.0, 52.0, 48.0, 51.0, 100_000),  # Only 100k volume
            MockBar(51.0, 53.0, 49.0, 52.0, 150_000),
        ]
        mock_data_client.return_value.get_stock_bars.return_value = MockBarsResponse(
            {"LOWVOL": bars}
        )

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["LOWVOL"])

        assert len(result) == 0

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_symbol_not_found(self, mock_trading_client, mock_data_client):
        """Test filtering handles missing symbol data."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_data_client.return_value.get_stock_latest_trade.return_value = {}

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["NOTFOUND"])

        assert len(result) == 0

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_no_bars(self, mock_trading_client, mock_data_client):
        """Test filtering handles missing bar data."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_data_client.return_value.get_stock_latest_trade.return_value = {
            "NOBARS": MockTrade(price=100.0)
        }
        mock_data_client.return_value.get_stock_bars.return_value = MockBarsResponse({})

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["NOBARS"])

        assert len(result) == 0

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_sorts_by_momentum(self, mock_trading_client, mock_data_client):
        """Test that results are sorted by momentum score."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        # Setup two symbols with different momentum
        def mock_latest_trade(request):
            symbols = request.symbol_or_symbols
            return {symbols[0]: MockTrade(price=100.0)}

        def mock_bars(request):
            symbol = request.symbol_or_symbols[0]
            if symbol == "HIGH_MOM":
                bars = [
                    MockBar(80.0, 85.0, 75.0, 85.0, 2_000_000),
                    MockBar(85.0, 105.0, 80.0, 100.0, 2_500_000),  # +25% momentum
                ]
            else:
                bars = [
                    MockBar(95.0, 100.0, 90.0, 98.0, 2_000_000),
                    MockBar(98.0, 105.0, 95.0, 100.0, 2_500_000),  # +2% momentum
                ]
            return MockBarsResponse({symbol: bars})

        mock_data_client.return_value.get_stock_latest_trade.side_effect = mock_latest_trade
        mock_data_client.return_value.get_stock_bars.side_effect = mock_bars

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["LOW_MOM", "HIGH_MOM"])

        assert len(result) == 2
        # High momentum should be first
        assert result[0]["symbol"] == "HIGH_MOM"
        assert result[1]["symbol"] == "LOW_MOM"

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_filter_by_criteria_handles_exception(self, mock_trading_client, mock_data_client):
        """Test filtering handles exceptions gracefully."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        mock_data_client.return_value.get_stock_latest_trade.side_effect = Exception("API Error")

        selector = SimpleSymbolSelector("key", "secret")
        result = selector.filter_by_criteria(["AAPL"])

        assert len(result) == 0


class TestSelectTopSymbols:
    """Test selecting top symbols for trading."""

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_select_top_symbols_success(self, mock_trading_client, mock_data_client):
        """Test successful selection of top symbols."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        # Mock latest trade for all requests
        mock_data_client.return_value.get_stock_latest_trade.return_value = {
            "AAPL": MockTrade(price=175.0)
        }

        # Mock bars with high volume
        bars = [
            MockBar(165.0, 180.0, 160.0, 170.0, 5_000_000),
            MockBar(170.0, 185.0, 165.0, 180.0, 6_000_000),
        ]
        mock_data_client.return_value.get_stock_bars.return_value = MockBarsResponse({"AAPL": bars})

        selector = SimpleSymbolSelector("key", "secret")

        # Patch get_most_active_stocks to return a small list for testing
        with patch.object(selector, "get_most_active_stocks", return_value=["AAPL"]):
            result = selector.select_top_symbols(top_n=5)

        assert "AAPL" in result

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_select_top_symbols_with_min_score(self, mock_trading_client, mock_data_client):
        """Test filtering by minimum score."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector("key", "secret")

        # Mock filter_by_criteria to return symbols with different scores
        mock_filtered = [
            {"symbol": "HIGH", "price": 100, "avg_volume": 2e6, "momentum_5d": 5.0, "score": 5.0},
            {"symbol": "LOW", "price": 100, "avg_volume": 2e6, "momentum_5d": 0.5, "score": 0.5},
        ]

        with patch.object(selector, "get_most_active_stocks", return_value=["HIGH", "LOW"]):
            with patch.object(selector, "filter_by_criteria", return_value=mock_filtered):
                result = selector.select_top_symbols(top_n=10, min_score=1.0)

        assert "HIGH" in result
        assert "LOW" not in result  # Score below min_score

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_select_top_symbols_respects_top_n(self, mock_trading_client, mock_data_client):
        """Test that result is limited to top_n symbols."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector("key", "secret")

        # Mock many filtered results
        mock_filtered = [
            {
                "symbol": f"SYM{i}",
                "price": 100,
                "avg_volume": 2e6,
                "momentum_5d": 10 - i,
                "score": 10 - i,
            }
            for i in range(20)
        ]

        with patch.object(
            selector, "get_most_active_stocks", return_value=[f"SYM{i}" for i in range(20)]
        ):
            with patch.object(selector, "filter_by_criteria", return_value=mock_filtered):
                result = selector.select_top_symbols(top_n=5)

        assert len(result) == 5

    @patch("utils.simple_symbol_selector.StockHistoricalDataClient")
    @patch("utils.simple_symbol_selector.TradingClient")
    def test_select_top_symbols_empty_result(self, mock_trading_client, mock_data_client):
        """Test when no symbols pass criteria."""
        from utils.simple_symbol_selector import SimpleSymbolSelector

        selector = SimpleSymbolSelector("key", "secret")

        with patch.object(selector, "get_most_active_stocks", return_value=["AAPL"]):
            with patch.object(selector, "filter_by_criteria", return_value=[]):
                result = selector.select_top_symbols(top_n=10)

        assert result == []
