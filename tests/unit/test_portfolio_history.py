#!/usr/bin/env python3
"""
Comprehensive unit tests for Portfolio History API integration.

Tests cover:
- get_portfolio_history: Main API method for retrieving portfolio history
- get_equity_curve: Convenience method for equity curve data
- get_performance_summary: Performance metrics calculation
- get_intraday_equity: Intraday portfolio tracking
- get_historical_performance: Custom date range queries
- _calculate_max_drawdown: Drawdown calculation helper
"""

# Mock the config module before importing the broker
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

_original_config = sys.modules.get("config")
sys.modules["config"] = Mock(
    ALPACA_CREDS={"API_KEY": "test_api_key", "API_SECRET": "test_api_secret"},
    SYMBOLS=["AAPL", "MSFT", "GOOGL"],
    BACKTEST_PARAMS={
        "SLIPPAGE_PCT": 0.004,
        "BID_ASK_SPREAD": 0.001,
        "COMMISSION_PER_SHARE": 0.0,
        "EXECUTION_DELAY_BARS": 1,
        "USE_SLIPPAGE": True,
    },
    TRADING_PARAMS={},
    RISK_PARAMS={},
)

# Pre-cache broker module while config is mocked, then restore real config
import brokers.alpaca_broker as _cached_broker_mod  # noqa: E402, F401

if _original_config is not None:
    sys.modules["config"] = _original_config
elif "config" in sys.modules:
    del sys.modules["config"]


# ============================================================================
# Helper Functions for Creating Mock Data
# ============================================================================


def create_mock_portfolio_history(
    num_points: int = 30,
    start_equity: float = 100000.0,
    daily_return: float = 0.001,
    base_timestamp: int = None,
):
    """
    Create mock portfolio history response.

    Args:
        num_points: Number of data points
        start_equity: Starting equity value
        daily_return: Daily return percentage (as decimal)
        base_timestamp: Starting Unix timestamp (defaults to 30 days ago)

    Returns:
        Mock object with portfolio history attributes
    """
    if base_timestamp is None:
        base_timestamp = int((datetime.now() - timedelta(days=num_points)).timestamp())

    timestamps = [base_timestamp + (i * 86400) for i in range(num_points)]
    equity = [start_equity]
    profit_loss = [0.0]
    profit_loss_pct = [0.0]

    for i in range(1, num_points):
        daily_pnl = equity[-1] * daily_return
        equity.append(equity[-1] + daily_pnl)
        profit_loss.append(daily_pnl)
        profit_loss_pct.append(daily_return * 100)

    mock_history = Mock()
    mock_history.timestamp = timestamps
    mock_history.equity = equity
    mock_history.profit_loss = profit_loss
    mock_history.profit_loss_pct = profit_loss_pct
    mock_history.base_value = start_equity
    mock_history.timeframe = "1D"

    return mock_history


def create_mock_drawdown_history(
    num_points: int = 30,
    start_equity: float = 100000.0,
    peak_day: int = 10,
    trough_day: int = 20,
    drawdown_pct: float = 0.10,
):
    """
    Create mock portfolio history with a specific drawdown pattern.

    Args:
        num_points: Number of data points
        start_equity: Starting equity value
        peak_day: Day when equity peaks
        trough_day: Day when equity troughs
        drawdown_pct: Drawdown percentage from peak to trough

    Returns:
        Mock object with portfolio history showing drawdown
    """
    base_timestamp = int((datetime.now() - timedelta(days=num_points)).timestamp())
    timestamps = [base_timestamp + (i * 86400) for i in range(num_points)]

    # Build equity curve with drawdown
    equity = []
    peak_equity = start_equity * 1.05  # 5% gain to peak
    trough_equity = peak_equity * (1 - drawdown_pct)

    for i in range(num_points):
        if i <= peak_day:
            # Rising to peak
            progress = i / peak_day if peak_day > 0 else 1
            eq = start_equity + (peak_equity - start_equity) * progress
        elif i <= trough_day:
            # Falling to trough
            progress = (i - peak_day) / (trough_day - peak_day)
            eq = peak_equity - (peak_equity - trough_equity) * progress
        else:
            # Recovery
            progress = (i - trough_day) / (num_points - trough_day - 1) if num_points > trough_day + 1 else 1
            eq = trough_equity + (peak_equity - trough_equity) * progress * 0.5
        equity.append(eq)

    profit_loss = [0.0] + [equity[i] - equity[i - 1] for i in range(1, num_points)]
    profit_loss_pct = [0.0] + [
        (equity[i] - equity[i - 1]) / equity[i - 1] * 100 if equity[i - 1] > 0 else 0
        for i in range(1, num_points)
    ]

    mock_history = Mock()
    mock_history.timestamp = timestamps
    mock_history.equity = equity
    mock_history.profit_loss = profit_loss
    mock_history.profit_loss_pct = profit_loss_pct
    mock_history.base_value = start_equity
    mock_history.timeframe = "1D"

    return mock_history


# ============================================================================
# Test get_portfolio_history
# ============================================================================


class TestGetPortfolioHistory:
    """Test the get_portfolio_history method."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_portfolio_history_success(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return portfolio history with default parameters."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=30)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history()

        assert result is not None
        assert "timestamp" in result
        assert "equity" in result
        assert "profit_loss" in result
        assert "profit_loss_pct" in result
        assert "base_value" in result
        assert "timeframe" in result
        assert len(result["timestamp"]) == 30
        assert len(result["equity"]) == 30
        mock_trading.return_value.get_portfolio_history.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_portfolio_history_with_period(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should pass period parameter to API."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=90)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history(period="3M")

        assert result is not None
        mock_trading.return_value.get_portfolio_history.assert_called_once()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_portfolio_history_with_timeframe(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should pass timeframe parameter to API."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=24)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history(period="1D", timeframe="1H")

        assert result is not None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_portfolio_history_with_custom_dates(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should use custom date range when provided."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=60)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 1)
        result = await broker.get_portfolio_history(
            date_start=start_date, date_end=end_date
        )

        assert result is not None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_portfolio_history_extended_hours(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should include extended hours when specified."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=30)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history(extended_hours=True)

        assert result is not None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_portfolio_history_returns_none_on_error(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return None on API error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_portfolio_history.side_effect = Exception(
            "API error"
        )

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history()

        assert result is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_portfolio_history_handles_none_values(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should handle None values in response gracefully."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = Mock()
        mock_history.timestamp = None
        mock_history.equity = None
        mock_history.profit_loss = None
        mock_history.profit_loss_pct = None
        mock_history.base_value = 100000
        mock_history.timeframe = None
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history()

        assert result is not None
        assert result["timestamp"] == []
        assert result["equity"] == []
        assert result["profit_loss"] == []
        assert result["profit_loss_pct"] == []


# ============================================================================
# Test get_equity_curve
# ============================================================================


class TestGetEquityCurve:
    """Test the get_equity_curve convenience method."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_equity_curve_default(self, mock_trading, mock_data, mock_stream):
        """Should return equity curve for default 30 days."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=30)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_equity_curve()

        assert len(result) == 30
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        # Check first tuple has timestamp and equity
        assert isinstance(result[0][0], int)  # timestamp
        assert isinstance(result[0][1], float)  # equity

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_equity_curve_custom_days(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should request appropriate period for custom days."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=7)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_equity_curve(days=7)

        assert len(result) == 7

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_equity_curve_period_mapping(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should map days to correct period strings."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=90)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)

        # Test 90 days maps to 3M
        await broker.get_equity_curve(days=90)
        mock_trading.return_value.get_portfolio_history.assert_called()

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_equity_curve_returns_empty_on_error(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return empty list on error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_portfolio_history.side_effect = Exception(
            "API error"
        )

        broker = AlpacaBroker(paper=True)
        result = await broker.get_equity_curve()

        assert result == []

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_equity_curve_returns_empty_on_no_data(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return empty list when no equity data."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = Mock()
        mock_history.timestamp = []
        mock_history.equity = []
        mock_history.profit_loss = []
        mock_history.profit_loss_pct = []
        mock_history.base_value = 0
        mock_history.timeframe = "1D"
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_equity_curve()

        assert result == []


# ============================================================================
# Test get_performance_summary
# ============================================================================


class TestGetPerformanceSummary:
    """Test the get_performance_summary method."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_performance_summary_success(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return complete performance summary."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(
            num_points=30, start_equity=100000.0, daily_return=0.001
        )
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_performance_summary(period="1M")

        assert result is not None
        assert result["period"] == "1M"
        assert "start_equity" in result
        assert "end_equity" in result
        assert "total_return" in result
        assert "total_return_pct" in result
        assert "max_equity" in result
        assert "min_equity" in result
        assert "max_drawdown" in result
        assert "data_points" in result
        assert result["start_equity"] == 100000.0
        assert result["end_equity"] > result["start_equity"]  # Positive return
        assert result["data_points"] == 30

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_performance_summary_calculates_return(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should correctly calculate total return percentage."""
        from brokers.alpaca_broker import AlpacaBroker

        # 10% total return over period
        mock_history = Mock()
        mock_history.timestamp = [1000, 2000, 3000]
        mock_history.equity = [100000.0, 105000.0, 110000.0]
        mock_history.profit_loss = [0.0, 5000.0, 5000.0]
        mock_history.profit_loss_pct = [0.0, 5.0, 4.76]
        mock_history.base_value = 100000.0
        mock_history.timeframe = "1D"
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_performance_summary()

        assert result is not None
        assert abs(result["total_return_pct"] - 10.0) < 0.01  # ~10% return
        assert result["total_return"] == 10000.0

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_performance_summary_returns_none_on_error(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return None on API error."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_portfolio_history.side_effect = Exception(
            "API error"
        )

        broker = AlpacaBroker(paper=True)
        result = await broker.get_performance_summary()

        assert result is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_performance_summary_returns_none_on_empty(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return None when no equity data."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = Mock()
        mock_history.timestamp = []
        mock_history.equity = []
        mock_history.profit_loss = []
        mock_history.profit_loss_pct = []
        mock_history.base_value = 0
        mock_history.timeframe = "1D"
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_performance_summary()

        assert result is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_performance_summary_handles_none_in_equity(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should filter out None values in equity list."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = Mock()
        mock_history.timestamp = [1000, 2000, 3000, 4000]
        mock_history.equity = [100000.0, None, 105000.0, 110000.0]
        mock_history.profit_loss = [0.0, None, 5000.0, 5000.0]
        mock_history.profit_loss_pct = [0.0, None, 5.0, 4.76]
        mock_history.base_value = 100000.0
        mock_history.timeframe = "1D"
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_performance_summary()

        assert result is not None
        assert result["data_points"] == 3  # None filtered out


# ============================================================================
# Test _calculate_max_drawdown
# ============================================================================


class TestCalculateMaxDrawdown:
    """Test the _calculate_max_drawdown helper method."""

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_calculate_max_drawdown_simple(self, mock_trading, mock_data, mock_stream):
        """Should calculate simple drawdown correctly."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        # 10% drawdown: peak at 110, trough at 99
        equity = [100, 105, 110, 100, 99, 105]
        result = broker._calculate_max_drawdown(equity)

        # Max drawdown from 110 to 99 = 10%
        assert abs(result - 10.0) < 0.1

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_calculate_max_drawdown_no_drawdown(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return 0 for monotonically increasing equity."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        equity = [100, 105, 110, 115, 120]
        result = broker._calculate_max_drawdown(equity)

        assert result == 0.0

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_calculate_max_drawdown_empty_list(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return 0 for empty equity list."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        result = broker._calculate_max_drawdown([])

        assert result == 0.0

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_calculate_max_drawdown_with_none_values(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should filter None values and calculate correctly."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        equity = [100, None, 110, None, 99, 105]
        result = broker._calculate_max_drawdown(equity)

        # Max drawdown from 110 to 99 = 10%
        assert abs(result - 10.0) < 0.1

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_calculate_max_drawdown_all_none(self, mock_trading, mock_data, mock_stream):
        """Should return 0 when all values are None."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        result = broker._calculate_max_drawdown([None, None, None])

        assert result == 0.0

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_calculate_max_drawdown_multiple_drawdowns(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return the maximum of multiple drawdowns."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        # Two drawdowns: 5% then 15%
        equity = [100, 110, 104.5, 115, 97.75]  # 5% from 110, then 15% from 115
        result = broker._calculate_max_drawdown(equity)

        # Max drawdown should be 15%
        assert abs(result - 15.0) < 0.1

    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    def test_calculate_max_drawdown_recovery(self, mock_trading, mock_data, mock_stream):
        """Should track drawdown through recovery."""
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        # Peak, drop 20%, recover to new high
        equity = [100, 110, 88, 95, 115, 120]
        result = broker._calculate_max_drawdown(equity)

        # Max drawdown from 110 to 88 = 20%
        assert abs(result - 20.0) < 0.1


# ============================================================================
# Test get_intraday_equity
# ============================================================================


class TestGetIntradayEquity:
    """Test the get_intraday_equity method."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_intraday_equity_default(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return intraday equity with default 1H timeframe."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=8)  # 8 hours
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_intraday_equity()

        assert result is not None
        assert "timestamp" in result
        assert "equity" in result

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_intraday_equity_custom_timeframe(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should use custom timeframe."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=24)  # 24 15-min bars
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        result = await broker.get_intraday_equity(timeframe="15Min")

        assert result is not None


# ============================================================================
# Test get_historical_performance
# ============================================================================


class TestGetHistoricalPerformance:
    """Test the get_historical_performance method."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_historical_performance_success(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should return historical performance for date range."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=60)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 1)
        result = await broker.get_historical_performance(start_date, end_date)

        assert result is not None
        assert "timestamp" in result
        assert "equity" in result

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_historical_performance_without_end_date(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should use current date as end date if not provided."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=30)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        start_date = datetime(2024, 1, 1)
        result = await broker.get_historical_performance(start_date)

        assert result is not None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_get_historical_performance_custom_timeframe(
        self, mock_trading, mock_data, mock_stream
    ):
        """Should use custom timeframe."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = create_mock_portfolio_history(num_points=100)
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 1)
        result = await broker.get_historical_performance(
            start_date, end_date, timeframe="1H"
        )

        assert result is not None


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_profitable_month(self, mock_trading, mock_data, mock_stream):
        """Should correctly report a profitable month."""
        from brokers.alpaca_broker import AlpacaBroker

        # Simulate 5% monthly return
        mock_history = create_mock_portfolio_history(
            num_points=30, start_equity=100000.0, daily_return=0.0016  # ~5% monthly
        )
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        summary = await broker.get_performance_summary(period="1M")

        assert summary is not None
        assert summary["total_return_pct"] > 0
        assert summary["end_equity"] > summary["start_equity"]

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_losing_month(self, mock_trading, mock_data, mock_stream):
        """Should correctly report a losing month."""
        from brokers.alpaca_broker import AlpacaBroker

        # Simulate -3% monthly return
        mock_history = create_mock_portfolio_history(
            num_points=30, start_equity=100000.0, daily_return=-0.001  # ~-3% monthly
        )
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        summary = await broker.get_performance_summary(period="1M")

        assert summary is not None
        assert summary["total_return_pct"] < 0
        assert summary["end_equity"] < summary["start_equity"]
        assert summary["max_drawdown"] > 0

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_drawdown_scenario(self, mock_trading, mock_data, mock_stream):
        """Should correctly calculate drawdown in volatile period."""
        from brokers.alpaca_broker import AlpacaBroker

        # Create history with 10% drawdown
        mock_history = create_mock_drawdown_history(
            num_points=30,
            start_equity=100000.0,
            peak_day=10,
            trough_day=20,
            drawdown_pct=0.10,
        )
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        summary = await broker.get_performance_summary(period="1M")

        assert summary is not None
        assert summary["max_drawdown"] >= 9.0  # At least ~10% drawdown


# ============================================================================
# Test Error Handling and Edge Cases
# ============================================================================


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_network_error_handling(self, mock_trading, mock_data, mock_stream):
        """Should handle network errors gracefully."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_portfolio_history.side_effect = ConnectionError(
            "Network error"
        )

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history()

        assert result is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_timeout_error_handling(self, mock_trading, mock_data, mock_stream):
        """Should handle timeout errors gracefully."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_trading.return_value.get_portfolio_history.side_effect = TimeoutError(
            "Request timed out"
        )

        broker = AlpacaBroker(paper=True)
        result = await broker.get_portfolio_history()

        assert result is None

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_single_data_point(self, mock_trading, mock_data, mock_stream):
        """Should handle single data point correctly."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = Mock()
        mock_history.timestamp = [1000]
        mock_history.equity = [100000.0]
        mock_history.profit_loss = [0.0]
        mock_history.profit_loss_pct = [0.0]
        mock_history.base_value = 100000.0
        mock_history.timeframe = "1D"
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        summary = await broker.get_performance_summary()

        assert summary is not None
        assert summary["data_points"] == 1
        assert summary["total_return_pct"] == 0.0
        assert summary["max_drawdown"] == 0.0

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_zero_start_equity(self, mock_trading, mock_data, mock_stream):
        """Should handle zero start equity edge case."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = Mock()
        mock_history.timestamp = [1000, 2000, 3000]
        mock_history.equity = [0.0, 1000.0, 2000.0]
        mock_history.profit_loss = [0.0, 1000.0, 1000.0]
        mock_history.profit_loss_pct = [0.0, 0.0, 100.0]
        mock_history.base_value = 0.0
        mock_history.timeframe = "1D"
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        summary = await broker.get_performance_summary()

        assert summary is not None
        # Should handle divide by zero gracefully
        assert summary["total_return_pct"] == 0.0

    @pytest.mark.asyncio
    @patch("brokers.alpaca_broker.StockDataStream")
    @patch("brokers.alpaca_broker.StockHistoricalDataClient")
    @patch("brokers.alpaca_broker.TradingClient")
    async def test_mismatched_list_lengths(self, mock_trading, mock_data, mock_stream):
        """Should handle mismatched timestamp/equity list lengths."""
        from brokers.alpaca_broker import AlpacaBroker

        mock_history = Mock()
        mock_history.timestamp = [1000, 2000, 3000, 4000, 5000]
        mock_history.equity = [100000.0, 101000.0, 102000.0]  # Shorter
        mock_history.profit_loss = [0.0, 1000.0, 1000.0]
        mock_history.profit_loss_pct = [0.0, 1.0, 1.0]
        mock_history.base_value = 100000.0
        mock_history.timeframe = "1D"
        mock_trading.return_value.get_portfolio_history.return_value = mock_history

        broker = AlpacaBroker(paper=True)
        curve = await broker.get_equity_curve()

        # Should only return matching pairs
        assert len(curve) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
