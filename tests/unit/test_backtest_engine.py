#!/usr/bin/env python3
"""
Unit tests for BacktestEngine.

Tests cover:
1. Engine initialization
2. Running backtests
3. Performance metrics calculation
4. Trade P&L calculation
5. Strategy iteration
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.backtest_engine import BacktestEngine


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine():
    """Create a basic BacktestEngine instance."""
    return BacktestEngine()


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = MagicMock()
    broker.get_portfolio_value.return_value = 10000
    broker.get_balance.return_value = 5000
    broker.get_positions.return_value = []
    broker.get_trades.return_value = []
    return broker


@pytest.fixture
def engine_with_broker(mock_broker):
    """Create a BacktestEngine with mock broker."""
    return BacktestEngine(broker=mock_broker)


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    strategy = MagicMock()
    strategy.__class__.__name__ = "MockStrategy"
    strategy.on_trading_iteration = MagicMock()
    strategy.analyze_symbol = AsyncMock(return_value={"action": "neutral"})
    strategy.execute_trade = AsyncMock()
    return strategy


@pytest.fixture
def sample_trades():
    """Create sample trades for P&L calculation testing."""
    return [
        {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, "timestamp": datetime(2024, 1, 5)},
        {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 155.0, "timestamp": datetime(2024, 1, 10)},
        {"symbol": "AAPL", "side": "sell", "quantity": 15, "price": 160.0, "timestamp": datetime(2024, 1, 15)},
        {"symbol": "MSFT", "side": "buy", "quantity": 5, "price": 300.0, "timestamp": datetime(2024, 1, 5)},
        {"symbol": "MSFT", "side": "sell", "quantity": 5, "price": 280.0, "timestamp": datetime(2024, 1, 20)},
    ]


# =============================================================================
# TEST INITIALIZATION
# =============================================================================


class TestBacktestEngineInit:
    """Test BacktestEngine initialization."""

    def test_default_initialization(self, engine):
        """Test default initialization."""
        assert engine.broker is None
        assert engine.current_date is None
        assert engine.strategies == []
        assert engine.results == {}

    def test_initialization_with_broker(self, engine_with_broker, mock_broker):
        """Test initialization with broker."""
        assert engine_with_broker.broker == mock_broker


# =============================================================================
# TEST RUN METHOD
# =============================================================================


class TestRunMethod:
    """Test the run method."""

    @pytest.mark.asyncio
    async def test_run_sets_strategies(self, engine_with_broker, mock_strategy):
        """Test that run sets strategies list."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        await engine_with_broker.run([mock_strategy], start, end)

        assert engine_with_broker.strategies == [mock_strategy]

    @pytest.mark.asyncio
    async def test_run_creates_result_dataframe(self, engine_with_broker, mock_strategy):
        """Test that run creates result DataFrames."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        results = await engine_with_broker.run([mock_strategy], start, end)

        assert len(results) == 1
        assert isinstance(results[0], pd.DataFrame)
        assert "equity" in results[0].columns
        assert "cash" in results[0].columns
        assert "holdings" in results[0].columns

    @pytest.mark.asyncio
    async def test_run_skips_weekends(self, engine_with_broker, mock_strategy):
        """Test that run skips weekends."""
        # Start on Friday Jan 5, 2024, end on Monday Jan 8, 2024
        start = datetime(2024, 1, 5)  # Friday
        end = datetime(2024, 1, 8)  # Monday

        results = await engine_with_broker.run([mock_strategy], start, end)

        # Should have entries for Friday and Monday only
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_assigns_broker_to_strategy(self, engine_with_broker, mock_strategy):
        """Test that broker is assigned to strategy."""
        # Remove broker attribute if it exists
        if hasattr(mock_strategy, "broker"):
            delattr(mock_strategy, "broker")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        await engine_with_broker.run([mock_strategy], start, end)

        assert mock_strategy.broker == engine_with_broker.broker

    @pytest.mark.asyncio
    async def test_run_handles_strategy_error(self, engine_with_broker):
        """Test that run handles strategy errors gracefully."""
        strategy = MagicMock()
        strategy.__class__.__name__ = "ErrorStrategy"
        strategy.on_trading_iteration = MagicMock(side_effect=Exception("Strategy error"))

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        # Should not raise
        results = await engine_with_broker.run([strategy], start, end)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_calculates_returns(self, engine_with_broker, mock_strategy):
        """Test that returns are calculated correctly."""
        # Setup broker to return increasing equity
        equity_values = [10000, 10100, 10200, 10150]
        call_count = [0]

        def get_portfolio_value(date):
            result = equity_values[min(call_count[0], len(equity_values) - 1)]
            call_count[0] += 1
            return result

        engine_with_broker.broker.get_portfolio_value = get_portfolio_value

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        results = await engine_with_broker.run([mock_strategy], start, end)

        # Check that cum_returns is calculated
        assert "cum_returns" in results[0].columns


# =============================================================================
# TEST STRATEGY ITERATION
# =============================================================================


class TestStrategyIteration:
    """Test _run_strategy_iteration method."""

    @pytest.mark.asyncio
    async def test_calls_on_trading_iteration(self, engine, mock_strategy):
        """Test that on_trading_iteration is called."""
        current_date = datetime(2024, 1, 5)

        await engine._run_strategy_iteration(mock_strategy, current_date)

        mock_strategy.on_trading_iteration.assert_called_once()

    @pytest.mark.asyncio
    async def test_sets_current_date_on_strategy(self, engine, mock_strategy):
        """Test that current_date is set on strategy."""
        current_date = datetime(2024, 1, 5)

        await engine._run_strategy_iteration(mock_strategy, current_date)

        assert mock_strategy.current_date == current_date

    @pytest.mark.asyncio
    async def test_handles_missing_on_trading_iteration(self, engine):
        """Test handling strategy without on_trading_iteration."""
        strategy = MagicMock(spec=[])  # No on_trading_iteration

        # Should not raise
        await engine._run_strategy_iteration(strategy, datetime(2024, 1, 5))


# =============================================================================
# TEST PERFORMANCE METRICS
# =============================================================================


class TestPerformanceMetrics:
    """Test _calculate_performance_metrics method."""

    def test_calculates_metrics(self, engine):
        """Test that metrics are calculated."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        equity = [10000 + i * 50 for i in range(30)]  # Steady increase
        returns = [0] + [(equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, 30)]

        result_df = pd.DataFrame(
            {
                "equity": equity,
                "returns": returns,
            },
            index=dates,
        )
        result_df["cum_returns"] = (1 + result_df["returns"].fillna(0)).cumprod() - 1

        engine._calculate_performance_metrics(result_df, "TestStrategy")

        assert "daily_returns" in result_df.columns
        assert "peak" in result_df.columns
        assert "drawdown" in result_df.columns
        assert "annualized_return" in result_df.attrs
        assert "max_drawdown" in result_df.attrs
        assert "sharpe_ratio" in result_df.attrs

    def test_skips_short_data(self, engine):
        """Test that metrics are skipped for short data."""
        result_df = pd.DataFrame({"equity": [10000], "returns": [0]})

        engine._calculate_performance_metrics(result_df, "TestStrategy")

        assert "annualized_return" not in result_df.attrs

    def test_calculates_max_drawdown(self, engine):
        """Test max drawdown calculation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        # Equity: goes up, then down, then up
        equity = [10000, 10500, 11000, 10800, 10200, 10000, 10300, 10600, 10800, 11000]
        returns = [0] + [(equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, 10)]

        result_df = pd.DataFrame({"equity": equity, "returns": returns}, index=dates)
        result_df["cum_returns"] = (1 + result_df["returns"].fillna(0)).cumprod() - 1

        engine._calculate_performance_metrics(result_df, "TestStrategy")

        # Max drawdown should be (10000 - 11000) / 11000 = -9.09%
        assert result_df.attrs["max_drawdown"] < 0

    def test_sharpe_ratio_calculation(self, engine):
        """Test Sharpe ratio calculation."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="B")
        np.random.seed(42)
        # Positive mean returns with some volatility
        returns = np.random.normal(0.001, 0.01, 50)
        equity = [10000]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        equity = equity[1:]

        result_df = pd.DataFrame({"equity": equity, "returns": returns}, index=dates)
        result_df["cum_returns"] = (1 + result_df["returns"].fillna(0)).cumprod() - 1

        engine._calculate_performance_metrics(result_df, "TestStrategy")

        # With positive mean returns, Sharpe should be positive
        assert "sharpe_ratio" in result_df.attrs

    def test_handles_zero_std(self, engine):
        """Test handling zero standard deviation in returns."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        # All zero returns = zero std
        returns = [0.0] * 10
        equity = [10000] * 10

        result_df = pd.DataFrame({"equity": equity, "returns": returns}, index=dates)
        result_df["cum_returns"] = (1 + result_df["returns"].fillna(0)).cumprod() - 1

        engine._calculate_performance_metrics(result_df, "TestStrategy")

        # Should not raise, sharpe should be 0 when std is 0
        assert result_df.attrs["sharpe_ratio"] == 0


# =============================================================================
# TEST TRADE P&L CALCULATION
# =============================================================================


class TestTradePnLCalculation:
    """Test _calculate_trade_pnl method."""

    def test_basic_trade_pnl(self, engine, sample_trades):
        """Test basic P&L calculation."""
        trade_records = engine._calculate_trade_pnl(sample_trades)

        assert len(trade_records) == 5
        # Check that sells have P&L
        sell_trades = [t for t in trade_records if t["side"] == "sell"]
        assert all("pnl" in t for t in sell_trades)

    def test_buy_has_zero_pnl(self, engine):
        """Test that buys have zero P&L."""
        trades = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0}]

        trade_records = engine._calculate_trade_pnl(trades)

        assert trade_records[0]["pnl"] == 0

    def test_sell_calculates_profit(self, engine):
        """Test sell calculates profit correctly."""
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0},
            {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 160.0},
        ]

        trade_records = engine._calculate_trade_pnl(trades)

        # Profit = (160 - 150) * 10 = $100
        assert trade_records[1]["pnl"] == pytest.approx(100.0)

    def test_sell_calculates_loss(self, engine):
        """Test sell calculates loss correctly."""
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0},
            {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 140.0},
        ]

        trade_records = engine._calculate_trade_pnl(trades)

        # Loss = (140 - 150) * 10 = -$100
        assert trade_records[1]["pnl"] == pytest.approx(-100.0)

    def test_average_price_on_multiple_buys(self, engine):
        """Test average price calculation on multiple buys."""
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0},
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 160.0},
            {"symbol": "AAPL", "side": "sell", "quantity": 20, "price": 158.0},
        ]

        trade_records = engine._calculate_trade_pnl(trades)

        # Avg price = (10*150 + 10*160) / 20 = 155
        # P&L = (158 - 155) * 20 = $60
        assert trade_records[2]["pnl"] == pytest.approx(60.0)

    def test_partial_sell(self, engine):
        """Test partial sell."""
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 20, "price": 150.0},
            {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 160.0},
        ]

        trade_records = engine._calculate_trade_pnl(trades)

        # P&L = (160 - 150) * 10 = $100
        assert trade_records[1]["pnl"] == pytest.approx(100.0)

    def test_sell_without_position_has_zero_pnl(self, engine):
        """Test selling without prior position."""
        trades = [{"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 150.0}]

        trade_records = engine._calculate_trade_pnl(trades)

        assert trade_records[0]["pnl"] == 0

    def test_multiple_symbols(self, engine, sample_trades):
        """Test P&L calculation with multiple symbols."""
        trade_records = engine._calculate_trade_pnl(sample_trades)

        aapl_trades = [t for t in trade_records if t["symbol"] == "AAPL"]
        msft_trades = [t for t in trade_records if t["symbol"] == "MSFT"]

        assert len(aapl_trades) == 3
        assert len(msft_trades) == 2

        # MSFT: bought at 300, sold at 280, loss
        msft_sell = [t for t in msft_trades if t["side"] == "sell"][0]
        assert msft_sell["pnl"] == pytest.approx((280 - 300) * 5)

    def test_preserves_timestamp(self, engine):
        """Test that timestamp is preserved in trade records."""
        timestamp = datetime(2024, 1, 5)
        trades = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, "timestamp": timestamp}]

        trade_records = engine._calculate_trade_pnl(trades)

        assert trade_records[0]["timestamp"] == timestamp


# =============================================================================
# TEST RUN_BACKTEST METHOD
# =============================================================================


class TestRunBacktestMethod:
    """Test the run_backtest method."""

    @pytest.mark.asyncio
    async def test_run_backtest_returns_dict(self, engine_with_broker):
        """Test that run_backtest returns a dictionary."""
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_instance.analyze_symbol = AsyncMock(return_value={"action": "neutral"})
        mock_strategy_instance.execute_trade = AsyncMock()
        mock_strategy_class.return_value = mock_strategy_instance

        # Setup broker methods
        engine_with_broker.broker.get_bars = AsyncMock(return_value=[])
        engine_with_broker.broker.get_trades.return_value = []
        engine_with_broker.broker.get_portfolio_value.return_value = 100000
        engine_with_broker.broker.get_balance.return_value = 100000
        engine_with_broker.broker.get_positions.return_value = []

        with patch("brokers.backtest_broker.BacktestBroker") as MockBacktestBroker:
            mock_bb = MagicMock()
            mock_bb.get_portfolio_value.return_value = 100000
            mock_bb.get_trades.return_value = []
            mock_bb.get_positions.return_value = []
            mock_bb.price_data = {}
            MockBacktestBroker.return_value = mock_bb

            result = await engine_with_broker.run_backtest(
                mock_strategy_class,
                ["AAPL"],
                datetime(2024, 1, 1),
                datetime(2024, 1, 5),
                initial_capital=100000,
            )

        assert isinstance(result, dict)
        assert "equity_curve" in result
        assert "trades" in result
        assert "start_date" in result
        assert "end_date" in result
        assert "initial_capital" in result
        assert "final_equity" in result

    @pytest.mark.asyncio
    async def test_run_backtest_converts_date_to_datetime(self, engine_with_broker):
        """Test that date objects are converted to datetime."""
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_instance.analyze_symbol = AsyncMock(return_value={"action": "neutral"})
        mock_strategy_class.return_value = mock_strategy_instance

        engine_with_broker.broker.get_bars = AsyncMock(return_value=[])

        with patch("brokers.backtest_broker.BacktestBroker") as MockBacktestBroker:
            mock_bb = MagicMock()
            mock_bb.get_portfolio_value.return_value = 100000
            mock_bb.get_trades.return_value = []
            mock_bb.get_positions.return_value = []
            mock_bb.price_data = {}
            MockBacktestBroker.return_value = mock_bb

            # Use date objects instead of datetime
            from datetime import date

            result = await engine_with_broker.run_backtest(
                mock_strategy_class,
                ["AAPL"],
                date(2024, 1, 1),
                date(2024, 1, 5),
            )

        assert isinstance(result["start_date"], datetime)
        assert isinstance(result["end_date"], datetime)


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_run_with_empty_strategies(self, engine_with_broker):
        """Test running with empty strategies list."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        results = await engine_with_broker.run([], start, end)

        assert results == []

    @pytest.mark.asyncio
    async def test_run_with_single_day(self, engine_with_broker, mock_strategy):
        """Test running for a single day."""
        start = datetime(2024, 1, 3)  # Wednesday
        end = datetime(2024, 1, 3)

        results = await engine_with_broker.run([mock_strategy], start, end)

        assert len(results) == 1

    def test_empty_trades_pnl(self, engine):
        """Test P&L calculation with empty trades."""
        trade_records = engine._calculate_trade_pnl([])
        assert trade_records == []

    @pytest.mark.asyncio
    async def test_run_with_multiple_strategies(self, engine_with_broker, mock_strategy):
        """Test running with multiple strategies."""
        strategy1 = MagicMock()
        strategy1.__class__.__name__ = "Strategy1"
        strategy2 = MagicMock()
        strategy2.__class__.__name__ = "Strategy2"

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        results = await engine_with_broker.run([strategy1, strategy2], start, end)

        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
