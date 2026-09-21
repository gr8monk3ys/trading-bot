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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from engine.backtest_engine import BacktestEngine

# =============================================================================
# FIXTURES
# =============================================================================


def _daily_bars(n):
    """Minimal bar objects: the engine refuses to run on empty data (ADR 0008)."""
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            timestamp=datetime(2024, 1, 1 + i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1e6,
        )
        for i in range(n)
    ]


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
    broker.get_positions = AsyncMock(return_value=[])
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
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.0,
            "timestamp": datetime(2024, 1, 5),
        },
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 155.0,
            "timestamp": datetime(2024, 1, 10),
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 15,
            "price": 160.0,
            "timestamp": datetime(2024, 1, 15),
        },
        {
            "symbol": "MSFT",
            "side": "buy",
            "quantity": 5,
            "price": 300.0,
            "timestamp": datetime(2024, 1, 5),
        },
        {
            "symbol": "MSFT",
            "side": "sell",
            "quantity": 5,
            "price": 280.0,
            "timestamp": datetime(2024, 1, 20),
        },
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


# =============================================================================
# TEST STRATEGY ITERATION
# =============================================================================


# =============================================================================
# TEST PERFORMANCE METRICS
# =============================================================================


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
        trades = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "price": 150.0,
                "timestamp": timestamp,
            }
        ]

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
        engine_with_broker.broker.get_bars = AsyncMock(return_value=_daily_bars(5))
        engine_with_broker.broker.get_trades.return_value = []
        engine_with_broker.broker.get_portfolio_value.return_value = 100000
        engine_with_broker.broker.get_balance.return_value = 100000
        engine_with_broker.broker.get_positions = AsyncMock(return_value=[])

        with patch("brokers.backtest.BacktestBroker") as MockBacktestBroker:
            mock_bb = MagicMock()
            mock_bb.get_portfolio_value.return_value = 100000
            mock_bb.get_trades.return_value = []
            mock_bb.get_positions = AsyncMock(return_value=[])
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

        engine_with_broker.broker.get_bars = AsyncMock(return_value=_daily_bars(5))

        with patch("brokers.backtest.BacktestBroker") as MockBacktestBroker:
            mock_bb = MagicMock()
            mock_bb.get_portfolio_value.return_value = 100000
            mock_bb.get_trades.return_value = []
            mock_bb.get_positions = AsyncMock(return_value=[])
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

    def test_empty_trades_pnl(self, engine):
        """Test P&L calculation with empty trades."""
        trade_records = engine._calculate_trade_pnl([])
        assert trade_records == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
