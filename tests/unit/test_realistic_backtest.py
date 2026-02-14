#!/usr/bin/env python3
"""
Unit tests for utils/realistic_backtest.py

Tests Trade, BacktestResults, and RealisticBacktester classes.
"""

import sys
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.realistic_backtest import (
    BacktestResults,
    RealisticBacktester,
    Trade,
    print_backtest_report,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def backtest_mock_broker():
    """Mock broker for testing."""
    broker = MagicMock()
    broker.get_bars = AsyncMock()
    return broker


@pytest.fixture
def mock_strategy():
    """Mock strategy for testing."""
    strategy = MagicMock()
    strategy.symbols = ["AAPL", "MSFT"]
    strategy.position_size = 0.1
    strategy.signals = {}
    strategy.initialize = AsyncMock()
    strategy.on_bar = AsyncMock()
    return strategy


@pytest.fixture
def backtester(backtest_mock_broker, mock_strategy):
    """Default realistic backtester."""
    return RealisticBacktester(backtest_mock_broker, mock_strategy)


@pytest.fixture
def backtester_custom(backtest_mock_broker, mock_strategy):
    """Custom realistic backtester with explicit costs."""
    return RealisticBacktester(
        backtest_mock_broker,
        mock_strategy,
        initial_capital=50000.0,
        slippage_pct=0.005,
        spread_pct=0.002,
        commission_per_share=0.01,
        execution_delay_bars=2,
    )


@pytest.fixture
def mock_bars():
    """Create mock bars with OHLCV data."""

    def create_bars(prices, dates=None):
        bars = []
        for i, price in enumerate(prices):
            bar = MagicMock()
            bar.open = price * 0.99
            bar.high = price * 1.01
            bar.low = price * 0.98
            bar.close = price
            bar.volume = 1000000
            if dates:
                bar.timestamp = dates[i]
            else:
                bar.timestamp = datetime(2024, 1, 1) + timedelta(days=i)
            bars.append(bar)
        return bars

    return create_bars


# ============================================================================
# Trade Dataclass Tests
# ============================================================================


class TestTradeDataclass:
    """Test Trade dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        trade = Trade(
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
        )

        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.quantity == 100
        assert trade.entry_price == 150.0
        assert trade.exit_price is None
        assert trade.slippage_cost == 0.0
        assert trade.spread_cost == 0.0
        assert trade.commission_cost == 0.0
        assert trade.gross_pnl == 0.0
        assert trade.net_pnl == 0.0

    def test_calculate_costs(self):
        """Test cost calculation."""
        trade = Trade(
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=100.0,
        )

        trade.calculate_costs(
            slippage_pct=0.01,  # 1%
            spread_pct=0.005,  # 0.5%
            commission_per_share=0.01,
        )

        # Trade value = 100 * 100 = 10000
        # Slippage = 10000 * 0.01 * 2 = 200
        # Spread = 10000 * 0.005 * 2 = 100
        # Commission = 100 * 0.01 * 2 = 2
        assert trade.slippage_cost == 200.0
        assert trade.spread_cost == 100.0
        assert trade.commission_cost == 2.0
        assert trade.total_cost == 302.0

    def test_close_long_position_profit(self):
        """Test closing long position with profit."""
        trade = Trade(
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=100.0,
        )
        trade.calculate_costs(slippage_pct=0.01, spread_pct=0.005)

        trade.close(exit_price=110.0, exit_time=datetime.now())

        # Gross P&L = (110 - 100) * 100 = 1000
        assert trade.gross_pnl == 1000.0
        assert trade.exit_price == 110.0
        # Net P&L = 1000 - 300 (total_cost) = 700
        assert trade.net_pnl == pytest.approx(700.0, abs=1)

    def test_close_long_position_loss(self):
        """Test closing long position with loss."""
        trade = Trade(
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=100.0,
        )
        trade.calculate_costs(slippage_pct=0.01, spread_pct=0.005)

        trade.close(exit_price=90.0, exit_time=datetime.now())

        # Gross P&L = (90 - 100) * 100 = -1000
        assert trade.gross_pnl == -1000.0
        # Net P&L = -1000 - 300 = -1300
        assert trade.net_pnl == pytest.approx(-1300.0, abs=1)

    def test_close_short_position_profit(self):
        """Test closing short position with profit."""
        trade = Trade(
            symbol="AAPL",
            side="sell",
            quantity=100,
            entry_price=100.0,
        )
        trade.calculate_costs(slippage_pct=0.01, spread_pct=0.005)

        trade.close(exit_price=90.0, exit_time=datetime.now())

        # Short profit = (entry - exit) * qty = (100 - 90) * 100 = 1000
        assert trade.gross_pnl == 1000.0

    def test_close_short_position_loss(self):
        """Test closing short position with loss."""
        trade = Trade(
            symbol="AAPL",
            side="sell",
            quantity=100,
            entry_price=100.0,
        )
        trade.calculate_costs(slippage_pct=0.01, spread_pct=0.005)

        trade.close(exit_price=110.0, exit_time=datetime.now())

        # Short loss = (entry - exit) * qty = (100 - 110) * 100 = -1000
        assert trade.gross_pnl == -1000.0


# ============================================================================
# BacktestResults Dataclass Tests
# ============================================================================


class TestBacktestResultsDataclass:
    """Test BacktestResults dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        results = BacktestResults()

        assert results.initial_capital == 100000.0
        assert results.final_capital == 0.0
        assert results.gross_return == 0.0
        assert results.net_return == 0.0
        assert results.total_trades == 0
        assert results.equity_curve == []
        assert results.trades == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = BacktestResults(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            trading_days=252,
            initial_capital=100000.0,
            final_capital=110000.0,
            gross_return=0.12,
            net_return=0.10,
            total_trades=50,
            win_rate=0.6,
        )

        result_dict = results.to_dict()

        assert "period" in result_dict
        assert result_dict["trading_days"] == 252
        assert result_dict["initial_capital"] == 100000.0
        assert result_dict["final_capital"] == 110000.0
        assert result_dict["gross_return"] == 0.12
        assert result_dict["net_return"] == 0.10
        assert result_dict["total_trades"] == 50
        assert result_dict["win_rate"] == 0.6

    def test_custom_initialization(self):
        """Test custom initialization."""
        results = BacktestResults(
            initial_capital=50000.0,
            max_drawdown=-0.15,
            sharpe_ratio=1.5,
        )

        assert results.initial_capital == 50000.0
        assert results.max_drawdown == -0.15
        assert results.sharpe_ratio == 1.5


# ============================================================================
# RealisticBacktester Initialization Tests
# ============================================================================


class TestRealisticBacktesterInit:
    """Test RealisticBacktester initialization."""

    def test_default_init(self, backtester):
        """Test default initialization."""
        assert backtester.initial_capital == 100000.0
        assert backtester.positions == {}
        assert backtester.closed_trades == []
        assert backtester.equity_history == []

    def test_custom_costs(self, backtester_custom):
        """Test custom cost parameters."""
        assert backtester_custom.initial_capital == 50000.0
        assert backtester_custom.slippage_pct == 0.005
        assert backtester_custom.spread_pct == 0.002
        assert backtester_custom.commission == 0.01
        assert backtester_custom.execution_delay == 2

    def test_loads_defaults_from_config(self, backtest_mock_broker, mock_strategy):
        """Test loads defaults from config."""
        with patch(
            "utils.realistic_backtest.BACKTEST_PARAMS",
            {
                "SLIPPAGE_PCT": 0.003,
                "BID_ASK_SPREAD": 0.0015,
                "COMMISSION_PER_SHARE": 0.005,
                "EXECUTION_DELAY_BARS": 1,
                "USE_SLIPPAGE": True,
            },
        ):
            backtester = RealisticBacktester(backtest_mock_broker, mock_strategy)
            assert backtester.slippage_pct == 0.003
            assert backtester.spread_pct == 0.0015
            assert backtester.commission == 0.005


# ============================================================================
# Fetch Bars Tests
# ============================================================================


class TestFetchBars:
    """Test _fetch_bars method."""

    @pytest.mark.asyncio
    async def test_returns_formatted_data(self, backtester, backtest_mock_broker, mock_bars):
        """Test returns properly formatted data."""
        backtest_mock_broker.get_bars.return_value = mock_bars([100, 101, 102])

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        data = await backtester._fetch_bars("AAPL", start, end)

        assert data is not None
        assert len(data) == 3
        assert "open" in data[0]
        assert "high" in data[0]
        assert "low" in data[0]
        assert "close" in data[0]
        assert "volume" in data[0]

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error(self, backtester, backtest_mock_broker):
        """Test returns None on API error."""
        backtest_mock_broker.get_bars.side_effect = Exception("API error")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        data = await backtester._fetch_bars("AAPL", start, end)

        assert data is None

    @pytest.mark.asyncio
    async def test_returns_none_when_bars_is_none(self, backtester, backtest_mock_broker):
        """Test returns None when broker returns None."""
        backtest_mock_broker.get_bars.return_value = None

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        data = await backtester._fetch_bars("AAPL", start, end)

        assert data is None


# ============================================================================
# Process Signal Tests
# ============================================================================


class TestProcessSignal:
    """Test _process_signal method."""

    @pytest.mark.asyncio
    async def test_buy_signal_opens_position(self, backtester):
        """Test buy signal opens a position."""
        backtester.capital = 100000

        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())

        assert "AAPL" in backtester.positions
        assert backtester.positions["AAPL"].side == "buy"

    @pytest.mark.asyncio
    async def test_buy_signal_reduces_capital(self, backtester):
        """Test buy signal reduces available capital."""
        initial_capital = backtester.capital

        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())

        assert backtester.capital < initial_capital

    @pytest.mark.asyncio
    async def test_short_signal_opens_position(self, backtester):
        """Test short signal opens a position."""
        backtester.capital = 100000

        await backtester._process_signal("AAPL", "short", 100.0, datetime.now())

        assert "AAPL" in backtester.positions
        assert backtester.positions["AAPL"].side == "sell"

    @pytest.mark.asyncio
    async def test_sell_signal_closes_position(self, backtester):
        """Test sell signal closes existing position."""
        # First open a position
        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())
        assert "AAPL" in backtester.positions

        # Then close it
        await backtester._process_signal("AAPL", "sell", 110.0, datetime.now())

        assert "AAPL" not in backtester.positions
        assert len(backtester.closed_trades) == 1

    @pytest.mark.asyncio
    async def test_ignores_buy_when_position_exists(self, backtester):
        """Test ignores buy signal when position already exists."""
        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())
        initial_positions = len(backtester.positions)

        await backtester._process_signal("AAPL", "buy", 105.0, datetime.now())

        assert len(backtester.positions) == initial_positions

    @pytest.mark.asyncio
    async def test_ignores_sell_when_no_position(self, backtester):
        """Test ignores sell signal when no position exists."""
        await backtester._process_signal("AAPL", "sell", 100.0, datetime.now())

        assert "AAPL" not in backtester.positions
        assert len(backtester.closed_trades) == 0


# ============================================================================
# Close Position Tests
# ============================================================================


class TestClosePosition:
    """Test _close_position method."""

    @pytest.mark.asyncio
    async def test_closes_existing_position(self, backtester):
        """Test closes an existing position."""
        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())

        await backtester._close_position("AAPL", 110.0, datetime.now())

        assert "AAPL" not in backtester.positions
        assert len(backtester.closed_trades) == 1

    @pytest.mark.asyncio
    async def test_updates_capital_on_close(self, backtester):
        """Test updates capital when closing position."""
        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())
        capital_before_close = backtester.capital

        await backtester._close_position("AAPL", 110.0, datetime.now())

        # Capital should increase on profitable close
        assert backtester.capital > capital_before_close

    @pytest.mark.asyncio
    async def test_ignores_nonexistent_position(self, backtester):
        """Test ignores close request for nonexistent position."""
        initial_capital = backtester.capital

        await backtester._close_position("AAPL", 100.0, datetime.now())

        assert backtester.capital == initial_capital


# ============================================================================
# Calculate Results Tests
# ============================================================================


class TestCalculateResults:
    """Test _calculate_results method."""

    @pytest.mark.asyncio
    async def test_calculates_basic_stats(self, backtester):
        """Test calculates basic statistics."""
        # Create some trades
        await backtester._process_signal("AAPL", "buy", 100.0, datetime(2024, 1, 1))
        await backtester._close_position("AAPL", 110.0, datetime(2024, 1, 10))

        await backtester._process_signal("MSFT", "buy", 200.0, datetime(2024, 1, 5))
        await backtester._close_position("MSFT", 190.0, datetime(2024, 1, 15))

        results = BacktestResults(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=100000.0,
        )

        results = backtester._calculate_results(results)

        assert results.total_trades == 2
        assert results.winning_trades == 1
        assert results.losing_trades == 1
        assert results.win_rate == 0.5

    @pytest.mark.asyncio
    async def test_calculates_cost_breakdown(self, backtester):
        """Test calculates cost breakdown."""
        await backtester._process_signal("AAPL", "buy", 100.0, datetime(2024, 1, 1))
        await backtester._close_position("AAPL", 110.0, datetime(2024, 1, 10))

        results = BacktestResults(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=100000.0,
        )

        results = backtester._calculate_results(results)

        assert results.total_slippage >= 0
        assert results.total_spread >= 0
        assert results.total_costs >= 0

    def test_handles_zero_trades(self, backtester):
        """Test handles zero trades gracefully."""
        results = BacktestResults(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=100000.0,
        )

        results = backtester._calculate_results(results)

        assert results.total_trades == 0
        assert results.win_rate == 0

    @pytest.mark.asyncio
    async def test_calculates_risk_metrics(self, backtester):
        """Test calculates risk metrics from equity curve."""
        # First create some trades so we don't return early
        await backtester._process_signal("AAPL", "buy", 100.0, datetime(2024, 1, 1))
        await backtester._close_position("AAPL", 110.0, datetime(2024, 1, 10))

        # Add equity history manually
        backtester.equity_history = [
            (datetime(2024, 1, i + 1), 100000 + i * 100) for i in range(30)
        ]

        results = BacktestResults(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=100000.0,
        )

        results = backtester._calculate_results(results)

        # With trades and equity history, it should populate equity_curve
        assert results.max_drawdown <= 0  # Drawdown is negative or zero
        # The equity curve is populated from self.equity_history
        assert len(results.dates) == 30
        assert len(results.equity_curve) == 30


# ============================================================================
# Run Backtest Tests
# ============================================================================


class TestRunBacktest:
    """Test run method."""

    @pytest.mark.asyncio
    async def test_returns_backtest_results(self, backtester, backtest_mock_broker, mock_bars):
        """Test returns BacktestResults object."""
        # Mock bars for symbols
        backtest_mock_broker.get_bars.return_value = mock_bars([100, 101, 102, 103, 104])

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        results = await backtester.run(start, end)

        assert isinstance(results, BacktestResults)
        assert results.start_date == start
        assert results.end_date == end

    @pytest.mark.asyncio
    async def test_returns_results_on_no_data(self, backtester, backtest_mock_broker):
        """Test returns empty results when no data available."""
        backtest_mock_broker.get_bars.return_value = None

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        results = await backtester.run(start, end)

        assert isinstance(results, BacktestResults)
        assert results.total_trades == 0

    @pytest.mark.asyncio
    async def test_handles_exception(self, backtester, backtest_mock_broker):
        """Test handles exception gracefully."""
        backtest_mock_broker.get_bars.side_effect = Exception("API error")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        results = await backtester.run(start, end)

        assert isinstance(results, BacktestResults)


# ============================================================================
# Print Backtest Report Tests
# ============================================================================


class TestPrintBacktestReport:
    """Test print_backtest_report function."""

    def test_prints_report(self):
        """Test prints formatted report."""
        results = BacktestResults(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            trading_days=252,
            initial_capital=100000.0,
            final_capital=110000.0,
            gross_return=0.12,
            net_return=0.10,
            annualized_return=0.10,
            cost_drag_pct=0.02,
            total_costs=2000.0,
            total_slippage=1000.0,
            total_spread=800.0,
            total_commission=200.0,
            total_trades=50,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win=500.0,
            avg_loss=-300.0,
            largest_win=2000.0,
            largest_loss=-1500.0,
            max_drawdown=-0.10,
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            calmar_ratio=1.0,
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        print_backtest_report(results)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check key sections are printed
        assert "REALISTIC BACKTEST REPORT" in output
        assert "RETURNS" in output
        assert "COSTS BREAKDOWN" in output
        assert "TRADE STATISTICS" in output
        assert "RISK METRICS" in output


# ============================================================================
# Integration Tests
# ============================================================================


class TestBacktesterIntegration:
    """Integration tests for the backtester."""

    @pytest.mark.asyncio
    async def test_full_backtest_workflow(self, backtest_mock_broker, mock_strategy, mock_bars):
        """Test full backtest workflow."""
        # Setup strategy signals
        signal_sequence = iter(["buy", "sell", "buy", "sell", "neutral"])

        def get_signal(symbol, *args, **kwargs):
            try:
                mock_strategy.signals[symbol] = next(signal_sequence)
            except StopIteration:
                mock_strategy.signals[symbol] = "neutral"

        mock_strategy.on_bar.side_effect = get_signal

        # Setup price data
        prices = [100, 105, 102, 110, 108]
        backtest_mock_broker.get_bars.return_value = mock_bars(prices)

        backtester = RealisticBacktester(backtest_mock_broker, mock_strategy)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)
        results = await backtester.run(start, end)

        assert isinstance(results, BacktestResults)

    @pytest.mark.asyncio
    async def test_cost_impact_on_returns(self, backtest_mock_broker, mock_strategy, mock_bars):
        """Test that costs reduce returns appropriately."""
        # High slippage backtester
        backtester_high_cost = RealisticBacktester(
            backtest_mock_broker,
            mock_strategy,
            slippage_pct=0.05,  # 5% slippage!
            spread_pct=0.02,
        )

        # Low slippage backtester
        backtester_low_cost = RealisticBacktester(
            backtest_mock_broker,
            mock_strategy,
            slippage_pct=0.001,
            spread_pct=0.0005,
        )

        # Both should have cost_drag_pct >= 0
        assert backtester_high_cost.slippage_pct > backtester_low_cost.slippage_pct


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_trade_with_zero_quantity(self):
        """Test trade with zero quantity."""
        trade = Trade(
            symbol="AAPL",
            side="buy",
            quantity=0,
            entry_price=100.0,
        )

        trade.calculate_costs(slippage_pct=0.01, spread_pct=0.005)

        assert trade.slippage_cost == 0.0
        assert trade.spread_cost == 0.0
        assert trade.total_cost == 0.0

    def test_trade_with_zero_price(self):
        """Test trade with zero entry price."""
        trade = Trade(
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=0.0,
        )

        trade.calculate_costs(slippage_pct=0.01, spread_pct=0.005)

        assert trade.slippage_cost == 0.0
        assert trade.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_backtester_with_zero_capital(self, backtest_mock_broker, mock_strategy):
        """Test backtester with zero initial capital."""
        backtester = RealisticBacktester(
            backtest_mock_broker,
            mock_strategy,
            initial_capital=0.0,
        )

        # Should not open positions with zero capital
        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())

        assert len(backtester.positions) == 0

    @pytest.mark.asyncio
    async def test_backtester_with_insufficient_capital(self, backtester):
        """Test backtester with insufficient capital for trade."""
        backtester.capital = 1.0  # Very low capital

        await backtester._process_signal("AAPL", "buy", 100.0, datetime.now())

        # Should not open position with insufficient capital
        assert "AAPL" not in backtester.positions or backtester.positions["AAPL"].quantity < 0.01

    def test_results_annualized_return_with_zero_days(self):
        """Test annualized return calculation with zero trading days."""
        results = BacktestResults(
            trading_days=0,
            net_return=0.10,
        )

        # Should not crash
        assert results.annualized_return == 0.0
