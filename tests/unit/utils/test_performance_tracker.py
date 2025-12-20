"""
Unit tests for PerformanceTracker.

Tests the real-time performance tracking system including:
- Trade logging and persistence
- Equity curve tracking
- Performance metrics calculation
- Drawdown analysis
- Performance report generation
"""

import os
import tempfile
from datetime import datetime, timedelta


class TestTradeDataclass:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a trade record."""
        from utils.performance_tracker import Trade

        trade = Trade(
            trade_id="trade_001",
            strategy="MomentumStrategy",
            symbol="AAPL",
            side="buy",
            entry_time=datetime(2024, 1, 1, 10, 0, 0),
            exit_time=datetime(2024, 1, 1, 14, 0, 0),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=0.0333,
            fees=10.0,
            holding_period_seconds=14400,
            is_winner=True,
            tags='{"reason": "trend_follow"}',
        )

        assert trade.trade_id == "trade_001"
        assert trade.strategy == "MomentumStrategy"
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.pnl == 500.0
        assert trade.is_winner is True


class TestPerformanceMetricsDataclass:
    """Test PerformanceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating performance metrics."""
        from utils.performance_tracker import PerformanceMetrics

        metrics = PerformanceMetrics(
            total_return=5000.0,
            total_return_pct=0.05,
            annualized_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=1000.0,
            max_drawdown_pct=0.01,
            current_drawdown_pct=0.005,
            recovery_factor=5.0,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win=150.0,
            avg_loss=-100.0,
            avg_win_pct=0.02,
            avg_loss_pct=-0.01,
            largest_win=500.0,
            largest_loss=-300.0,
            avg_holding_period_hours=4.5,
            total_fees=200.0,
            trading_days=30,
        )

        assert metrics.total_return == 5000.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.win_rate == 0.6


class TestPerformanceTrackerInit:
    """Test PerformanceTracker initialization."""

    def test_init_creates_directory(self):
        """Test initialization creates database directory."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            assert os.path.exists(os.path.dirname(db_path))
            assert tracker.db_path == db_path
            assert tracker.trades == []
            assert tracker.equity_curve == []

    def test_init_creates_database(self):
        """Test initialization creates SQLite database."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            PerformanceTracker(db_path=db_path)

            assert os.path.exists(db_path)


class TestInitDatabase:
    """Test database initialization."""

    def test_creates_tables(self):
        """Test that all required tables are created."""
        import sqlite3

        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            PerformanceTracker(db_path=db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check trades table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            assert cursor.fetchone() is not None

            # Check equity_curve table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='equity_curve'"
            )
            assert cursor.fetchone() is not None

            # Check performance_snapshots table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='performance_snapshots'"
            )
            assert cursor.fetchone() is not None

            conn.close()


class TestLogTrade:
    """Test trade logging functionality."""

    def test_log_trade_adds_to_memory(self):
        """Test that logged trades are added to memory."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="MomentumStrategy",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=150.0,
                exit_price=155.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.0333,
                fees=10.0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",
            )

            tracker.log_trade(trade)

            assert len(tracker.trades) == 1
            assert tracker.trades[0].trade_id == "trade_001"

    def test_log_trade_persists_to_database(self):
        """Test that logged trades are saved to database."""
        import sqlite3

        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="MomentumStrategy",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=150.0,
                exit_price=155.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.0333,
                fees=10.0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",
            )

            tracker.log_trade(trade)

            # Verify in database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", ("trade_001",))
            row = cursor.fetchone()
            conn.close()

            assert row is not None
            assert row[0] == "trade_001"

    def test_log_multiple_trades(self):
        """Test logging multiple trades."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            for i in range(5):
                trade = Trade(
                    trade_id=f"trade_{i:03d}",
                    strategy="MomentumStrategy",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 1, 14, 0),
                    entry_price=150.0,
                    exit_price=155.0 if i % 2 == 0 else 145.0,
                    quantity=100,
                    pnl=500.0 if i % 2 == 0 else -500.0,
                    pnl_pct=0.0333 if i % 2 == 0 else -0.0333,
                    fees=10.0,
                    holding_period_seconds=14400,
                    is_winner=i % 2 == 0,
                    tags="",
                )
                tracker.log_trade(trade)

            assert len(tracker.trades) == 5


class TestLoadTrades:
    """Test trade loading from database."""

    def test_load_trades_on_init(self):
        """Test that trades are loaded from database on initialization."""

        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")

            # Create tracker and add trade
            tracker1 = PerformanceTracker(db_path=db_path)
            trade = Trade(
                trade_id="trade_001",
                strategy="MomentumStrategy",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=150.0,
                exit_price=155.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.0333,
                fees=10.0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",
            )
            tracker1.log_trade(trade)

            # Create new tracker and verify trade is loaded
            tracker2 = PerformanceTracker(db_path=db_path)
            assert len(tracker2.trades) == 1
            assert tracker2.trades[0].trade_id == "trade_001"


class TestUpdateEquity:
    """Test equity curve updating."""

    def test_update_equity_adds_to_memory(self):
        """Test that equity updates are added to memory."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            now = datetime.now()
            tracker.update_equity(now, 100000)

            assert len(tracker.equity_curve) == 1
            assert tracker.equity_curve[0][1] == 100000

    def test_update_equity_calculates_daily_pnl(self):
        """Test that daily P/L is calculated correctly."""
        import sqlite3

        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            now = datetime.now()
            tracker.update_equity(now, 100000)
            tracker.update_equity(now + timedelta(days=1), 105000)

            # Verify in database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT daily_pnl, daily_pnl_pct FROM equity_curve ORDER BY timestamp")
            rows = cursor.fetchall()
            conn.close()

            # First entry has no daily P/L
            assert rows[0][0] is None
            # Second entry has calculated daily P/L
            assert rows[1][0] == 5000
            assert abs(rows[1][1] - 0.05) < 0.001

    def test_update_equity_handles_zero_prev_equity(self):
        """Test handling of zero previous equity."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            now = datetime.now()
            tracker.update_equity(now, 0)
            tracker.update_equity(now + timedelta(days=1), 100000)

            # Should not raise division by zero
            assert len(tracker.equity_curve) == 2


class TestCalculateMetrics:
    """Test metrics calculation."""

    def test_empty_trades(self):
        """Test metrics with no trades."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            metrics = tracker.calculate_metrics()

            assert metrics.total_return == 0
            assert metrics.total_trades == 0
            assert metrics.win_rate == 0
            assert metrics.sharpe_ratio == 0

    def test_single_winning_trade(self):
        """Test metrics with single winning trade."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="MomentumStrategy",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=150.0,
                exit_price=155.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.0333,
                fees=10.0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade)

            metrics = tracker.calculate_metrics(starting_equity=100000)

            assert metrics.total_return == 500.0
            assert metrics.total_trades == 1
            assert metrics.winning_trades == 1
            assert metrics.win_rate == 1.0
            assert metrics.total_fees == 10.0

    def test_mixed_trades(self):
        """Test metrics with winning and losing trades."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Add winning trade
            trade1 = Trade(
                trade_id="trade_001",
                strategy="MomentumStrategy",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 2, 10, 0),
                entry_price=150.0,
                exit_price=160.0,
                quantity=100,
                pnl=1000.0,
                pnl_pct=0.0667,
                fees=10.0,
                holding_period_seconds=86400,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade1)

            # Add losing trade
            trade2 = Trade(
                trade_id="trade_002",
                strategy="MomentumStrategy",
                symbol="MSFT",
                side="buy",
                entry_time=datetime(2024, 1, 3, 10, 0),
                exit_time=datetime(2024, 1, 4, 10, 0),
                entry_price=300.0,
                exit_price=290.0,
                quantity=50,
                pnl=-500.0,
                pnl_pct=-0.0333,
                fees=10.0,
                holding_period_seconds=86400,
                is_winner=False,
                tags="",
            )
            tracker.log_trade(trade2)

            metrics = tracker.calculate_metrics(starting_equity=100000)

            assert metrics.total_return == 500.0  # 1000 - 500
            assert metrics.total_trades == 2
            assert metrics.winning_trades == 1
            assert metrics.losing_trades == 1
            assert metrics.win_rate == 0.5
            assert metrics.avg_win == 1000.0
            assert metrics.avg_loss == -500.0
            assert metrics.largest_win == 1000.0
            assert metrics.largest_loss == -500.0

    def test_profit_factor(self):
        """Test profit factor calculation."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Add 2 winning trades: $500 each
            for i in range(2):
                trade = Trade(
                    trade_id=f"win_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 1, 14, 0),
                    entry_price=100.0,
                    exit_price=105.0,
                    quantity=100,
                    pnl=500.0,
                    pnl_pct=0.05,
                    fees=0,
                    holding_period_seconds=14400,
                    is_winner=True,
                    tags="",
                )
                tracker.log_trade(trade)

            # Add 1 losing trade: $500
            trade = Trade(
                trade_id="loss_0",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 3, 10, 0),
                exit_time=datetime(2024, 1, 3, 14, 0),
                entry_price=100.0,
                exit_price=95.0,
                quantity=100,
                pnl=-500.0,
                pnl_pct=-0.05,
                fees=0,
                holding_period_seconds=14400,
                is_winner=False,
                tags="",
            )
            tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            # Profit factor = total wins / total losses = 1000 / 500 = 2.0
            assert metrics.profit_factor == 2.0

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Add multiple trades with varying returns
            for i in range(10):
                pnl_pct = 0.02 if i % 2 == 0 else -0.01
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 1, 14, 0),
                    entry_price=100.0,
                    exit_price=100.0 * (1 + pnl_pct),
                    quantity=100,
                    pnl=pnl_pct * 10000,
                    pnl_pct=pnl_pct,
                    fees=0,
                    holding_period_seconds=14400,
                    is_winner=pnl_pct > 0,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            # Sharpe ratio should be calculated
            assert metrics.sharpe_ratio != 0

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Add trades with varied negative returns to avoid zero std
            pnl_pcts = [0.02, -0.01, 0.03, -0.015, 0.025, -0.02, 0.015, -0.005, 0.02, -0.01]
            for i, pnl_pct in enumerate(pnl_pcts):
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 1, 14, 0),
                    entry_price=100.0,
                    exit_price=100.0 * (1 + pnl_pct),
                    quantity=100,
                    pnl=pnl_pct * 10000,
                    pnl_pct=pnl_pct,
                    fees=0,
                    holding_period_seconds=14400,
                    is_winner=pnl_pct > 0,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            # Sortino ratio should be calculated (non-zero with varied downside returns)
            assert metrics.sortino_ratio != 0

    def test_all_winning_trades_sortino(self):
        """Test Sortino ratio when all trades are winners."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            for i in range(5):
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 1, 14, 0),
                    entry_price=100.0,
                    exit_price=105.0,
                    quantity=100,
                    pnl=500.0,
                    pnl_pct=0.05,
                    fees=0,
                    holding_period_seconds=14400,
                    is_winner=True,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            # When no negative returns, Sortino equals Sharpe
            assert metrics.sortino_ratio == metrics.sharpe_ratio


class TestCalculateEquityCurve:
    """Test equity curve calculation."""

    def test_equity_curve_from_trades(self):
        """Test building equity curve from trades."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Add trades
            trade1 = Trade(
                trade_id="trade_001",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=100.0,
                exit_price=105.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.05,
                fees=10.0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade1)

            trade2 = Trade(
                trade_id="trade_002",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 2, 10, 0),
                exit_time=datetime(2024, 1, 2, 14, 0),
                entry_price=100.0,
                exit_price=95.0,
                quantity=100,
                pnl=-500.0,
                pnl_pct=-0.05,
                fees=10.0,
                holding_period_seconds=14400,
                is_winner=False,
                tags="",
            )
            tracker.log_trade(trade2)

            equity_curve = tracker._calculate_equity_curve(100000)

            assert len(equity_curve) == 3
            assert equity_curve[0] == 100000
            # After first trade: 100000 + 500 - 10 = 100490
            assert equity_curve[1] == 100490
            # After second trade: 100490 - 500 - 10 = 99980
            assert equity_curve[2] == 99980


class TestCalculateDrawdown:
    """Test drawdown calculation."""

    def test_drawdown_empty_values(self):
        """Test drawdown with empty equity values."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            max_dd, max_dd_pct, current_dd_pct = tracker._calculate_drawdown([])

            assert max_dd == 0
            assert max_dd_pct == 0
            assert current_dd_pct == 0

    def test_drawdown_no_losses(self):
        """Test drawdown when equity only goes up."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            equity_values = [100000, 101000, 102000, 103000]
            max_dd, max_dd_pct, current_dd_pct = tracker._calculate_drawdown(equity_values)

            assert max_dd == 0
            assert max_dd_pct == 0
            assert current_dd_pct == 0

    def test_drawdown_with_losses(self):
        """Test drawdown with actual losses."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # 100000 -> 110000 -> 100000 -> 105000
            # Max drawdown: 10000 from peak of 110000
            equity_values = [100000, 110000, 100000, 105000]
            max_dd, max_dd_pct, current_dd_pct = tracker._calculate_drawdown(equity_values)

            assert max_dd == 10000
            assert abs(max_dd_pct - 0.0909) < 0.001  # 10000/110000

    def test_current_drawdown(self):
        """Test current drawdown calculation."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Currently at 105000 from peak of 110000 = 5000 current DD
            equity_values = [100000, 110000, 100000, 105000]
            max_dd, max_dd_pct, current_dd_pct = tracker._calculate_drawdown(equity_values)

            assert abs(current_dd_pct - 0.0455) < 0.001  # 5000/110000


class TestGetPerformanceReport:
    """Test performance report generation."""

    def test_report_with_no_trades(self):
        """Test report generation with no trades."""
        from utils.performance_tracker import PerformanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            report = tracker.get_performance_report()

            assert "PERFORMANCE REPORT" in report
            assert "Total Return" in report
            assert "$0.00" in report

    def test_report_with_trades(self):
        """Test report generation with trades."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="MomentumStrategy",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 10, 14, 0),
                entry_price=150.0,
                exit_price=160.0,
                quantity=100,
                pnl=1000.0,
                pnl_pct=0.0667,
                fees=20.0,
                holding_period_seconds=86400 * 9 + 14400,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade)

            report = tracker.get_performance_report()

            assert "PERFORMANCE REPORT" in report
            assert "RETURNS" in report
            assert "RISK-ADJUSTED RETURNS" in report
            assert "DRAWDOWN" in report
            assert "TRADE STATISTICS" in report
            assert "WIN/LOSS ANALYSIS" in report
            assert "Total Trades: 1" in report

    def test_report_format(self):
        """Test report formatting."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 2, 10, 0),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                pnl=1000.0,
                pnl_pct=0.10,
                fees=10.0,
                holding_period_seconds=86400,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade)

            report = tracker.get_performance_report()

            # Check separators
            assert "=" * 80 in report


class TestAnnualizedReturn:
    """Test annualized return calculation."""

    def test_annualized_return_with_short_period(self):
        """Test annualized return with less than a year of trading."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 2, 1, 10, 0),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                pnl=1000.0,
                pnl_pct=0.10,
                fees=0,
                holding_period_seconds=86400 * 31,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade)

            metrics = tracker.calculate_metrics(starting_equity=10000)

            # 10% in ~1 month should annualize to more than 100%
            assert metrics.annualized_return > 1.0

    def test_annualized_return_zero_days(self):
        """Test annualized return when trading days is zero."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),  # Same day
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                pnl=1000.0,
                pnl_pct=0.10,
                fees=0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade)

            metrics = tracker.calculate_metrics(starting_equity=10000)

            assert metrics.annualized_return == 0


class TestHoldingPeriod:
    """Test holding period calculations."""

    def test_average_holding_period(self):
        """Test average holding period calculation."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Add trades with different holding periods
            trade1 = Trade(
                trade_id="trade_001",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=100.0,
                exit_price=105.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.05,
                fees=0,
                holding_period_seconds=14400,  # 4 hours
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade1)

            trade2 = Trade(
                trade_id="trade_002",
                strategy="Test",
                symbol="MSFT",
                side="buy",
                entry_time=datetime(2024, 1, 2, 10, 0),
                exit_time=datetime(2024, 1, 2, 18, 0),
                entry_price=300.0,
                exit_price=310.0,
                quantity=50,
                pnl=500.0,
                pnl_pct=0.0333,
                fees=0,
                holding_period_seconds=28800,  # 8 hours
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade2)

            metrics = tracker.calculate_metrics()

            # Average: (4 + 8) / 2 = 6 hours
            assert metrics.avg_holding_period_hours == 6.0


class TestCalmarRatio:
    """Test Calmar ratio calculation."""

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Add trades that create drawdown
            trades_data = [
                (datetime(2024, 1, 1), datetime(2024, 1, 2), 1000),
                (datetime(2024, 1, 3), datetime(2024, 1, 4), -500),
                (datetime(2024, 1, 5), datetime(2024, 1, 6), 800),
            ]

            for i, (entry, exit, pnl) in enumerate(trades_data):
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=entry,
                    exit_time=exit,
                    entry_price=100.0,
                    exit_price=100.0 + pnl / 100,
                    quantity=100,
                    pnl=float(pnl),
                    pnl_pct=pnl / 10000,
                    fees=0,
                    holding_period_seconds=86400,
                    is_winner=pnl > 0,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            # Calmar ratio should be computed
            # It's annualized_return / max_drawdown_pct
            assert metrics.calmar_ratio >= 0

    def test_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio when max drawdown is zero."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            # Only winning trades, no drawdown
            for i in range(3):
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 2, 10, 0),
                    entry_price=100.0,
                    exit_price=105.0,
                    quantity=100,
                    pnl=500.0,
                    pnl_pct=0.05,
                    fees=0,
                    holding_period_seconds=86400,
                    is_winner=True,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            assert metrics.calmar_ratio == 0


class TestRecoveryFactor:
    """Test recovery factor calculation."""

    def test_recovery_factor(self):
        """Test recovery factor calculation."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trades_data = [
                (datetime(2024, 1, 1), datetime(2024, 1, 2), 2000),
                (datetime(2024, 1, 3), datetime(2024, 1, 4), -1000),  # Drawdown
                (datetime(2024, 1, 5), datetime(2024, 1, 6), 1500),
            ]

            for i, (entry, exit, pnl) in enumerate(trades_data):
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=entry,
                    exit_time=exit,
                    entry_price=100.0,
                    exit_price=100.0 + pnl / 100,
                    quantity=100,
                    pnl=float(pnl),
                    pnl_pct=pnl / 100000,
                    fees=0,
                    holding_period_seconds=86400,
                    is_winner=pnl > 0,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            # Recovery factor = total P/L / max drawdown
            assert metrics.recovery_factor > 0

    def test_recovery_factor_zero_drawdown(self):
        """Test recovery factor when max drawdown is zero."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            for i in range(3):
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 2, 10, 0),
                    entry_price=100.0,
                    exit_price=105.0,
                    quantity=100,
                    pnl=500.0,
                    pnl_pct=0.05,
                    fees=0,
                    holding_period_seconds=86400,
                    is_winner=True,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            assert metrics.recovery_factor == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_trade_sharpe(self):
        """Test Sharpe ratio with single trade."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=100.0,
                exit_price=105.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.05,
                fees=0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",
            )
            tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            # Sharpe is 0 with single trade (can't calculate std)
            assert metrics.sharpe_ratio == 0

    def test_all_losing_trades(self):
        """Test metrics with all losing trades."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            for i in range(5):
                trade = Trade(
                    trade_id=f"trade_{i}",
                    strategy="Test",
                    symbol="AAPL",
                    side="buy",
                    entry_time=datetime(2024, 1, i + 1, 10, 0),
                    exit_time=datetime(2024, 1, i + 1, 14, 0),
                    entry_price=100.0,
                    exit_price=95.0,
                    quantity=100,
                    pnl=-500.0,
                    pnl_pct=-0.05,
                    fees=10.0,
                    holding_period_seconds=14400,
                    is_winner=False,
                    tags="",
                )
                tracker.log_trade(trade)

            metrics = tracker.calculate_metrics()

            assert metrics.total_return == -2500.0
            assert metrics.win_rate == 0.0
            assert metrics.profit_factor == 0.0
            assert metrics.avg_win == 0

    def test_null_tags(self):
        """Test handling of null tags."""
        from utils.performance_tracker import PerformanceTracker, Trade

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "trading.db")
            tracker = PerformanceTracker(db_path=db_path)

            trade = Trade(
                trade_id="trade_001",
                strategy="Test",
                symbol="AAPL",
                side="buy",
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 14, 0),
                entry_price=100.0,
                exit_price=105.0,
                quantity=100,
                pnl=500.0,
                pnl_pct=0.05,
                fees=0,
                holding_period_seconds=14400,
                is_winner=True,
                tags="",  # Empty string instead of None
            )
            tracker.log_trade(trade)

            # Reload tracker
            tracker2 = PerformanceTracker(db_path=db_path)
            assert len(tracker2.trades) == 1
