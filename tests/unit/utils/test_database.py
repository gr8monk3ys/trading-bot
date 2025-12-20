"""
Unit tests for the async database module.

Tests cover:
- Database initialization and table creation
- Trade CRUD operations
- Daily metrics operations
- Position tracking
- Analytics queries
- Error handling
"""

import asyncio
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from utils.database import (
    DailyMetrics,
    DatabaseError,
    Trade,
    TradingDatabase,
    create_database,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_trading.db")


@pytest.fixture
async def db(temp_db_path):
    """Create and initialize a test database."""
    database = TradingDatabase(temp_db_path)
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
        id=None,
        symbol="AAPL",
        side="buy",
        qty=100.0,
        price=150.0,
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        strategy="MomentumStrategy",
        order_id="order_123",
        status="filled",
        pnl=None,
    )


@pytest.fixture
def sample_daily_metrics():
    """Create sample daily metrics for testing."""
    return DailyMetrics(
        id=None,
        date=date(2024, 1, 15),
        starting_equity=100000.0,
        ending_equity=101500.0,
        pnl=1500.0,
        pnl_pct=0.015,
        trades_count=5,
        winning_trades=3,
        losing_trades=2,
        win_rate=0.6,
        max_drawdown=0.005,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestDatabaseInitialization:
    """Tests for database initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_database_file(self, temp_db_path):
        """Test that initialize creates the database file."""
        db = TradingDatabase(temp_db_path)
        await db.initialize()

        assert Path(temp_db_path).exists()
        await db.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_parent_directories(self, temp_db_path):
        """Test that initialize creates parent directories if needed."""
        nested_path = os.path.join(temp_db_path, "nested", "dir", "test.db")
        db = TradingDatabase(nested_path)

        # Should not raise
        await db.initialize()
        assert Path(nested_path).exists()
        await db.close()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, temp_db_path):
        """Test that initialize can be called multiple times safely."""
        db = TradingDatabase(temp_db_path)
        await db.initialize()
        await db.initialize()  # Should not raise

        await db.close()

    @pytest.mark.asyncio
    async def test_create_database_helper(self, temp_db_path):
        """Test the create_database convenience function."""
        db = await create_database(temp_db_path)

        assert Path(temp_db_path).exists()
        await db.close()


# =============================================================================
# TRADE OPERATIONS TESTS
# =============================================================================


class TestTradeOperations:
    """Tests for trade CRUD operations."""

    @pytest.mark.asyncio
    async def test_insert_trade(self, db, sample_trade):
        """Test inserting a trade."""
        trade_id = await db.insert_trade(sample_trade)

        assert trade_id > 0

    @pytest.mark.asyncio
    async def test_insert_trade_returns_unique_ids(self, db, sample_trade):
        """Test that each insert returns a unique ID."""
        ids = []
        for i in range(3):
            trade = Trade(
                id=None,
                symbol="AAPL",
                side="buy",
                qty=100.0,
                price=150.0,
                timestamp=datetime.now(),
                strategy="Test",
                order_id=f"order_{i}",
                status="filled",
                pnl=None,
            )
            ids.append(await db.insert_trade(trade))

        assert len(set(ids)) == 3  # All IDs are unique

    @pytest.mark.asyncio
    async def test_insert_trade_duplicate_order_id_raises(self, db, sample_trade):
        """Test that inserting duplicate order_id raises error."""
        await db.insert_trade(sample_trade)

        with pytest.raises(DatabaseError) as exc_info:
            await db.insert_trade(sample_trade)

        assert "Duplicate" in str(exc_info.value) or "order_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_trade_by_order_id(self, db, sample_trade):
        """Test retrieving a trade by order ID."""
        await db.insert_trade(sample_trade)

        trade = await db.get_trade_by_order_id("order_123")

        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.qty == 100.0
        assert trade.order_id == "order_123"

    @pytest.mark.asyncio
    async def test_get_trade_by_order_id_not_found(self, db):
        """Test get_trade_by_order_id returns None for non-existent order."""
        trade = await db.get_trade_by_order_id("nonexistent")
        assert trade is None

    @pytest.mark.asyncio
    async def test_get_trades_no_filter(self, db):
        """Test getting all trades without filters."""
        # Insert several trades
        for i in range(5):
            trade = Trade(
                id=None,
                symbol=f"SYM{i}",
                side="buy",
                qty=100.0,
                price=100.0 + i,
                timestamp=datetime.now(),
                strategy="Test",
                order_id=f"order_{i}",
                status="filled",
                pnl=10.0 * i,
            )
            await db.insert_trade(trade)

        trades = await db.get_trades()

        assert len(trades) == 5

    @pytest.mark.asyncio
    async def test_get_trades_filter_by_symbol(self, db):
        """Test filtering trades by symbol."""
        symbols = ["AAPL", "AAPL", "MSFT", "GOOGL"]
        for i, symbol in enumerate(symbols):
            trade = Trade(
                id=None,
                symbol=symbol,
                side="buy",
                qty=100.0,
                price=150.0,
                timestamp=datetime.now(),
                strategy="Test",
                order_id=f"order_{symbol}_{i}",
                status="filled",
                pnl=None,
            )
            await db.insert_trade(trade)

        aapl_trades = await db.get_trades(symbol="AAPL")

        assert len(aapl_trades) == 2

    @pytest.mark.asyncio
    async def test_get_trades_filter_by_strategy(self, db):
        """Test filtering trades by strategy."""
        strategies = ["MomentumStrategy", "MomentumStrategy", "MeanReversion"]
        for i, strategy in enumerate(strategies):
            trade = Trade(
                id=None,
                symbol="AAPL",
                side="buy",
                qty=100.0,
                price=150.0,
                timestamp=datetime.now(),
                strategy=strategy,
                order_id=f"order_{i}",
                status="filled",
                pnl=None,
            )
            await db.insert_trade(trade)

        momentum_trades = await db.get_trades(strategy="MomentumStrategy")

        assert len(momentum_trades) == 2

    @pytest.mark.asyncio
    async def test_get_trades_filter_by_date_range(self, db):
        """Test filtering trades by date range."""
        dates = [
            datetime(2024, 1, 10),
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
            datetime(2024, 1, 25),
        ]
        for i, dt in enumerate(dates):
            trade = Trade(
                id=None,
                symbol="AAPL",
                side="buy",
                qty=100.0,
                price=150.0,
                timestamp=dt,
                strategy="Test",
                order_id=f"order_{i}",
                status="filled",
                pnl=None,
            )
            await db.insert_trade(trade)

        trades = await db.get_trades(
            start_date=date(2024, 1, 14),
            end_date=date(2024, 1, 21),
        )

        assert len(trades) == 2  # Jan 15 and Jan 20

    @pytest.mark.asyncio
    async def test_get_trades_with_limit(self, db):
        """Test limiting number of trades returned."""
        for i in range(10):
            trade = Trade(
                id=None,
                symbol="AAPL",
                side="buy",
                qty=100.0,
                price=150.0,
                timestamp=datetime.now(),
                strategy="Test",
                order_id=f"order_{i}",
                status="filled",
                pnl=None,
            )
            await db.insert_trade(trade)

        trades = await db.get_trades(limit=5)

        assert len(trades) == 5

    @pytest.mark.asyncio
    async def test_get_trades_with_offset(self, db):
        """Test pagination with offset."""
        for i in range(10):
            trade = Trade(
                id=None,
                symbol="AAPL",
                side="buy",
                qty=100.0,
                price=150.0 + i,  # Unique price for each
                timestamp=datetime.now() - timedelta(minutes=i),
                strategy="Test",
                order_id=f"order_{i}",
                status="filled",
                pnl=None,
            )
            await db.insert_trade(trade)

        page1 = await db.get_trades(limit=5, offset=0)
        page2 = await db.get_trades(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        # Ensure different trades (by checking no overlap in prices)
        page1_prices = {t.price for t in page1}
        page2_prices = {t.price for t in page2}
        assert page1_prices.isdisjoint(page2_prices)

    @pytest.mark.asyncio
    async def test_update_trade_pnl(self, db, sample_trade):
        """Test updating trade P&L."""
        await db.insert_trade(sample_trade)

        result = await db.update_trade_pnl("order_123", 250.0)

        assert result is True

        # Verify update
        trade = await db.get_trade_by_order_id("order_123")
        assert trade.pnl == 250.0

    @pytest.mark.asyncio
    async def test_update_trade_pnl_not_found(self, db):
        """Test updating P&L for non-existent trade."""
        result = await db.update_trade_pnl("nonexistent", 100.0)
        assert result is False


# =============================================================================
# DAILY METRICS TESTS
# =============================================================================


class TestDailyMetricsOperations:
    """Tests for daily metrics operations."""

    @pytest.mark.asyncio
    async def test_insert_daily_metrics(self, db, sample_daily_metrics):
        """Test inserting daily metrics."""
        metrics_id = await db.insert_daily_metrics(sample_daily_metrics)
        assert metrics_id > 0

    @pytest.mark.asyncio
    async def test_insert_daily_metrics_upsert(self, db, sample_daily_metrics):
        """Test that inserting metrics for same date updates existing."""
        await db.insert_daily_metrics(sample_daily_metrics)

        # Insert updated metrics for same date
        updated = DailyMetrics(
            id=None,
            date=sample_daily_metrics.date,
            starting_equity=100000.0,
            ending_equity=102000.0,  # Different value
            pnl=2000.0,
            pnl_pct=0.02,
            trades_count=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            max_drawdown=0.01,
        )
        await db.insert_daily_metrics(updated)

        # Should have only one record for that date
        metrics = await db.get_daily_metrics(
            sample_daily_metrics.date,
            sample_daily_metrics.date,
        )

        assert len(metrics) == 1
        assert metrics[0].ending_equity == 102000.0

    @pytest.mark.asyncio
    async def test_get_daily_metrics_range(self, db):
        """Test getting daily metrics for a date range."""
        for i in range(5):
            metrics = DailyMetrics(
                id=None,
                date=date(2024, 1, 10 + i),
                starting_equity=100000.0,
                ending_equity=100000.0 + i * 100,
                pnl=i * 100,
                pnl_pct=0.001 * i,
                trades_count=i,
                winning_trades=i,
                losing_trades=0,
                win_rate=1.0 if i > 0 else 0.0,
                max_drawdown=0.0,
            )
            await db.insert_daily_metrics(metrics)

        metrics_list = await db.get_daily_metrics(
            date(2024, 1, 11),
            date(2024, 1, 13),
        )

        assert len(metrics_list) == 3

    @pytest.mark.asyncio
    async def test_get_latest_metrics(self, db):
        """Test getting the most recent daily metrics."""
        for i in range(3):
            metrics = DailyMetrics(
                id=None,
                date=date(2024, 1, 10 + i),
                starting_equity=100000.0,
                ending_equity=100000.0 + i * 100,
                pnl=i * 100,
                pnl_pct=0.001 * i,
                trades_count=i,
                winning_trades=i,
                losing_trades=0,
                win_rate=1.0 if i > 0 else 0.0,
                max_drawdown=0.0,
            )
            await db.insert_daily_metrics(metrics)

        latest = await db.get_latest_metrics()

        assert latest is not None
        assert latest.date == date(2024, 1, 12)  # Most recent

    @pytest.mark.asyncio
    async def test_get_latest_metrics_empty(self, db):
        """Test get_latest_metrics returns None when no metrics exist."""
        latest = await db.get_latest_metrics()
        assert latest is None


# =============================================================================
# POSITION OPERATIONS TESTS
# =============================================================================


class TestPositionOperations:
    """Tests for position tracking operations."""

    @pytest.mark.asyncio
    async def test_save_position_new(self, db):
        """Test saving a new position."""
        position_id = await db.save_position(
            symbol="AAPL",
            qty=100.0,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 15, 10, 30, 0),
            strategy="MomentumStrategy",
        )

        assert position_id > 0

    @pytest.mark.asyncio
    async def test_save_position_update_existing(self, db):
        """Test that saving position with same symbol/strategy updates existing."""
        await db.save_position(
            symbol="AAPL",
            qty=100.0,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 15, 10, 30, 0),
            strategy="MomentumStrategy",
        )

        # Update position
        await db.save_position(
            symbol="AAPL",
            qty=200.0,  # Increased
            entry_price=145.0,  # Average down
            entry_time=datetime(2024, 1, 15, 14, 30, 0),
            strategy="MomentumStrategy",
        )

        positions = await db.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["qty"] == 200.0

    @pytest.mark.asyncio
    async def test_get_open_positions(self, db):
        """Test getting all open positions."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            await db.save_position(
                symbol=symbol,
                qty=100.0,
                entry_price=150.0,
                entry_time=datetime.now(),
                strategy="Test",
            )

        positions = await db.get_open_positions()

        assert len(positions) == 3

    @pytest.mark.asyncio
    async def test_get_open_positions_filter_by_symbol(self, db):
        """Test filtering open positions by symbol."""
        for symbol in ["AAPL", "MSFT"]:
            await db.save_position(
                symbol=symbol,
                qty=100.0,
                entry_price=150.0,
                entry_time=datetime.now(),
                strategy="Test",
            )

        positions = await db.get_open_positions(symbol="AAPL")

        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_open_positions_filter_by_strategy(self, db):
        """Test filtering open positions by strategy."""
        await db.save_position(
            symbol="AAPL",
            qty=100.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            strategy="Momentum",
        )
        await db.save_position(
            symbol="MSFT",
            qty=100.0,
            entry_price=350.0,
            entry_time=datetime.now(),
            strategy="MeanReversion",
        )

        positions = await db.get_open_positions(strategy="Momentum")

        assert len(positions) == 1
        assert positions[0]["strategy"] == "Momentum"

    @pytest.mark.asyncio
    async def test_close_position(self, db):
        """Test closing a position."""
        await db.save_position(
            symbol="AAPL",
            qty=100.0,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 15, 10, 30, 0),
            strategy="MomentumStrategy",
        )

        pnl = await db.close_position(
            symbol="AAPL",
            exit_price=160.0,  # +10 per share
            exit_time=datetime(2024, 1, 16, 10, 30, 0),
        )

        assert pnl == 1000.0  # 100 shares * $10 profit

        # Verify position is closed
        open_positions = await db.get_open_positions()
        assert len(open_positions) == 0

    @pytest.mark.asyncio
    async def test_close_position_short(self, db):
        """Test closing a short position."""
        await db.save_position(
            symbol="AAPL",
            qty=-100.0,  # Short position
            entry_price=160.0,
            entry_time=datetime.now(),
            strategy="Test",
        )

        pnl = await db.close_position(
            symbol="AAPL",
            exit_price=150.0,  # Price dropped (profit for short)
            exit_time=datetime.now(),
        )

        assert pnl == 1000.0  # 100 shares * $10 profit (short)

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, db):
        """Test closing non-existent position returns None."""
        pnl = await db.close_position(
            symbol="AAPL",
            exit_price=160.0,
            exit_time=datetime.now(),
        )

        assert pnl is None

    @pytest.mark.asyncio
    async def test_close_position_with_strategy_filter(self, db):
        """Test closing position with strategy filter."""
        # Open positions with different strategies
        await db.save_position(
            symbol="AAPL",
            qty=100.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            strategy="Momentum",
        )
        await db.save_position(
            symbol="AAPL",
            qty=50.0,
            entry_price=155.0,
            entry_time=datetime.now(),
            strategy="MeanReversion",
        )

        # Close only the Momentum position
        pnl = await db.close_position(
            symbol="AAPL",
            exit_price=160.0,
            exit_time=datetime.now(),
            strategy="Momentum",
        )

        assert pnl == 1000.0

        # MeanReversion position should still be open
        positions = await db.get_open_positions(strategy="MeanReversion")
        assert len(positions) == 1


# =============================================================================
# ANALYTICS TESTS
# =============================================================================


class TestAnalytics:
    """Tests for analytics and reporting functions."""

    @pytest.mark.asyncio
    async def test_get_strategy_performance(self, db):
        """Test getting performance metrics for a strategy."""
        # Insert trades with P&L
        trades_data = [
            ("AAPL", "buy", 100.0, 150.0, 200.0),  # Win
            ("MSFT", "buy", 50.0, 300.0, -100.0),  # Loss
            ("GOOGL", "buy", 20.0, 2800.0, 500.0),  # Win
        ]
        for i, (symbol, side, qty, price, pnl) in enumerate(trades_data):
            trade = Trade(
                id=None,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                timestamp=datetime.now(),
                strategy="MomentumStrategy",
                order_id=f"order_{i}",
                status="filled",
                pnl=pnl,
            )
            await db.insert_trade(trade)

        perf = await db.get_strategy_performance("MomentumStrategy")

        assert perf["strategy"] == "MomentumStrategy"
        assert perf["total_trades"] == 3
        assert perf["winning_trades"] == 2
        assert perf["losing_trades"] == 1
        assert perf["total_pnl"] == 600.0  # 200 - 100 + 500
        assert perf["win_rate"] == pytest.approx(2 / 3, rel=0.01)

    @pytest.mark.asyncio
    async def test_get_strategy_performance_empty(self, db):
        """Test strategy performance with no trades."""
        perf = await db.get_strategy_performance("NonexistentStrategy")

        assert perf["total_trades"] == 0
        assert perf["total_pnl"] == 0.0
        assert perf["win_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_symbol_performance(self, db):
        """Test getting performance metrics for a symbol."""
        # Insert multiple trades for AAPL
        for i, pnl in enumerate([100.0, -50.0, 200.0, -25.0]):
            trade = Trade(
                id=None,
                symbol="AAPL",
                side="buy",
                qty=10.0,
                price=150.0,
                timestamp=datetime.now(),
                strategy="Test",
                order_id=f"aapl_order_{i}",
                status="filled",
                pnl=pnl,
            )
            await db.insert_trade(trade)

        perf = await db.get_symbol_performance("AAPL")

        assert perf["symbol"] == "AAPL"
        assert perf["total_trades"] == 4
        assert perf["winning_trades"] == 2
        assert perf["losing_trades"] == 2
        assert perf["total_pnl"] == 225.0  # 100 - 50 + 200 - 25
        assert perf["win_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_get_summary_stats(self, db):
        """Test getting overall summary statistics."""
        # Insert trades for multiple symbols and strategies
        for i in range(5):
            trade = Trade(
                id=None,
                symbol=["AAPL", "MSFT", "GOOGL"][i % 3],
                side="buy",
                qty=100.0,
                price=150.0,
                timestamp=datetime.now() - timedelta(days=i),
                strategy=["Momentum", "MeanReversion"][i % 2],
                order_id=f"order_{i}",
                status="filled",
                pnl=100.0 if i % 2 == 0 else -50.0,
            )
            await db.insert_trade(trade)

        # Open some positions
        await db.save_position("TSLA", 50.0, 200.0, datetime.now(), "Test")

        stats = await db.get_summary_stats()

        assert stats["total_trades"] == 5
        assert stats["unique_symbols"] == 3
        assert stats["unique_strategies"] == 2
        assert stats["total_pnl"] == 200.0  # 100 - 50 + 100 - 50 + 100
        assert stats["open_positions"] == 1


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_operations_without_initialize_raises(self, temp_db_path):
        """Test that operations without initialize raise DatabaseError."""
        db = TradingDatabase(temp_db_path)

        with pytest.raises(DatabaseError) as exc_info:
            await db.get_trades()

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_trade_dataclass_to_dict(self, sample_trade):
        """Test Trade dataclass to_dict method."""
        d = sample_trade.to_dict()

        assert d["symbol"] == "AAPL"
        assert d["side"] == "buy"
        assert d["qty"] == 100.0
        assert "timestamp" in d

    @pytest.mark.asyncio
    async def test_daily_metrics_dataclass_to_dict(self, sample_daily_metrics):
        """Test DailyMetrics dataclass to_dict method."""
        d = sample_daily_metrics.to_dict()

        assert d["date"] == "2024-01-15"
        assert d["pnl"] == 1500.0
        assert d["win_rate"] == 0.6


# =============================================================================
# CONCURRENCY TESTS
# =============================================================================


class TestConcurrency:
    """Tests for concurrent database access."""

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, db):
        """Test that concurrent inserts are handled correctly."""

        async def insert_trade(i):
            trade = Trade(
                id=None,
                symbol="AAPL",
                side="buy",
                qty=100.0,
                price=150.0,
                timestamp=datetime.now(),
                strategy="Test",
                order_id=f"concurrent_order_{i}",
                status="filled",
                pnl=None,
            )
            return await db.insert_trade(trade)

        # Insert 10 trades concurrently
        tasks = [insert_trade(i) for i in range(10)]
        ids = await asyncio.gather(*tasks)

        assert len(ids) == 10
        assert len(set(ids)) == 10  # All unique

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, db, sample_trade):
        """Test that concurrent reads work correctly."""
        await db.insert_trade(sample_trade)

        async def read_trade():
            return await db.get_trade_by_order_id("order_123")

        # Read concurrently
        tasks = [read_trade() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(r is not None for r in results)
        assert all(r.symbol == "AAPL" for r in results)
