"""
Async SQLite Database Manager for Trading Bot

Provides persistent storage for:
- Trade history with P&L tracking
- Daily performance metrics
- Position tracking (open/closed)
- Strategy and symbol performance analytics

Uses aiosqlite for non-blocking database operations compatible
with the bot's async architecture.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a trade execution."""

    id: Optional[int]
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    price: float
    timestamp: datetime
    strategy: str
    order_id: str
    status: str  # 'filled', 'partial', 'cancelled'
    pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "order_id": self.order_id,
            "status": self.status,
            "pnl": self.pnl,
        }


@dataclass
class DailyMetrics:
    """Daily performance metrics snapshot."""

    id: Optional[int]
    date: date
    starting_equity: float
    ending_equity: float
    pnl: float
    pnl_pct: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "trades_count": self.trades_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class Position:
    """Position tracking record."""

    id: Optional[int]
    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    strategy: str
    status: str  # 'open' or 'closed'
    pnl: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "strategy": self.strategy,
            "status": self.status,
            "pnl": self.pnl,
        }


class DatabaseError(Exception):
    """Exception raised for database operation failures."""

    pass


class TradingDatabase:
    """
    Async SQLite database manager for trading bot persistence.

    Provides async methods for:
    - Trade logging and retrieval
    - Daily metrics storage
    - Position tracking
    - Performance analytics by strategy/symbol

    Usage:
        db = TradingDatabase("data/trading.db")
        await db.initialize()

        # Log a trade
        trade = Trade(id=None, symbol="AAPL", side="buy", ...)
        trade_id = await db.insert_trade(trade)

        # Get trade history
        trades = await db.get_trades(symbol="AAPL", limit=50)

        # Clean up
        await db.close()
    """

    def __init__(self, db_path: str = "data/trading_bot.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Directory will be created if needed.
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Create database and tables if they don't exist.

        This method is idempotent and safe to call multiple times.
        """
        if self._connection is not None:
            return

        # Create directory if needed
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        try:
            conn = await aiosqlite.connect(self.db_path)
            self._connection = conn
            # Enable foreign keys and WAL mode for better concurrency
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")

            # Create tables
            await self._create_tables(conn)
            await conn.commit()

            self.logger.info(f"Database initialized: {self.db_path}")

        except Exception as e:
            if self._connection is not None:
                try:
                    await self._connection.close()
                except Exception:
                    pass
                finally:
                    self._connection = None
            self.logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}") from e

    async def _create_tables(self, conn: aiosqlite.Connection) -> None:
        """Create all required database tables."""

        # Trades table - records every trade execution
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                qty REAL NOT NULL CHECK (qty > 0),
                price REAL NOT NULL CHECK (price > 0),
                timestamp DATETIME NOT NULL,
                strategy TEXT,
                order_id TEXT UNIQUE,
                status TEXT NOT NULL CHECK (status IN ('filled', 'partial', 'cancelled')),
                pnl REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Daily metrics table - daily performance rollup
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                starting_equity REAL NOT NULL,
                ending_equity REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                trades_count INTEGER NOT NULL DEFAULT 0,
                winning_trades INTEGER NOT NULL DEFAULT 0,
                losing_trades INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0,
                max_drawdown REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Positions table - track open and closed positions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL CHECK (qty != 0),
                entry_price REAL NOT NULL CHECK (entry_price > 0),
                entry_time DATETIME NOT NULL,
                exit_price REAL,
                exit_time DATETIME,
                strategy TEXT,
                status TEXT DEFAULT 'open' CHECK (status IN ('open', 'closed')),
                pnl REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for common query patterns
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date)"
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy)"
        )

    async def close(self) -> None:
        """Close database connection."""
        conn = self._connection
        if conn is None:
            return
        if conn:
            try:
                await conn.close()
                self._connection = None
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database: {e}")

    def _ensure_connection(self) -> aiosqlite.Connection:
        """Return an active connection or raise if not initialized."""
        if self._connection is None:
            raise DatabaseError("Database not initialized. Call initialize() first.")
        return self._connection

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise DatabaseError(f"Invalid datetime value: {value!r}")

    @staticmethod
    def _parse_date(value: Any) -> date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        raise DatabaseError(f"Invalid date value: {value!r}")

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def insert_trade(self, trade: Trade) -> int:
        """
        Insert a new trade record.

        Args:
            trade: Trade object to insert

        Returns:
            The auto-generated trade ID
        """
        conn = self._ensure_connection()

        async with self._lock:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO trades (symbol, side, qty, price, timestamp, strategy, order_id, status, pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.symbol,
                        trade.side,
                        trade.qty,
                        trade.price,
                        trade.timestamp.isoformat(),
                        trade.strategy,
                        trade.order_id,
                        trade.status,
                        trade.pnl,
                    ),
                )
                await conn.commit()

                trade_id = cursor.lastrowid
                if trade_id is None:
                    raise DatabaseError("Insert trade returned no row id")
                self.logger.debug(
                    f"Inserted trade {trade_id}: {trade.symbol} {trade.side} {trade.qty}@{trade.price}"
                )
                return int(trade_id)

            except aiosqlite.IntegrityError as e:
                self.logger.warning(f"Trade with order_id {trade.order_id} already exists: {e}")
                raise DatabaseError(f"Duplicate order_id: {trade.order_id}") from e
            except Exception as e:
                self.logger.error(f"Failed to insert trade: {e}")
                raise DatabaseError(f"Insert trade failed: {e}") from e

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """
        Get trade history with optional filters.

        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy name
            start_date: Filter trades on or after this date
            end_date: Filter trades on or before this date
            status: Filter by status ('filled', 'partial', 'cancelled')
            limit: Maximum number of trades to return
            offset: Number of trades to skip (for pagination)

        Returns:
            List of Trade objects matching filters
        """
        conn = self._ensure_connection()

        query = "SELECT id, symbol, side, qty, price, timestamp, strategy, order_id, status, pnl FROM trades WHERE 1=1"
        params: List[Any] = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date.isoformat())

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            trades = []
            for row in rows:
                row_seq = cast(Sequence[Any], row)
                trades.append(
                    Trade(
                        id=cast(Optional[int], row_seq[0]),
                        symbol=cast(str, row_seq[1]),
                        side=cast(str, row_seq[2]),
                        qty=float(row_seq[3]),
                        price=float(row_seq[4]),
                        timestamp=self._parse_datetime(row_seq[5]),
                        strategy=cast(str, row_seq[6]),
                        order_id=cast(str, row_seq[7]),
                        status=cast(str, row_seq[8]),
                        pnl=cast(Optional[float], row_seq[9]),
                    )
                )

            return trades

        except Exception as e:
            self.logger.error(f"Failed to get trades: {e}")
            raise DatabaseError(f"Get trades failed: {e}") from e

    async def get_trade_by_order_id(self, order_id: str) -> Optional[Trade]:
        """
        Get trade by Alpaca order ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            Trade object if found, None otherwise
        """
        conn = self._ensure_connection()

        try:
            async with conn.execute(
                """
                SELECT id, symbol, side, qty, price, timestamp, strategy, order_id, status, pnl
                FROM trades WHERE order_id = ?
                """,
                (order_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                row_seq = cast(Sequence[Any], row)
                return Trade(
                    id=cast(Optional[int], row_seq[0]),
                    symbol=cast(str, row_seq[1]),
                    side=cast(str, row_seq[2]),
                    qty=float(row_seq[3]),
                    price=float(row_seq[4]),
                    timestamp=self._parse_datetime(row_seq[5]),
                    strategy=cast(str, row_seq[6]),
                    order_id=cast(str, row_seq[7]),
                    status=cast(str, row_seq[8]),
                    pnl=cast(Optional[float], row_seq[9]),
                )
            return None

        except Exception as e:
            self.logger.error(f"Failed to get trade by order_id {order_id}: {e}")
            raise DatabaseError(f"Get trade failed: {e}") from e

    async def update_trade_pnl(self, order_id: str, pnl: float) -> bool:
        """
        Update P&L for a completed trade.

        Args:
            order_id: Alpaca order ID
            pnl: Realized profit/loss

        Returns:
            True if updated, False if trade not found
        """
        conn = self._ensure_connection()

        async with self._lock:
            try:
                cursor = await conn.execute(
                    "UPDATE trades SET pnl = ? WHERE order_id = ?", (pnl, order_id)
                )
                await conn.commit()

                if cursor.rowcount > 0:
                    self.logger.debug(f"Updated P&L for order {order_id}: ${pnl:+,.2f}")
                    return True
                return False

            except Exception as e:
                self.logger.error(f"Failed to update trade P&L: {e}")
                raise DatabaseError(f"Update trade P&L failed: {e}") from e

    # =========================================================================
    # Daily Metrics Operations
    # =========================================================================

    async def insert_daily_metrics(self, metrics: DailyMetrics) -> int:
        """
        Insert or update daily metrics (upsert).

        If metrics for the date already exist, they will be updated.

        Args:
            metrics: DailyMetrics object to insert/update

        Returns:
            The metrics row ID
        """
        conn = self._ensure_connection()

        async with self._lock:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO daily_metrics (
                        date, starting_equity, ending_equity, pnl, pnl_pct,
                        trades_count, winning_trades, losing_trades, win_rate, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        starting_equity = excluded.starting_equity,
                        ending_equity = excluded.ending_equity,
                        pnl = excluded.pnl,
                        pnl_pct = excluded.pnl_pct,
                        trades_count = excluded.trades_count,
                        winning_trades = excluded.winning_trades,
                        losing_trades = excluded.losing_trades,
                        win_rate = excluded.win_rate,
                        max_drawdown = excluded.max_drawdown
                    """,
                    (
                        metrics.date.isoformat(),
                        metrics.starting_equity,
                        metrics.ending_equity,
                        metrics.pnl,
                        metrics.pnl_pct,
                        metrics.trades_count,
                        metrics.winning_trades,
                        metrics.losing_trades,
                        metrics.win_rate,
                        metrics.max_drawdown,
                    ),
                )
                await conn.commit()

                self.logger.debug(
                    f"Saved daily metrics for {metrics.date}: P&L ${metrics.pnl:+,.2f}"
                )
                row_id = cursor.lastrowid
                if row_id is None:
                    raise DatabaseError("Insert daily metrics returned no row id")
                return int(row_id)

            except Exception as e:
                self.logger.error(f"Failed to insert daily metrics: {e}")
                raise DatabaseError(f"Insert daily metrics failed: {e}") from e

    async def get_daily_metrics(
        self,
        start_date: date,
        end_date: date,
    ) -> List[DailyMetrics]:
        """
        Get daily metrics for date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of DailyMetrics objects
        """
        conn = self._ensure_connection()

        try:
            async with conn.execute(
                """
                SELECT id, date, starting_equity, ending_equity, pnl, pnl_pct,
                       trades_count, winning_trades, losing_trades, win_rate, max_drawdown
                FROM daily_metrics
                WHERE date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ) as cursor:
                rows = await cursor.fetchall()

            metrics_list = []
            for row in rows:
                row_seq = cast(Sequence[Any], row)
                metrics_list.append(
                    DailyMetrics(
                        id=cast(Optional[int], row_seq[0]),
                        date=self._parse_date(row_seq[1]),
                        starting_equity=float(row_seq[2]),
                        ending_equity=float(row_seq[3]),
                        pnl=float(row_seq[4]),
                        pnl_pct=float(row_seq[5]),
                        trades_count=int(row_seq[6]),
                        winning_trades=int(row_seq[7]),
                        losing_trades=int(row_seq[8]),
                        win_rate=float(row_seq[9]),
                        max_drawdown=cast(Optional[float], row_seq[10]),
                    )
                )

            return metrics_list

        except Exception as e:
            self.logger.error(f"Failed to get daily metrics: {e}")
            raise DatabaseError(f"Get daily metrics failed: {e}") from e

    async def get_latest_metrics(self) -> Optional[DailyMetrics]:
        """
        Get most recent daily metrics.

        Returns:
            DailyMetrics object or None if no metrics exist
        """
        conn = self._ensure_connection()

        try:
            async with conn.execute("""
                SELECT id, date, starting_equity, ending_equity, pnl, pnl_pct,
                       trades_count, winning_trades, losing_trades, win_rate, max_drawdown
                FROM daily_metrics
                ORDER BY date DESC
                LIMIT 1
                """) as cursor:
                row = await cursor.fetchone()

            if row:
                row_seq = cast(Sequence[Any], row)
                return DailyMetrics(
                    id=cast(Optional[int], row_seq[0]),
                    date=self._parse_date(row_seq[1]),
                    starting_equity=float(row_seq[2]),
                    ending_equity=float(row_seq[3]),
                    pnl=float(row_seq[4]),
                    pnl_pct=float(row_seq[5]),
                    trades_count=int(row_seq[6]),
                    winning_trades=int(row_seq[7]),
                    losing_trades=int(row_seq[8]),
                    win_rate=float(row_seq[9]),
                    max_drawdown=cast(Optional[float], row_seq[10]),
                )
            return None

        except Exception as e:
            self.logger.error(f"Failed to get latest metrics: {e}")
            raise DatabaseError(f"Get latest metrics failed: {e}") from e

    # =========================================================================
    # Position Operations
    # =========================================================================

    async def save_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        entry_time: datetime,
        strategy: str,
    ) -> int:
        """
        Save or update an open position.

        If an open position for this symbol/strategy exists, it will be updated.

        Args:
            symbol: Stock symbol
            qty: Position quantity (positive for long, negative for short)
            entry_price: Average entry price
            entry_time: Position entry timestamp
            strategy: Strategy name

        Returns:
            Position row ID
        """
        conn = self._ensure_connection()

        async with self._lock:
            try:
                # Check for existing open position
                async with conn.execute(
                    """
                    SELECT id FROM positions
                    WHERE symbol = ? AND strategy = ? AND status = 'open'
                    """,
                    (symbol, strategy),
                ) as cursor:
                    existing = await cursor.fetchone()

                if existing:
                    existing_seq = cast(Sequence[Any], existing)
                    existing_id = int(existing_seq[0])
                    # Update existing position
                    await conn.execute(
                        """
                        UPDATE positions
                        SET qty = ?, entry_price = ?, entry_time = ?
                        WHERE id = ?
                        """,
                        (qty, entry_price, entry_time.isoformat(), existing_id),
                    )
                    await conn.commit()
                    self.logger.debug(f"Updated position {symbol}: {qty}@{entry_price}")
                    return existing_id
                else:
                    # Insert new position
                    cursor = await conn.execute(
                        """
                        INSERT INTO positions (symbol, qty, entry_price, entry_time, strategy, status)
                        VALUES (?, ?, ?, ?, ?, 'open')
                        """,
                        (symbol, qty, entry_price, entry_time.isoformat(), strategy),
                    )
                    await conn.commit()
                    self.logger.debug(f"Opened position {symbol}: {qty}@{entry_price}")
                    row_id = cursor.lastrowid
                    if row_id is None:
                        raise DatabaseError("Insert position returned no row id")
                    return int(row_id)

            except Exception as e:
                self.logger.error(f"Failed to save position: {e}")
                raise DatabaseError(f"Save position failed: {e}") from e

    async def get_open_positions(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all open positions.

        Args:
            symbol: Optional filter by symbol
            strategy: Optional filter by strategy

        Returns:
            List of position dictionaries
        """
        conn = self._ensure_connection()

        query = """
            SELECT id, symbol, qty, entry_price, entry_time, strategy
            FROM positions WHERE status = 'open'
        """
        params: List[Any] = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY entry_time DESC"

        try:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            positions = []
            for row in rows:
                row_seq = cast(Sequence[Any], row)
                positions.append(
                    {
                        "id": row_seq[0],
                        "symbol": row_seq[1],
                        "qty": row_seq[2],
                        "entry_price": row_seq[3],
                        "entry_time": self._parse_datetime(row_seq[4]),
                        "strategy": row_seq[5],
                        "status": "open",
                    }
                )

            return positions

        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            raise DatabaseError(f"Get open positions failed: {e}") from e

    async def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        strategy: Optional[str] = None,
    ) -> Optional[float]:
        """
        Mark position as closed and calculate P&L.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            strategy: Optional strategy filter (closes matching position)

        Returns:
            Realized P&L, or None if no matching open position
        """
        conn = self._ensure_connection()

        async with self._lock:
            try:
                # Find open position
                query = """
                    SELECT id, qty, entry_price FROM positions
                    WHERE symbol = ? AND status = 'open'
                """
                params: List[Any] = [symbol]

                if strategy:
                    query += " AND strategy = ?"
                    params.append(strategy)

                query += " LIMIT 1"

                async with conn.execute(query, params) as cursor:
                    row = await cursor.fetchone()

                if not row:
                    self.logger.warning(f"No open position found for {symbol}")
                    return None

                row_seq = cast(Sequence[Any], row)
                position_id = int(row_seq[0])
                qty = float(row_seq[1])
                entry_price = float(row_seq[2])

                # Calculate P&L
                if qty > 0:  # Long position
                    pnl = (exit_price - entry_price) * qty
                else:  # Short position
                    pnl = (entry_price - exit_price) * abs(qty)

                # Update position
                await conn.execute(
                    """
                    UPDATE positions
                    SET exit_price = ?, exit_time = ?, status = 'closed', pnl = ?
                    WHERE id = ?
                    """,
                    (exit_price, exit_time.isoformat(), pnl, position_id),
                )
                await conn.commit()

                self.logger.info(f"Closed position {symbol}: P&L ${pnl:+,.2f}")
                return pnl

            except Exception as e:
                self.logger.error(f"Failed to close position: {e}")
                raise DatabaseError(f"Close position failed: {e}") from e

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_strategy_performance(self, strategy: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            Dictionary with performance metrics
        """
        conn = self._ensure_connection()

        try:
            # Get trade statistics
            async with conn.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN pnl = 0 THEN 1 ELSE 0 END) as breakeven_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss
                FROM trades
                WHERE strategy = ? AND pnl IS NOT NULL
                """,
                (strategy,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise DatabaseError("Strategy performance query returned no row")

            row_seq = cast(Sequence[Any], row)
            if int(row_seq[0] or 0) == 0:
                return {
                    "strategy": strategy,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "profit_factor": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                }

            total_trades = int(row_seq[0] or 0)
            winning_trades = int(row_seq[1] or 0)
            losing_trades = int(row_seq[2] or 0)
            total_pnl = float(row_seq[4] or 0.0)
            avg_win = float(row_seq[8] or 0.0)
            avg_loss = abs(float(row_seq[9] or 0.0))

            # Calculate derived metrics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            profit_factor = (
                (avg_win * winning_trades) / (avg_loss * losing_trades)
                if losing_trades > 0 and avg_loss > 0
                else 0.0
            )

            return {
                "strategy": strategy,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "breakeven_trades": int(row_seq[3] or 0),
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": float(row_seq[5] or 0.0),
                "profit_factor": profit_factor,
                "best_trade": float(row_seq[6] or 0.0),
                "worst_trade": float(row_seq[7] or 0.0),
                "avg_win": avg_win,
                "avg_loss": -avg_loss,  # Return as negative for clarity
            }

        except Exception as e:
            self.logger.error(f"Failed to get strategy performance: {e}")
            raise DatabaseError(f"Get strategy performance failed: {e}") from e

    async def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with performance metrics
        """
        conn = self._ensure_connection()

        try:
            async with conn.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade,
                    SUM(qty * price) as total_volume
                FROM trades
                WHERE symbol = ? AND pnl IS NOT NULL
                """,
                (symbol,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise DatabaseError("Symbol performance query returned no row")

            row_seq = cast(Sequence[Any], row)
            total_trades = int(row_seq[0] or 0)
            if total_trades == 0:
                return {
                    "symbol": symbol,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                    "total_volume": 0.0,
                }

            winning_trades = int(row_seq[1] or 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            return {
                "symbol": symbol,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": int(row_seq[2] or 0),
                "win_rate": win_rate,
                "total_pnl": float(row_seq[3] or 0.0),
                "avg_pnl": float(row_seq[4] or 0.0),
                "best_trade": float(row_seq[5] or 0.0),
                "worst_trade": float(row_seq[6] or 0.0),
                "total_volume": float(row_seq[7] or 0.0),
            }

        except Exception as e:
            self.logger.error(f"Failed to get symbol performance: {e}")
            raise DatabaseError(f"Get symbol performance failed: {e}") from e

    async def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get overall trading summary statistics.

        Returns:
            Dictionary with summary stats
        """
        conn = self._ensure_connection()

        try:
            # Trade summary
            async with conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT strategy) as unique_strategies,
                    SUM(pnl) as total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade
                FROM trades
                WHERE pnl IS NOT NULL
                """) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise DatabaseError("Summary stats query returned no row")
            row_seq = cast(Sequence[Any], row)

            # Open positions count
            async with conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status = 'open'"
            ) as cursor:
                open_row = await cursor.fetchone()

            if open_row is None:
                raise DatabaseError("Open positions query returned no row")
            open_positions = int(cast(Sequence[Any], open_row)[0] or 0)

            total_trades = int(row_seq[0] or 0)
            winning_trades = int(row_seq[4] or 0)

            return {
                "total_trades": total_trades,
                "unique_symbols": int(row_seq[1] or 0),
                "unique_strategies": int(row_seq[2] or 0),
                "total_pnl": float(row_seq[3] or 0.0),
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
                "open_positions": open_positions,
                "first_trade": row_seq[5],
                "last_trade": row_seq[6],
            }

        except Exception as e:
            self.logger.error(f"Failed to get summary stats: {e}")
            raise DatabaseError(f"Get summary stats failed: {e}") from e


# Convenience function for creating database instance
async def create_database(db_path: str = "data/trading_bot.db") -> TradingDatabase:
    """
    Create and initialize a TradingDatabase instance.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Initialized TradingDatabase instance
    """
    db = TradingDatabase(db_path)
    await db.initialize()
    return db
