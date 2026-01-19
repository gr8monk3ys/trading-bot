#!/usr/bin/env python3
"""
Real-Time Performance Tracker

Tracks all trades and calculates comprehensive performance metrics:
- Total return, daily/weekly/monthly returns
- Sharpe ratio, Sortino ratio
- Maximum drawdown, recovery time
- Win rate, profit factor
- Average win/loss
- Trade statistics

Stores everything in SQLite database for persistence.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a completed trade."""

    trade_id: str
    strategy: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    holding_period_seconds: int
    is_winner: bool
    tags: str  # JSON string of tags


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    recovery_factor: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # Win/Loss statistics
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win: float
    largest_loss: float

    # Other
    avg_holding_period_hours: float
    total_fees: float
    trading_days: int


class PerformanceTracker:
    """
    Real-time performance tracking with database persistence.

    Features:
    - Automatic trade logging to SQLite
    - Real-time metric calculation
    - Historical performance analysis
    - Export to CSV/JSON
    """

    def __init__(self, db_path: str = "data/trading_history.db"):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # In-memory cache
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        # Load existing trades
        self._load_trades()

        logger.info(f"Performance tracker initialized: {db_path}")
        logger.info(f"Loaded {len(self.trades)} historical trades")

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                fees REAL DEFAULT 0,
                holding_period_seconds INTEGER NOT NULL,
                is_winner BOOLEAN NOT NULL,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Equity curve table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp TIMESTAMP PRIMARY KEY,
                equity REAL NOT NULL,
                daily_pnl REAL,
                daily_pnl_pct REAL
            )
        """
        )

        # Performance snapshots (daily rollup)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                date DATE PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                equity REAL
            )
        """
        )

        conn.commit()
        conn.close()

    def _load_trades(self):
        """Load existing trades from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM trades
            ORDER BY exit_time
        """
        )

        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            trade = Trade(
                trade_id=row[0],
                strategy=row[1],
                symbol=row[2],
                side=row[3],
                entry_time=datetime.fromisoformat(row[4]),
                exit_time=datetime.fromisoformat(row[5]),
                entry_price=row[6],
                exit_price=row[7],
                quantity=row[8],
                pnl=row[9],
                pnl_pct=row[10],
                fees=row[11],
                holding_period_seconds=row[12],
                is_winner=bool(row[13]),
                tags=row[14] or "",
            )
            self.trades.append(trade)

    def log_trade(self, trade: Trade):
        """
        Log a completed trade.

        Args:
            trade: Trade object to log
        """
        # Add to memory
        self.trades.append(trade)

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trade.trade_id,
                trade.strategy,
                trade.symbol,
                trade.side,
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                trade.entry_price,
                trade.exit_price,
                trade.quantity,
                trade.pnl,
                trade.pnl_pct,
                trade.fees,
                trade.holding_period_seconds,
                trade.is_winner,
                trade.tags,
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Trade logged: {trade.symbol} ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2%})")

    def update_equity(self, timestamp: datetime, equity: float):
        """
        Update equity curve.

        Args:
            timestamp: Current timestamp
            equity: Current equity value
        """
        # Add to memory
        self.equity_curve.append((timestamp, equity))

        # Calculate daily P/L if we have previous data
        daily_pnl = None
        daily_pnl_pct = None

        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2][1]
            daily_pnl = equity - prev_equity
            daily_pnl_pct = (daily_pnl / prev_equity) if prev_equity > 0 else 0

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO equity_curve (timestamp, equity, daily_pnl, daily_pnl_pct)
            VALUES (?, ?, ?, ?)
        """,
            (timestamp.isoformat(), equity, daily_pnl, daily_pnl_pct),
        )

        conn.commit()
        conn.close()

    def calculate_metrics(self, starting_equity: float = 100000) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            starting_equity: Initial capital

        Returns:
            PerformanceMetrics object
        """
        if not self.trades:
            # Return zero metrics
            return PerformanceMetrics(
                total_return=0,
                total_return_pct=0,
                annualized_return=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                current_drawdown_pct=0,
                recovery_factor=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                largest_win=0,
                largest_loss=0,
                avg_holding_period_hours=0,
                total_fees=0,
                trading_days=0,
            )

        # Calculate returns
        total_pnl = sum(t.pnl for t in self.trades)
        total_return_pct = total_pnl / starting_equity

        # Calculate annualized return
        first_trade = min(t.entry_time for t in self.trades)
        last_trade = max(t.exit_time for t in self.trades)
        trading_days = (last_trade - first_trade).days

        if trading_days > 0:
            years = trading_days / 365.0
            annualized_return = ((1 + total_return_pct) ** (1 / years)) - 1 if years > 0 else 0
        else:
            annualized_return = 0

        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_std = np.std(downside_returns)
            sortino_ratio = (
                (np.mean(returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            )
        else:
            sortino_ratio = sharpe_ratio

        # Calculate drawdown
        equity_values = self._calculate_equity_curve(starting_equity)
        max_drawdown, max_drawdown_pct, current_drawdown_pct = self._calculate_drawdown(
            equity_values
        )

        # Calmar ratio
        calmar_ratio = abs(annualized_return / max_drawdown_pct) if max_drawdown_pct != 0 else 0

        # Recovery factor
        recovery_factor = abs(total_pnl / max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]

        win_rate = len(winners) / len(self.trades) if self.trades else 0

        # Profit factor
        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Average win/loss
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0
        avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0

        # Largest win/loss
        largest_win = max([t.pnl for t in winners]) if winners else 0
        largest_loss = min([t.pnl for t in losers]) if losers else 0

        # Holding period
        avg_holding_seconds = np.mean([t.holding_period_seconds for t in self.trades])
        avg_holding_hours = avg_holding_seconds / 3600

        # Total fees
        total_fees = sum(t.fees for t in self.trades)

        return PerformanceMetrics(
            total_return=total_pnl,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            current_drawdown_pct=current_drawdown_pct,
            recovery_factor=recovery_factor,
            total_trades=len(self.trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_period_hours=avg_holding_hours,
            total_fees=total_fees,
            trading_days=trading_days,
        )

    def _calculate_equity_curve(self, starting_equity: float) -> List[float]:
        """Calculate equity curve from trades."""
        equity = starting_equity
        equity_values = [equity]

        for trade in sorted(self.trades, key=lambda t: t.exit_time):
            equity += trade.pnl - trade.fees
            equity_values.append(equity)

        return equity_values

    def _calculate_drawdown(self, equity_values: List[float]) -> Tuple[float, float, float]:
        """Calculate maximum drawdown and current drawdown."""
        if not equity_values:
            return 0, 0, 0

        equity_array = np.array(equity_values)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = running_max - equity_array
        drawdown_pct = drawdown / running_max

        max_drawdown = np.max(drawdown)
        max_drawdown_pct = np.max(drawdown_pct)
        current_drawdown_pct = drawdown_pct[-1]

        return max_drawdown, max_drawdown_pct, current_drawdown_pct

    def get_performance_report(self, starting_equity: float = 100000) -> str:
        """
        Generate formatted performance report.

        Args:
            starting_equity: Initial capital

        Returns:
            Formatted string report
        """
        metrics = self.calculate_metrics(starting_equity)

        report = []
        report.append("\n" + "=" * 80)
        report.append("📊 PERFORMANCE REPORT")
        report.append("=" * 80)

        report.append("\n💰 RETURNS:")
        report.append(
            f"   Total Return: ${metrics.total_return:+,.2f} ({metrics.total_return_pct:+.2%})"
        )
        report.append(f"   Annualized Return: {metrics.annualized_return:+.2%}")
        report.append(f"   Trading Days: {metrics.trading_days}")

        report.append("\n📈 RISK-ADJUSTED RETURNS:")
        report.append(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
        report.append(f"   Calmar Ratio: {metrics.calmar_ratio:.2f}")

        report.append("\n📉 DRAWDOWN:")
        report.append(
            f"   Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2%})"
        )
        report.append(f"   Current Drawdown: {metrics.current_drawdown_pct:.2%}")
        report.append(f"   Recovery Factor: {metrics.recovery_factor:.2f}")

        report.append("\n🎯 TRADE STATISTICS:")
        report.append(f"   Total Trades: {metrics.total_trades}")
        report.append(f"   Winners: {metrics.winning_trades} ({metrics.win_rate:.1%})")
        report.append(f"   Losers: {metrics.losing_trades}")
        report.append(f"   Profit Factor: {metrics.profit_factor:.2f}")

        report.append("\n💵 WIN/LOSS ANALYSIS:")
        report.append(f"   Average Win: ${metrics.avg_win:,.2f} ({metrics.avg_win_pct:+.2%})")
        report.append(f"   Average Loss: ${metrics.avg_loss:,.2f} ({metrics.avg_loss_pct:+.2%})")
        report.append(f"   Largest Win: ${metrics.largest_win:,.2f}")
        report.append(f"   Largest Loss: ${metrics.largest_loss:,.2f}")

        report.append("\n⏱️  OTHER:")
        report.append(f"   Avg Holding Period: {metrics.avg_holding_period_hours:.1f} hours")
        report.append(f"   Total Fees: ${metrics.total_fees:,.2f}")

        report.append("=" * 80 + "\n")

        return "\n".join(report)
