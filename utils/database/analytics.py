"""
Async SQLite Database Manager — Analytics & aggregation queries.

Provides :class:`TradingDatabaseAnalyticsMixin` containing multi-table
aggregation queries over the trades/positions/daily_metrics tables:

- :meth:`get_strategy_performance` — per-strategy P&L statistics
- :meth:`get_symbol_performance` — per-symbol P&L statistics
- :meth:`get_summary_stats` — bot-wide summary

Connection state and CRUD live in :mod:`utils.database.core`.
"""

import logging
from typing import Any, Dict, Sequence, cast

import aiosqlite

from utils.database.core import DatabaseError

logger = logging.getLogger(__name__)


class TradingDatabaseAnalyticsMixin:
    """
    Analytics queries for the trading database.

    Mixed into :class:`utils.database.TradingDatabase` alongside
    :class:`utils.database.core.TradingDatabaseCore`. Relies on the
    core mixin to provide ``self._ensure_connection()`` and
    ``self.logger``.
    """

    # Attribute / method stubs supplied by TradingDatabaseCore at runtime.
    logger: logging.Logger

    def _ensure_connection(self) -> aiosqlite.Connection:  # pragma: no cover - provided by core
        raise NotImplementedError

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
