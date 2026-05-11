"""
Async SQLite Database Manager for Trading Bot.

Provides persistent storage for:
- Trade history with P&L tracking
- Daily performance metrics
- Position tracking (open/closed)
- Strategy and symbol performance analytics

Uses aiosqlite for non-blocking database operations compatible
with the bot's async architecture.

Public API (re-exported here):
- :class:`TradingDatabase`
- :class:`Trade`, :class:`DailyMetrics`, :class:`Position`
- :class:`DatabaseError`
- :func:`create_database`

This module is a thin facade. Implementation is split across:
- :mod:`utils.database.core` — dataclasses, connection lifecycle, CRUD
- :mod:`utils.database.analytics` — aggregation/analytics queries
"""

from utils.database.analytics import TradingDatabaseAnalyticsMixin
from utils.database.core import (
    DailyMetrics,
    DatabaseError,
    Position,
    Trade,
    TradingDatabaseCore,
)


class TradingDatabase(TradingDatabaseCore, TradingDatabaseAnalyticsMixin):
    """
    Async SQLite database manager for trading bot persistence.

    Combines core CRUD operations from
    :class:`utils.database.core.TradingDatabaseCore` with aggregation
    analytics from
    :class:`utils.database.analytics.TradingDatabaseAnalyticsMixin`.
    """

    pass


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


__all__ = [
    "DailyMetrics",
    "DatabaseError",
    "Position",
    "Trade",
    "TradingDatabase",
    "create_database",
]
