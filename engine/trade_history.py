"""Trade history: the one store of completed trades.

Written only by the TradeRecorder (through ``record_trade``), read by the web
dashboard, the CLI dashboard and anything else that wants performance. The
audit trail is a different thing: a tamper-evidence event chain that is never
queried for performance (ADR 0007). Two adapters: SQLite for a running bot,
memory for tests and backtests.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    strategy: str
    side: str  # "long" | "short"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        d["is_winner"] = self.is_winner
        return d


class MemoryStore:
    def __init__(self) -> None:
        self._rows: List[TradeRecord] = []

    def append(self, rec: TradeRecord) -> None:
        self._rows.append(rec)

    def all(self) -> List[TradeRecord]:
        return list(self._rows)

    def close(self) -> None:
        pass


class SqliteStore:
    _DDL = """CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, strategy TEXT NOT NULL, side TEXT NOT NULL,
        entry_time TEXT NOT NULL, exit_time TEXT NOT NULL,
        entry_price REAL NOT NULL, exit_price REAL NOT NULL, quantity REAL NOT NULL,
        pnl REAL NOT NULL, pnl_pct REAL NOT NULL)"""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(self._DDL)
        self._conn.commit()

    def append(self, rec: TradeRecord) -> None:
        self._conn.execute(
            "INSERT INTO trades (symbol, strategy, side, entry_time, exit_time, entry_price, "
            "exit_price, quantity, pnl, pnl_pct) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                rec.symbol,
                rec.strategy,
                rec.side,
                rec.entry_time.isoformat(),
                rec.exit_time.isoformat(),
                rec.entry_price,
                rec.exit_price,
                rec.quantity,
                rec.pnl,
                rec.pnl_pct,
            ),
        )
        self._conn.commit()

    def all(self) -> List[TradeRecord]:
        rows = self._conn.execute(
            "SELECT symbol, strategy, side, entry_time, exit_time, entry_price, exit_price, "
            "quantity, pnl, pnl_pct FROM trades ORDER BY exit_time, id"
        ).fetchall()
        return [
            TradeRecord(
                r[0],
                r[1],
                r[2],
                datetime.fromisoformat(r[3]),
                datetime.fromisoformat(r[4]),
                r[5],
                r[6],
                r[7],
                r[8],
                r[9],
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()


class TradeHistory:
    def __init__(self, store: Any = None) -> None:
        self.store = store or MemoryStore()

    # -- the writer ----------------------------------------------------------

    def record(self, rec: TradeRecord) -> None:
        self.store.append(rec)

    def record_trade(self, trade: Any, strategy: str = "") -> TradeRecord:
        """Adapter for the recorder's utils.kelly Trade."""
        rec = TradeRecord(
            symbol=trade.symbol,
            strategy=strategy,
            side="short" if getattr(trade, "side", "long") == "short" else "long",
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            entry_price=float(trade.entry_price),
            exit_price=float(trade.exit_price),
            quantity=float(trade.quantity),
            pnl=float(trade.pnl),
            pnl_pct=float(trade.pnl_pct),
        )
        self.record(rec)
        return rec

    # -- the readers ---------------------------------------------------------

    def trades(self, limit: Optional[int] = None) -> List[TradeRecord]:
        """Newest first."""
        rows = sorted(self.store.all(), key=lambda r: r.exit_time, reverse=True)
        return rows[:limit] if limit else rows

    def summary(self) -> Dict[str, Any]:
        rows = self.store.all()
        wins = [r.pnl for r in rows if r.pnl > 0]
        losses = [r.pnl for r in rows if r.pnl <= 0]
        gross_loss = -sum(losses)
        return {
            "total_trades": len(rows),
            "winning_trades": len(wins),
            "win_rate": (len(wins) / len(rows)) if rows else 0.0,
            "total_pnl": sum(r.pnl for r in rows),
            "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else None,
            "avg_win": statistics.mean(wins) if wins else None,
            "avg_loss": statistics.mean(losses) if losses else None,
            "unique_symbols": len({r.symbol for r in rows}),
            "unique_strategies": len({r.strategy for r in rows}),
            "first_trade": min((r.exit_time for r in rows), default=None),
            "last_trade": max((r.exit_time for r in rows), default=None),
        }

    def daily(self, start: date, end: date) -> List[Dict[str, Any]]:
        """Realised P&L per exit date, inclusive of both ends, ascending."""
        by_day: Dict[date, Dict[str, Any]] = {}
        for r in self.store.all():
            d = r.exit_time.date()
            if start <= d <= end:
                entry = by_day.setdefault(d, {"date": d.isoformat(), "pnl": 0.0, "trades": 0})
                entry["pnl"] += r.pnl
                entry["trades"] += 1
        return [by_day[d] for d in sorted(by_day)]

    def close(self) -> None:
        self.store.close()
