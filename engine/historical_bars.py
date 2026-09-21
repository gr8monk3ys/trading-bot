"""Historical bars: one module you ask for bars over a window.

Two adapters sit behind it, Alpaca (any broker with ``async get_bars``) and
yfinance. Every symbol comes back with a data outcome, loaded / empty /
failed(cause), and a failed fetch is never reported as an empty backtest.
See docs/adr/0008-data-failures-fail-the-run.md.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class DataOutcome:
    """What loading one symbol produced."""

    symbol: str
    status: str  # "loaded" | "empty" | "failed"
    bars: List[Bar] = field(default_factory=list)
    cause: str = ""

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"


@dataclass
class BarsResult:
    source: str
    outcomes: Dict[str, DataOutcome]

    @property
    def loaded(self) -> List[str]:
        return [s for s, o in self.outcomes.items() if o.status == "loaded"]

    @property
    def failed(self) -> List[str]:
        return [s for s, o in self.outcomes.items() if o.status == "failed"]

    @property
    def empty(self) -> List[str]:
        return [s for s, o in self.outcomes.items() if o.status == "empty"]

    @property
    def ok(self) -> bool:
        """Every requested symbol loaded. Partial success is not a run."""
        return bool(self.outcomes) and len(self.loaded) == len(self.outcomes)

    def frames(self) -> Dict[str, pd.DataFrame]:
        """OHLCV frames per loaded symbol (volume as float, for TA-Lib)."""
        out = {}
        for symbol in self.loaded:
            bars = self.outcomes[symbol].bars
            out[symbol] = pd.DataFrame(
                {
                    "open": [float(b.open) for b in bars],
                    "high": [float(b.high) for b in bars],
                    "low": [float(b.low) for b in bars],
                    "close": [float(b.close) for b in bars],
                    "volume": [float(b.volume) for b in bars],
                },
                index=pd.DatetimeIndex([b.timestamp for b in bars]),
            )
        return out

    def sessions(self) -> List[datetime]:
        """Sorted trading sessions across every loaded symbol, one per date."""
        by_date: Dict[Any, datetime] = {}
        for symbol in self.loaded:
            for b in self.outcomes[symbol].bars:
                if b.timestamp is None:
                    continue
                ts = pd.Timestamp(b.timestamp).to_pydatetime()
                by_date.setdefault(ts.date(), ts)
        return [by_date[d] for d in sorted(by_date)]

    def report(self) -> Dict[str, Dict[str, Any]]:
        """Per-symbol data-quality report in the shape the engine has always returned."""
        rep = {}
        for symbol, o in self.outcomes.items():
            entry: Dict[str, Any] = {"rows": len(o.bars), "loaded": o.loaded, "status": o.status}
            if o.cause:
                entry["error"] = o.cause
            rep[symbol] = entry
        return rep

    def describe_failures(self) -> str:
        parts = [f"{s}: {self.outcomes[s].cause or 'no rows'}" for s in self.failed + self.empty]
        return "; ".join(parts)


class DataUnavailableError(RuntimeError):
    """Raised when a run cannot proceed because bars are missing for any symbol."""

    def __init__(self, result: BarsResult, message: Optional[str] = None):
        self.result = result
        super().__init__(
            message
            or f"Data unavailable from {result.source}: {result.describe_failures() or 'no symbols'}"
        )


# --- adapters ----------------------------------------------------------------


class BrokerBars:
    """Adapter over any broker exposing ``async get_bars(symbol, start=, end=, timeframe=)``."""

    def __init__(self, broker: Any, name: str = "alpaca"):
        self.broker = broker
        self.name = name

    async def get_bars(self, symbol: str, start: str, end: str) -> List[Bar]:
        raw = await self.broker.get_bars(symbol, start=start, end=end, timeframe="1Day")
        return [
            Bar(
                timestamp=getattr(b, "timestamp", None),
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(getattr(b, "volume", 0.0) or 0.0),
            )
            for b in (raw or [])
        ]


class YFinanceBars:
    """Read-only daily bars from yfinance, run in a thread so the loop is not blocked."""

    name = "yfinance"

    def __init__(self) -> None:
        import yfinance as yf

        self._yf = yf

    async def get_bars(self, symbol: str, start: str, end: str) -> List[Bar]:
        return await asyncio.to_thread(self._sync_get_bars, symbol, start, end)

    def _sync_get_bars(self, symbol: str, start: str, end: str) -> List[Bar]:
        # Errors propagate: a network failure is a failed outcome, never an empty one.
        df = self._yf.download(
            symbol, start=start, end=end, progress=False, auto_adjust=False, threads=False
        )
        if df is None or df.empty:
            return []
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            try:
                df = df.xs(symbol, axis=1, level=-1)
            except KeyError:
                df.columns = [c[0] for c in df.columns]
        bars = []
        for ts, row in df.iterrows():
            try:
                o, h, lo, c = (float(row[k]) for k in ("Open", "High", "Low", "Close"))
                v = float(row["Volume"]) if not _isnan(row["Volume"]) else 0.0
            except (KeyError, TypeError, ValueError):
                continue
            if any(_isnan(x) for x in (o, h, lo, c)):
                continue
            bars.append(
                Bar(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts, o, h, lo, c, v)
            )
        return bars


def _isnan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


# --- the interface ------------------------------------------------------------


async def load_bars(source: Any, symbols: List[str], start: str, end: str) -> BarsResult:
    """Load daily bars for every symbol; each gets an outcome, nothing is swallowed."""

    async def one(symbol: str) -> DataOutcome:
        try:
            bars = await source.get_bars(symbol, start, end)
        except Exception as exc:
            logger.warning("Bar fetch failed for %s: %s", symbol, exc)
            return DataOutcome(symbol, "failed", cause=f"{type(exc).__name__}: {exc}")
        if not bars:
            logger.warning("No bars for %s in %s..%s", symbol, start, end)
            return DataOutcome(symbol, "empty")
        return DataOutcome(symbol, "loaded", bars=list(bars))

    outcomes = await asyncio.gather(*(one(s) for s in symbols))
    return BarsResult(
        getattr(source, "name", type(source).__name__), {o.symbol: o for o in outcomes}
    )


async def resolve_bars_source(probe_symbol: str = "SPY") -> Tuple[Any, str]:
    """Alpaca when credentials work, else yfinance. Raises DataUnavailableError when neither."""
    tried = []
    if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
        try:
            from brokers.alpaca_broker import AlpacaBroker

            broker = AlpacaBroker(paper=True)
            adapter = BrokerBars(broker, "alpaca")
            probe = await adapter.get_bars(probe_symbol, "2024-01-01", "2024-01-10")
            if probe:
                return adapter, "alpaca"
            tried.append("alpaca: probe returned no bars")
        except Exception as exc:
            tried.append(f"alpaca: {type(exc).__name__}: {exc}")
    else:
        tried.append("alpaca: credentials not set")
    try:
        adapter = YFinanceBars()
        probe = await adapter.get_bars(probe_symbol, "2024-01-01", "2024-01-10")
        if probe:
            return adapter, "yfinance"
        tried.append("yfinance: probe returned no bars")
    except Exception as exc:
        tried.append(f"yfinance: {type(exc).__name__}: {exc}")
    raise DataUnavailableError(
        BarsResult("none", {}), "No data source available. " + " | ".join(tried)
    )


# --- benchmarks ---------------------------------------------------------------


def compute_buy_and_hold(
    symbol: str,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
) -> dict:
    """Compute buy-and-hold metrics for `symbol` over [start, end] via yfinance.

    Returns a dict with total_return (fraction), cagr, sharpe (rf=0),
    max_drawdown (negative fraction), final_equity, and n_days. Missing data
    returns None fields rather than raising — benchmarks are nice-to-have,
    not load-bearing.
    """
    try:
        import yfinance as yf
    except Exception as exc:
        logger.warning("yfinance import failed for benchmark %s: %s", symbol, exc)
        return _empty_benchmark(symbol, start, end, reason=f"yfinance import failed: {exc}")

    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as exc:
        logger.warning("yfinance fetch failed for benchmark %s: %s", symbol, exc)
        return _empty_benchmark(symbol, start, end, reason=f"yfinance fetch failed: {exc}")

    if df is None or df.empty:
        return _empty_benchmark(symbol, start, end, reason="no data")

    # Single ticker download can come back as MultiIndex columns in newer yfinance.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        try:
            df = df.xs(symbol, axis=1, level=-1)
        except KeyError:
            df.columns = [c[0] for c in df.columns]

    closes = df["Close"].squeeze().dropna()
    if len(closes) < 2:
        return _empty_benchmark(symbol, start, end, reason="<2 closes")

    first = float(closes.iloc[0])
    last = float(closes.iloc[-1])
    if first <= 0:
        return _empty_benchmark(symbol, start, end, reason="non-positive start price")

    total_return = (last / first) - 1.0
    final_equity = initial_capital * (last / first)

    # CAGR from the period spanned by the actual close dates.
    try:
        first_ts = closes.index[0]
        last_ts = closes.index[-1]
        years = max((last_ts - first_ts).days / 365.25, 1e-9)
    except Exception:
        years = max(len(closes) / 252.0, 1e-9)
    cagr = ((last / first) ** (1.0 / years)) - 1.0 if last > 0 else None

    # Daily returns, Sharpe (rf=0, annualized), max drawdown.
    daily_returns = closes.pct_change().dropna()
    if len(daily_returns) >= 2:
        mean = float(daily_returns.mean())
        std = float(daily_returns.std(ddof=1))
        sharpe = (mean / std) * math.sqrt(252) if std > 0 else None
    else:
        sharpe = None

    running_max = closes.cummax()
    drawdowns = (closes / running_max) - 1.0
    max_dd = float(drawdowns.min()) if len(drawdowns) else None

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": final_equity,
        "n_days": int(len(closes)),
        "reason": None,
    }


def _empty_benchmark(symbol: str, start: str, end: str, reason: str) -> dict:
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "total_return": None,
        "cagr": None,
        "sharpe": None,
        "max_drawdown": None,
        "final_equity": None,
        "n_days": 0,
        "reason": reason,
    }
