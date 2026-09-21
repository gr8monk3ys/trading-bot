"""
BacktestEngine core mixin.

Houses the foundational orchestration surface:

    - Session resolution helpers (weekday fallback, bar-derived sessions,
      cached-session validation, async data-broker fetch).
    - ``run()`` — the simpler strategy.run() orchestration loop that records
      per-strategy daily equity/cash/holdings/returns.
    - ``_run_strategy_iteration`` — invokes a strategy's
      ``on_trading_iteration`` and awaits it if coroutine.
    - ``_calculate_performance_metrics`` — drawdown / annualized return /
      Sharpe computed inline for a single result DataFrame.
    - ``_process_symbol_signal`` — per-symbol concurrency unit shared with
      the comprehensive run_backtest driver in runner.py.
    - ``_calculate_trade_pnl`` — signed-quantity position state machine that
      realizes PnL on both long and short legs (Step 2B fix).

This mixin is composed onto ``BacktestEngine`` along with the runner
mixin.  Tests rely on the bound-method form (``engine._calculate_trade_pnl(...)``
etc.), which is why mixin composition is preferred over plain helper modules.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class BacktestCoreMixin:
    """Session resolution, per-bar iteration, inline metrics, signed-qty PnL."""

    # ------------------------------------------------------------------
    # Session resolution
    # ------------------------------------------------------------------

    def _build_weekday_sessions(self, start_dt: datetime, end_dt: datetime) -> List[datetime]:
        """Build a weekday-only fallback session list when market data is unavailable."""
        if start_dt > end_dt:
            return []

        sessions = []
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:
                sessions.append(current)
            current += timedelta(days=1)
        return sessions

    def _extract_trading_sessions_from_price_data(
        self,
        start_dt: datetime,
        end_dt: datetime,
        price_data: Dict[str, pd.DataFrame] | None,
    ) -> List[datetime]:
        """Derive actual market sessions from loaded daily bar timestamps."""
        if not price_data or start_dt > end_dt:
            return []

        sessions_by_date = {}
        start_date = start_dt.date()
        end_date = end_dt.date()

        for df in price_data.values():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            for timestamp in df.index:
                session_ts = pd.Timestamp(timestamp).to_pydatetime()
                session_date = session_ts.date()
                if start_date <= session_date <= end_date:
                    sessions_by_date.setdefault(session_date, session_ts)

        return [sessions_by_date[session_date] for session_date in sorted(sessions_by_date)]

    def _resolve_trading_sessions(
        self,
        start_dt: datetime,
        end_dt: datetime,
        price_data: Dict[str, pd.DataFrame] | None = None,
    ) -> List[datetime]:
        """Use actual bar timestamps when available, otherwise fall back to weekdays."""
        sessions = self._extract_trading_sessions_from_price_data(start_dt, end_dt, price_data)
        return sessions if sessions else self._build_weekday_sessions(start_dt, end_dt)

    # ------------------------------------------------------------------
    # Strategy.run() orchestration (simple loop)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Inline performance metrics for run() result frames
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-symbol signal processing (used by run_backtest in runner.py)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Trade-level PnL accounting (signed-position state machine, Step 2B)
    # ------------------------------------------------------------------

    def _calculate_trade_pnl(self, trades: List[Dict]) -> List[Dict]:
        """
        Calculate P&L for each trade by matching opening and closing legs.

        Tracks signed position state per symbol so both long and short legs
        record realized PnL:

        - ``position_tracker[symbol] = {"qty": int, "avg_price": float}`` where
          a positive ``qty`` is a long position, negative is a short, and 0 is
          flat.
        - ``buy`` reduces (covers) any existing short, then any remaining
          quantity opens / adds to a long at the trade price (weighted avg).
        - ``sell`` reduces (closes) any existing long, then any remaining
          quantity opens / adds to a short at the trade price (weighted avg).

        Realized PnL from the closing portion of a leg is recorded against the
        single output record for that trade (one output record per input).

        Args:
            trades: List of raw trade records

        Returns:
            List of trade records with P&L calculated
        """
        trade_records = []
        position_tracker: Dict[str, Dict[str, float]] = {}

        for trade in trades:
            symbol = trade["symbol"]
            side = trade["side"]
            quantity = trade["quantity"]
            price = trade["price"]

            state = position_tracker.setdefault(symbol, {"qty": 0, "avg_price": 0.0})
            old_qty = state["qty"]
            old_avg = state["avg_price"]
            pnl = 0.0
            remaining = quantity

            if side == "buy":
                # First, cover any open short (qty < 0).
                if old_qty < 0 and remaining > 0:
                    cover_qty = min(remaining, -old_qty)
                    # Short PnL: short was opened at old_avg, covered at price.
                    pnl += (old_avg - price) * cover_qty
                    new_qty = old_qty + cover_qty
                    state["qty"] = new_qty
                    if new_qty == 0:
                        state["avg_price"] = 0.0
                    # avg_price unchanged while still short
                    remaining -= cover_qty
                    old_qty = new_qty
                # Any leftover quantity adds to / opens a long.
                if remaining > 0:
                    new_qty = old_qty + remaining
                    if new_qty > 0:
                        # Weighted average across existing long and new buy.
                        prior_long = max(old_qty, 0)
                        state["avg_price"] = (prior_long * old_avg + remaining * price) / new_qty
                    state["qty"] = new_qty

            else:  # sell
                # First, close any open long (qty > 0).
                if old_qty > 0 and remaining > 0:
                    close_qty = min(remaining, old_qty)
                    pnl += (price - old_avg) * close_qty
                    new_qty = old_qty - close_qty
                    state["qty"] = new_qty
                    if new_qty == 0:
                        state["avg_price"] = 0.0
                    # avg_price unchanged while still long
                    remaining -= close_qty
                    old_qty = new_qty
                # Any leftover quantity adds to / opens a short.
                if remaining > 0:
                    new_qty = old_qty - remaining
                    if new_qty < 0:
                        # Weighted average across existing short and new sell.
                        prior_short = -min(old_qty, 0)
                        state["avg_price"] = (prior_short * old_avg + remaining * price) / (
                            prior_short + remaining
                        )
                    state["qty"] = new_qty

            trade_records.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "timestamp": trade.get("timestamp"),
                    "pnl": pnl,
                }
            )

        return trade_records
