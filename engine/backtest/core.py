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

This mixin is composed onto ``BacktestEngine`` along with the runner and
walk-forward mixins.  Tests rely on the bound-method form (``engine.run``,
``engine._calculate_trade_pnl(...)``, etc.), which is why mixin composition
is preferred over plain helper modules.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List

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

    def _cached_sessions_cover_requested_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        sessions: List[datetime],
    ) -> bool:
        """Only trust cached sessions when they span the full requested window."""
        if not sessions or start_dt > end_dt:
            return False

        return sessions[0].date() <= start_dt.date() and sessions[-1].date() >= end_dt.date()

    async def _fetch_trading_sessions_from_data_broker(
        self,
        data_broker,
        symbols: List[str],
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[datetime]:
        """Fetch session timestamps for a date range to build an exchange-aware fold calendar."""
        cached_price_data = getattr(data_broker, "price_data", None)
        cached_sessions = self._extract_trading_sessions_from_price_data(
            start_dt, end_dt, cached_price_data
        )
        if self._cached_sessions_cover_requested_range(start_dt, end_dt, cached_sessions):
            return cached_sessions

        if not hasattr(data_broker, "get_bars"):
            return self._build_weekday_sessions(start_dt, end_dt)

        sessions_by_date = {}
        start_date = start_dt.date()
        end_date = end_dt.date()

        async def _load_symbol_sessions(symbol: str) -> None:
            try:
                bars = await data_broker.get_bars(
                    symbol,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    timeframe="1Day",
                )
            except Exception as exc:
                logger.warning(f"Failed to load session calendar for {symbol}: {exc}")
                return

            for bar in bars or []:
                timestamp = getattr(bar, "timestamp", None)
                if timestamp is None:
                    continue
                session_ts = pd.Timestamp(timestamp).to_pydatetime()
                session_date = session_ts.date()
                if start_date <= session_date <= end_date:
                    sessions_by_date.setdefault(session_date, session_ts)

        await asyncio.gather(
            *[_load_symbol_sessions(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        return (
            [sessions_by_date[session_date] for session_date in sorted(sessions_by_date)]
            if sessions_by_date
            else self._build_weekday_sessions(start_dt, end_dt)
        )

    # ------------------------------------------------------------------
    # Strategy.run() orchestration (simple loop)
    # ------------------------------------------------------------------

    async def run(self, strategies, start_date, end_date):
        """
        Run backtest for strategies over the given period.

        Args:
            strategies: List of strategy instances to backtest
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            List of result DataFrames, one per strategy
        """
        self.strategies = strategies
        self.current_date = start_date
        trading_days = self._resolve_trading_sessions(
            start_date,
            end_date,
            getattr(self.broker, "price_data", None),
        )
        dates = pd.DatetimeIndex(trading_days)

        # Initialize result tracking for each strategy
        results = []
        for _strategy in strategies:
            # Create daily results dataframe with date index
            result_df = pd.DataFrame(index=dates)
            result_df["equity"] = pd.Series(index=dates, dtype="float64")
            result_df["cash"] = pd.Series(index=dates, dtype="float64")
            result_df["holdings"] = pd.Series(index=dates, dtype="float64")
            result_df["returns"] = pd.Series(index=dates, dtype="float64")
            result_df["trades"] = pd.Series(0, index=dates, dtype="int64")
            results.append(result_df)

        # Initialize strategies with broker
        for strategy in strategies:
            if not hasattr(strategy, "broker"):
                strategy.broker = self.broker

        # Run backtest day by day
        for current_date in trading_days:
            logger.debug(f"Processing date: {current_date.date()}")
            self.current_date = current_date

            # Process each strategy for this day
            for i, strategy in enumerate(strategies):
                try:
                    # Run one iteration of the strategy
                    await self._run_strategy_iteration(strategy, current_date)

                    # Record daily results
                    result_df = results[i]
                    if current_date in result_df.index:
                        portfolio_value = self.broker.get_portfolio_value(current_date)
                        cash = self.broker.get_balance()
                        holdings = portfolio_value - cash

                        result_df.loc[current_date, "equity"] = portfolio_value
                        result_df.loc[current_date, "cash"] = cash
                        result_df.loc[current_date, "holdings"] = holdings

                        # Calculate daily returns
                        if current_date != start_date:
                            prev_date = current_date - timedelta(days=1)
                            while prev_date not in result_df.index and prev_date >= start_date:
                                prev_date = prev_date - timedelta(days=1)

                            if (
                                prev_date in result_df.index
                                and result_df.loc[prev_date, "equity"] > 0
                            ):
                                prev_equity = result_df.loc[prev_date, "equity"]
                                result_df.loc[current_date, "returns"] = (
                                    portfolio_value / prev_equity
                                ) - 1
                except Exception as e:
                    logger.error(
                        f"Error in strategy {strategy.__class__.__name__} on {current_date.date()}: {e}"
                    )

        # Post-process results to fill missing values and calculate metrics
        for i, result_df in enumerate(results):
            # Forward fill equity values for non-trading days
            result_df.ffill(inplace=True)

            # Calculate cumulative returns
            result_df["cum_returns"] = (1 + result_df["returns"].fillna(0)).cumprod() - 1

            # Calculate additional metrics
            strategy_name = strategies[i].__class__.__name__
            self._calculate_performance_metrics(result_df, strategy_name)

        return results

    async def _run_strategy_iteration(self, strategy, current_date):
        """Run a single iteration of a strategy for the given date."""
        # Call the strategy's on_trading_iteration method if it exists
        if hasattr(strategy, "on_trading_iteration"):
            strategy.current_date = current_date  # Set current date for the strategy
            result = strategy.on_trading_iteration()
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Inline performance metrics for run() result frames
    # ------------------------------------------------------------------

    def _calculate_performance_metrics(self, result_df, strategy_name):
        """Calculate performance metrics for a strategy."""
        # Skip if not enough data
        if len(result_df) < 2:
            return

        equity = pd.to_numeric(result_df.get("equity"), errors="coerce").astype(float)
        returns = pd.to_numeric(result_df.get("returns"), errors="coerce").astype(float)
        if "cum_returns" in result_df.columns:
            cum_returns = pd.to_numeric(result_df["cum_returns"], errors="coerce").astype(float)
        else:
            cum_returns = (1.0 + returns.fillna(0.0)).cumprod() - 1.0

        result_df["equity"] = equity
        result_df["returns"] = returns
        result_df["cum_returns"] = cum_returns

        # Calculate daily, monthly, and annual returns
        result_df["daily_returns"] = returns

        # Calculate drawdowns
        peak_values: list[float] = []
        drawdown_values: list[float] = []
        running_peak: float | None = None
        for value in equity.tolist():
            if pd.isna(value) or float(value) <= 0.0:
                peak_values.append(float("nan"))
                drawdown_values.append(float("nan"))
                continue

            current_equity = float(value)
            running_peak = (
                current_equity if running_peak is None else max(running_peak, current_equity)
            )
            peak_values.append(running_peak)
            drawdown_values.append((current_equity / running_peak) - 1.0)

        result_df["peak"] = pd.Series(peak_values, index=result_df.index, dtype=float)
        result_df["drawdown"] = pd.Series(drawdown_values, index=result_df.index, dtype=float)

        # Maximum drawdown
        valid_drawdowns = [
            float(value) for value in drawdown_values if value is not None and not pd.isna(value)
        ]
        max_drawdown = min(valid_drawdowns) if valid_drawdowns else 0.0

        # Annualized return (assuming 252 trading days in a year)
        days = (result_df.index[-1] - result_df.index[0]).days
        if days > 0:
            years = days / 365
            valid_cum_returns = [
                float(value)
                for value in result_df["cum_returns"].tolist()
                if value is not None and not pd.isna(value)
            ]
            total_return = valid_cum_returns[-1] if valid_cum_returns else 0.0
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        daily_return_values = [
            float(value)
            for value in result_df["daily_returns"].tolist()
            if value is not None and not pd.isna(value)
        ]
        if len(daily_return_values) >= 2:
            mean_return = sum(daily_return_values) / len(daily_return_values)
            variance = sum((value - mean_return) ** 2 for value in daily_return_values) / (
                len(daily_return_values) - 1
            )
            daily_std = math.sqrt(variance) if variance > 0.0 else 0.0
        else:
            daily_std = 0.0

        if daily_std > 0.0:
            sharpe_ratio = (mean_return / daily_std) * math.sqrt(252.0)
        else:
            sharpe_ratio = 0.0

        # Add metrics to dataframe
        result_df.attrs["strategy"] = strategy_name
        result_df.attrs["annualized_return"] = annualized_return
        result_df.attrs["max_drawdown"] = max_drawdown
        result_df.attrs["sharpe_ratio"] = sharpe_ratio

        logger.info(
            f"Strategy {strategy_name} - Annualized Return: {annualized_return:.2%}, "
            f"Max Drawdown: {max_drawdown:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}"
        )

    # ------------------------------------------------------------------
    # Per-symbol signal processing (used by run_backtest in runner.py)
    # ------------------------------------------------------------------

    async def _process_symbol_signal(
        self, symbol: str, strategy, backtest_broker, day_num: int
    ) -> Dict[str, Any]:
        """Process a single symbol's signal in parallel.

        Performance optimization: This method allows multiple symbols to be
        analyzed and traded concurrently using asyncio.gather().

        Args:
            symbol: The symbol to process
            strategy: The strategy instance
            backtest_broker: The backtest broker instance
            day_num: Current day number (for debug logging)
        """
        event: Dict[str, Any] = {
            "event_type": "decision",
            "symbol": symbol,
            "day_num": day_num,
            "action": "neutral",
            "trade_attempted": False,
            "trade_executed": False,
            "error": None,
        }

        if symbol not in backtest_broker.price_data:
            event["action"] = "no_data"
            return event

        try:
            signal = await strategy.analyze_symbol(symbol)
            event["signal"] = (
                signal if isinstance(signal, (dict, list, str, int, float, bool)) else str(signal)
            )
            if signal:
                # Handle both string and dict signal formats
                if isinstance(signal, str):
                    action = signal
                else:
                    action = signal.get("action") if isinstance(signal, dict) else "neutral"
                event["action"] = action

                if day_num < 5:  # Log first few days for debugging
                    logger.debug(f"  {symbol} signal: {action}")

                if action not in ["hold", "neutral", None]:
                    event["trade_attempted"] = True
                    logger.info(f"  Trade signal: {symbol} - {action}")
                    # Convert string signal to dict for execute_trade
                    if isinstance(signal, str):
                        signal = {"action": signal, "symbol": symbol}
                    await strategy.execute_trade(symbol, signal)
                    event["trade_executed"] = True
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
            event["error"] = str(e)

        return event

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
                        state["avg_price"] = (
                            prior_long * old_avg + remaining * price
                        ) / new_qty
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
                        state["avg_price"] = (
                            prior_short * old_avg + remaining * price
                        ) / (prior_short + remaining)
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
