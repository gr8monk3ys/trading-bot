"""
BacktestEngine comprehensive run_backtest driver mixin.

Owns the full single-run lifecycle:

    - Parallel historical bar loading (per-symbol get_bars under
      asyncio.gather).
    - BacktestBroker instantiation with the requested execution profile.
    - Strategy instantiation + OrderSubmission wiring (every order, including
      the end-of-run liquidation, goes through one ``OrderSubmission`` so the strategy's
      ``submit_entry_order`` / ``submit_exit_order`` calls don't fail with
      "No OrderGateway configured").
    - Day-by-day signal processing via ``_process_symbol_signal`` (defined
      on the core mixin), gap-risk modeling, and decision-event logging.
    - End-of-period liquidation (Step 2C: ``_liquidate_open_positions``)
      so headline equity reflects realized PnL only.
    - Trade-level PnL via ``_calculate_trade_pnl`` (signed-qty state
      machine, defined on the core mixin).
    - Result assembly including gap statistics, stress test, and run
      metadata.

This mixin depends on helpers provided by ``BacktestCoreMixin`` and is
composed onto ``BacktestEngine`` together with it.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Type

import pandas as pd
import pytz

from engine.historical_bars import BrokerBars, DataUnavailableError, YFinanceBars, load_bars
from engine.order_submission import OrderIntent
from utils.portfolio_stress import run_portfolio_stress_test

logger = logging.getLogger(__name__)


class BacktestRunnerMixin:
    """Comprehensive run_backtest driver + end-of-period liquidation."""

    async def run_backtest(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date,
        end_date,
        initial_capital: float = 100000,
        strategy_params: Dict[str, Any] | None = None,
        execution_profile: str = "realistic",
        run_id: str | None = None,
        artifacts_dir: str = "results/runs",
    ) -> Dict[str, Any]:
        """
        Run a comprehensive backtest for a strategy.

        Args:
            strategy_class: Strategy class to instantiate and test
            symbols: List of symbols to trade
            start_date: Start date for backtest (date or datetime)
            end_date: End date for backtest (date or datetime)
            initial_capital: Starting capital
            strategy_params: Optional strategy parameter overrides
            execution_profile: Backtest execution realism profile
            run_id: Optional externally provided run ID
            artifacts_dir: Base path for run artifact directories

        Returns:
            Dictionary with backtest results including equity_curve and trades
        """
        from brokers.alpaca_broker import AlpacaBroker
        from brokers.backtest import BacktestBroker

        run_started_at = datetime.utcnow()
        run_id = run_id or f"backtest_{run_started_at.strftime('%Y%m%d_%H%M%S')}"
        decision_event_count = 0
        decision_error_count = 0

        # Create backtest broker
        backtest_broker = BacktestBroker(
            initial_balance=initial_capital,
            execution_profile=execution_profile,
            run_id=run_id,
        )

        # Use existing broker for data if available, otherwise create one
        data_broker = self.broker if self.broker else AlpacaBroker(paper=True)

        # NOTE: No survivorship-bias correction. Backtests trade the full symbol
        # list for all dates; rerun on a curated symbol set if survivorship matters.

        # Convert dates to datetime if they are date objects
        if hasattr(start_date, "strftime") and not hasattr(start_date, "hour"):
            start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            start_dt = start_date

        if hasattr(end_date, "strftime") and not hasattr(end_date, "hour"):
            end_dt = datetime.combine(end_date, datetime.min.time())
        else:
            end_dt = end_date

        # Fetch historical data for all symbols
        logger.info(
            f"Loading historical data for {len(symbols)} symbols from {start_dt.date()} to {end_dt.date()}..."
        )
        # A bars source (YFinanceBars / BrokerBars) is used as-is; a raw broker
        # with ``get_bars(symbol, start=, end=, timeframe=)`` gets the adapter.
        source = (
            data_broker
            if isinstance(data_broker, (YFinanceBars, BrokerBars))
            else BrokerBars(data_broker, type(data_broker).__name__)
        )
        bars = await load_bars(
            source,
            list(symbols),
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )
        if not bars.ok:
            # Partial success is not a run (ADR 0008).
            raise DataUnavailableError(bars)
        loaded_price_data = bars.frames()
        for symbol, data in loaded_price_data.items():
            backtest_broker.set_price_data(symbol, data)
            logger.debug(f"Loaded {len(data)} bars for {symbol}")
        data_quality_reports = bars.report()
        loaded_sessions = bars.sessions()

        # Instantiate strategy with backtest broker and symbols
        params = {"symbols": symbols}
        if strategy_params:
            params.update(strategy_params)
        strategy = strategy_class(broker=backtest_broker, parameters=params)

        # One order submission for the run: the strategy's entries and exits
        # and the end-of-run liquidation all go through it (ADR 0004).
        from engine.order_submission import OrderSubmission

        order_submission = OrderSubmission(backtest_broker)
        strategy.order_submission = order_submission

        # Initialize the strategy if it has an initialize method
        if hasattr(strategy, "initialize"):
            try:
                await strategy.initialize()
            except Exception as e:
                logger.warning(f"Strategy initialization warning: {e}")

        # Track equity curve and daily gross exposure (deployed notional /
        # portfolio value). Exposure is the denominator any comparison against
        # buy-and-hold needs: the 2020-2024 baselines ran ~25% deployed and
        # nothing measured it.
        equity_curve = [initial_capital]
        exposure_curve = []

        trading_days = loaded_sessions or self._resolve_trading_sessions(
            start_dt, end_dt, loaded_price_data
        )

        logger.info(f"Running backtest over {len(trading_days)} trading days...")

        # Run day by day
        for day_num, current_date in enumerate(trading_days):
            try:
                # Update the backtest broker's current date for price lookups
                backtest_broker.advance_to(current_date)

                # ==========================================
                # GAP RISK MODELING: Process overnight gaps
                # ==========================================
                # Before processing today's signals, check if any positions
                # were affected by overnight gaps (stops gapped through, etc.)
                if day_num > 0:  # Skip first day (no previous close)
                    gap_events = await backtest_broker.process_day_start_gaps(current_date)
                    if gap_events:
                        for gap in gap_events:
                            if gap.stop_triggered:
                                logger.warning(
                                    f"  Gap risk event: {gap.symbol} - "
                                    f"gap {gap.gap_pct:+.2%}, stop gapped through "
                                    f"(slippage: ${gap.slippage_from_stop:.2f})"
                                )

                # Populate price history and current_data for the strategy
                # Make current_date timezone-aware for comparison with UTC index
                current_date_utc = (
                    current_date.replace(tzinfo=pytz.UTC)
                    if current_date.tzinfo is None
                    else current_date
                )

                # Ensure strategy has current_data attribute
                if not hasattr(strategy, "current_data"):
                    strategy.current_data = {}

                # No survivorship-bias correction: trade the full symbol list each day.
                tradeable_symbols = symbols

                for symbol in tradeable_symbols:
                    if symbol in backtest_broker.price_data:
                        df = backtest_broker.price_data[symbol]
                        # CRITICAL: Use strictly LESS THAN (<) to prevent look-ahead bias
                        # Strategy should only see data from BEFORE the current trading day
                        # Using <= would allow seeing today's close when making today's decision
                        try:
                            historical = df[df.index < current_date_utc]
                        except TypeError:
                            # If comparison fails, try normalizing the index
                            df_naive = df.copy()
                            df_naive.index = df_naive.index.tz_localize(None)
                            historical = df_naive[
                                df_naive.index < current_date.replace(tzinfo=None)
                            ]

                        if len(historical) > 0:
                            prices = historical["close"].tolist()
                            if hasattr(strategy, "price_history"):
                                strategy.price_history[symbol] = prices[-30:]  # Keep last 30 days
                            # Also populate current_data with the historical DataFrame
                            strategy.current_data[symbol] = historical

                # Generate signals if the strategy has a generate_signals method
                if hasattr(strategy, "generate_signals"):
                    try:
                        await strategy.generate_signals()
                    except Exception as e:
                        logger.debug(f"Error in generate_signals: {e}")

                # Performance optimization: Process all symbols in parallel using asyncio.gather
                # This significantly speeds up backtests with many symbols by running
                # analyze_symbol and execute_trade concurrently
                # Only process symbols that were tradeable on this date
                decision_events = await asyncio.gather(
                    *[
                        self._process_symbol_signal(symbol, strategy, backtest_broker, day_num)
                        for symbol in tradeable_symbols
                    ],
                    return_exceptions=True,  # Don't fail on individual symbol errors
                )

                for event in decision_events:
                    if isinstance(event, Exception):
                        decision_error_count += 1
                        continue

                    if not isinstance(event, dict):
                        continue

                    decision_event_count += 1
                    if event.get("error"):
                        decision_error_count += 1

                # Record equity and gross exposure at end of day
                portfolio_value = backtest_broker.get_portfolio_value(current_date)
                equity_curve.append(portfolio_value)
                try:
                    exposure_curve.append(float(backtest_broker.get_gross_exposure(current_date)))
                except (TypeError, ValueError, AttributeError):
                    # Broker without the exposure API (mocks, custom brokers):
                    # record 0.0 rather than poisoning the series.
                    exposure_curve.append(0.0)

                # ==========================================
                # GAP RISK MODELING: Update previous closes
                # ==========================================
                # Store today's closes for gap calculation tomorrow
                backtest_broker.update_prev_day_closes(current_date)

                # Progress logging every 50 days
                if day_num % 50 == 0:
                    logger.info(
                        f"  Day {day_num}/{len(trading_days)}: Equity = ${portfolio_value:,.2f}"
                    )

            except Exception as e:
                logger.error(f"Error on {current_date.date()}: {e}")
                equity_curve.append(equity_curve[-1] if equity_curve else initial_capital)
                exposure_curve.append(exposure_curve[-1] if exposure_curve else 0.0)

        # Liquidate all remaining open positions at the final trading day's
        # close. Headline equity must reflect realized PnL — unrealized MTM
        # on still-open positions overstates what an operator could capture
        # (see results/honest_backtest_2020-2024.md and the 2026-05 follow-up:
        # "Add an end-of-backtest liquidation pass").
        if trading_days:
            await self._liquidate_open_positions(
                order_submission,
                broker=backtest_broker,
                final_date=trading_days[-1],
                execution_profile=execution_profile,
            )

            # Recompute the closing equity snapshot now that positions are flat.
            final_portfolio_value = backtest_broker.get_portfolio_value(trading_days[-1])
            if equity_curve:
                equity_curve[-1] = final_portfolio_value
            else:
                equity_curve.append(final_portfolio_value)

        # Process trades to calculate P&L
        trades = backtest_broker.get_trades()
        trade_records = self._calculate_trade_pnl(trades)

        final_equity = equity_curve[-1] if equity_curve else initial_capital
        total_return = (final_equity / initial_capital) - 1

        logger.info(f"Backtest complete: Final equity = ${final_equity:,.2f} ({total_return:+.2%})")
        logger.info(f"Total trades: {len(trade_records)}")
        open_positions = await backtest_broker.get_positions()
        stress_test = run_portfolio_stress_test(
            open_positions,
            equity=final_equity,
        )

        # ==========================================
        # GAP RISK STATISTICS
        # ==========================================
        def _safe_numeric(value, default=0.0):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            return default

        gap_stats_obj = (
            backtest_broker.get_gap_statistics()
            if hasattr(backtest_broker, "get_gap_statistics")
            else None
        )
        gap_events_raw = (
            backtest_broker.get_gap_events() if hasattr(backtest_broker, "get_gap_events") else []
        )
        gap_events = gap_events_raw if isinstance(gap_events_raw, list) else []

        total_gaps = int(_safe_numeric(getattr(gap_stats_obj, "total_gaps", 0), 0))
        gaps_exceeding_2pct = int(
            _safe_numeric(getattr(gap_stats_obj, "gaps_exceeding_2pct", 0), 0)
        )
        stops_gapped_through = int(
            _safe_numeric(getattr(gap_stats_obj, "stops_gapped_through", 0), 0)
        )
        total_gap_slippage = float(
            _safe_numeric(getattr(gap_stats_obj, "total_gap_slippage", 0.0), 0.0)
        )
        largest_gap_pct = float(_safe_numeric(getattr(gap_stats_obj, "largest_gap_pct", 0.0), 0.0))
        average_gap_pct = float(_safe_numeric(getattr(gap_stats_obj, "average_gap_pct", 0.0), 0.0))

        if total_gaps > 0:
            logger.info(
                f"Gap Risk Analysis: {total_gaps} gap events, "
                f"{stops_gapped_through} stops gapped through, "
                f"total gap slippage: ${total_gap_slippage:.2f}"
            )
            if largest_gap_pct > 0.05:  # >5% gap
                logger.warning(
                    f"  WARNING: Largest gap was {largest_gap_pct:.1%} - "
                    f"consider tighter position sizing"
                )

        run_completed_at = datetime.utcnow()

        if trading_days:
            equity_curve_series = pd.Series(
                [float(value) for value in equity_curve[1:]],
                index=pd.DatetimeIndex(trading_days),
                dtype=float,
            )
        elif equity_curve:
            equity_curve_series = pd.Series(
                [float(equity_curve[-1])],
                index=pd.DatetimeIndex([pd.Timestamp(start_dt)]),
                dtype=float,
            )
        else:
            equity_curve_series = pd.Series(dtype=float)

        daily_returns = (
            equity_curve_series.pct_change().fillna(0.0)
            if not equity_curve_series.empty
            else pd.Series(dtype=float)
        )

        result = {
            "equity_curve": equity_curve,
            "exposure_curve": exposure_curve,
            "avg_gross_exposure": (
                sum(exposure_curve) / len(exposure_curve) if exposure_curve else 0.0
            ),
            "peak_gross_exposure": max(exposure_curve) if exposure_curve else 0.0,
            "equity_curve_series": equity_curve_series,
            "daily_returns": daily_returns,
            "trades": trade_records,
            "start_date": start_dt,
            "end_date": end_dt,
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "positions": open_positions,
            "total_trades": len(trade_records),
            "stress_test": stress_test,
            "run_metadata": {
                "run_id": run_id,
                "started_at": run_started_at.isoformat(),
                "completed_at": run_completed_at.isoformat(),
                "decision_events": decision_event_count,
                "decision_errors": decision_error_count,
                "execution_profile": execution_profile,
            },
            "data_quality": {
                "reports": data_quality_reports,
                "source": bars.source,
                "symbols_loaded": len(bars.loaded),
                "symbols_requested": len(symbols),
                "symbols_rejected": len(symbols) - len(bars.loaded),
            },
            # Gap risk modeling results
            "gap_statistics": {
                "total_gaps": total_gaps,
                "gaps_exceeding_2pct": gaps_exceeding_2pct,
                "stops_gapped_through": stops_gapped_through,
                "total_gap_slippage": total_gap_slippage,
                "largest_gap_pct": largest_gap_pct,
                "average_gap_pct": average_gap_pct,
            },
            "gap_events": [
                {
                    "symbol": (
                        e.get("symbol") if isinstance(e, dict) else getattr(e, "symbol", None)
                    ),
                    "date": (
                        e.get("date").isoformat()
                        if isinstance(e, dict) and hasattr(e.get("date"), "isoformat")
                        else (
                            e.get("date")
                            if isinstance(e, dict)
                            else (
                                e.date.isoformat()
                                if hasattr(getattr(e, "date", None), "isoformat")
                                else str(getattr(e, "date", ""))
                            )
                        )
                    ),
                    "prev_close": (
                        e.get("prev_close")
                        if isinstance(e, dict)
                        else getattr(e, "prev_close", None)
                    ),
                    "open_price": (
                        e.get("open_price")
                        if isinstance(e, dict)
                        else getattr(e, "open_price", None)
                    ),
                    "gap_pct": (
                        e.get("gap_pct") if isinstance(e, dict) else getattr(e, "gap_pct", None)
                    ),
                    "position_side": (
                        e.get("position_side")
                        if isinstance(e, dict)
                        else getattr(e, "position_side", None)
                    ),
                    "position_qty": (
                        e.get("position_qty")
                        if isinstance(e, dict)
                        else getattr(e, "position_qty", None)
                    ),
                    "stop_price": (
                        e.get("stop_price")
                        if isinstance(e, dict)
                        else getattr(e, "stop_price", None)
                    ),
                    "stop_triggered": (
                        e.get("stop_triggered")
                        if isinstance(e, dict)
                        else getattr(e, "stop_triggered", None)
                    ),
                    "slippage_from_stop": (
                        e.get("slippage_from_stop")
                        if isinstance(e, dict)
                        else getattr(e, "slippage_from_stop", None)
                    ),
                }
                for e in gap_events
            ],
        }

        return result

    async def _liquidate_open_positions(
        self,
        order_submission,
        broker,
        final_date: datetime,
        execution_profile: str = "realistic",
    ) -> int:
        """Close every open position at the final trading day's close.

        Routes through `broker.place_order(...)` so liquidation fills incur the
        same realistic execution profile (spread, market impact, slippage) as
        any other trade, and so the fills land in `broker.get_trades()` and
        flow through `_calculate_trade_pnl`.

        Long positions (qty > 0) are sold; short positions (qty < 0) are
        covered with a buy. The positions are read once up front because
        each fill mutates the broker's ledger during the loop.

        Args:
            broker: The backtest broker holding the positions to liquidate.
            final_date: The trading session used as the timestamp/price
                anchor for the liquidation fills.
            execution_profile: Pass-through for logging only; the broker has
                already been configured with this profile at run start.

        Returns:
            The number of positions that were liquidated.
        """
        # Pin the broker's effective "now" so price lookups + slippage use the
        # final session, not wall-clock time.
        advance = getattr(broker, "advance_to", None)
        if advance is not None:
            advance(final_date)

        # Read once: place_order mutates the ledger during iteration.
        open_positions = list(await broker.get_positions())
        if not open_positions:
            return 0

        liquidated = 0
        for position in open_positions:
            symbol = position.symbol
            qty = position.qty
            if symbol is None or qty == 0:
                continue

            side = "sell" if qty > 0 else "buy"
            abs_qty = abs(qty)
            if abs_qty <= 0:
                continue

            outcome = await order_submission.submit(
                OrderIntent(
                    symbol=symbol,
                    side=side,
                    qty=abs_qty,
                    is_exit=True,
                    strategy_name="backtest_engine",
                    reason="end_of_backtest_liquidation",
                )
            )
            if outcome.ok:
                liquidated += 1
            else:
                logger.warning(f"End-of-backtest liquidation failed for {symbol}: {outcome.reason}")

        if liquidated:
            logger.info(f"Liquidated {liquidated} open positions at end of backtest")
        return liquidated
