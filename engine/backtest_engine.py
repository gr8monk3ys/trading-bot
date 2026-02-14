import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Type

import pandas as pd
import pytz

from utils.data_quality import validate_ohlcv_frame
from utils.historical_universe import HistoricalUniverse
from utils.portfolio_stress import run_portfolio_stress_test
from utils.run_artifacts import (
    JsonlWriter,
    ensure_run_directory,
    generate_run_id,
    write_json,
)

# Set up logging
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Engine for backtesting trading strategies using historical data.
    """

    def __init__(self, broker=None):
        """
        Initialize the backtest engine.

        Args:
            broker: The broker instance to use for market data. If None, create a new one.
        """
        self.broker = broker
        self.current_date = None
        self.strategies = []
        self.results = {}

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

        # Initialize result tracking for each strategy
        results = []
        for _strategy in strategies:
            # Create daily results dataframe with date index
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
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
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() > 4:  # Saturday = 5, Sunday = 6
                current_date += timedelta(days=1)
                continue

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

            # Move to next day
            current_date += timedelta(days=1)

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

    def _calculate_performance_metrics(self, result_df, strategy_name):
        """Calculate performance metrics for a strategy."""
        # Skip if not enough data
        if len(result_df) < 2:
            return

        # Calculate daily, monthly, and annual returns
        result_df["daily_returns"] = result_df["returns"]

        # Calculate drawdowns
        result_df["peak"] = result_df["equity"].cummax()
        result_df["drawdown"] = (result_df["equity"] / result_df["peak"]) - 1

        # Maximum drawdown
        max_drawdown = result_df["drawdown"].min()

        # Annualized return (assuming 252 trading days in a year)
        days = (result_df.index[-1] - result_df.index[0]).days
        if days > 0:
            years = days / 365
            total_return = result_df["cum_returns"].iloc[-1]
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        if result_df["daily_returns"].std() > 0:
            sharpe_ratio = (
                result_df["daily_returns"].mean() / result_df["daily_returns"].std()
            ) * (252**0.5)
        else:
            sharpe_ratio = 0

        # Add metrics to dataframe
        result_df.attrs["strategy"] = strategy_name
        result_df.attrs["annualized_return"] = annualized_return
        result_df.attrs["max_drawdown"] = max_drawdown
        result_df.attrs["sharpe_ratio"] = sharpe_ratio

        logger.info(
            f"Strategy {strategy_name} - Annualized Return: {annualized_return:.2%}, "
            f"Max Drawdown: {max_drawdown:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}"
        )

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
        persist_artifacts: bool = False,
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
            persist_artifacts: If True, persist run artifacts to disk
            artifacts_dir: Base path for run artifact directories

        Returns:
            Dictionary with backtest results including equity_curve and trades
        """
        from brokers.alpaca_broker import AlpacaBroker
        from brokers.backtest_broker import BacktestBroker

        run_started_at = datetime.utcnow()
        run_id = run_id or generate_run_id("backtest")
        run_dir = ensure_run_directory(artifacts_dir, run_id) if persist_artifacts else None
        decision_log_writer = (
            JsonlWriter(run_dir / "decision_events.jsonl") if run_dir is not None else None
        )
        trades_log_writer = JsonlWriter(run_dir / "trades.jsonl") if run_dir is not None else None
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

        # Initialize historical universe for survivorship bias correction
        # This ensures we only trade symbols that were actually tradeable on each date
        historical_universe = HistoricalUniverse(broker=data_broker)
        await historical_universe.initialize()
        logger.info(
            f"Survivorship bias correction enabled: "
            f"{historical_universe.get_statistics()['total_symbols']} symbols tracked"
        )

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
        data_quality_reports: dict[str, dict] = {}

        async def _load_symbol_data(symbol: str) -> None:
            """Load historical data for a single symbol."""
            try:
                bars = await data_broker.get_bars(
                    symbol,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    timeframe="1Day",
                )

                if bars and len(bars) > 0:
                    # Convert to DataFrame format expected by BacktestBroker
                    # Note: volume must be float for talib SMA compatibility
                    data = pd.DataFrame(
                        {
                            "open": [float(b.open) for b in bars],
                            "high": [float(b.high) for b in bars],
                            "low": [float(b.low) for b in bars],
                            "close": [float(b.close) for b in bars],
                            "volume": [float(b.volume) for b in bars],
                        },
                        index=pd.DatetimeIndex([b.timestamp for b in bars]),
                    )

                    quality = validate_ohlcv_frame(
                        data,
                        symbol=symbol,
                        stale_after_days=10,
                        reference_time=end_dt,
                    )
                    data_quality_reports[symbol] = quality.to_dict()
                    if quality.has_errors:
                        logger.warning(
                            f"Data quality errors for {symbol}: "
                            f"{quality.error_count} errors, {quality.warning_count} warnings. "
                            "Skipping symbol."
                        )
                        return

                    backtest_broker.set_price_data(symbol, data)
                    logger.debug(f"Loaded {len(bars)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    data_quality_reports[symbol] = {
                        "symbol": symbol,
                        "rows": 0,
                        "has_errors": True,
                        "error_count": 1,
                        "warning_count": 0,
                        "issues": [
                            {
                                "severity": "error",
                                "code": "no_data",
                                "message": "No historical bars returned",
                                "symbol": symbol,
                                "affected_rows": 0,
                            }
                        ],
                    }

            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                data_quality_reports[symbol] = {
                    "symbol": symbol,
                    "rows": 0,
                    "has_errors": True,
                    "error_count": 1,
                    "warning_count": 0,
                    "issues": [
                        {
                            "severity": "error",
                            "code": "data_load_error",
                            "message": f"Failed to load data: {e}",
                            "symbol": symbol,
                            "affected_rows": 0,
                        }
                    ],
                }

        # Load all symbols in parallel for faster data fetching
        await asyncio.gather(
            *[_load_symbol_data(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        # Instantiate strategy with backtest broker and symbols
        params = {"symbols": symbols}
        if strategy_params:
            params.update(strategy_params)
        strategy = strategy_class(broker=backtest_broker, parameters=params)

        # Initialize the strategy if it has an initialize method
        if hasattr(strategy, "initialize"):
            try:
                await strategy.initialize()
            except Exception as e:
                logger.warning(f"Strategy initialization warning: {e}")

        # Track equity curve
        equity_curve = [initial_capital]

        # Generate trading days
        current_date = start_dt
        trading_days = []
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Skip weekends
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"Running backtest over {len(trading_days)} trading days...")

        # Run day by day
        for day_num, current_date in enumerate(trading_days):
            try:
                # Update the backtest broker's current date for price lookups
                backtest_broker._current_date = current_date

                # ==========================================
                # GAP RISK MODELING: Process overnight gaps
                # ==========================================
                # Before processing today's signals, check if any positions
                # were affected by overnight gaps (stops gapped through, etc.)
                if day_num > 0:  # Skip first day (no previous close)
                    gap_events = backtest_broker.process_day_start_gaps(current_date)
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

                # SURVIVORSHIP BIAS CORRECTION: Filter to only tradeable symbols
                # This prevents backtesting on symbols that weren't actually tradeable
                # on this date (IPO hadn't happened, stock was delisted, etc.)
                tradeable_symbols = historical_universe.get_tradeable_symbols(
                    current_date.date(), symbols
                )

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
                        if decision_log_writer:
                            decision_log_writer.write(
                                {
                                    "event_type": "decision",
                                    "run_id": run_id,
                                    "date": current_date.date().isoformat(),
                                    "error": str(event),
                                }
                            )
                        continue

                    if not isinstance(event, dict):
                        continue

                    decision_event_count += 1
                    if event.get("error"):
                        decision_error_count += 1

                    if decision_log_writer:
                        decision_log_writer.write(
                            {
                                "run_id": run_id,
                                "date": current_date.date().isoformat(),
                                "strategy": strategy_class.__name__,
                                "execution_profile": execution_profile,
                                **event,
                            }
                        )

                # Record equity at end of day
                portfolio_value = backtest_broker.get_portfolio_value(current_date)
                equity_curve.append(portfolio_value)

                if decision_log_writer:
                    decision_log_writer.write(
                        {
                            "event_type": "portfolio_snapshot",
                            "run_id": run_id,
                            "date": current_date.date().isoformat(),
                            "equity": portfolio_value,
                            "cash": backtest_broker.get_balance(),
                            "open_positions": len(backtest_broker.get_positions()),
                        }
                    )

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

        # Process trades to calculate P&L
        trades = backtest_broker.get_trades()
        trade_records = self._calculate_trade_pnl(trades)

        final_equity = equity_curve[-1] if equity_curve else initial_capital
        total_return = (final_equity / initial_capital) - 1

        logger.info(f"Backtest complete: Final equity = ${final_equity:,.2f} ({total_return:+.2%})")
        logger.info(f"Total trades: {len(trade_records)}")
        stress_test = run_portfolio_stress_test(
            backtest_broker.get_positions(),
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

        for trade in trade_records:
            if trades_log_writer:
                trades_log_writer.write(
                    {
                        "event_type": "trade",
                        "run_id": run_id,
                        "date": str(trade.get("timestamp", ""))[:10],
                        **trade,
                    }
                )

        order_records = (
            backtest_broker.get_orders() if hasattr(backtest_broker, "get_orders") else []
        )
        for order in order_records:
            if trades_log_writer:
                trades_log_writer.write(
                    {
                        "event_type": "order",
                        "run_id": run_id,
                        "date": str(order.get("created_at", ""))[:10],
                        **order,
                    }
                )

        result = {
            "equity_curve": equity_curve,
            "trades": trade_records,
            "start_date": start_dt,
            "end_date": end_dt,
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "positions": backtest_broker.get_positions(),
            "total_trades": len(trade_records),
            "stress_test": stress_test,
            "run_metadata": {
                "run_id": run_id,
                "started_at": run_started_at.isoformat(),
                "completed_at": run_completed_at.isoformat(),
                "persist_artifacts": persist_artifacts,
                "artifacts_dir": str(run_dir) if run_dir else None,
                "decision_events": decision_event_count,
                "decision_errors": decision_error_count,
                "execution_profile": execution_profile,
            },
            "data_quality": {
                "reports": data_quality_reports,
                "symbols_loaded": len(backtest_broker.price_data),
                "symbols_requested": len(symbols),
                "symbols_rejected": len(symbols) - len(backtest_broker.price_data),
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

        if run_dir is not None:
            summary_payload = {
                "run_id": run_id,
                "strategy": strategy_class.__name__,
                "symbols": symbols,
                "start_date": start_dt,
                "end_date": end_dt,
                "initial_capital": initial_capital,
                "final_equity": final_equity,
                "total_return": total_return,
                "total_trades": len(trade_records),
                "execution_profile": execution_profile,
                "decision_events": decision_event_count,
                "decision_errors": decision_error_count,
                "data_quality": result.get("data_quality", {}),
                "gap_statistics": result.get("gap_statistics", {}),
                "stress_test": result.get("stress_test", {}),
                "started_at": run_started_at,
                "completed_at": run_completed_at,
            }

            write_json(run_dir / "summary.json", summary_payload)
            write_json(
                run_dir / "manifest.json",
                {
                    "run_id": run_id,
                    "generated_at": run_completed_at,
                    "artifacts": {
                        "summary": "summary.json",
                        "decision_events": "decision_events.jsonl",
                        "trades": "trades.jsonl",
                    },
                },
            )
            result["run_metadata"].update(
                {
                    "summary_path": str(run_dir / "summary.json"),
                    "manifest_path": str(run_dir / "manifest.json"),
                    "decision_events_path": str(run_dir / "decision_events.jsonl"),
                    "trades_path": str(run_dir / "trades.jsonl"),
                }
            )

        if decision_log_writer:
            decision_log_writer.close()
        if trades_log_writer:
            trades_log_writer.close()

        return result

    def _calculate_trade_pnl(self, trades: List[Dict]) -> List[Dict]:
        """
        Calculate P&L for each trade by matching buys and sells.

        Args:
            trades: List of raw trade records

        Returns:
            List of trade records with P&L calculated
        """
        trade_records = []
        position_tracker = {}  # Track average entry price per symbol

        for trade in trades:
            symbol = trade["symbol"]
            side = trade["side"]
            quantity = trade["quantity"]
            price = trade["price"]

            if side == "buy":
                # Update position tracker with new buy
                if symbol not in position_tracker:
                    position_tracker[symbol] = {"qty": 0, "avg_price": 0}

                old_qty = position_tracker[symbol]["qty"]
                old_avg = position_tracker[symbol]["avg_price"]
                new_qty = old_qty + quantity

                if new_qty > 0:
                    position_tracker[symbol]["avg_price"] = (
                        old_qty * old_avg + quantity * price
                    ) / new_qty
                position_tracker[symbol]["qty"] = new_qty

                trade_records.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "timestamp": trade.get("timestamp"),
                        "pnl": 0,  # Buys don't have immediate P&L
                    }
                )

            else:  # sell
                pnl = 0
                if symbol in position_tracker and position_tracker[symbol]["qty"] > 0:
                    entry_price = position_tracker[symbol]["avg_price"]
                    pnl = (price - entry_price) * quantity
                    position_tracker[symbol]["qty"] -= quantity

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

    # =========================================================================
    # WALK-FORWARD BACKTEST MODE
    # =========================================================================

    async def run_walk_forward_backtest(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date,
        end_date,
        initial_capital: float = 100000,
        n_folds: int = 5,
        embargo_days: int = 5,
        train_pct: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Run a walk-forward backtest to detect overfitting.

        Walk-forward analysis splits data into multiple train/test periods,
        optimizes on training data, and validates on out-of-sample test data.
        This prevents curve-fitting and gives a realistic performance estimate.

        Args:
            strategy_class: Strategy class to instantiate and test
            symbols: List of symbols to trade
            start_date: Start date for backtest (date or datetime)
            end_date: End date for backtest (date or datetime)
            initial_capital: Starting capital per fold
            n_folds: Number of walk-forward folds
            embargo_days: Days to skip between train and test (prevents lookahead)
            train_pct: Percentage of each fold used for training (rest is test)

        Returns:
            Dictionary with walk-forward results including:
            - fold_results: Detailed results per fold
            - is_sharpe: Average in-sample Sharpe ratio
            - oos_sharpe: Average out-of-sample Sharpe ratio
            - degradation: Percentage degradation from IS to OOS
            - overfit_detected: True if OOS Sharpe < 50% of IS Sharpe
        """
        import numpy as np

        # Convert dates to datetime
        if hasattr(start_date, "strftime") and not hasattr(start_date, "hour"):
            start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            start_dt = start_date

        if hasattr(end_date, "strftime") and not hasattr(end_date, "hour"):
            end_dt = datetime.combine(end_date, datetime.min.time())
        else:
            end_dt = end_date

        logger.info(
            f"Running walk-forward backtest with {n_folds} folds, "
            f"{train_pct:.0%} train / {1-train_pct:.0%} test split, "
            f"{embargo_days} day embargo"
        )

        # Generate trading days
        current = start_dt
        trading_days = []
        while current <= end_dt:
            if current.weekday() < 5:  # Skip weekends
                trading_days.append(current)
            current += timedelta(days=1)

        total_days = len(trading_days)
        fold_size = total_days // n_folds

        logger.info(f"Total trading days: {total_days}, ~{fold_size} days per fold")

        # Results storage
        fold_results = []
        is_sharpes = []
        oos_sharpes = []
        is_returns = []
        oos_returns = []

        for fold_idx in range(n_folds):
            fold_start_idx = fold_idx * fold_size
            fold_end_idx = (
                min(fold_start_idx + fold_size, total_days)
                if fold_idx < n_folds - 1
                else total_days
            )

            # Split fold into train and test
            fold_days = trading_days[fold_start_idx:fold_end_idx]
            n_train = int(len(fold_days) * train_pct)

            train_days = fold_days[:n_train]
            # Add embargo period between train and test
            test_start_idx = n_train + embargo_days
            test_days = fold_days[test_start_idx:] if test_start_idx < len(fold_days) else []

            if len(train_days) < 20 or len(test_days) < 10:
                logger.warning(f"Fold {fold_idx + 1}: Insufficient data, skipping")
                continue

            train_start = train_days[0]
            train_end = train_days[-1]
            test_start = test_days[0]
            test_end = test_days[-1]

            logger.info(
                f"Fold {fold_idx + 1}/{n_folds}: "
                f"Train {train_start.date()} to {train_end.date()} ({len(train_days)} days), "
                f"Test {test_start.date()} to {test_end.date()} ({len(test_days)} days)"
            )

            # Run in-sample backtest
            try:
                is_result = await self.run_backtest(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=initial_capital,
                )
                is_equity = is_result.get("equity_curve", [initial_capital])
                is_return = (is_equity[-1] / is_equity[0]) - 1 if len(is_equity) > 1 else 0
                is_sharpe = self._calculate_sharpe_from_equity(is_equity)
            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} IS backtest failed: {e}")
                is_return = 0
                is_sharpe = 0
                is_result = {}

            # Run out-of-sample backtest
            try:
                oos_result = await self.run_backtest(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=initial_capital,
                )
                oos_equity = oos_result.get("equity_curve", [initial_capital])
                oos_return = (oos_equity[-1] / oos_equity[0]) - 1 if len(oos_equity) > 1 else 0
                oos_sharpe = self._calculate_sharpe_from_equity(oos_equity)
            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} OOS backtest failed: {e}")
                oos_return = 0
                oos_sharpe = 0
                oos_result = {}

            # Store fold results
            fold_result = {
                "fold": fold_idx + 1,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "train_days": len(train_days),
                "test_days": len(test_days),
                "is_return": is_return,
                "is_sharpe": is_sharpe,
                "oos_return": oos_return,
                "oos_sharpe": oos_sharpe,
                "is_trades": is_result.get("total_trades", 0),
                "oos_trades": oos_result.get("total_trades", 0),
            }
            fold_results.append(fold_result)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)
            is_returns.append(is_return)
            oos_returns.append(oos_return)

            logger.info(
                f"  IS: {is_return:+.2%} return, {is_sharpe:.2f} Sharpe | "
                f"OOS: {oos_return:+.2%} return, {oos_sharpe:.2f} Sharpe"
            )

        # Calculate aggregate metrics
        if not fold_results:
            logger.error("No valid folds completed")
            return {
                "fold_results": [],
                "is_sharpe": 0,
                "oos_sharpe": 0,
                "degradation": 0,
                "overfit_detected": True,
                "error": "No valid folds",
            }

        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
        avg_is_return = np.mean(is_returns) if is_returns else 0
        avg_oos_return = np.mean(oos_returns) if oos_returns else 0

        # Calculate degradation
        if avg_is_sharpe > 0:
            sharpe_degradation = 1 - (avg_oos_sharpe / avg_is_sharpe)
        else:
            sharpe_degradation = 0

        if avg_is_return > 0:
            return_degradation = 1 - (avg_oos_return / avg_is_return)
        else:
            return_degradation = 0

        # Detect overfitting: OOS Sharpe < 50% of IS Sharpe
        overfit_detected = avg_is_sharpe > 0 and avg_oos_sharpe < (avg_is_sharpe * 0.5)

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("WALK-FORWARD BACKTEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Folds completed: {len(fold_results)}/{n_folds}")
        logger.info(f"Average IS Sharpe:  {avg_is_sharpe:.2f}")
        logger.info(f"Average OOS Sharpe: {avg_oos_sharpe:.2f}")
        logger.info(f"Sharpe Degradation: {sharpe_degradation:.1%}")
        logger.info(f"Average IS Return:  {avg_is_return:+.2%}")
        logger.info(f"Average OOS Return: {avg_oos_return:+.2%}")
        logger.info(f"Return Degradation: {return_degradation:.1%}")

        if overfit_detected:
            logger.warning(
                "⚠️ OVERFITTING DETECTED: OOS Sharpe < 50% of IS Sharpe. "
                "Strategy may be curve-fitted to historical data."
            )
        else:
            logger.info("✓ No significant overfitting detected")

        logger.info(f"{'='*60}\n")

        return {
            "fold_results": fold_results,
            "n_folds": n_folds,
            "embargo_days": embargo_days,
            "train_pct": train_pct,
            "is_sharpe": avg_is_sharpe,
            "oos_sharpe": avg_oos_sharpe,
            "is_return": avg_is_return,
            "oos_return": avg_oos_return,
            "sharpe_degradation": sharpe_degradation,
            "return_degradation": return_degradation,
            "overfit_detected": overfit_detected,
            "overfit_threshold": 0.5,  # OOS < 50% of IS = overfit
        }

    def _calculate_sharpe_from_equity(
        self, equity_curve: List[float], risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio from an equity curve.

        Args:
            equity_curve: List of daily portfolio values
            risk_free_rate: Annual risk-free rate (default 0)

        Returns:
            Annualized Sharpe ratio
        """
        import numpy as np

        if len(equity_curve) < 2:
            return 0.0

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        sharpe = np.mean(excess_returns) / np.std(excess_returns)

        # Annualize
        return sharpe * np.sqrt(252)
