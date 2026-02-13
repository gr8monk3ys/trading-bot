#!/usr/bin/env python3
"""
Alpaca Trading Bot - Main Entry Point

This script initializes and runs the Alpaca trading bot with multiple strategies.
It handles strategy evaluation, selection, and execution based on market conditions.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path to allow imports from project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.alpaca_broker import AlpacaBroker
from config import RISK_PARAMS, SYMBOL_SELECTION, SYMBOLS, require_alpaca_credentials
from engine.strategy_manager import StrategyManager
from utils.circuit_breaker import CircuitBreaker
from utils.data_quality import (
    should_halt_trading_for_data_quality,
    summarize_quality_reports,
    validate_ohlcv_frame,
)
from utils.order_reconciliation import OrderReconciler
from utils.reconciliation import PositionReconciler
from utils.incident_tracker import IncidentTracker
from utils.run_artifacts import JsonlWriter, ensure_run_directory, generate_run_id, write_json
from utils.simple_symbol_selector import SimpleSymbolSelector
from utils.slo_alerting import build_slo_alert_notifier
from utils.slo_monitor import SLOMonitor

logger = logging.getLogger(__name__)


_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """Configure application logging exactly once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root_logger = logging.getLogger()
    if root_logger.handlers:
        _LOGGING_CONFIGURED = True
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"),
        ],
    )
    _LOGGING_CONFIGURED = True


async def run_walk_forward_validation(strategy_class, strategy_manager, symbols, start_date, end_date, args):
    """
    Run walk-forward validation to detect overfitting.

    Returns:
        Tuple of (passed: bool, results: dict)
    """
    from engine.walk_forward import WalkForwardValidator

    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 60)

    validator = WalkForwardValidator(
        train_ratio=args.wf_train_ratio,
        n_splits=args.wf_splits,
        min_train_days=30,
        gap_days=1,  # 1-day gap to prevent look-ahead
    )

    # Create backtest function wrapper
    async def backtest_fn(syms, start_str, end_str, **kwargs):
        s_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        e_date = datetime.strptime(end_str, "%Y-%m-%d").date()
        result = await strategy_manager.backtest_engine.run_backtest(
            strategy_class=strategy_class,
            symbols=syms,
            start_date=s_date,
            end_date=e_date,
            initial_capital=args.capital,
            execution_profile=args.execution_profile,
            persist_artifacts=False,
        )
        metrics = strategy_manager.perf_metrics.calculate_metrics(result)
        return metrics

    # Run validation
    validation_result = await validator.validate(
        backtest_fn,
        symbols=symbols,
        start_date_str=start_date.strftime("%Y-%m-%d"),
        end_date_str=end_date.strftime("%Y-%m-%d"),
    )

    # Analyze results
    avg_is_return = validation_result.get("avg_is_return", 0)
    avg_oos_return = validation_result.get("avg_oos_return", 0)
    avg_overfit_ratio = validation_result.get("avg_overfit_ratio", float('inf'))

    # Determine if validation passes
    passed = avg_overfit_ratio <= args.overfit_threshold

    # Print results
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 60)
    print(f"Average In-Sample Return:    {avg_is_return:.2%}")
    print(f"Average Out-of-Sample Return: {avg_oos_return:.2%}")
    print(f"Overfitting Ratio (IS/OOS):   {avg_overfit_ratio:.2f}")
    print(f"Threshold:                    {args.overfit_threshold:.2f}")
    print("-" * 60)

    if passed:
        print("✅ VALIDATION PASSED - Strategy shows acceptable out-of-sample performance")
    else:
        print("❌ VALIDATION FAILED - Strategy appears to be overfit")
        print(f"   In-sample performance is {avg_overfit_ratio:.1f}x better than out-of-sample")
        print("   This suggests the strategy may not perform well in live trading")

    print("=" * 60 + "\n")

    return passed, validation_result


async def run_backtest(args):
    """Run backtest mode with selected strategies."""
    strategy_manager = None
    try:
        logger.info(f"Starting backtest from {args.start_date} to {args.end_date}")

        # Initialize broker
        broker = AlpacaBroker(paper=True)

        # Initialize strategy manager
        strategy_manager = StrategyManager(broker=broker)

        # Get available strategies
        available_strategies = strategy_manager.get_available_strategy_names()
        logger.info(f"Available strategies: {available_strategies}")

        # Select strategies to backtest
        strategies_to_test = []
        if args.strategy == "all":
            strategies_to_test = available_strategies
        elif args.strategy in available_strategies:
            strategies_to_test = [args.strategy]
        else:
            logger.error(f"Strategy '{args.strategy}' not found")
            return

        # Convert date strings to datetime objects
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

        # Get symbols
        symbols = args.symbols.split(",") if args.symbols else SYMBOLS

        # Run walk-forward validation if requested (skip if using validated backtest)
        if args.walk_forward and not args.validated:
            for strategy_name in strategies_to_test:
                strategy_class = strategy_manager.available_strategies[strategy_name]

                passed, wf_results = await run_walk_forward_validation(
                    strategy_class, strategy_manager, symbols, start_date, end_date, args
                )

                if not passed and not args.force:
                    print(f"\n⚠️  Walk-forward validation FAILED for {strategy_name}")
                    print("Use --force to run backtest anyway, but be aware of overfitting risk.")
                    logger.warning(f"Walk-forward validation failed for {strategy_name}")
                    continue

        # Run backtests
        results = {}
        metrics = {}

        for strategy_name in strategies_to_test:
            try:
                logger.info(f"Backtesting strategy: {strategy_name}")

                # Get strategy class
                strategy_class = strategy_manager.available_strategies[strategy_name]

                if args.validated:
                    from engine.validated_backtest import (
                        ValidatedBacktestRunner,
                        format_validated_backtest_report,
                    )

                    runner = ValidatedBacktestRunner(broker)
                    validated_result = await runner.run_validated_backtest(
                        strategy_class=strategy_class,
                        symbols=args.symbols.split(",") if args.symbols else SYMBOLS,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        initial_capital=args.capital,
                    )

                    print(format_validated_backtest_report(validated_result))

                    if not validated_result.eligible_for_trading and not args.force:
                        print(
                            "Profitability gates FAILED. Use --force to proceed anyway."
                        )
                        continue

                    results[strategy_name] = {
                        "equity_curve": validated_result.equity_curve,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                    metrics[strategy_name] = {
                        "total_return": validated_result.total_return,
                        "annualized_return": validated_result.total_return,
                        "sharpe_ratio": validated_result.sharpe_ratio,
                        "max_drawdown": validated_result.max_drawdown,
                        "win_rate": validated_result.win_rate,
                        "profit_factor": None,
                        "num_trades": validated_result.num_trades,
                    }
                else:
                    # Run backtest
                    result = await strategy_manager.backtest_engine.run_backtest(
                        strategy_class=strategy_class,
                        symbols=args.symbols.split(",") if args.symbols else SYMBOLS,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=args.capital,
                        execution_profile=args.execution_profile,
                        run_id=args.run_id,
                        persist_artifacts=not args.no_run_artifacts,
                        artifacts_dir=args.artifacts_dir,
                    )

                    # Calculate metrics
                    strategy_metrics = strategy_manager.perf_metrics.calculate_metrics(result)

                    # Store results
                    results[strategy_name] = result
                    metrics[strategy_name] = strategy_metrics

                    # Print summary
                    print(f"\n--- {strategy_name} Performance Summary ---")
                    print(f"Total Return: {strategy_metrics['total_return']:.2%}")
                    print(f"Annualized Return: {strategy_metrics['annualized_return']:.2%}")
                    print(f"Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {strategy_metrics['max_drawdown']:.2%}")
                    print(f"Win Rate: {strategy_metrics['win_rate']:.2%}")
                    print(f"Average Win: {strategy_metrics['avg_win']:.2%}")
                    print(f"Average Loss: {strategy_metrics['avg_loss']:.2%}")
                    print(f"Profit Factor: {strategy_metrics['profit_factor']:.2f}")
                    print(f"Number of Trades: {strategy_metrics['num_trades']}")

            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {e}", exc_info=True)

        # Compare strategies if more than one was tested
        if len(metrics) > 1:
            print("\n--- Strategy Comparison ---")
            comparison = pd.DataFrame(
                {
                    k: {
                        "Total Return": v["total_return"],
                        "Annualized Return": v["annualized_return"],
                        "Sharpe Ratio": v["sharpe_ratio"],
                        "Max Drawdown": v["max_drawdown"],
                        "Win Rate": v["win_rate"],
                        "Profit Factor": v["profit_factor"],
                        "Number of Trades": v["num_trades"],
                    }
                    for k, v in metrics.items()
                }
            ).T

            pd.set_option("display.float_format", "{:.2%}".format)
            numeric_cols = ["Sharpe Ratio", "Profit Factor", "Number of Trades"]
            for col in numeric_cols:
                comparison[col] = pd.to_numeric(comparison[col])
                pd.set_option("display.float_format", "{:.2f}".format)

            print(comparison)

            # Generate plots if requested
            if args.plot:
                try:
                    import matplotlib.pyplot as plt

                    # Plot equity curves
                    plt.figure(figsize=(12, 6))
                    for name, result in results.items():
                        curve = result.get("equity_curve") or result.get("portfolio_value")
                        if curve is None:
                            continue
                        plt.plot(curve, label=name)

                    plt.title("Equity Curves")
                    plt.xlabel("Date")
                    plt.ylabel("Portfolio Value")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig("backtest_equity_curves.png")
                    plt.close()

                    logger.info("Generated equity curve plot: backtest_equity_curves.png")

                except Exception as e:
                    logger.error(f"Error generating plots: {e}")

        logger.info("Backtest completed")

    except Exception as e:
        logger.error(f"Error in backtest mode: {e}", exc_info=True)
    finally:
        if strategy_manager is not None:
            strategy_manager.close()


async def select_trading_symbols(broker):
    """
    Select symbols for trading - either dynamic or static.

    Returns list of symbols to trade.
    """
    if SYMBOL_SELECTION.get("USE_DYNAMIC_SELECTION", False):
        logger.info("🔍 Using dynamic symbol selection...")
        try:
            creds = require_alpaca_credentials("dynamic symbol selection")
            selector = SimpleSymbolSelector(
                api_key=str(creds["API_KEY"]),
                secret_key=str(creds["API_SECRET"]),
                paper=bool(creds["PAPER"]),
            )
            symbols = selector.select_top_symbols(
                top_n=SYMBOL_SELECTION.get("TOP_N_SYMBOLS", 20),
                min_score=SYMBOL_SELECTION.get("MIN_MOMENTUM_SCORE", 1.0),
            )
            logger.info(f"✅ Dynamic selection: {len(symbols)} symbols selected")
            return symbols
        except Exception as e:
            logger.error(f"Dynamic selection failed, falling back to static list: {e}")
            return SYMBOLS
    else:
        logger.info(f"Using static symbol list: {len(SYMBOLS)} symbols")
        return SYMBOLS


async def run_live(args):
    """Run live trading mode."""
    strategy_manager = None
    session_run_id = args.run_id or generate_run_id("live")
    session_run_dir = ensure_run_directory(args.artifacts_dir, session_run_id)
    slo_monitor = None
    data_quality_writer = None
    try:
        logger.info("Starting live trading mode")
        logger.info(f"Live session run_id={session_run_id}")
        logger.info(f"Live artifacts directory={session_run_dir}")

        # P0 FIX: Safety confirmation for live trading with real money
        paper = not args.real
        if not paper:  # Real money trading
            print("\n" + "=" * 60)
            print("⚠️  WARNING: LIVE TRADING WITH REAL MONEY")
            print("=" * 60)
            print("\nYou are about to start trading with REAL MONEY.")
            print("This will execute actual trades on your Alpaca account.")
            print("\nBefore proceeding, confirm you understand:")
            print("  1. Losses can occur and are your responsibility")
            print("  2. The bot will trade autonomously")
            print("  3. This is NOT paper trading mode")
            print("\n" + "=" * 60)

            confirmation = input("\nType 'CONFIRM LIVE TRADING' to proceed: ")
            if confirmation != "CONFIRM LIVE TRADING":
                print("\n❌ Live trading cancelled. Use paper trading for testing.")
                print("   Run without --real flag for paper trading mode.")
                logger.warning("Live trading cancelled by user - confirmation not provided")
                return

            print("\n✅ Confirmation received. Starting live trading...")
            logger.warning("=" * 40)
            logger.warning("LIVE TRADING MODE ACTIVATED - REAL MONEY AT RISK")
            logger.warning("=" * 40)
        else:
            logger.info("Paper trading mode - no real money at risk")

        # Initialize broker
        broker = AlpacaBroker(paper=paper)
        await broker.start_websocket()
        logger.info("Websocket started (trade updates enabled)")

        # Select symbols for trading (dynamic or static)
        trading_symbols = await select_trading_symbols(broker)
        logger.info(
            f"Trading universe: {', '.join(trading_symbols[:10])}"
            + (f" ... and {len(trading_symbols)-10} more" if len(trading_symbols) > 10 else "")
        )

        # Initialize circuit breaker (CRITICAL SAFETY FEATURE)
        circuit_breaker = CircuitBreaker(
            max_daily_loss=0.03,  # 3% max daily loss
            auto_close_positions=True,  # Automatically close positions on trigger
        )
        await circuit_breaker.initialize(broker)
        logger.info("Circuit breaker initialized and armed")

        # Check if market is open
        market_status = await broker.get_market_status()
        logger.info(f"Market status: {market_status}")

        if not market_status.get("is_open", False) and not args.force:
            logger.warning("Market is closed. Use --force to run anyway.")
            print("Market is closed. Use --force to run anyway.")
            return

        # Initialize strategy manager
        strategy_manager = StrategyManager(
            broker=broker,
            max_strategies=args.max_strategies,
            max_allocation=args.max_allocation,
            circuit_breaker=circuit_breaker,
        )

        # Sync internal position ownership with broker on startup (restart recovery)
        await strategy_manager.position_manager.sync_with_broker(
            broker, default_strategy="recovered"
        )

        # Reconciliation loop (detect broker/internal drift)
        data_quality_writer = JsonlWriter(session_run_dir / "data_quality_events.jsonl")
        reconciler = PositionReconciler(
            broker=broker,
            internal_tracker=strategy_manager.position_manager,
            halt_on_mismatch=False,
            sync_to_broker=True,
            audit_log=strategy_manager.audit_log,
            events_path=session_run_dir / "position_reconciliation_events.jsonl",
            run_id=session_run_id,
        )
        order_reconciler = OrderReconciler(
            broker=broker,
            lifecycle_tracker=strategy_manager.order_gateway.lifecycle_tracker,
            audit_log=strategy_manager.audit_log,
            mismatch_halt_threshold=RISK_PARAMS.get("ORDER_RECON_MISMATCH_HALT_RUNS", 3),
            events_path=session_run_dir / "order_reconciliation_events.jsonl",
            run_id=session_run_id,
        )
        incident_tracker = IncidentTracker(
            events_path=session_run_dir / "incident_events.jsonl",
            run_id=session_run_id,
            ack_sla_minutes=RISK_PARAMS.get("INCIDENT_ACK_SLA_MINUTES", 15),
        )
        slo_alert_notifier = build_slo_alert_notifier(RISK_PARAMS, source="main.run_live")
        if slo_alert_notifier:
            logger.info(
                "SLO paging alerts enabled (severity>=%s)",
                RISK_PARAMS.get("SLO_PAGING_MIN_SEVERITY", "critical"),
            )
        slo_monitor = SLOMonitor(
            audit_log=strategy_manager.audit_log,
            events_path=session_run_dir / "ops_slo_events.jsonl",
            alert_notifier=slo_alert_notifier,
            incident_tracker=incident_tracker,
            recon_mismatch_halt_runs=RISK_PARAMS.get("ORDER_RECON_MISMATCH_HALT_RUNS", 3),
            max_data_quality_errors=RISK_PARAMS.get("DATA_QUALITY_MAX_ERRORS", 0),
            max_stale_data_warnings=RISK_PARAMS.get("DATA_QUALITY_MAX_STALE_WARNINGS", 0),
            shadow_drift_warning_threshold=RISK_PARAMS.get("PAPER_LIVE_SHADOW_DRIFT_WARNING", 0.12),
            shadow_drift_critical_threshold=RISK_PARAMS.get("PAPER_LIVE_SHADOW_DRIFT_MAX", 0.15),
        )
        quality_symbols = (
            [s.strip() for s in args.symbols.split(",") if s.strip()]
            if args.symbols
            else list(trading_symbols)
        )

        async def run_data_quality_gate():
            reference_time = datetime.utcnow()
            start = (reference_time - timedelta(days=30)).strftime("%Y-%m-%d")
            end = reference_time.strftime("%Y-%m-%d")
            stale_after_days = RISK_PARAMS.get("DATA_QUALITY_STALE_AFTER_DAYS", 3)
            reports = []

            for symbol in quality_symbols:
                try:
                    bars = await broker.get_bars(
                        symbol,
                        start=start,
                        end=end,
                        timeframe="1Day",
                    )
                    if not bars:
                        reports.append(
                            {
                                "symbol": symbol,
                                "rows": 0,
                                "error_count": 1,
                                "warning_count": 0,
                                "issues": [
                                    {
                                        "severity": "error",
                                        "code": "no_data",
                                        "message": "No bars returned in data quality gate",
                                    }
                                ],
                            }
                        )
                        continue

                    frame = pd.DataFrame(
                        {
                            "open": [float(b.open) for b in bars],
                            "high": [float(b.high) for b in bars],
                            "low": [float(b.low) for b in bars],
                            "close": [float(b.close) for b in bars],
                            "volume": [float(b.volume) for b in bars],
                        },
                        index=pd.DatetimeIndex([b.timestamp for b in bars]),
                    )
                    reports.append(
                        validate_ohlcv_frame(
                            frame,
                            symbol=symbol,
                            stale_after_days=stale_after_days,
                            reference_time=reference_time,
                        )
                    )
                except Exception as e:
                    reports.append(
                        {
                            "symbol": symbol,
                            "rows": 0,
                            "error_count": 1,
                            "warning_count": 0,
                            "issues": [
                                {
                                    "severity": "error",
                                    "code": "data_quality_exception",
                                    "message": str(e),
                                }
                            ],
                        }
                    )

            summary = summarize_quality_reports(reports)
            payload = {
                "event_type": "data_quality_snapshot",
                "run_id": session_run_id,
                "timestamp": reference_time.isoformat(),
                **summary,
            }
            write_json(session_run_dir / "data_quality_latest.json", payload)
            if data_quality_writer:
                data_quality_writer.write(payload)

            if slo_monitor:
                breaches = slo_monitor.record_data_quality_summary(summary)
                if SLOMonitor.has_critical_breach(breaches):
                    strategy_manager.order_gateway.activate_kill_switch(
                        reason="Critical data quality SLO breach",
                        source="slo_monitor",
                    )
                    return

            should_halt, reason = should_halt_trading_for_data_quality(
                summary,
                max_errors=RISK_PARAMS.get("DATA_QUALITY_MAX_ERRORS", 0),
                max_stale_warnings=RISK_PARAMS.get("DATA_QUALITY_MAX_STALE_WARNINGS", 0),
            )
            if should_halt and reason:
                strategy_manager.order_gateway.activate_kill_switch(
                    reason=reason,
                    source="data_quality",
                )

        async def periodic_housekeeping():
            state_interval = 60
            reconciliation_interval = 300
            order_recon_interval = 120
            counter = 0
            try:
                while True:
                    await asyncio.sleep(1)
                    counter += 1
                    if counter % state_interval == 0:
                        await strategy_manager.save_runtime_state()
                        if slo_monitor:
                            slo_monitor.check_incident_ack_sla()
                    if counter % reconciliation_interval == 0:
                        try:
                            await reconciler.reconcile()
                        except Exception as e:
                            logger.error(f"Reconciliation error: {e}")
                        try:
                            await run_data_quality_gate()
                        except Exception as e:
                            logger.error(f"Data quality gate error: {e}")
                    if counter % order_recon_interval == 0:
                        try:
                            await order_reconciler.reconcile()
                            if slo_monitor:
                                breaches = slo_monitor.record_order_reconciliation_health(
                                    order_reconciler.get_health_snapshot()
                                )
                                if SLOMonitor.has_critical_breach(breaches):
                                    strategy_manager.order_gateway.activate_kill_switch(
                                        reason="Critical order reconciliation SLO breach",
                                        source="slo_monitor",
                                    )
                            if order_reconciler.should_halt_trading():
                                health = order_reconciler.get_health_snapshot()
                                reason = (
                                    health.get("halt_reason")
                                    or "Order reconciliation drift threshold breached"
                                )
                                strategy_manager.order_gateway.activate_kill_switch(
                                    reason=reason,
                                    source="order_reconciliation",
                                )
                        except Exception as e:
                            logger.error(f"Order reconciliation error: {e}")
            except asyncio.CancelledError:
                return

        housekeeping_task = asyncio.create_task(periodic_housekeeping())

        # Get available strategies
        available_strategies = strategy_manager.get_available_strategy_names()
        logger.info(f"Available strategies: {available_strategies}")

        # Select strategies to run
        strategies_to_run = []
        if args.strategy == "auto":
            # Auto-select the best strategies
            logger.info("Auto-selecting the best strategies...")
            await strategy_manager.evaluate_all_strategies()
            strategies_to_run = await strategy_manager.select_top_strategies(
                n=args.max_strategies, min_score=args.min_score
            )
        elif args.strategy == "all":
            strategies_to_run = available_strategies
        elif args.strategy in available_strategies:
            strategies_to_run = [args.strategy]
        else:
            logger.error(f"Strategy '{args.strategy}' not found")
            return

        logger.info(f"Selected strategies: {strategies_to_run}")

        # Optimize allocations if auto-allocation
        if args.auto_allocate:
            allocations = await strategy_manager.optimize_allocations(strategies_to_run)
            logger.info(f"Optimized allocations: {allocations}")
        else:
            # Equal allocation
            equal_alloc = args.max_allocation / len(strategies_to_run)
            allocations = dict.fromkeys(strategies_to_run, equal_alloc)
            logger.info(f"Equal allocations: {allocations}")

        # Start strategies
        started = []
        for strategy_name in strategies_to_run:
            allocation = allocations.get(strategy_name, 0.1)
            # Use command-line symbols if provided, otherwise use dynamically selected symbols
            symbols_to_use = args.symbols.split(",") if args.symbols else trading_symbols
            success = await strategy_manager.start_strategy(
                strategy_name=strategy_name, allocation=allocation, symbols=symbols_to_use
            )
            if success:
                started.append(strategy_name)

        if not started:
            logger.error("Failed to start any strategies")
            return

        logger.info(f"Started {len(started)} strategies: {started}")

        # Set up periodic tasks
        async def periodic_evaluation():
            """Run periodic strategy evaluation and rebalancing."""
            try:
                while True:
                    # Wait for the next evaluation period
                    await asyncio.sleep(args.evaluation_interval * 3600)  # Convert hours to seconds

                    logger.info("Running periodic strategy evaluation and rebalancing")

                    # Re-evaluate strategies
                    if args.strategy == "auto":
                        scores = await strategy_manager.evaluate_all_strategies()
                        logger.info(f"Updated strategy scores: {scores}")

                        # Re-optimize allocations
                        if args.auto_allocate:
                            allocations = await strategy_manager.optimize_allocations()
                            logger.info(f"Updated allocations: {allocations}")

                            # Rebalance
                            await strategy_manager.rebalance_strategies()

                    # Generate performance report
                    report = await strategy_manager.generate_performance_report(days=7)
                    logger.info(f"Weekly performance report: {report}")

            except Exception as e:
                logger.error(f"Error in periodic evaluation: {e}", exc_info=True)

        # Start periodic evaluation task
        evaluation_task = asyncio.create_task(periodic_evaluation())

        # Keep running until interrupted
        try:
            logger.info("Trading bot running. Press Ctrl+C to stop.")
            check_counter = 0
            last_rebalance_hour = datetime.now().hour

            while True:
                await asyncio.sleep(1)

                # Check circuit breaker every 60 seconds
                check_counter += 1
                if check_counter >= 60:
                    check_counter = 0

                    # CRITICAL: Check if circuit breaker triggered
                    if await circuit_breaker.check_and_halt():
                        logger.critical("=" * 80)
                        logger.critical("CIRCUIT BREAKER TRIGGERED - TRADING HALTED")
                        logger.critical("Daily loss limit exceeded. Stopping all strategies.")
                        logger.critical("=" * 80)

                        # Stop all strategies immediately
                        await strategy_manager.stop_all_strategies(liquidate=True)
                        logger.critical("All strategies stopped and positions liquidated")

                        # Exit the trading loop
                        break

                # Portfolio rebalancing every 4 hours
                current_hour = datetime.now().hour
                if (
                    len(started) > 1
                    and current_hour % 4 == 0
                    and current_hour != last_rebalance_hour
                ):
                    last_rebalance_hour = current_hour
                    try:
                        logger.info("⚖️  Running portfolio rebalancing...")
                        await strategy_manager.rebalance_strategies()
                        logger.info("✅ Portfolio rebalanced successfully")
                    except Exception as e:
                        logger.error(f"Error during rebalancing: {e}")

        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        finally:
            # Cancel evaluation task
            evaluation_task.cancel()
            housekeeping_task.cancel()

            # Stop all strategies
            await strategy_manager.stop_all_strategies(liquidate=args.liquidate_on_exit)
            logger.info("All strategies stopped")

            # Persist runtime state on shutdown
            try:
                await strategy_manager.save_runtime_state()
            except Exception as e:
                logger.error(f"Error saving runtime state: {e}")

            # Stop websocket
            try:
                await broker.stop_websocket()
            except Exception as e:
                logger.error(f"Error stopping websocket: {e}")

            try:
                reconciler.close()
            except Exception:
                pass
            try:
                order_reconciler.close()
            except Exception:
                pass
            if slo_monitor:
                slo_monitor.close()
            if data_quality_writer:
                data_quality_writer.close()

            write_json(
                session_run_dir / "live_session_summary.json",
                {
                    "run_id": session_run_id,
                    "strategy_mode": args.strategy,
                    "paper": paper,
                    "symbols": quality_symbols,
                    "ended_at": datetime.utcnow().isoformat(),
                    "slo_status": slo_monitor.get_status_snapshot() if slo_monitor else {},
                },
            )

    except Exception as e:
        logger.error(f"Error in live trading mode: {e}", exc_info=True)
    finally:
        if slo_monitor:
            slo_monitor.close()
        if data_quality_writer:
            data_quality_writer.close()
        if strategy_manager is not None:
            strategy_manager.close()


async def optimize_parameters(args):
    """Optimize strategy parameters."""
    strategy_manager = None
    try:
        logger.info(f"Starting parameter optimization for strategy: {args.strategy}")

        # Initialize broker
        broker = AlpacaBroker(paper=True)

        # Initialize strategy manager
        strategy_manager = StrategyManager(broker=broker)

        # Get available strategies
        available_strategies = strategy_manager.get_available_strategy_names()

        if args.strategy not in available_strategies:
            logger.error(f"Strategy '{args.strategy}' not found")
            return

        # Get strategy class
        strategy_class = strategy_manager.available_strategies[args.strategy]

        # Create strategy instance to get default parameters
        temp_strategy = strategy_class(broker=broker, symbols=[])
        default_params = temp_strategy.default_parameters()

        # Parse parameter ranges
        param_ranges = {}
        if args.param_ranges:
            try:
                param_ranges = json.loads(args.param_ranges)
            except json.JSONDecodeError:
                logger.error("Invalid JSON format for parameter ranges")
                return

        # If no ranges specified, use defaults with some variation
        if not param_ranges:
            for param, value in default_params.items():
                # Only optimize numeric parameters
                if isinstance(value, (int, float)) and param != "allocation":
                    if isinstance(value, int):
                        param_ranges[param] = {
                            "min": max(1, int(value * 0.5)),
                            "max": int(value * 1.5),
                            "step": 1,
                        }
                    else:
                        param_ranges[param] = {
                            "min": value * 0.5,
                            "max": value * 1.5,
                            "step": value * 0.1,
                        }

        logger.info(f"Parameter ranges to optimize: {param_ranges}")

        # Convert date strings to datetime objects
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

        symbols = args.symbols.split(",") if args.symbols else SYMBOLS

        # Generate parameter combinations
        import itertools

        param_values = {}
        for param, range_info in param_ranges.items():
            min_val = range_info["min"]
            max_val = range_info["max"]
            step = range_info["step"]

            if isinstance(min_val, int) and isinstance(max_val, int):
                values = list(range(min_val, max_val + 1, step))
            else:
                # Generate float range
                values = []
                val = min_val
                while val <= max_val:
                    values.append(val)
                    val += step

            param_values[param] = values

        # Generate combinations
        param_names = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[p] for p in param_names]))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        # Run backtests for each combination
        results = []

        for i, combo in enumerate(combinations):
            params = default_params.copy()
            for j, param in enumerate(param_names):
                params[param] = combo[j]

            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")

            # Run backtest
            result = await strategy_manager.backtest_engine.run_backtest(
                strategy_class=strategy_class,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=args.capital,
                strategy_params=params,
                execution_profile=args.execution_profile,
                persist_artifacts=False,
            )

            # Calculate metrics
            metrics = strategy_manager.perf_metrics.calculate_metrics(result)

            # Store results
            results.append({"params": params, "metrics": metrics})

        # Find best parameters
        if args.optimize_for == "sharpe":
            best_result = max(results, key=lambda x: x["metrics"]["sharpe_ratio"])
        elif args.optimize_for == "return":
            best_result = max(results, key=lambda x: x["metrics"]["total_return"])
        elif args.optimize_for == "drawdown":
            best_result = min(results, key=lambda x: x["metrics"]["max_drawdown"])
        else:
            best_result = max(results, key=lambda x: x["metrics"]["sharpe_ratio"])

        # Print results
        print("\n--- Parameter Optimization Results ---")
        print(f"Best parameters for {args.strategy} optimized for {args.optimize_for}:")
        for param, value in best_result["params"].items():
            if param in param_ranges:
                print(f"{param}: {value}")

        print("\nPerformance with optimized parameters:")
        print(f"Total Return: {best_result['metrics']['total_return']:.2%}")
        print(f"Annualized Return: {best_result['metrics']['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {best_result['metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {best_result['metrics']['max_drawdown']:.2%}")
        print(f"Win Rate: {best_result['metrics']['win_rate']:.2%}")

        # Save results to file
        output_file = f"{args.strategy}_optimized_params.json"
        with open(output_file, "w") as f:
            json.dump(
                {"optimized_params": best_result["params"], "performance": best_result["metrics"]},
                f,
                indent=4,
                default=str,
            )

        logger.info(f"Saved optimized parameters to {output_file}")

    except Exception as e:
        logger.error(f"Error in parameter optimization: {e}", exc_info=True)
    finally:
        if strategy_manager is not None:
            strategy_manager.close()


def _load_json_object(value: str | None, field_name: str) -> dict[str, Any]:
    """Load JSON object from inline JSON string or file path."""
    if not value:
        return {}

    raw = value
    try:
        candidate = Path(value)
        if candidate.exists() and candidate.is_file():
            raw = candidate.read_text(encoding="utf-8")
    except OSError:
        # Inline JSON payloads can exceed filesystem name limits.
        raw = value

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {field_name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")

    return payload


def _parse_csv_list(value: str | None) -> list[str]:
    """Parse comma-separated values into a cleaned list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def run_research(args) -> int:
    """
    Run research registry operations from a single CLI entry point.

    Returns:
        Process-style exit code (0 success, non-zero failure)
    """
    from research.research_registry import ResearchRegistry, print_experiment_summary

    registry = ResearchRegistry(
        registry_path=args.research_registry_path,
        production_path=args.research_production_path,
        parameter_registry_path=args.research_parameter_registry_path,
        artifacts_path=args.research_artifacts_path,
    )

    action = args.research_action

    if action == "create":
        if not args.name or not args.description or not args.author:
            raise ValueError("create action requires --name, --description, and --author")
        exp_id = registry.create_experiment(
            name=args.name,
            description=args.description,
            author=args.author,
            parameters=_load_json_object(args.parameters_json, "parameters_json"),
            tags=_parse_csv_list(args.tags),
        )
        print(f"Experiment created: {exp_id}")
        print_experiment_summary(registry, exp_id)
        return 0

    if not args.experiment_id:
        raise ValueError(f"{action} action requires --experiment-id")

    if action == "snapshot":
        if not args.parameters_json:
            raise ValueError("snapshot action requires --parameters-json")
        snapshot = registry.record_parameter_snapshot(
            args.experiment_id,
            parameters=_load_json_object(args.parameters_json, "parameters_json"),
            source=args.source,
            notes=args.notes or "",
            make_active=not args.no_make_active,
        )
        print(
            f"Recorded parameter snapshot for {args.experiment_id}: "
            f"{snapshot['snapshot_id']} ({snapshot['hash']})"
        )
        return 0

    if action == "record-backtest":
        if not args.backtest_json:
            raise ValueError("record-backtest action requires --backtest-json")
        registry.record_backtest_results(
            args.experiment_id,
            _load_json_object(args.backtest_json, "backtest_json"),
        )
        print(f"Recorded backtest results for {args.experiment_id}")
        return 0

    if action == "record-validation":
        if not args.validation_json:
            raise ValueError("record-validation action requires --validation-json")
        registry.record_validation_results(
            args.experiment_id,
            _load_json_object(args.validation_json, "validation_json"),
        )
        print(f"Recorded validation results for {args.experiment_id}")
        return 0

    if action == "record-paper":
        if not args.paper_json:
            raise ValueError("record-paper action requires --paper-json")
        registry.record_paper_results(
            args.experiment_id,
            _load_json_object(args.paper_json, "paper_json"),
        )
        print(f"Recorded paper results for {args.experiment_id}")
        return 0

    if action == "store-walk-forward":
        if not args.walk_forward_json:
            raise ValueError("store-walk-forward action requires --walk-forward-json")
        paths = registry.store_walk_forward_artifacts(
            args.experiment_id,
            _load_json_object(args.walk_forward_json, "walk_forward_json"),
            source_run_id=args.source_run_id,
        )
        print(f"Stored walk-forward artifacts for {args.experiment_id}")
        print(f"  - {paths['results_path']}")
        print(f"  - {paths['summary_path']}")
        return 0

    if action == "approve-review":
        if not args.reviewer:
            raise ValueError("approve-review action requires --reviewer")
        registry.approve_manual_review(
            args.experiment_id,
            reviewer=args.reviewer,
            notes=args.notes or "",
        )
        print(f"Approved manual review for {args.experiment_id} by {args.reviewer}")
        return 0

    if action == "summary":
        print_experiment_summary(registry, args.experiment_id)
        return 0

    checklist = registry.generate_promotion_checklist(args.experiment_id)
    blockers = registry.get_promotion_blockers(args.experiment_id, strict=args.strict)
    ready = registry.is_promotion_ready(args.experiment_id, strict=args.strict)

    if action == "blockers":
        print(f"Promotion blockers for {args.experiment_id}:")
        if blockers:
            for blocker in blockers:
                print(f"  - {blocker}")
        else:
            print("  none")
        return 0 if not blockers else 1

    if action == "check":
        payload = {
            "experiment_id": args.experiment_id,
            "strict": args.strict,
            "ready": ready,
            "blockers": blockers,
            "checklist": checklist,
            "summary": registry.get_experiment_summary(args.experiment_id),
        }
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(
            f"Promotion check for {args.experiment_id}: "
            f"{'READY' if ready else 'NOT READY'} (strict={args.strict})"
        )
        if blockers:
            print("Blockers:")
            for blocker in blockers:
                print(f"  - {blocker}")
        return 0 if ready else 1

    if action == "promote":
        if args.strict and not args.force and not ready:
            print(
                f"Strict promotion blocked for {args.experiment_id}: "
                f"{len(blockers)} blockers"
            )
            for blocker in blockers:
                print(f"  - {blocker}")
            return 1
        promoted = registry.promote_to_production(args.experiment_id, force=args.force)
        print(
            f"Promotion {'succeeded' if promoted else 'blocked'} for "
            f"{args.experiment_id} (force={args.force})"
        )
        return 0 if promoted else 1

    raise ValueError(f"Unknown research action: {action}")


def main():
    """Entry point for the trading bot."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot")

    # Mode selection
    parser.add_argument(
        "mode",
        choices=["live", "backtest", "optimize", "replay", "research"],
        help="Operation mode: live trading, backtesting, optimization, replay, or research registry",
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        default="auto",
        help='Strategy to use (name, "all", or "auto" for automatic selection)',
    )

    # Symbol selection
    parser.add_argument("--symbols", default=None, help="Comma-separated list of symbols to trade")
    parser.add_argument("--symbol", default=None, help="Single symbol filter (replay mode)")

    # Live trading options
    parser.add_argument(
        "--real", action="store_true", help="Use real trading instead of paper trading"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force execution even if market is closed"
    )
    parser.add_argument(
        "--max-strategies",
        type=int,
        default=3,
        help="Maximum number of strategies to run simultaneously",
    )
    parser.add_argument(
        "--max-allocation", type=float, default=0.9, help="Maximum capital allocation (0.0 to 1.0)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.5, help="Minimum strategy score for selection"
    )
    parser.add_argument(
        "--auto-allocate",
        action="store_true",
        help="Automatically allocate capital based on performance",
    )
    parser.add_argument(
        "--evaluation-interval", type=int, default=24, help="Hours between strategy evaluations"
    )
    parser.add_argument(
        "--liquidate-on-exit", action="store_true", help="Liquidate all positions on exit"
    )

    # Backtest options
    parser.add_argument(
        "--start-date",
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital", type=float, default=100000, help="Initial capital for backtest"
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots for backtest results")
    parser.add_argument(
        "--execution-profile",
        choices=["idealistic", "realistic", "stressed"],
        default="realistic",
        help="Execution realism profile for backtest fills",
    )
    parser.add_argument(
        "--validated",
        action="store_true",
        help="Run validated backtest (walk-forward + profitability gates + permutation tests)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID for backtest artifacts (or replay target run ID)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="results/runs",
        help="Directory for run artifacts and replay inputs",
    )
    parser.add_argument(
        "--no-run-artifacts",
        action="store_true",
        help="Disable writing run artifacts in backtest mode",
    )

    # Walk-forward validation options
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation to detect overfitting before backtest",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip walk-forward validation for live trading (not recommended)",
    )
    parser.add_argument(
        "--wf-splits",
        type=int,
        default=5,
        help="Number of walk-forward validation splits (default: 5)",
    )
    parser.add_argument(
        "--wf-train-ratio",
        type=float,
        default=0.7,
        help="Training ratio for walk-forward (default: 0.7)",
    )
    parser.add_argument(
        "--overfit-threshold",
        type=float,
        default=1.5,
        help="Maximum acceptable in-sample/out-of-sample ratio (default: 1.5)",
    )

    # Optimization options
    parser.add_argument(
        "--param-ranges", default=None, help="JSON string with parameter ranges to optimize"
    )
    parser.add_argument(
        "--optimize-for",
        choices=["sharpe", "return", "drawdown"],
        default="sharpe",
        help="Metric to optimize for",
    )
    parser.add_argument(
        "--replay-date",
        default=None,
        help="Filter replay events by YYYY-MM-DD date prefix",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Replay only events that contain errors",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of replay events to display",
    )
    parser.add_argument(
        "--research-action",
        choices=[
            "create",
            "snapshot",
            "record-backtest",
            "record-validation",
            "record-paper",
            "store-walk-forward",
            "approve-review",
            "summary",
            "blockers",
            "check",
            "promote",
        ],
        default="check",
        help="Research registry action (research mode)",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment ID for research actions",
    )
    parser.add_argument("--name", default=None, help="Experiment name (create action)")
    parser.add_argument("--description", default=None, help="Experiment description (create action)")
    parser.add_argument("--author", default=None, help="Experiment author (create action)")
    parser.add_argument("--tags", default=None, help="Comma-separated experiment tags")
    parser.add_argument(
        "--parameters-json",
        default=None,
        help="JSON object or file path for parameter payload",
    )
    parser.add_argument(
        "--source",
        default="manual",
        help="Source label for parameter snapshots",
    )
    parser.add_argument("--notes", default="", help="Optional notes for review/snapshots")
    parser.add_argument(
        "--reviewer",
        default=None,
        help="Reviewer name for approve-review action",
    )
    parser.add_argument(
        "--backtest-json",
        default=None,
        help="JSON object or file path for backtest results",
    )
    parser.add_argument(
        "--validation-json",
        default=None,
        help="JSON object or file path for validation results",
    )
    parser.add_argument(
        "--paper-json",
        default=None,
        help="JSON object or file path for paper trading results",
    )
    parser.add_argument(
        "--walk-forward-json",
        default=None,
        help="JSON object or file path for walk-forward results",
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Optional source run ID for stored walk-forward artifacts",
    )
    parser.add_argument(
        "--research-registry-path",
        default=".research/experiments",
        help="Research experiment registry path",
    )
    parser.add_argument(
        "--research-production-path",
        default=".research/production",
        help="Research production registry path",
    )
    parser.add_argument(
        "--research-parameter-registry-path",
        default=".research/parameters",
        help="Research parameter registry path",
    )
    parser.add_argument(
        "--research-artifacts-path",
        default=".research/artifacts",
        help="Research artifacts path",
    )
    parser.add_argument(
        "--no-make-active",
        action="store_true",
        help="When snapshotting parameters, do not set snapshot as active config",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict promotion checks for research actions",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for check action payload",
    )

    args = parser.parse_args()

    try:
        if args.mode == "live":
            asyncio.run(run_live(args))
        elif args.mode == "backtest":
            asyncio.run(run_backtest(args))
        elif args.mode == "optimize":
            asyncio.run(optimize_parameters(args))
        elif args.mode == "replay":
            from utils.run_replay import (
                filter_events,
                format_replay_report,
                load_run_artifacts,
            )

            if not args.run_id:
                logger.error("Replay mode requires --run-id")
                sys.exit(2)

            artifacts = load_run_artifacts(args.run_id, artifacts_dir=args.artifacts_dir)
            decisions = filter_events(
                artifacts["decisions"],
                symbol=args.symbol,
                date_prefix=args.replay_date,
                errors_only=args.errors_only,
                limit=args.limit,
            )
            trades = filter_events(
                artifacts["trades"],
                symbol=args.symbol,
                date_prefix=args.replay_date,
                limit=args.limit,
            )
            print(
                format_replay_report(
                    artifacts["summary"],
                    decisions=decisions,
                    trades=trades,
                    order_reconciliation=artifacts.get("order_reconciliation"),
                    position_reconciliation=artifacts.get("position_reconciliation"),
                    slo_events=artifacts.get("slo_events"),
                    incident_events=artifacts.get("incident_events"),
                    data_quality_events=artifacts.get("data_quality_events"),
                    limit=args.limit,
                )
            )
        elif args.mode == "research":
            exit_code = run_research(args)
            if exit_code != 0:
                sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error running {args.mode} mode: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
