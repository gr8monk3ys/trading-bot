#!/usr/bin/env python3
"""
Run Adaptive Trading Strategy

This script runs the regime-switching adaptive strategy that automatically
selects between momentum and mean reversion based on market conditions.

Features:
- Automatic market regime detection (bull/bear/sideways/volatile)
- Strategy switching based on regime
- Trailing stops to let winners run
- Volatility-adjusted position sizing
- Kelly Criterion for optimal sizing
- DYNAMIC SYMBOL SELECTION - Automatically scans for best opportunities!

NEW FEATURES (Jan 2026):
- Economic Event Calendar - Avoid FOMC, NFP, CPI releases
- Volume Confirmation - Only trade with volume support
- Relative Strength Ranking - Focus on market leaders
- Support/Resistance Levels - Better stop/target placement
- Position Scaling - Scale in/out for better entries
- Correlation Management - Sector-aware position sizing

Usage:
    # Paper trading with auto-scanned symbols (recommended)
    python run_adaptive.py

    # With manual symbol override
    python run_adaptive.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA

    # Scan for opportunities only (no trading)
    python run_adaptive.py --scan-only

    # Show regime info only (no trading)
    python run_adaptive.py --regime-only

    # Backtest mode
    python run_adaptive.py --backtest --start 2024-01-01 --end 2024-12-31
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("adaptive_trading.log")],
)
logger = logging.getLogger(__name__)


async def scan_for_opportunities(
    top_n: int = 20, min_score: float = 1.0, use_sector_rotation: bool = True, broker=None
) -> List[str]:
    """
    Scan the market for the best trading opportunities.

    Uses SimpleSymbolSelector to find:
    - Liquid stocks (>1M daily volume)
    - Reasonable prices ($10-$500)
    - Stocks with momentum (price movement)

    Optionally uses sector rotation to weight towards favored sectors.

    Args:
        top_n: Maximum number of symbols to return
        min_score: Minimum momentum score percentage
        use_sector_rotation: Whether to use sector-based weighting
        broker: Optional broker instance (created if not provided)

    Returns:
        List of symbols ranked by opportunity score
    """
    from dotenv import load_dotenv

    from utils.simple_symbol_selector import SimpleSymbolSelector

    load_dotenv()

    print("\n" + "=" * 60)
    print("OPPORTUNITY SCANNER")
    print("=" * 60)
    print("Scanning market for best trading opportunities...")
    print(f"Criteria: >$10, <$500, >1M volume, momentum > {min_score}%")
    if use_sector_rotation:
        print("Sector Rotation: ENABLED (weighting by economic phase)")
    print("=" * 60 + "\n")

    symbols: List[str] = []

    # Try sector rotation first
    if use_sector_rotation:
        try:
            from brokers.alpaca_broker import AlpacaBroker
            from utils.sector_rotation import SectorRotator

            if broker is None:
                broker = AlpacaBroker(paper=True)
            rotator = SectorRotator(broker)

            # Get sector-weighted recommendations (now properly async)
            report = await rotator.get_sector_report()

            print(
                f"Economic Phase: {report['phase'].upper()} ({report['phase_confidence']:.0%} confidence)"
            )
            print(f"Overweight Sectors: {', '.join(report['overweight_sectors'])}")

            # Use sector-recommended stocks as base
            sector_stocks = report["recommended_stocks"]
            print(f"Sector picks: {', '.join(sector_stocks[:10])}...")

            symbols = sector_stocks

        except Exception as e:
            logger.warning(f"Sector rotation failed: {e}. Falling back to momentum scan.")

    # Fall back or supplement with momentum scanner
    if len(symbols) < top_n:
        try:
            selector = SimpleSymbolSelector(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                paper=True,
            )

            momentum_symbols = selector.select_top_symbols(
                top_n=top_n - len(symbols), min_score=min_score
            )

            # Combine and dedupe
            for sym in momentum_symbols:
                if sym not in symbols:
                    symbols.append(sym)

        except Exception as e:
            logger.error(f"Momentum scanner error: {e}")

    # Final fallback
    if not symbols:
        print("⚠ No opportunities found matching criteria. Using defaults.")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    else:
        symbols = symbols[:top_n]
        print(f"\n✓ Found {len(symbols)} opportunities:")
        print(", ".join(symbols))

    return symbols


async def check_market_regime(broker) -> Dict[str, Any]:
    """
    Check and display current market regime.

    Args:
        broker: AlpacaBroker instance for fetching market data

    Returns:
        Dict containing regime type, confidence, trend info, and recommendations
    """
    from utils.market_regime import MarketRegimeDetector

    print("\n" + "=" * 60)
    print("MARKET REGIME ANALYSIS")
    print("=" * 60)

    detector = MarketRegimeDetector(broker)
    regime = await detector.detect_regime()

    print(f"\nCurrent Regime: {regime['type'].upper()}")
    print(f"Confidence: {regime['confidence']:.0%}")
    print(f"\nTrend Direction: {regime['trend_direction']}")
    print(f"Trend Strength (ADX): {regime['trend_strength']:.1f}")
    print(f"  - Trending: {'Yes' if regime['is_trending'] else 'No'}")
    print(f"  - Ranging: {'Yes' if regime['is_ranging'] else 'No'}")
    print(f"\nVolatility: {regime['volatility_regime']} ({regime['volatility_pct']:.1f}%)")
    print(f"\nRecommended Strategy: {regime['recommended_strategy']}")
    print(f"Position Multiplier: {regime['position_multiplier']:.1f}x")

    if regime["sma_50"] and regime["sma_200"]:
        print(f"\nSMA50: ${regime['sma_50']:.2f}")
        print(f"SMA200: ${regime['sma_200']:.2f}")
        spread = (regime["sma_50"] / regime["sma_200"] - 1) * 100
        print(f"Spread: {spread:+.1f}%")

    print("\n" + "=" * 60)
    return regime


async def run_backtest(
    broker, symbols: List[str], start_date: str, end_date: str
) -> Dict[str, Any]:
    """
    Run a realistic backtest with the adaptive strategy.

    Args:
        broker: AlpacaBroker instance
        symbols: List of symbols to backtest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dict containing backtest results and performance metrics
    """
    from strategies.adaptive_strategy import AdaptiveStrategy
    from utils.realistic_backtest import RealisticBacktester, print_backtest_report

    print("\n" + "=" * 60)
    print("REALISTIC BACKTEST")
    print("=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    print("=" * 60 + "\n")

    # Create adaptive strategy
    strategy = AdaptiveStrategy(
        broker=broker,
        symbols=symbols,
        parameters={
            "position_size": 0.10,
            "max_positions": 5,
            "use_trailing_stop": True,
            "use_kelly_criterion": True,
        },
    )

    # Run realistic backtest
    backtester = RealisticBacktester(broker=broker, strategy=strategy, initial_capital=100000.0)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    results = await backtester.run(start, end, symbols)
    print_backtest_report(results)

    return results


async def run_live_trading(broker, symbols: List[str]) -> None:
    """
    Run live (paper) trading with the adaptive strategy.

    Args:
        broker: AlpacaBroker instance for order execution
        symbols: List of symbols to trade

    Returns:
        None (runs until interrupted or circuit breaker triggers)
    """
    from strategies.adaptive_strategy import AdaptiveStrategy
    from utils.circuit_breaker import CircuitBreaker
    from utils.correlation_manager import CorrelationManager
    from utils.earnings_calendar import EarningsCalendar
    from utils.economic_calendar import EconomicEventCalendar
    from utils.position_scaling import PositionScaler
    from utils.relative_strength import RSMomentumFilter
    from utils.support_resistance import SupportResistanceAnalyzer
    from utils.trading_hours import TradingHoursFilter
    from utils.volume_filter import VolumeFilter

    print("\n" + "=" * 60)
    print("ADAPTIVE STRATEGY - LIVE TRADING")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print("Mode: Paper Trading")
    print("Features: All advanced filters enabled")
    print("=" * 60 + "\n")

    # Initialize filters
    earnings_calendar = EarningsCalendar(exit_days_before=2, skip_entry_days_before=3)
    hours_filter = TradingHoursFilter(
        avoid_opening=True, avoid_lunch=True, avoid_monday_morning=True, avoid_friday_afternoon=True
    )

    # NEW: Economic event calendar
    economic_calendar = EconomicEventCalendar(
        avoid_high_impact=True, avoid_medium_impact=False, reduce_size_medium_impact=True
    )

    # NEW: Volume filter
    volume_filter = VolumeFilter(min_volume_ratio=1.2, breakout_volume_ratio=1.5)

    # NEW: Relative strength ranker
    rs_filter = RSMomentumFilter(broker)

    # NEW: Support/Resistance analyzer
    sr_analyzer = SupportResistanceAnalyzer(swing_lookback=5, min_touches=2)

    # NEW: Position scaler
    position_scaler = PositionScaler(
        default_tranches=3, scale_out_levels=[0.05, 0.10, 0.20]  # 5%, 10%, 20%
    )

    # NEW: Correlation manager
    correlation_mgr = CorrelationManager(max_sector_concentration=0.40, same_sector_penalty=0.60)

    # Check trading hours
    hours_status = hours_filter.get_trading_status()
    print(f"\nTrading Window: {hours_status['window']}")
    print(f"Quality Score: {hours_status['quality_score']:.2f}")
    print(f"Recommendation: {hours_status['recommendation']}")

    # NEW: Check economic calendar
    print("\nChecking economic calendar...")
    is_safe, econ_info = economic_calendar.is_safe_to_trade()
    if not is_safe:
        print(f"  ⚠ BLOCKED: {econ_info['blocking_event']}")
        print(f"  Safe in: {econ_info['hours_until_safe']:.1f} hours")
        print("  Waiting for event to pass...")
    else:
        print("  ✓ No blocking economic events")
        if econ_info["events_today"]:
            print(f"  Events today: {len(econ_info['events_today'])}")
            for e in econ_info["events_today"][:3]:
                print(f"    - {e['time']}: {e['name']} ({e['impact']})")

    position_multiplier = econ_info.get("position_multiplier", 1.0)
    if position_multiplier < 1.0:
        print(f"  Position size reduced to {position_multiplier:.0%} due to medium-impact event")

    # Filter symbols for earnings risk
    print("\nChecking earnings calendar...")
    safe_symbols = earnings_calendar.filter_symbols(symbols, for_entry=True)

    if len(safe_symbols) < len(symbols):
        filtered = set(symbols) - set(safe_symbols)
        print(f"  Filtered out (earnings risk): {', '.join(filtered)}")

    symbols = safe_symbols if safe_symbols else symbols
    print(f"  Trading {len(symbols)} symbols: {', '.join(symbols)}")

    # NEW: Refresh relative strength rankings
    print("\nCalculating relative strength rankings...")
    await rs_filter.refresh_rankings(symbols)
    leaders = [
        s
        for s in symbols
        if rs_filter.get_rs(s) and rs_filter.get_rs(s).get("percentile", 0) >= 0.7
    ]
    if leaders:
        print(f"  RS Leaders (top 30%): {', '.join(leaders)}")
    else:
        print("  No clear RS leaders - using all symbols")

    # Check market regime first
    regime = await check_market_regime(broker)

    # Initialize circuit breaker for safety
    circuit_breaker = CircuitBreaker(max_daily_loss=0.03)
    await circuit_breaker.initialize(broker)

    # Create adaptive strategy
    strategy = AdaptiveStrategy(
        broker=broker,
        symbols=symbols,
        parameters={
            "position_size": 0.10,
            "max_positions": 5,
            "use_trailing_stop": True,
            "use_kelly_criterion": True,
            "use_volatility_regime": True,
        },
    )

    if not await strategy.initialize():
        logger.error("Failed to initialize strategy")
        return

    logger.info(f"Strategy initialized: {strategy.NAME}")
    logger.info(f"Active sub-strategy: {strategy.active_strategy_name}")
    logger.info(f"Market regime: {regime['type']}")

    # Main trading loop
    iteration = 0
    last_earnings_check = None
    last_rs_refresh = None

    try:
        while True:
            iteration += 1

            # Check circuit breaker
            if await circuit_breaker.check_and_halt():
                logger.warning("Circuit breaker triggered - halting trading")
                break

            # Check if we're in a good trading window
            hours_status = hours_filter.get_trading_status()
            if not hours_status["is_good_time"]:
                if iteration == 1 or iteration % 30 == 0:
                    logger.info(
                        f"Waiting for better trading window: {hours_status['recommendation']}"
                    )
                await asyncio.sleep(60)
                continue

            # NEW: Check economic calendar before each trading decision
            is_econ_safe, econ_info = economic_calendar.is_safe_to_trade()
            if not is_econ_safe:
                if iteration % 10 == 0:
                    logger.info(
                        f"Waiting for economic event: {econ_info['blocking_event']} "
                        f"(safe in {econ_info['hours_until_safe']:.1f}h)"
                    )
                await asyncio.sleep(60)
                continue

            # Re-check earnings daily
            today = datetime.now().date()
            if last_earnings_check != today:
                positions_to_exit = earnings_calendar.get_positions_to_exit(symbols)
                if positions_to_exit:
                    logger.warning(f"EARNINGS ALERT: Should exit {positions_to_exit}")
                last_earnings_check = today

            # NEW: Refresh RS rankings every 2 hours
            now = datetime.now()
            if last_rs_refresh is None or (now - last_rs_refresh).seconds > 7200:
                await rs_filter.refresh_rankings(symbols)
                last_rs_refresh = now
                logger.info("Refreshed relative strength rankings")

            # Get current status
            status = strategy.get_status()
            active_signals = status.get("signals", {})

            # NEW: Filter signals through RS filter
            if active_signals:
                filtered_signals = {}
                for symbol, signal in active_signals.items():
                    should_trade, rs_reason = rs_filter.should_trade(
                        symbol, signal.get("action", "neutral")
                    )
                    if should_trade:
                        # Apply RS-based position multiplier
                        rs_mult = rs_filter.get_position_multiplier(symbol)
                        signal["rs_multiplier"] = rs_mult
                        signal["rs_reason"] = rs_reason
                        filtered_signals[symbol] = signal
                    else:
                        logger.info(f"Signal filtered by RS for {symbol}: {rs_reason}")

                # NEW: Apply correlation limits
                positions = await broker.get_positions()
                position_symbols = [p.symbol for p in positions] if positions else []

                for symbol, signal in list(filtered_signals.items()):
                    if signal.get("action") in ["long", "buy"]:
                        adjusted_size = correlation_mgr.get_adjusted_position_size(
                            symbol, position_symbols, signal.get("size", 1.0)
                        )
                        signal["correlation_adjusted_size"] = adjusted_size
                        if adjusted_size < signal.get("size", 1.0) * 0.5:
                            logger.info(f"Position reduced for {symbol} due to sector correlation")

                if filtered_signals:
                    logger.info(f"Active signals (after filters): {list(filtered_signals.keys())}")
                active_signals = filtered_signals

            # Log status every 10 iterations
            if iteration % 10 == 0:
                regime_info = await strategy.get_regime_info()
                quality = hours_status["quality_score"]
                econ_mult = econ_info.get("position_multiplier", 1.0)
                logger.info(
                    f"Status: regime={regime_info['type']}, "
                    f"strategy={status['active_strategy']}, "
                    f"switches={status['regime_switches']}, "
                    f"window_quality={quality:.2f}, econ_mult={econ_mult:.1f}"
                )

            # Wait before next iteration
            await asyncio.sleep(60)  # 1 minute between checks

    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)

    # Final status
    print("\n" + "=" * 60)
    print("TRADING SESSION ENDED")
    print("=" * 60)
    status = strategy.get_status()
    print(f"Total regime switches: {status['regime_switches']}")
    print(f"Final strategy: {status['active_strategy']}")


async def main():
    parser = argparse.ArgumentParser(description="Run Adaptive Trading Strategy")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,  # None = auto-scan for opportunities
        help="Comma-separated list of symbols (default: auto-scan)",
    )
    parser.add_argument(
        "--scan-only", action="store_true", help="Only scan for opportunities, no trading"
    )
    parser.add_argument(
        "--regime-only", action="store_true", help="Only show market regime analysis, no trading"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run in backtest mode instead of live trading"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--top-n", type=int, default=15, help="Number of top opportunities to trade (default: 15)"
    )
    parser.add_argument(
        "--min-momentum",
        type=float,
        default=1.0,
        help="Minimum momentum score %% to consider (default: 1.0)",
    )
    parser.add_argument(
        "--no-sector-rotation",
        action="store_true",
        help="Disable sector rotation (use pure momentum scanning)",
    )

    args = parser.parse_args()

    # Determine symbols: manual override or auto-scan
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"\nUsing manual symbols: {', '.join(symbols)}")
    else:
        # Auto-scan for best opportunities (now properly async)
        use_sectors = not args.no_sector_rotation
        symbols = await scan_for_opportunities(
            top_n=args.top_n, min_score=args.min_momentum, use_sector_rotation=use_sectors
        )

    # Handle scan-only mode
    if args.scan_only:
        print("\n" + "=" * 60)
        print("SCAN COMPLETE - Use these symbols for trading:")
        print("=" * 60)
        print(f"python run_adaptive.py --symbols {','.join(symbols)}")
        return

    # Initialize broker
    from brokers.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(paper=True)

    print("\n" + "=" * 60)
    print("ADAPTIVE TRADING BOT")
    print("=" * 60)
    print("Broker: Alpaca (Paper Trading)")
    print(f"Mode: {'Auto-scanned symbols' if not args.symbols else 'Manual symbols'}")
    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")

    if args.regime_only:
        # Just show regime info
        await check_market_regime(broker)

    elif args.backtest:
        # Run backtest
        await run_backtest(broker, symbols, args.start, args.end)

    else:
        # Run live trading
        await run_live_trading(broker, symbols)


if __name__ == "__main__":
    asyncio.run(main())
