#!/usr/bin/env python3
"""
Live Trading Launcher

Launches strategies in live paper trading with real-time monitoring.

Features:
- Real-time trade execution
- Live performance tracking
- Position monitoring
- P/L updates
- Risk monitoring (circuit breaker)
- Trade logging

Usage:
    python live_trader.py --strategy momentum --symbols AAPL MSFT GOOGL
    python live_trader.py --strategy mean_reversion --capital 50000
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime

from brokers.alpaca_broker import AlpacaBroker
from config import RISK_PARAMS, SYMBOLS
from strategies.bracket_momentum_strategy import BracketMomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.risk_manager import RiskManager
from utils.audit_log import AuditEventType, AuditLog
from utils.circuit_breaker import CircuitBreaker
from utils.order_gateway import OrderGateway
from utils.order_reconciliation import OrderReconciler
from utils.position_manager import PositionManager
from utils.reconciliation import PositionReconciler
from utils.runtime_state import RuntimeStateStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    ],
)

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live trading manager with real-time monitoring.

    Handles:
    - Strategy initialization
    - WebSocket connection
    - Real-time monitoring
    - Graceful shutdown
    - Performance tracking
    """

    def __init__(self, strategy_name: str, symbols: list, parameters: dict):
        """
        Initialize live trader.

        Args:
            strategy_name: Name of strategy to run
            symbols: List of symbols to trade
            parameters: Strategy parameters
        """
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.parameters = parameters

        self.broker = None
        self.strategy = None
        self.circuit_breaker = None
        self.order_gateway = None
        self.audit_log = None
        self.position_manager = None
        self.reconciler = None
        self.order_reconciler = None
        self.state_store = RuntimeStateStore("data/live_trader_state.json")
        self._pending_strategy_state = {}
        self.running = False

        # Performance tracking
        self.start_time = None
        self.start_equity = None
        self.trades_executed = 0

        # Shutdown handling
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize broker and strategy."""
        try:
            logger.info("=" * 80)
            logger.info("🚀 LIVE TRADING INITIALIZATION")
            logger.info("=" * 80)

            # Initialize broker
            logger.info("1. Connecting to Alpaca (Paper Trading)...")
            self.broker = AlpacaBroker(paper=True)

            # Get account info
            account = await self.broker.get_account()
            self.start_equity = float(account.equity)

            logger.info(f"✅ Connected to account: {account.id}")
            logger.info(f"   Starting Capital: ${self.start_equity:,.2f}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}\n")

            # Initialize circuit breaker (CRITICAL SAFETY FEATURE)
            logger.info("1.5. Initializing circuit breaker...")
            self.circuit_breaker = CircuitBreaker(
                max_daily_loss=0.03,  # 3% max daily loss
                auto_close_positions=True,  # Automatically close positions on trigger
            )
            await self.circuit_breaker.initialize(self.broker)
            logger.info("✅ Circuit breaker armed")
            logger.info("   Max Daily Loss: 3%")
            logger.info("   Auto-close enabled: YES\n")

            # Initialize audit log
            self.audit_log = AuditLog(log_dir="./audit_logs", auto_verify=True)
            self.audit_log.log(
                AuditEventType.SYSTEM_START,
                {"component": "LiveTrader", "strategy": self.strategy_name},
            )
            if hasattr(self.broker, "set_audit_log"):
                self.broker.set_audit_log(self.audit_log)

            # Initialize order gateway (CRITICAL SAFETY FEATURE)
            logger.info("1.6. Initializing OrderGateway...")
            position_manager = PositionManager()
            risk_manager = RiskManager(
                max_portfolio_risk=RISK_PARAMS.get("MAX_PORTFOLIO_RISK", 0.02),
                max_position_risk=RISK_PARAMS.get("MAX_POSITION_RISK", 0.01),
            )
            self.position_manager = position_manager
            self.order_gateway = OrderGateway(
                broker=self.broker,
                circuit_breaker=self.circuit_breaker,
                position_manager=position_manager,
                risk_manager=risk_manager,
                audit_log=self.audit_log,
                enforce_gateway=True,
            )
            logger.info("✅ OrderGateway initialized (mandatory routing enabled)\n")
            if hasattr(self.broker, "set_position_manager"):
                self.broker.set_position_manager(self.position_manager)

            # Load persisted runtime state (if any) and sync with broker
            state = await self.state_store.load()
            if state:
                await self.position_manager.import_state(state.position_manager)
                self._pending_strategy_state = state.strategy_states or {}
                logger.info("Runtime state restored into PositionManager")
            await self.position_manager.sync_with_broker(
                self.broker, default_strategy="recovered"
            )

            # Initialize reconciler
            self.reconciler = PositionReconciler(
                broker=self.broker,
                internal_tracker=self.position_manager,
                halt_on_mismatch=False,
                sync_to_broker=True,
                audit_log=self.audit_log,
            )
            self.order_reconciler = OrderReconciler(
                broker=self.broker,
                lifecycle_tracker=self.order_gateway.lifecycle_tracker,
                audit_log=self.audit_log,
            )

            # Initialize strategy
            logger.info(f"2. Initializing {self.strategy_name} strategy...")

            strategy_class = self._get_strategy_class()
            self.strategy = strategy_class(
                broker=self.broker,
                parameters={"symbols": self.symbols, **self.parameters},
                order_gateway=self.order_gateway,
            )

            # Initialize strategy
            success = await self.strategy.initialize()
            if not success:
                raise RuntimeError("Strategy initialization failed")

            # Restore strategy state if available
            if hasattr(self.strategy, "import_state"):
                saved = self._pending_strategy_state.get(self.strategy_name)
                if saved:
                    await self.strategy.import_state(saved)

            logger.info("✅ Strategy initialized")
            logger.info(f"   Trading: {', '.join(self.symbols)}")
            logger.info(f"   Parameters: {self.parameters}\n")

            # Check market status
            clock = await self.broker.get_clock()
            logger.info("3. Market Status:")
            logger.info(f"   Market Open: {'YES ✅' if clock.is_open else 'NO ❌'}")
            logger.info(f"   Next Open: {clock.next_open}")
            logger.info(f"   Next Close: {clock.next_close}\n")

            logger.info("=" * 80)
            logger.info("✅ INITIALIZATION COMPLETE - READY TO TRADE")
            logger.info("=" * 80 + "\n")

            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False

    def _get_strategy_class(self):
        """Get strategy class by name."""
        strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "bracket_momentum": BracketMomentumStrategy,
        }

        if self.strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. Available: {list(strategies.keys())}"
            )

        return strategies[self.strategy_name]

    async def start_trading(self):
        """Start live trading."""
        try:
            self.running = True
            self.start_time = datetime.now()

            logger.info("\n" + "=" * 80)
            logger.info("📈 STARTING LIVE TRADING")
            logger.info("=" * 80)
            logger.info(f"Strategy: {self.strategy_name}")
            logger.info(f"Symbols: {', '.join(self.symbols)}")
            logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("\nPress Ctrl+C to stop trading gracefully")
            logger.info("=" * 80 + "\n")

            # Start WebSocket for real-time data
            await self.broker.start_websocket()

            # Start monitoring loop
            monitor_task = asyncio.create_task(self.monitor_performance())
            housekeeping_task = asyncio.create_task(self._housekeeping_loop())

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            # Cancel monitoring
            monitor_task.cancel()
            housekeeping_task.cancel()

        except Exception as e:
            logger.error(f"Error during trading: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def monitor_performance(self):
        """Monitor and log performance periodically."""
        try:
            while self.running:
                await asyncio.sleep(60)  # Check every minute

                # CRITICAL: Check circuit breaker FIRST
                if await self.circuit_breaker.check_and_halt():
                    logger.critical("=" * 80)
                    logger.critical("🚨 CIRCUIT BREAKER TRIGGERED 🚨")
                    logger.critical("Daily loss limit exceeded!")
                    logger.critical("All positions will be closed automatically.")
                    logger.critical("Trading will HALT for the rest of the day.")
                    logger.critical("=" * 80)

                    # Trigger shutdown
                    self.running = False
                    self.shutdown_event.set()
                    break

                # Get current account state
                account = await self.broker.get_account()
                current_equity = float(account.equity)

                # Calculate P/L
                pnl = current_equity - self.start_equity
                pnl_pct = (pnl / self.start_equity) * 100

                # Get positions
                positions = await self.broker.get_positions()

                # Log status
                runtime = datetime.now() - self.start_time

                logger.info("\n" + "-" * 80)
                logger.info(f"📊 PERFORMANCE UPDATE - Runtime: {str(runtime).split('.')[0]}")
                logger.info("-" * 80)
                logger.info(f"Equity: ${current_equity:,.2f} (P/L: ${pnl:+,.2f} / {pnl_pct:+.2f}%)")
                logger.info(f"Positions: {len(positions)}")

                if positions:
                    logger.info("\nOpen Positions:")
                    for pos in positions:
                        logger.info(
                            f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} "
                            f"(Current: ${float(pos.current_price):.2f}, "
                            f"P/L: ${float(pos.unrealized_pl):+,.2f} / {float(pos.unrealized_plpc)*100:+.2f}%)"
                        )

                logger.info("-" * 80 + "\n")

        except asyncio.CancelledError:
            logger.info("Performance monitoring stopped")
        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}", exc_info=True)

    async def shutdown(self):
        """Graceful shutdown."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("🛑 SHUTTING DOWN LIVE TRADING")
            logger.info("=" * 80)

            self.running = False

            # Get final performance
            account = await self.broker.get_account()
            final_equity = float(account.equity)
            total_pnl = final_equity - self.start_equity
            total_pnl_pct = (total_pnl / self.start_equity) * 100

            runtime = datetime.now() - self.start_time

            logger.info("\n📊 FINAL PERFORMANCE SUMMARY:")
            logger.info(f"   Runtime: {str(runtime).split('.')[0]}")
            logger.info(f"   Starting Equity: ${self.start_equity:,.2f}")
            logger.info(f"   Final Equity: ${final_equity:,.2f}")
            logger.info(f"   Total P/L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")

            # Get final positions
            positions = await self.broker.get_positions()
            if positions:
                logger.info(f"\n   Open Positions: {len(positions)}")
                for pos in positions:
                    logger.info(
                        f"     {pos.symbol}: {pos.qty} shares, "
                        f"P/L: ${float(pos.unrealized_pl):+,.2f}"
                    )

            logger.info("\n✅ Shutdown complete")
            logger.info("=" * 80 + "\n")

            if self.position_manager:
                await self.state_store.save(
                    self.position_manager,
                    strategy_states={
                        self.strategy_name: await self.strategy.export_state()
                        if hasattr(self.strategy, "export_state")
                        else {}
                    },
                )

            if self.audit_log:
                self.audit_log.log(
                    AuditEventType.SYSTEM_STOP,
                    {"component": "LiveTrader", "strategy": self.strategy_name},
                )
                self.audit_log.close()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def _housekeeping_loop(self):
        """Persist runtime state and run reconciliation periodically."""
        try:
            state_interval = 60
            reconciliation_interval = 300
            counter = 0
            while self.running:
                await asyncio.sleep(1)
                counter += 1
                if counter % state_interval == 0 and self.position_manager:
                    await self.state_store.save(self.position_manager)
                if (
                    counter % reconciliation_interval == 0
                    and self.reconciler is not None
                ):
                    try:
                        await self.reconciler.reconcile()
                    except Exception as e:
                        logger.error(f"Reconciliation error: {e}")
                if (
                    counter % 120 == 0
                    and self.order_reconciler is not None
                ):
                    try:
                        await self.order_reconciler.reconcile()
                    except Exception as e:
                        logger.error(f"Order reconciliation error: {e}")
        except asyncio.CancelledError:
            return

    def handle_shutdown_signal(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\n\n⚠️  Shutdown signal received...")
        self.shutdown_event.set()


async def main():
    """Main entry point for live trading."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Live Trading Bot")
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        choices=["momentum", "mean_reversion", "bracket_momentum"],
        help="Strategy to run",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None, help="Symbols to trade (default: from config)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.10,
        help="Position size as fraction of capital (default: 0.10)",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=0.02, help="Stop loss percentage (default: 0.02)"
    )
    parser.add_argument(
        "--take-profit", type=float, default=0.05, help="Take profit percentage (default: 0.05)"
    )

    args = parser.parse_args()

    # Use provided symbols or default from config
    symbols = args.symbols if args.symbols else SYMBOLS[:3]  # Default to first 3

    # Strategy parameters
    parameters = {
        "position_size": args.position_size,
        "stop_loss": args.stop_loss,
        "take_profit": args.take_profit,
    }

    # Create live trader
    trader = LiveTrader(strategy_name=args.strategy, symbols=symbols, parameters=parameters)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, trader.handle_shutdown_signal)
    signal.signal(signal.SIGTERM, trader.handle_shutdown_signal)

    # Initialize
    if not await trader.initialize():
        logger.error("Failed to initialize. Exiting.")
        return 1

    # Start trading
    await trader.start_trading()

    return 0


if __name__ == "__main__":
    # Create logs directory if needed
    import os

    os.makedirs("logs", exist_ok=True)

    # Run
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
