#!/usr/bin/env python3
"""
Single-Strategy Paper Trading Launcher

Launches a single momentum or mean-reversion strategy against Alpaca paper.
This entrypoint is paper-only; use `main.py live --real` for real-money
operation (which is also strongly discouraged for this experimental repo).

Features:
- Real-time trade execution via WebSocket
- Live performance tracking
- Position monitoring + P/L updates
- Circuit-breaker daily-loss halt
- Risk-profile-based runtime parameters

Usage:
    python live_trader.py --strategy momentum --symbols AAPL MSFT GOOGL
    python live_trader.py --strategy mean_reversion --risk-profile balanced
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from brokers.alpaca_broker import AlpacaBroker
from config import SYMBOLS
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from utils.audit_log import AuditEventType, AuditLog
from utils.circuit_breaker import CircuitBreaker

# Set up logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)


RISK_PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "custom": {},
    "conservative": {
        "position_size": 0.02,
        "max_position_size": 0.03,
        "stop_loss": 0.01,
        "take_profit": 0.02,
        "max_daily_loss": 0.025,
    },
    "balanced": {
        "position_size": 0.05,
        "max_position_size": 0.08,
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "max_daily_loss": 0.03,
    },
    "aggressive": {
        "position_size": 0.10,
        "max_position_size": 0.15,
        "stop_loss": 0.03,
        "take_profit": 0.08,
        "max_daily_loss": 0.04,
    },
}


def _resolve_runtime_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """Merge risk-profile defaults with CLI overrides."""
    profile = RISK_PROFILE_DEFAULTS.get(args.risk_profile, {}).copy()

    cli_overrides = {
        "position_size": args.position_size,
        "max_position_size": args.max_position_size,
        "stop_loss": args.stop_loss,
        "take_profit": args.take_profit,
        "max_daily_loss": args.max_daily_loss,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            profile[key] = float(value)

    return profile


class LiveTrader:
    """Simple single-strategy paper trader."""

    def __init__(self, strategy_name: str, symbols: list[str], parameters: dict[str, Any]):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.parameters = parameters

        self.broker: AlpacaBroker | None = None
        self.strategy = None
        self.circuit_breaker: CircuitBreaker | None = None
        self.audit_log: AuditLog | None = None
        self.running = False
        self.start_time: datetime | None = None
        self.start_equity: float | None = None
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> bool:
        """Initialize broker, circuit breaker, and strategy."""
        try:
            logger.info("=" * 80)
            logger.info("PAPER TRADING INITIALIZATION")
            logger.info("=" * 80)

            paper = os.getenv("PAPER", "true").lower() == "true"
            self.broker = AlpacaBroker(paper=paper)
            logger.info(f"1. Connecting to Alpaca (paper={paper})...")

            account = await self.broker.get_account()
            self.start_equity = float(account.equity)
            logger.info(f"   Connected. Account {account.id}; equity ${self.start_equity:,.2f}")

            logger.info("2. Initializing circuit breaker...")
            self.circuit_breaker = CircuitBreaker(
                max_daily_loss=float(self.parameters.get("max_daily_loss", 0.03)),
                auto_close_positions=True,
            )
            await self.circuit_breaker.initialize(self.broker)
            logger.info(
                "   Circuit breaker armed. Max daily loss %.2f%%",
                float(self.parameters.get("max_daily_loss", 0.03)) * 100,
            )

            self.audit_log = AuditLog(log_dir="./audit_logs", auto_verify=True)
            self.audit_log.log(
                AuditEventType.SYSTEM_START,
                {"component": "LiveTrader", "strategy": self.strategy_name},
            )
            if hasattr(self.broker, "set_audit_log"):
                self.broker.set_audit_log(self.audit_log)

            logger.info(f"3. Initializing {self.strategy_name} strategy...")
            strategy_class = self._get_strategy_class()
            self.strategy = strategy_class(
                broker=self.broker,
                parameters={"symbols": self.symbols, **self.parameters},
            )
            success = await self.strategy.initialize()
            if not success:
                raise RuntimeError("Strategy initialization failed")
            logger.info(f"   Strategy ready. Trading: {', '.join(self.symbols)}")

            logger.info("4. Checking market status...")
            clock = await self.broker.get_clock()
            logger.info(f"   Market open: {clock.is_open}")
            logger.info(f"   Next open:   {clock.next_open}")
            logger.info(f"   Next close:  {clock.next_close}")

            logger.info("=" * 80)
            logger.info("INITIALIZATION COMPLETE")
            logger.info("=" * 80)
            return True
        except Exception as exc:
            logger.error(f"Initialization failed: {exc}", exc_info=True)
            return False

    def _get_strategy_class(self):
        strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
        }
        if self.strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy '{self.strategy_name}'. Available: {list(strategies)}"
            )
        return strategies[self.strategy_name]

    async def start_trading(self) -> None:
        """Run the trading loop until shutdown."""
        try:
            self.running = True
            self.start_time = datetime.now()

            logger.info("=" * 80)
            logger.info("STARTING PAPER TRADING")
            logger.info("=" * 80)
            logger.info(f"Strategy: {self.strategy_name}")
            logger.info(f"Symbols:  {', '.join(self.symbols)}")
            logger.info("Press Ctrl+C for graceful shutdown")
            logger.info("=" * 80)

            await self.broker.start_websocket(self.symbols)
            monitor_task = asyncio.create_task(self._monitor_performance())
            await self.shutdown_event.wait()
            monitor_task.cancel()
        except Exception as exc:
            logger.error(f"Error during trading: {exc}", exc_info=True)
        finally:
            await self.shutdown()

    async def _monitor_performance(self) -> None:
        """Log P&L every 60s and watch the circuit breaker."""
        try:
            while self.running:
                await asyncio.sleep(60)
                if await self.circuit_breaker.check_and_halt():
                    logger.critical("CIRCUIT BREAKER TRIGGERED - daily loss limit hit; halting.")
                    self.running = False
                    self.shutdown_event.set()
                    break

                account = await self.broker.get_account()
                current_equity = float(account.equity)
                pnl = current_equity - self.start_equity
                pnl_pct = (pnl / self.start_equity) * 100 if self.start_equity else 0.0
                positions = await self.broker.get_positions()
                runtime = datetime.now() - self.start_time
                logger.info("-" * 80)
                logger.info(f"PERFORMANCE - Runtime {str(runtime).split('.')[0]}")
                logger.info(
                    f"Equity ${current_equity:,.2f}  P/L ${pnl:+,.2f} ({pnl_pct:+.2f}%)  "
                    f"Positions {len(positions)}"
                )
                for pos in positions or []:
                    logger.info(
                        f"  {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}  "
                        f"P/L ${float(pos.unrealized_pl):+,.2f}"
                    )
                logger.info("-" * 80)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"Monitor error: {exc}", exc_info=True)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        try:
            logger.info("=" * 80)
            logger.info("SHUTTING DOWN")
            logger.info("=" * 80)
            self.running = False

            if self.broker:
                try:
                    account = await self.broker.get_account()
                    final_equity = float(account.equity)
                    if self.start_equity:
                        total_pnl = final_equity - self.start_equity
                        total_pnl_pct = (total_pnl / self.start_equity) * 100
                        logger.info(
                            f"Final equity ${final_equity:,.2f}  "
                            f"Total P/L ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)"
                        )
                    positions = await self.broker.get_positions()
                    logger.info(f"Open positions at shutdown: {len(positions)}")
                except Exception as exc:
                    logger.warning(f"Could not collect shutdown stats: {exc}")
                try:
                    await self.broker.stop_websocket()
                except Exception as exc:
                    logger.warning(f"Failed to stop websocket: {exc}")

            if self.audit_log:
                self.audit_log.log(
                    AuditEventType.SYSTEM_STOP,
                    {"component": "LiveTrader", "strategy": self.strategy_name},
                )
                self.audit_log.close()
            logger.info("Shutdown complete.")
        except Exception as exc:
            logger.error(f"Error during shutdown: {exc}", exc_info=True)

    def handle_shutdown_signal(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.shutdown_event.set()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Single-strategy paper trading bot")
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        choices=["momentum", "mean_reversion"],
        help="Strategy to run",
    )
    parser.add_argument(
        "--risk-profile",
        type=str,
        default="custom",
        choices=["custom", "conservative", "balanced", "aggressive"],
        help="Risk profile preset",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to trade (default: first 3 from config.SYMBOLS)",
    )
    parser.add_argument("--position-size", type=float, default=None)
    parser.add_argument("--max-position-size", type=float, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--max-daily-loss", type=float, default=None)

    args = parser.parse_args()

    symbols = args.symbols if args.symbols else SYMBOLS[:3]
    parameters = _resolve_runtime_parameters(args)

    trader = LiveTrader(strategy_name=args.strategy, symbols=symbols, parameters=parameters)
    signal.signal(signal.SIGINT, trader.handle_shutdown_signal)
    signal.signal(signal.SIGTERM, trader.handle_shutdown_signal)

    if not await trader.initialize():
        logger.error("Initialization failed. Exiting.")
        return 1

    await trader.start_trading()
    return 0


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
