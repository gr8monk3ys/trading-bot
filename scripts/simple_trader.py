#!/usr/bin/env python3
"""
Simple Live Trader - No Dependencies Conflicts

Direct trading without lumibot dependency issues.
"""

import asyncio
import logging
import signal
import sys

from brokers.alpaca_broker import AlpacaBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


class SimpleTrader:
    """Simple momentum trader."""

    def __init__(self, symbols, position_size=0.08, stop_loss=0.02, take_profit=0.05):
        self.symbols = symbols
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.broker = None
        self.running = True
        self.shutdown_event = asyncio.Event()
        self.start_equity = None

    async def initialize(self):
        """Initialize broker."""
        logger.info("=" * 80)
        logger.info("🚀 INITIALIZING SIMPLE TRADER")
        logger.info("=" * 80)

        self.broker = AlpacaBroker(paper=True)

        # Get account
        account = await self.broker.get_account()
        self.start_equity = float(account.equity)

        logger.info("✅ Connected to Alpaca")
        logger.info(f"   Account: {account.id}")
        logger.info(f"   Equity: ${self.start_equity:,.2f}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Position Size: {self.position_size:.1%}")
        logger.info(f"   Stop Loss: {self.stop_loss:.1%}")
        logger.info(f"   Take Profit: {self.take_profit:.1%}")

        # Check market
        clock = await self.broker.get_clock()
        logger.info(f"\n   Market: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'}")
        logger.info(f"   Next Open: {clock.next_open}")

        logger.info("=" * 80 + "\n")

        return True

    async def start(self):
        """Start trading."""
        logger.info("📈 STARTING LIVE TRADING")
        logger.info("Press Ctrl+C to stop\n")

        # Start monitoring
        monitor_task = asyncio.create_task(self.monitor())

        # Start WebSocket
        await self.broker.start_websocket()

        # Wait for shutdown
        await self.shutdown_event.wait()

        monitor_task.cancel()

    async def monitor(self):
        """Monitor positions periodically."""
        try:
            while self.running:
                await asyncio.sleep(60)  # Every minute

                account = await self.broker.get_account()
                equity = float(account.equity)
                pnl = equity - self.start_equity
                pnl_pct = (pnl / self.start_equity) * 100

                positions = await self.broker.get_positions()

                logger.info("-" * 80)
                logger.info(
                    f"💰 Equity: ${equity:,.2f} | P/L: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | Positions: {len(positions)}"
                )

                for pos in positions:
                    logger.info(
                        f"   {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f} "
                        f"→ ${float(pos.current_price):.2f} "
                        f"(${float(pos.unrealized_pl):+,.2f})"
                    )

                logger.info("-" * 80)

        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """Shutdown."""
        logger.info("\n" + "=" * 80)
        logger.info("🛑 SHUTTING DOWN")
        logger.info("=" * 80)

        account = await self.broker.get_account()
        final_equity = float(account.equity)
        pnl = final_equity - self.start_equity
        pnl_pct = (pnl / self.start_equity) * 100

        logger.info(f"\nFinal Equity: ${final_equity:,.2f}")
        logger.info(f"Total P/L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

        positions = await self.broker.get_positions()
        if positions:
            logger.info(f"\nOpen Positions: {len(positions)}")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.qty} shares")

        logger.info("\n✅ Shutdown complete")
        logger.info("=" * 80)

    def handle_shutdown(self, signum, frame):
        """Handle Ctrl+C."""
        logger.info("\n⚠️  Shutdown signal received...")
        self.running = False
        self.shutdown_event.set()


async def main():
    """Main entry."""
    symbols = ["SPY", "QQQ", "AAPL"]

    trader = SimpleTrader(symbols=symbols, position_size=0.08, stop_loss=0.02, take_profit=0.05)

    signal.signal(signal.SIGINT, trader.handle_shutdown)
    signal.signal(signal.SIGTERM, trader.handle_shutdown)

    if not await trader.initialize():
        return 1

    try:
        await trader.start()
    finally:
        await trader.shutdown()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
