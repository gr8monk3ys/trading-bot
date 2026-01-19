"""
Circuit Breaker - Daily Loss Limit Protection

CRITICAL SAFETY FEATURE: Automatically halts trading if daily losses exceed threshold.
This prevents catastrophic losses and protects capital from runaway strategies.

Usage:
    circuit_breaker = CircuitBreaker(max_daily_loss=0.03)  # 3% max daily loss
    await circuit_breaker.initialize(broker)

    # In trading loop:
    if await circuit_breaker.check_and_halt():
        logger.critical("TRADING HALTED - Daily loss limit exceeded!")
        break
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker that halts trading when daily losses exceed threshold.

    Features:
    - Automatic daily loss tracking
    - Configurable loss threshold (default 3%)
    - Automatic reset at market open
    - Emergency position closing on trigger
    """

    def __init__(self, max_daily_loss: float = 0.03, auto_close_positions: bool = True):
        """
        Initialize circuit breaker.

        Args:
            max_daily_loss: Maximum allowed daily loss as decimal (0.03 = 3%)
            auto_close_positions: Whether to automatically close all positions when triggered
        """
        self.max_daily_loss = max_daily_loss
        self.auto_close_positions = auto_close_positions

        # State tracking
        self.trading_halted = False
        self.starting_balance = None
        self.starting_equity = None
        self.peak_equity_today = None
        self.halt_triggered_at = None
        self.last_reset_date = None
        self.broker = None

        logger.info(f"Circuit Breaker initialized: max daily loss = {max_daily_loss:.1%}")

    async def initialize(self, broker):
        """
        Initialize with broker and set starting balance.

        Args:
            broker: Broker instance to monitor
        """
        self.broker = broker

        try:
            account = await broker.get_account()
            self.starting_balance = float(account.cash)
            self.starting_equity = float(account.equity)
            self.peak_equity_today = float(account.equity)
            self.last_reset_date = datetime.now().date()

            logger.info(
                f"Circuit Breaker armed: "
                f"Starting equity: ${self.starting_equity:,.2f}, "
                f"Will halt if equity drops below ${self.starting_equity * (1 - self.max_daily_loss):,.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker: {e}")
            raise

    async def check_and_halt(self) -> bool:
        """
        Check if daily loss limit has been exceeded and halt trading if needed.

        Returns:
            True if trading is halted, False if trading can continue

        Raises:
            RuntimeError: If broker not initialized
        """
        if not self.broker:
            raise RuntimeError("Circuit breaker not initialized - call initialize() first")

        # Auto-reset at start of new trading day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            await self._reset_for_new_day()

        # If already halted, stay halted until reset
        if self.trading_halted:
            logger.debug("Trading remains halted from previous trigger")
            return True

        try:
            # Get current account status
            account = await self.broker.get_account()
            current_equity = float(account.equity)

            # Update peak equity for the day
            if current_equity > self.peak_equity_today:
                self.peak_equity_today = current_equity

            # Calculate daily loss from starting equity
            daily_loss = (self.starting_equity - current_equity) / self.starting_equity

            # Calculate drawdown from peak equity today
            drawdown_from_peak = (self.peak_equity_today - current_equity) / self.peak_equity_today

            # Check if loss limit exceeded
            if daily_loss >= self.max_daily_loss:
                await self._trigger_halt(current_equity, daily_loss, "daily_loss_limit")
                return True

            # Also check for rapid drawdown from peak (additional safety)
            rapid_drawdown_threshold = self.max_daily_loss * 0.67  # 2% if daily limit is 3%
            if drawdown_from_peak >= rapid_drawdown_threshold:
                await self._trigger_halt(current_equity, drawdown_from_peak, "rapid_drawdown")
                return True

            # Log status periodically (every 1% loss)
            if daily_loss > 0 and int(daily_loss * 100) % 1 == 0:
                logger.info(
                    f"Daily P/L: {-daily_loss:.1%} "
                    f"(${current_equity - self.starting_equity:,.2f}) | "
                    f"Loss limit: {self.max_daily_loss:.1%} | "
                    f"Remaining: {(self.max_daily_loss - daily_loss):.1%}"
                )

            return False

        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            # FAIL SAFE: Halt trading on error
            self.trading_halted = True
            return True

    async def _trigger_halt(self, current_equity: float, loss_pct: float, reason: str):
        """
        Trigger the circuit breaker and halt trading.

        Args:
            current_equity: Current account equity
            loss_pct: Loss percentage that triggered halt
            reason: Reason for halt ('daily_loss_limit' or 'rapid_drawdown')
        """
        self.trading_halted = True
        self.halt_triggered_at = datetime.now()

        loss_amount = self.starting_equity - current_equity

        logger.critical("=" * 80)
        logger.critical("🚨 CIRCUIT BREAKER TRIGGERED 🚨")
        logger.critical(f"Reason: {reason.replace('_', ' ').upper()}")
        logger.critical(f"Daily Loss: {loss_pct:.2%} (${loss_amount:,.2f})")
        logger.critical(f"Max Allowed: {self.max_daily_loss:.2%}")
        logger.critical(f"Starting Equity: ${self.starting_equity:,.2f}")
        logger.critical(f"Current Equity: ${current_equity:,.2f}")
        logger.critical(f"Triggered At: {self.halt_triggered_at.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.critical("TRADING HALTED FOR THE DAY")
        logger.critical("=" * 80)

        # Close all positions if configured
        if self.auto_close_positions:
            await self._emergency_close_positions()

    async def _emergency_close_positions(self):
        """Emergency close all positions when circuit breaker triggers."""
        try:
            logger.warning("EMERGENCY: Closing all positions...")

            positions = await self.broker.get_positions()

            if not positions:
                logger.info("No positions to close")
                return

            from brokers.order_builder import OrderBuilder

            for position in positions:
                try:
                    symbol = position.symbol
                    quantity = abs(float(position.qty))

                    logger.warning(f"Emergency closing {quantity} shares of {symbol}")

                    # Market order to close immediately
                    order = OrderBuilder(symbol, "sell", quantity).market().day().build()

                    result = await self.broker.submit_order_advanced(order)
                    logger.info(f"Emergency sell order submitted for {symbol}: {result.id}")

                except Exception as e:
                    logger.error(f"Failed to close position {position.symbol}: {e}")

            logger.warning(f"Emergency close complete - {len(positions)} positions closed")

        except Exception as e:
            logger.error(f"Error during emergency position close: {e}")

    async def _reset_for_new_day(self):
        """Reset circuit breaker for a new trading day."""
        try:
            account = await self.broker.get_account()
            new_equity = float(account.equity)

            old_equity = self.starting_equity

            logger.info("=" * 80)
            logger.info("🔄 CIRCUIT BREAKER DAILY RESET")
            logger.info(f"Previous Day Equity: ${old_equity:,.2f}")
            logger.info(f"New Starting Equity: ${new_equity:,.2f}")

            if old_equity:
                overnight_change = (new_equity - old_equity) / old_equity
                logger.info(
                    f"Overnight Change: {overnight_change:+.2%} (${new_equity - old_equity:+,.2f})"
                )

            logger.info("Trading is ENABLED for new day")
            logger.info("=" * 80)

            # Reset state
            self.starting_equity = new_equity
            self.peak_equity_today = new_equity
            self.trading_halted = False
            self.halt_triggered_at = None
            self.last_reset_date = datetime.now().date()

        except Exception as e:
            logger.error(f"Error resetting circuit breaker: {e}")

    def is_halted(self) -> bool:
        """Check if trading is currently halted."""
        return self.trading_halted

    def get_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            Dict with status information
        """
        return {
            "halted": self.trading_halted,
            "max_daily_loss": self.max_daily_loss,
            "starting_equity": self.starting_equity,
            "peak_equity_today": self.peak_equity_today,
            "halt_triggered_at": (
                self.halt_triggered_at.isoformat() if self.halt_triggered_at else None
            ),
            "last_reset_date": self.last_reset_date.isoformat() if self.last_reset_date else None,
        }

    async def manual_reset(self, confirmation_token: str = None, force: bool = False):
        """
        Manually reset the circuit breaker (use with extreme caution!).

        P0 FIX: Added safety controls to prevent accidental resets.

        Args:
            confirmation_token: Must be "CONFIRM_RESET" to proceed
            force: If True, bypasses cooldown (still requires token)

        Returns:
            bool: True if reset successful, False if rejected

        Raises:
            ValueError: If confirmation token is invalid
        """
        # P0 SAFETY: Require explicit confirmation token
        if confirmation_token != "CONFIRM_RESET":
            logger.error(
                "CIRCUIT BREAKER RESET REJECTED: Invalid confirmation token. "
                "Use confirmation_token='CONFIRM_RESET' to confirm."
            )
            raise ValueError(
                "Manual reset requires confirmation_token='CONFIRM_RESET' to proceed. "
                "This is a safety measure to prevent accidental resets."
            )

        # P0 SAFETY: Rate limiting - prevent multiple resets within 5 minutes
        if hasattr(self, "_last_manual_reset") and self._last_manual_reset:
            cooldown_seconds = 300  # 5 minutes
            time_since_last = (datetime.now() - self._last_manual_reset).total_seconds()
            if time_since_last < cooldown_seconds and not force:
                remaining = int(cooldown_seconds - time_since_last)
                logger.error(
                    f"CIRCUIT BREAKER RESET REJECTED: Cooldown active. "
                    f"Wait {remaining} seconds or use force=True."
                )
                return False

        # P0 SAFETY: Audit logging with full context
        logger.warning("=" * 60)
        logger.warning("MANUAL CIRCUIT BREAKER RESET INITIATED")
        logger.warning(f"  Timestamp: {datetime.now().isoformat()}")
        logger.warning(f"  Was halted: {self.trading_halted}")
        if self.halt_triggered_at:
            logger.warning(f"  Halt triggered at: {self.halt_triggered_at.isoformat()}")
        if self.starting_equity and self.broker:
            try:
                account = await self.broker.get_account()
                current_equity = float(account.equity)
                loss_pct = (self.starting_equity - current_equity) / self.starting_equity
                logger.warning(f"  Current loss: {loss_pct:.2%}")
            except Exception as e:
                logger.warning(f"  Could not determine current loss: {e}")
        logger.warning("=" * 60)

        # Perform the reset
        await self._reset_for_new_day()

        # Track reset time for rate limiting
        self._last_manual_reset = datetime.now()

        logger.warning("CIRCUIT BREAKER MANUALLY RESET - TRADING RE-ENABLED")
        logger.warning("Monitor closely for continued losses!")

        return True
