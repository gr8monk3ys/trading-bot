"""
Circuit Breaker - Daily Loss Limit Protection + Economic Event Safety

CRITICAL SAFETY FEATURE: Automatically halts trading if:
1. Daily losses exceed threshold (protects capital from runaway strategies)
2. High-impact economic events are imminent (FOMC, NFP, CPI)

INSTITUTIONAL ENHANCEMENT: Circuit breaker now RAISES exceptions instead of
returning bools. This ensures orders are atomically blocked when halted.

Usage:
    circuit_breaker = CircuitBreaker(max_daily_loss=0.03)  # 3% max daily loss
    await circuit_breaker.initialize(broker)

    # In OrderGateway (recommended):
    try:
        await circuit_breaker.check_before_order()
    except TradingHaltedException as e:
        return reject_order(f"Circuit breaker: {e}")

    # Legacy usage (returns bool):
    if await circuit_breaker.check_and_halt():
        logger.critical("TRADING HALTED - Daily loss limit exceeded!")
        break
"""

import logging
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TradingHaltedException(Exception):
    """
    Raised when circuit breaker has halted trading.

    INSTITUTIONAL SAFETY: This exception is raised instead of returning a bool
    to ensure orders are atomically blocked. Callers cannot accidentally ignore
    a halted state.

    Attributes:
        reason: Why trading was halted (daily_loss_limit, rapid_drawdown, economic_event)
        loss_pct: Current loss percentage (if applicable)
        event_name: Name of economic event (if applicable)
    """

    def __init__(
        self,
        message: str,
        reason: str = "unknown",
        loss_pct: float = None,
        event_name: str = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.loss_pct = loss_pct
        self.event_name = event_name


# Lazy import for economic calendar
EconomicEventCalendar = None

# Named constants for magic numbers
RAPID_DRAWDOWN_RATIO = 0.67  # Rapid drawdown triggers at 67% of max daily loss


class CircuitBreaker:
    """
    Circuit breaker that halts trading when daily losses exceed threshold.

    Features:
    - Automatic daily loss tracking
    - Configurable loss threshold (default 3%)
    - Automatic reset at market open
    - Emergency position closing on trigger
    """

    def __init__(
        self,
        max_daily_loss: float = 0.03,
        auto_close_positions: bool = True,
        use_economic_calendar: bool = True,
        block_high_impact_events: bool = True,
        reduce_position_medium_impact: bool = True,
    ):
        """
        Initialize circuit breaker.

        Args:
            max_daily_loss: Maximum allowed daily loss as decimal (0.03 = 3%)
            auto_close_positions: Whether to automatically close all positions when triggered
            use_economic_calendar: Enable economic event protection
            block_high_impact_events: Block new entries before FOMC, NFP, CPI
            reduce_position_medium_impact: Reduce position size for medium-impact events
        """
        self.max_daily_loss = max_daily_loss
        self.auto_close_positions = auto_close_positions

        # Economic calendar settings
        self.use_economic_calendar = use_economic_calendar
        self.block_high_impact_events = block_high_impact_events
        self.reduce_position_medium_impact = reduce_position_medium_impact
        self.economic_calendar = None  # Initialized lazily

        # State tracking
        self.trading_halted = False
        self.starting_balance = None
        self.starting_equity = None
        self.peak_equity_today = None
        self._true_peak_equity = None  # INSTITUTIONAL: Never resets during recovery
        self.halt_triggered_at = None
        self.last_reset_date = None
        self.broker = None
        self._last_logged_loss_pct = 0  # Track last logged loss for throttling
        self._halt_reason = None  # Track why we halted
        self._halt_loss_pct = None  # Track loss at halt time

        # Account data cache with TTL to reduce redundant API calls
        self._account_cache = None
        self._account_cache_time = None
        self._account_cache_ttl = 5  # seconds

        # Faster cache for per-order checks (1-second TTL)
        self._order_check_cache = None
        self._order_check_time = None
        self._order_check_ttl = 1  # seconds - fresh check before each order

        # Economic calendar cache (30-second TTL - events don't change rapidly)
        self._calendar_cache = None
        self._calendar_cache_time = None
        self._calendar_cache_ttl = 30  # seconds

        logger.info(
            f"Circuit Breaker initialized: max daily loss = {max_daily_loss:.1%}, "
            f"economic calendar = {use_economic_calendar}"
        )

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

            # Initialize economic calendar if enabled
            if self.use_economic_calendar:
                self._init_economic_calendar()

            logger.info(
                f"Circuit Breaker armed: "
                f"Starting equity: ${self.starting_equity:,.2f}, "
                f"Will halt if equity drops below ${self.starting_equity * (1 - self.max_daily_loss):,.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker: {e}")
            raise

    def _init_economic_calendar(self):
        """Lazy-load and initialize economic calendar."""
        global EconomicEventCalendar
        if EconomicEventCalendar is None:
            try:
                from utils.economic_calendar import EconomicEventCalendar
            except ImportError:
                logger.warning("Economic calendar not available - feature disabled")
                self.use_economic_calendar = False
                return

        self.economic_calendar = EconomicEventCalendar(
            avoid_high_impact=self.block_high_impact_events,
            avoid_medium_impact=False,  # Only reduce size, don't block
            reduce_size_medium_impact=self.reduce_position_medium_impact,
        )
        logger.info(
            f"Economic calendar initialized: "
            f"block_high_impact={self.block_high_impact_events}, "
            f"reduce_medium={self.reduce_position_medium_impact}"
        )

    async def _get_account_cached(self):
        """Get account data with simple TTL cache to reduce API calls.

        Returns cached account data if the cache is less than _account_cache_ttl
        seconds old. Otherwise fetches fresh data from the broker.
        """
        now = time.monotonic()
        if (
            self._account_cache is not None
            and self._account_cache_time is not None
            and (now - self._account_cache_time) < self._account_cache_ttl
        ):
            return self._account_cache

        account = await self.broker.get_account()
        self._account_cache = account
        self._account_cache_time = now
        return account

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
            # Get current account status (cached for ~5 seconds to reduce API calls)
            account = await self._get_account_cached()
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
            rapid_drawdown_threshold = self.max_daily_loss * RAPID_DRAWDOWN_RATIO
            if drawdown_from_peak >= rapid_drawdown_threshold:
                await self._trigger_halt(current_equity, drawdown_from_peak, "rapid_drawdown")
                return True

            # Log status when loss changes by 1 percentage point (throttle logging)
            current_loss_pct = int(daily_loss * 100)
            if daily_loss > 0 and current_loss_pct != self._last_logged_loss_pct:
                self._last_logged_loss_pct = current_loss_pct
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
        self._halt_reason = reason  # INSTITUTIONAL: Store for exception messages
        self._halt_loss_pct = loss_pct  # INSTITUTIONAL: Store for exception messages

        loss_amount = self.starting_equity - current_equity

        logger.critical("=" * 80)
        logger.critical("🚨 CIRCUIT BREAKER TRIGGERED 🚨")
        logger.critical(f"Reason: {reason.replace('_', ' ').upper()}")
        logger.critical(f"Daily Loss: {loss_pct:.2%} (${loss_amount:,.2f})")
        logger.critical(f"Max Allowed: {self.max_daily_loss:.2%}")
        logger.critical(f"Starting Equity: ${self.starting_equity:,.2f}")
        logger.critical(f"Current Equity: ${current_equity:,.2f}")
        if self._true_peak_equity:
            logger.critical(f"True Peak Equity: ${self._true_peak_equity:,.2f}")
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
            self._true_peak_equity = new_equity  # INSTITUTIONAL: Reset true peak for new day
            self.trading_halted = False
            self.halt_triggered_at = None
            self._halt_reason = None  # INSTITUTIONAL: Clear halt reason
            self._halt_loss_pct = None  # INSTITUTIONAL: Clear halt loss
            self.last_reset_date = datetime.now().date()

            # Clear caches
            self._order_check_cache = None
            self._order_check_time = None

        except Exception as e:
            logger.error(f"Error resetting circuit breaker: {e}")

    def is_halted(self) -> bool:
        """Check if trading is currently halted."""
        return self.trading_halted

    def quick_check(self) -> bool:
        """
        Fast synchronous check if trading is halted.

        This is a CACHED check that doesn't make API calls. Use for:
        - Quick pre-flight checks before expensive operations
        - High-frequency checks in hot paths
        - When you need immediate response

        NOTE: This may be slightly stale (up to 5 seconds). For order-time
        checks, use check_before_order() instead.

        Returns:
            True if trading is halted (should NOT place orders)
            False if trading appears to be allowed (but verify with check_before_order)
        """
        # If halted flag is set, definitely halted
        if self.trading_halted:
            return True

        # Check if we need a day reset (could be stale)
        current_date = datetime.now().date()
        if self.last_reset_date and current_date != self.last_reset_date:
            # New day - might be reset, but we can't do async here
            # Be conservative: allow trading (async check will verify)
            return False

        return False

    async def enforce_before_order(self, is_exit_order: bool = False) -> None:
        """
        INSTITUTIONAL ATOMIC CHECK: Enforce circuit breaker before order submission.

        This is the RECOMMENDED method for OrderGateway. Raises TradingHaltedException
        when trading should be blocked, ensuring orders are atomically rejected.

        Uses a 1-second TTL cache to balance freshness with API efficiency.
        Also checks economic calendar for high-impact events (FOMC, NFP, CPI).
        Exit orders bypass event blocking (we want to close positions, not hold them).

        Args:
            is_exit_order: True if this is an exit/close order (bypasses event blocking)

        Raises:
            TradingHaltedException: If trading is halted (order MUST be rejected)
            RuntimeError: If broker not initialized
        """
        if not self.broker:
            raise RuntimeError("Circuit breaker not initialized - call initialize() first")

        # If already halted, raise immediately with cached reason
        if self.trading_halted:
            raise TradingHaltedException(
                f"Trading halted: {self._halt_reason or 'daily loss limit exceeded'}",
                reason=self._halt_reason or "daily_loss_limit",
                loss_pct=self._halt_loss_pct,
            )

        # Check economic calendar FIRST (before hitting broker API)
        # Exit orders bypass event blocking - we want to close positions during events
        if not is_exit_order:
            is_blocked, event_name, hours_until_safe = self.is_blocked_by_event()
            if is_blocked:
                raise TradingHaltedException(
                    f"High-impact event '{event_name}' imminent - "
                    f"new entries blocked for {hours_until_safe:.1f}h",
                    reason="economic_event",
                    event_name=event_name,
                )

        # Auto-reset at start of new trading day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            await self._reset_for_new_day()
            return  # Fresh day, trading allowed

        # Check if we have a fresh order-time cache (1-second TTL)
        now = time.monotonic()
        if (
            self._order_check_cache is not None
            and self._order_check_time is not None
            and (now - self._order_check_time) < self._order_check_ttl
        ):
            if self._order_check_cache:
                raise TradingHaltedException(
                    f"Trading halted: {self._halt_reason or 'circuit breaker active'}",
                    reason=self._halt_reason or "cached_halt",
                    loss_pct=self._halt_loss_pct,
                )
            return  # Cached: trading allowed

        try:
            # Fresh check with 1-second TTL
            account = await self.broker.get_account()
            current_equity = float(account.equity)

            # INSTITUTIONAL FIX: Track true peak (never resets during recovery)
            # This prevents the bug where partial recovery masks total drawdown
            if self._true_peak_equity is None:
                self._true_peak_equity = current_equity
            else:
                self._true_peak_equity = max(self._true_peak_equity, current_equity)

            # Also update legacy peak_equity_today for backward compatibility
            if current_equity > self.peak_equity_today:
                self.peak_equity_today = current_equity

            # Calculate losses using TRUE peak (not recoverable peak)
            daily_loss = (self.starting_equity - current_equity) / self.starting_equity
            drawdown_from_true_peak = (
                self._true_peak_equity - current_equity
            ) / self._true_peak_equity

            # Check thresholds
            is_halted = False
            halt_reason = None
            halt_loss = None

            if daily_loss >= self.max_daily_loss:
                await self._trigger_halt(current_equity, daily_loss, "daily_loss_limit")
                is_halted = True
                halt_reason = "daily_loss_limit"
                halt_loss = daily_loss
            elif drawdown_from_true_peak >= self.max_daily_loss * RAPID_DRAWDOWN_RATIO:
                await self._trigger_halt(current_equity, drawdown_from_true_peak, "rapid_drawdown")
                is_halted = True
                halt_reason = "rapid_drawdown"
                halt_loss = drawdown_from_true_peak

            # Cache result
            self._order_check_cache = is_halted
            self._order_check_time = now

            if is_halted:
                raise TradingHaltedException(
                    f"Circuit breaker triggered: {halt_reason.replace('_', ' ')} at {halt_loss:.2%}",
                    reason=halt_reason,
                    loss_pct=halt_loss,
                )

        except TradingHaltedException:
            raise  # Re-raise our exceptions
        except Exception as e:
            logger.error(f"Error in enforce_before_order: {e}")
            # FAIL SAFE: Block orders on error
            raise TradingHaltedException(
                f"Circuit breaker check failed: {e}",
                reason="check_error",
            ) from e

    async def check_before_order(self, is_exit_order: bool = False) -> bool:
        """
        Check circuit breaker status before submitting an order.

        DEPRECATED: Use enforce_before_order() for atomic enforcement.
        This method exists for backward compatibility but logs a warning.

        Args:
            is_exit_order: True if this is an exit/close order (bypasses event blocking)

        Returns:
            True if trading is halted (order should be REJECTED)
            False if trading is allowed (order can proceed)

        Raises:
            RuntimeError: If broker not initialized
        """
        try:
            await self.enforce_before_order(is_exit_order=is_exit_order)
            return False  # Trading allowed
        except TradingHaltedException as e:
            logger.debug(f"check_before_order returning True: {e}")
            return True  # Trading halted

    def get_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            Dict with status information
        """
        status = {
            "halted": self.trading_halted,
            "max_daily_loss": self.max_daily_loss,
            "starting_equity": self.starting_equity,
            "peak_equity_today": self.peak_equity_today,
            "halt_triggered_at": (
                self.halt_triggered_at.isoformat() if self.halt_triggered_at else None
            ),
            "last_reset_date": self.last_reset_date.isoformat() if self.last_reset_date else None,
        }

        # Add economic calendar status
        if self.economic_calendar:
            calendar_status = self.check_economic_events()
            status["economic_calendar"] = calendar_status

        return status

    def check_economic_events(self) -> Dict:
        """
        Check for upcoming economic events that might affect trading.

        Uses a 30-second cache to reduce redundant calculations.

        Returns:
            Dict with:
                - is_safe: bool - whether new entries are allowed
                - blocking_event: str or None - name of blocking event
                - hours_until_safe: float - time until trading is safe
                - position_multiplier: float - position size multiplier (1.0 = normal)
                - events_today: list - today's economic events
        """
        if not self.economic_calendar:
            return {
                "is_safe": True,
                "blocking_event": None,
                "hours_until_safe": 0,
                "position_multiplier": 1.0,
                "events_today": [],
            }

        # Check cache
        now = time.monotonic()
        if (
            self._calendar_cache is not None
            and self._calendar_cache_time is not None
            and (now - self._calendar_cache_time) < self._calendar_cache_ttl
        ):
            return self._calendar_cache

        # Fresh check from calendar
        is_safe, info = self.economic_calendar.is_safe_to_trade()

        result = {
            "is_safe": is_safe,
            "blocking_event": info.get("blocking_event"),
            "hours_until_safe": info.get("hours_until_safe", 0),
            "position_multiplier": info.get("position_multiplier", 1.0),
            "events_today": info.get("events_today", []),
        }

        # Cache result
        self._calendar_cache = result
        self._calendar_cache_time = now

        return result

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on economic events.

        Call this before calculating position sizes to reduce exposure
        around medium-impact events.

        Returns:
            Multiplier between 0.0 and 1.0 (1.0 = full size, 0.5 = half size)
        """
        if not self.economic_calendar:
            return 1.0

        event_info = self.check_economic_events()
        return event_info.get("position_multiplier", 1.0)

    def is_blocked_by_event(self) -> Tuple[bool, Optional[str], float]:
        """
        Check if trading is blocked by an economic event.

        This is a synchronous quick check using cached data.

        Returns:
            Tuple of (is_blocked, event_name, hours_until_safe)
        """
        if not self.economic_calendar:
            return False, None, 0

        event_info = self.check_economic_events()

        if not event_info["is_safe"]:
            return True, event_info["blocking_event"], event_info["hours_until_safe"]

        return False, None, 0

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
