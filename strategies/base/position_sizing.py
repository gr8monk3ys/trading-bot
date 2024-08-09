"""
BaseStrategy position-sizing / risk-permission mixin.

Contains:
    - check_trading_allowed (circuit-breaker gate)
    - enforce_position_size_limit (hard cap on % of equity)
    - calculate_kelly_position_size (Kelly Criterion sizing)
    - track_position_entry / record_completed_trade (Kelly accounting)
    - apply_volatility_adjustments / apply_streak_adjustments
    - is_short_position / get_position_pnl (position queries)

These methods rely on attributes/objects initialized in
``strategies/base/strategy.py`` (``self.circuit_breaker``, ``self.kelly``,
``self.volatility_regime``, ``self.streak_sizer``, ``self.broker``,
``self.closed_positions``, ``self.position_size``,
``self.max_position_size``) and therefore must be mixed into the same
concrete class.
"""

import logging
from datetime import datetime

from utils.kelly_criterion import Trade

logger = logging.getLogger(__name__)


class BasePositionSizingMixin:
    """Methods for trading-permission checks and position-size calculation."""

    async def check_trading_allowed(self) -> bool:
        """
        CRITICAL SAFETY: Check if trading is allowed (circuit breaker not triggered).

        ALL strategies must call this before executing any trades.

        Returns:
            True if trading is allowed, False if halted

        Example:
            if not await self.check_trading_allowed():
                logger.warning("Trading halted by circuit breaker")
                return
        """
        is_halted = await self.circuit_breaker.check_and_halt()
        return not is_halted

    async def enforce_position_size_limit(self, symbol, desired_position_value, current_price):
        """
        CRITICAL SAFETY: Enforce maximum position size limit.

        Prevents over-concentration in a single position which could lead to
        catastrophic losses. Default limit is 5% of portfolio value.

        Args:
            symbol: Stock symbol
            desired_position_value: Dollar value of desired position
            current_price: Current stock price

        Returns:
            Tuple of (capped_position_value, capped_quantity) that respects limits

        Raises:
            ValueError: If account information cannot be retrieved
        """
        # P0 FIX: Validate current_price to prevent division by zero
        if not current_price or current_price <= 0:
            self.logger.error(
                f"SAFETY: Invalid current_price for {symbol}: {current_price}. "
                "Returning 0 to prevent division by zero."
            )
            return 0, 0

        try:
            # Get current account value
            account = await self.broker.get_account()
            account_value = float(account.equity)

            # Calculate maximum allowed position value
            max_position_value = account_value * self.max_position_size

            # Check if desired position exceeds limit
            if desired_position_value > max_position_value:
                self.logger.warning(
                    f"POSITION SIZE LIMIT ENFORCED for {symbol}: "
                    f"Requested ${desired_position_value:,.2f} exceeds "
                    f"max ${max_position_value:,.2f} ({self.max_position_size:.1%} of ${account_value:,.2f})"
                )
                capped_value = max_position_value
            else:
                capped_value = desired_position_value

            # Calculate capped quantity
            capped_quantity = capped_value / current_price

            self.logger.debug(
                f"Position size check for {symbol}: "
                f"${capped_value:,.2f} ({capped_quantity:.2f} shares) "
                f"= {(capped_value/account_value):.1%} of portfolio"
            )

            return capped_value, capped_quantity

        except Exception as e:
            self.logger.error(f"Error enforcing position size limit for {symbol}: {e}")
            # FAIL SAFE: Return 0 to prevent trading on error
            return 0, 0

    async def calculate_kelly_position_size(self, symbol, current_price):
        """
        Calculate optimal position size using Kelly Criterion.

        If Kelly is enabled and we have sufficient trade history, uses Kelly formula
        for optimal position sizing. Otherwise falls back to fixed position_size parameter.

        Args:
            symbol: Stock symbol
            current_price: Current stock price

        Returns:
            Tuple of (position_value, position_fraction, quantity)
        """
        # P0 FIX: Validate current_price to prevent division by zero
        if not current_price or current_price <= 0:
            self.logger.error(
                f"SAFETY: Invalid current_price for {symbol}: {current_price}. "
                "Returning 0 position size to prevent division by zero."
            )
            return 0, 0, 0

        try:
            # Get current account value
            account = await self.broker.get_account()
            account_value = float(account.equity)

            # If Kelly not enabled, use fixed position sizing
            if not self.kelly:
                position_fraction = self.position_size
                position_value = account_value * position_fraction
                quantity = position_value / current_price
                self.logger.debug(
                    f"Fixed position sizing: {position_fraction:.1%} = ${position_value:,.2f}"
                )
                return position_value, position_fraction, quantity

            # Use Kelly Criterion for optimal sizing
            position_value, position_fraction = self.kelly.calculate_position_size(
                current_capital=account_value, current_price=current_price
            )

            quantity = position_value / current_price

            self.logger.info(
                f"📊 Kelly position size for {symbol}: "
                f"{position_fraction:.1%} = ${position_value:,.2f} ({quantity:.2f} shares) "
                f"[Win rate: {self.kelly.win_rate:.1%}, Profit factor: {self.kelly.profit_factor:.2f}]"
            )

            return position_value, position_fraction, quantity

        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}")
            # P0 FIX: Safe fallback - return zeros if we can't calculate
            # account_value may not be defined if error occurred early
            return 0, 0, 0

    def track_position_entry(self, symbol, entry_price, entry_time=None):
        """
        Track position entry for Kelly Criterion trade recording.

        Call this when opening a position. Will be used to calculate P/L when position closes.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            entry_time: Entry timestamp (defaults to now)
        """
        if entry_time is None:
            entry_time = datetime.now()

        self.closed_positions[symbol] = {"entry_price": entry_price, "entry_time": entry_time}

        self.logger.debug(f"Tracking entry for {symbol} at ${entry_price:.2f}")

    def record_completed_trade(self, symbol, exit_price, exit_time, quantity, side="long"):
        """
        Record a completed trade for Kelly Criterion analysis.

        Call this when closing a position. Calculates P/L and adds to Kelly trade history.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            quantity: Number of shares traded
            side: 'long' or 'short'
        """
        if not self.kelly:
            return  # Kelly not enabled

        # Check if we tracked the entry
        if symbol not in self.closed_positions:
            self.logger.warning(f"No entry tracked for {symbol}, cannot record trade for Kelly")
            return

        entry_info = self.closed_positions[symbol]
        entry_price = entry_info["entry_price"]
        entry_time = entry_info["entry_time"]

        # Calculate P/L
        if side == "long":
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price

        is_winner = pnl > 0

        # Create Trade object
        trade = Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_winner=is_winner,
        )

        # Add to Kelly history
        self.kelly.add_trade(trade)

        # Add to streak sizer history
        if self.streak_sizer:
            self.streak_sizer.record_trade(is_winner=is_winner, pnl_pct=pnl_pct, symbol=symbol)

        self.logger.info(
            f"📈 Trade recorded: {symbol} {side.upper()} "
            f"{'WIN' if is_winner else 'LOSS'} {pnl_pct:+.2%} "
            f"(Total trades: {len(self.kelly.trades)})"
        )

        # Remove from tracking
        del self.closed_positions[symbol]

    async def apply_volatility_adjustments(self, base_position_size: float, base_stop_loss: float):
        """
        Apply volatility regime adjustments to position sizing and stop-loss.

        If volatility regime detection is enabled, adjusts parameters based on current market conditions.
        Otherwise, returns base values unchanged.

        Args:
            base_position_size: Base position size (e.g., 0.10 for 10%)
            base_stop_loss: Base stop-loss (e.g., 0.03 for 3%)

        Returns:
            Tuple of (adjusted_position_size, adjusted_stop_loss, regime_name)
        """
        if not self.volatility_regime:
            return base_position_size, base_stop_loss, "normal"

        try:
            # Get current regime
            regime, adjustments = await self.volatility_regime.get_current_regime()

            # Apply adjustments
            adjusted_position_size = self.volatility_regime.adjust_position_size(
                base_position_size, adjustments["pos_mult"]
            )

            adjusted_stop_loss = self.volatility_regime.adjust_stop_loss(
                base_stop_loss, adjustments["stop_mult"]
            )

            self.logger.debug(
                f"Volatility adjustments ({regime.upper()}): "
                f"Position: {base_position_size:.1%} → {adjusted_position_size:.1%}, "
                f"Stop: {base_stop_loss:.1%} → {adjusted_stop_loss:.1%}"
            )

            return adjusted_position_size, adjusted_stop_loss, regime

        except Exception as e:
            self.logger.error(f"Error applying volatility adjustments: {e}", exc_info=True)
            return base_position_size, base_stop_loss, "normal"

    def apply_streak_adjustments(self, base_position_size: float) -> float:
        """
        Apply streak-based adjustments to position sizing.

        If streak sizing is enabled, adjusts position size based on recent win/loss performance.
        Otherwise, returns base value unchanged.

        Args:
            base_position_size: Base position size (e.g., 0.10 for 10%)

        Returns:
            Adjusted position size
        """
        if not self.streak_sizer:
            return base_position_size

        try:
            # Get adjusted size based on streak
            adjusted_position_size = self.streak_sizer.adjust_for_streak(base_position_size)

            if adjusted_position_size != base_position_size:
                self.logger.debug(
                    f"Streak adjustments ({self.streak_sizer.current_streak.upper()}): "
                    f"Position: {base_position_size:.1%} → {adjusted_position_size:.1%}"
                )

            return adjusted_position_size

        except Exception as e:
            self.logger.error(f"Error applying streak adjustments: {e}", exc_info=True)
            return base_position_size

    async def is_short_position(self, symbol):
        """
        Check if we currently have a short position in a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if we have a short position (negative quantity), False otherwise
        """
        try:
            positions = await self.broker.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if position:
                qty = float(position.qty)
                return qty < 0

            return False

        except Exception as e:
            self.logger.error(f"Error checking short position for {symbol}: {e}")
            return False

    async def get_position_pnl(self, symbol):
        """
        Get current P/L for a position (works for both long and short).

        Args:
            symbol: Stock symbol

        Returns:
            Dict with unrealized_pl (dollar amount) and unrealized_plpc (percentage)
            or None if no position exists
        """
        try:
            positions = await self.broker.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if position:
                return {
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "qty": float(position.qty),
                    "avg_entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price),
                    "market_value": float(position.market_value),
                    "is_short": float(position.qty) < 0,
                }

            return None

        except Exception as e:
            self.logger.error(f"Error getting position P/L for {symbol}: {e}")
            return None
