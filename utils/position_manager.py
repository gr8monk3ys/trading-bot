"""
PositionManager - Centralized position ownership and conflict detection.

Provides:
1. Atomic position reservation before order submission
2. Strategy ownership tracking (which strategy owns which position)
3. Conflict detection (prevents multiple strategies sizing same symbol)
4. Thread-safe locking for concurrent strategies

This prevents the scenario where two strategies simultaneously open
positions in the same symbol, leading to unintended double exposure.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PositionReservation:
    """Tracks a position reservation before order submission."""

    symbol: str
    strategy_name: str
    intended_qty: float
    intended_side: str  # 'buy' or 'sell'
    reserved_at: datetime
    order_id: Optional[str] = None  # Filled after order submission
    status: str = "pending"  # pending, submitted, filled, cancelled, expired


@dataclass
class PositionOwnership:
    """Tracks which strategy owns a position."""

    symbol: str
    strategy_name: str
    qty: float
    entry_price: float
    opened_at: datetime
    side: str = "long"  # 'long' or 'short'


class PositionManager:
    """
    Centralized position manager that prevents strategy conflicts.

    THREAD-SAFETY: Uses asyncio.Lock for atomic operations.

    Usage:
        pm = PositionManager()

        # Before placing order:
        reservation = await pm.reserve_position("AAPL", "MomentumStrategy", 100, "buy")
        if reservation is None:
            # Another strategy already has this position
            return

        # After order submitted:
        await pm.confirm_order_submitted("AAPL", "MomentumStrategy", order_id)

        # After order filled:
        await pm.confirm_fill("AAPL", "MomentumStrategy", 100, 150.00)

        # When closing position:
        await pm.release_position("AAPL", "MomentumStrategy")
    """

    # Reservation expiry time (seconds) - if order not submitted, release
    RESERVATION_TIMEOUT = 30

    def __init__(self):
        self._lock = asyncio.Lock()
        self._reservations: Dict[str, PositionReservation] = {}  # symbol -> reservation
        self._ownership: Dict[str, PositionOwnership] = {}  # symbol -> ownership
        self._strategy_positions: Dict[str, Set[str]] = {}  # strategy -> set of symbols

    async def reserve_position(
        self,
        symbol: str,
        strategy_name: str,
        intended_qty: float,
        side: str = "buy",
    ) -> Optional[PositionReservation]:
        """
        Attempt to reserve a position for a strategy.

        ATOMIC: Only one strategy can hold reservation for a symbol.

        Args:
            symbol: Stock symbol
            strategy_name: Name of the strategy requesting the position
            intended_qty: Number of shares intended to buy/sell
            side: 'buy' or 'sell'

        Returns:
            PositionReservation if successful, None if conflict detected
        """
        async with self._lock:
            # Check for existing reservation
            if symbol in self._reservations:
                existing = self._reservations[symbol]

                # Allow same strategy to re-reserve (updating qty)
                if existing.strategy_name == strategy_name:
                    existing.intended_qty = intended_qty
                    existing.intended_side = side
                    existing.reserved_at = datetime.now()
                    logger.debug(f"Updated reservation for {symbol} by {strategy_name}")
                    return existing

                # Check if reservation expired
                elapsed = (datetime.now() - existing.reserved_at).total_seconds()
                if elapsed < self.RESERVATION_TIMEOUT:
                    logger.warning(
                        f"POSITION CONFLICT: {strategy_name} tried to reserve {symbol} "
                        f"but {existing.strategy_name} holds reservation "
                        f"(expires in {self.RESERVATION_TIMEOUT - elapsed:.0f}s)"
                    )
                    return None

                # Expired - allow new reservation
                logger.info(f"Expired reservation for {symbol} from {existing.strategy_name}")

            # Check for existing ownership by different strategy
            if symbol in self._ownership:
                owner = self._ownership[symbol]
                if owner.strategy_name != strategy_name:
                    logger.warning(
                        f"POSITION CONFLICT: {strategy_name} tried to reserve {symbol} "
                        f"but {owner.strategy_name} owns position of {owner.qty} shares"
                    )
                    return None

            # Create reservation
            reservation = PositionReservation(
                symbol=symbol,
                strategy_name=strategy_name,
                intended_qty=intended_qty,
                intended_side=side,
                reserved_at=datetime.now(),
            )
            self._reservations[symbol] = reservation

            logger.debug(f"Reserved {symbol} for {strategy_name}: {side} {intended_qty} shares")
            return reservation

    async def confirm_order_submitted(
        self,
        symbol: str,
        strategy_name: str,
        order_id: str,
    ) -> bool:
        """
        Mark reservation as submitted (order went to broker).

        Args:
            symbol: Stock symbol
            strategy_name: Name of the strategy
            order_id: Broker order ID

        Returns:
            True if successful, False if reservation not found or mismatch
        """
        async with self._lock:
            if symbol not in self._reservations:
                logger.error(f"No reservation found for {symbol}")
                return False

            reservation = self._reservations[symbol]
            if reservation.strategy_name != strategy_name:
                logger.error(
                    f"Reservation mismatch: {reservation.strategy_name} != {strategy_name}"
                )
                return False

            reservation.order_id = order_id
            reservation.status = "submitted"
            logger.debug(f"Order {order_id} submitted for {symbol} by {strategy_name}")
            return True

    async def confirm_fill(
        self,
        symbol: str,
        strategy_name: str,
        filled_qty: float,
        fill_price: float,
        side: str = "long",
    ) -> bool:
        """
        Confirm order filled - transfer from reservation to ownership.

        Args:
            symbol: Stock symbol
            strategy_name: Name of the strategy
            filled_qty: Actual filled quantity
            fill_price: Average fill price
            side: 'long' or 'short'

        Returns:
            True if successful
        """
        async with self._lock:
            # Update ownership
            self._ownership[symbol] = PositionOwnership(
                symbol=symbol,
                strategy_name=strategy_name,
                qty=filled_qty,
                entry_price=fill_price,
                opened_at=datetime.now(),
                side=side,
            )

            # Track strategy's positions
            if strategy_name not in self._strategy_positions:
                self._strategy_positions[strategy_name] = set()
            self._strategy_positions[strategy_name].add(symbol)

            # Clear reservation
            if symbol in self._reservations:
                del self._reservations[symbol]

            logger.info(
                f"Position confirmed: {strategy_name} owns {filled_qty} {symbol} "
                f"@ ${fill_price:.2f} ({side})"
            )
            return True

    async def release_position(self, symbol: str, strategy_name: str) -> bool:
        """
        Release position ownership when closed.

        Args:
            symbol: Stock symbol
            strategy_name: Name of the strategy

        Returns:
            True if successful, False if not owned by this strategy
        """
        async with self._lock:
            if symbol in self._ownership:
                owner = self._ownership[symbol]
                if owner.strategy_name != strategy_name:
                    logger.error(
                        f"Cannot release {symbol}: owned by {owner.strategy_name}, "
                        f"not {strategy_name}"
                    )
                    return False
                del self._ownership[symbol]
                logger.info(f"Position released: {symbol} by {strategy_name}")

            if strategy_name in self._strategy_positions:
                self._strategy_positions[strategy_name].discard(symbol)

            if symbol in self._reservations:
                del self._reservations[symbol]

            return True

    async def release_reservation(self, symbol: str, strategy_name: str) -> bool:
        """
        Release a reservation without filling (e.g., order rejected).

        Args:
            symbol: Stock symbol
            strategy_name: Name of the strategy

        Returns:
            True if reservation was released
        """
        async with self._lock:
            if symbol in self._reservations:
                reservation = self._reservations[symbol]
                if reservation.strategy_name == strategy_name:
                    del self._reservations[symbol]
                    logger.debug(f"Reservation released for {symbol} by {strategy_name}")
                    return True
            return False

    async def get_strategy_positions(self, strategy_name: str) -> Set[str]:
        """
        Get all symbols owned by a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Set of symbols owned by the strategy
        """
        async with self._lock:
            return self._strategy_positions.get(strategy_name, set()).copy()

    async def is_position_available(self, symbol: str, strategy_name: str) -> bool:
        """
        Check if a position is available for a strategy to trade.

        Args:
            symbol: Stock symbol
            strategy_name: Name of the strategy

        Returns:
            True if the strategy can trade this symbol
        """
        async with self._lock:
            # Check reservation
            if symbol in self._reservations:
                reservation = self._reservations[symbol]
                if reservation.strategy_name != strategy_name:
                    elapsed = (datetime.now() - reservation.reserved_at).total_seconds()
                    if elapsed < self.RESERVATION_TIMEOUT:
                        return False

            # Check ownership
            if symbol in self._ownership:
                if self._ownership[symbol].strategy_name != strategy_name:
                    return False

            return True

    async def get_position_owner(self, symbol: str) -> Optional[str]:
        """
        Get the strategy that owns a position.

        Args:
            symbol: Stock symbol

        Returns:
            Strategy name or None if not owned
        """
        async with self._lock:
            if symbol in self._ownership:
                return self._ownership[symbol].strategy_name
            return None

    async def get_all_positions(self) -> Dict[str, PositionOwnership]:
        """
        Get all position ownerships.

        Returns:
            Dict of symbol -> PositionOwnership
        """
        async with self._lock:
            return self._ownership.copy()

    async def get_positions(self) -> List[PositionOwnership]:
        """
        Return positions as a list for reconciliation compatibility.

        Returns:
            List of PositionOwnership records
        """
        async with self._lock:
            return list(self._ownership.values())

    async def get_all_reservations(self) -> Dict[str, PositionReservation]:
        """
        Get all active reservations.

        Returns:
            Dict of symbol -> PositionReservation
        """
        async with self._lock:
            return self._reservations.copy()

    async def export_state(self) -> Dict:
        """
        Export current ownership/reservation state for persistence.

        Returns:
            Dict suitable for JSON serialization.
        """
        async with self._lock:
            ownership = [
                {
                    "symbol": o.symbol,
                    "strategy_name": o.strategy_name,
                    "qty": o.qty,
                    "entry_price": o.entry_price,
                    "opened_at": o.opened_at.isoformat(),
                    "side": o.side,
                }
                for o in self._ownership.values()
            ]
            reservations = [
                {
                    "symbol": r.symbol,
                    "strategy_name": r.strategy_name,
                    "intended_qty": r.intended_qty,
                    "intended_side": r.intended_side,
                    "reserved_at": r.reserved_at.isoformat(),
                    "order_id": r.order_id,
                    "status": r.status,
                }
                for r in self._reservations.values()
            ]
            strategy_positions = {
                name: list(symbols) for name, symbols in self._strategy_positions.items()
            }
            return {
                "ownership": ownership,
                "reservations": reservations,
                "strategy_positions": strategy_positions,
            }

    async def import_state(self, state: Dict) -> None:
        """
        Import ownership/reservation state from persistence.

        Args:
            state: Dict produced by export_state()
        """
        async with self._lock:
            self._ownership.clear()
            self._reservations.clear()
            self._strategy_positions.clear()

            for item in state.get("ownership", []):
                self._ownership[item["symbol"]] = PositionOwnership(
                    symbol=item["symbol"],
                    strategy_name=item["strategy_name"],
                    qty=float(item["qty"]),
                    entry_price=float(item["entry_price"]),
                    opened_at=datetime.fromisoformat(item["opened_at"]),
                    side=item.get("side", "long"),
                )

            for item in state.get("reservations", []):
                self._reservations[item["symbol"]] = PositionReservation(
                    symbol=item["symbol"],
                    strategy_name=item["strategy_name"],
                    intended_qty=float(item["intended_qty"]),
                    intended_side=item["intended_side"],
                    reserved_at=datetime.fromisoformat(item["reserved_at"]),
                    order_id=item.get("order_id"),
                    status=item.get("status", "pending"),
                )

            for name, symbols in state.get("strategy_positions", {}).items():
                self._strategy_positions[name] = set(symbols)

    async def sync_with_broker(self, broker, default_strategy: str = "unknown"):
        """
        Sync internal state with actual broker positions.

        Call on startup to recover from crashes or sync with
        positions opened outside the bot.

        Args:
            broker: Broker instance with get_positions() method
            default_strategy: Strategy name to assign to unowned positions
        """
        try:
            positions = await broker.get_positions()

            async with self._lock:
                for pos in positions:
                    symbol = pos.symbol
                    if symbol not in self._ownership:
                        # Assign to default strategy
                        self._ownership[symbol] = PositionOwnership(
                            symbol=symbol,
                            strategy_name=default_strategy,
                            qty=float(pos.qty),
                            entry_price=float(pos.avg_entry_price),
                            opened_at=datetime.now(),
                            side="long" if float(pos.qty) > 0 else "short",
                        )

                        if default_strategy not in self._strategy_positions:
                            self._strategy_positions[default_strategy] = set()
                        self._strategy_positions[default_strategy].add(symbol)

                        logger.info(
                            f"Synced position: {symbol} ({pos.qty} shares) -> {default_strategy}"
                        )

        except Exception as e:
            logger.error(f"Error syncing with broker: {e}")

    def sync_positions(
        self, broker_positions: Dict[str, float], default_strategy: str = "reconciled"
    ):
        """
        Sync internal ownership directly from broker position quantities.

        Args:
            broker_positions: Dict of symbol -> qty
            default_strategy: Strategy name assigned to reconciled positions
        """
        self._ownership.clear()
        self._strategy_positions.clear()
        for symbol, qty in broker_positions.items():
            self._ownership[symbol] = PositionOwnership(
                symbol=symbol,
                strategy_name=default_strategy,
                qty=float(qty),
                entry_price=0.0,
                opened_at=datetime.now(),
                side="long" if float(qty) > 0 else "short",
            )
            self._strategy_positions.setdefault(default_strategy, set()).add(symbol)

    async def apply_fill(
        self,
        symbol: str,
        strategy_name: str,
        side: str,
        filled_qty: float,
        fill_price: float,
        delta_qty: float,
    ) -> None:
        """
        Update ownership based on a fill event.

        Args:
            symbol: Stock symbol
            strategy_name: Strategy responsible for the order
            side: 'buy' or 'sell'
            filled_qty: Cumulative filled quantity
            fill_price: Average fill price
        """
        async with self._lock:
            existing = self._ownership.get(symbol)

            if side == "buy":
                if existing and existing.side == "short":
                    # Cover short
                    remaining = existing.qty + delta_qty  # qty is negative
                    if remaining >= 0:
                        del self._ownership[symbol]
                        self._strategy_positions.get(existing.strategy_name, set()).discard(symbol)
                    else:
                        existing.qty = remaining
                else:
                    # Long entry or add
                    if existing and existing.side == "long":
                        new_qty = existing.qty + delta_qty
                        if new_qty > 0:
                            existing.entry_price = (
                                existing.entry_price * existing.qty + fill_price * delta_qty
                            ) / new_qty
                            existing.qty = new_qty
                        else:
                            del self._ownership[symbol]
                            self._strategy_positions.get(existing.strategy_name, set()).discard(
                                symbol
                            )
                    else:
                        self._ownership[symbol] = PositionOwnership(
                            symbol=symbol,
                            strategy_name=strategy_name,
                            qty=float(delta_qty),
                            entry_price=float(fill_price),
                            opened_at=datetime.now(),
                            side="long",
                        )
                        self._strategy_positions.setdefault(strategy_name, set()).add(symbol)
            else:  # sell
                if existing and existing.side == "long":
                    remaining = existing.qty - delta_qty
                    if remaining <= 0:
                        del self._ownership[symbol]
                        self._strategy_positions.get(existing.strategy_name, set()).discard(symbol)
                    else:
                        existing.qty = remaining
                else:
                    # Short entry or add
                    if existing and existing.side == "short":
                        new_qty = existing.qty - delta_qty  # qty negative
                        if new_qty < 0:
                            existing.entry_price = (
                                abs(existing.entry_price * existing.qty) + fill_price * delta_qty
                            ) / abs(new_qty)
                            existing.qty = new_qty
                        else:
                            del self._ownership[symbol]
                            self._strategy_positions.get(existing.strategy_name, set()).discard(
                                symbol
                            )
                    else:
                        self._ownership[symbol] = PositionOwnership(
                            symbol=symbol,
                            strategy_name=strategy_name,
                            qty=-float(delta_qty),
                            entry_price=float(fill_price),
                            opened_at=datetime.now(),
                            side="short",
                        )
                        self._strategy_positions.setdefault(strategy_name, set()).add(symbol)

    async def cleanup_expired_reservations(self):
        """
        Clean up expired reservations.

        Call periodically to prevent stale reservations from blocking trades.
        """
        async with self._lock:
            now = datetime.now()
            expired = []

            for symbol, reservation in self._reservations.items():
                elapsed = (now - reservation.reserved_at).total_seconds()
                if elapsed > self.RESERVATION_TIMEOUT:
                    expired.append(symbol)

            for symbol in expired:
                logger.warning(
                    f"Cleaning up expired reservation for {symbol} "
                    f"(strategy: {self._reservations[symbol].strategy_name})"
                )
                del self._reservations[symbol]

            return len(expired)

    def __repr__(self) -> str:
        return (
            f"PositionManager("
            f"reservations={len(self._reservations)}, "
            f"positions={len(self._ownership)}, "
            f"strategies={len(self._strategy_positions)})"
        )
