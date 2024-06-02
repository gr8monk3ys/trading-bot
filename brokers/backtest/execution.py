"""
Order-execution simulation for :class:`BacktestBroker`.

Provides :class:`BacktestBrokerExecutionMixin`, which implements:
- Slippage modeling (Almgren-Chriss inspired market impact).
- Partial fill simulation for large orders.
- Execution latency sampling.
- :meth:`place_order` — the main entry point that records orders,
  applies slippage/partial fills, updates positions and cash balance,
  and records trades.

Assumes the host class provides the attributes and methods from
:class:`brokers.backtest.core.BacktestBrokerCore` (positions, orders,
trades, ``_calculate_dynamic_spread``, ``_get_actual_daily_volume``,
``_get_stock_volatility``, ``get_price``, ``execution_profile``, ...).
"""

import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class BacktestBrokerExecutionMixin:
    """Mixin providing order placement and fill simulation."""

    def _calculate_slippage(self, symbol, quantity, side, base_price, order_type):
        """
        Calculate realistic slippage based on order characteristics.

        Uses Almgren-Chriss inspired market impact model with:
        - Dynamic spread based on liquidity/volatility
        - Actual historical volume (not hardcoded)
        - Temporary + permanent impact components

        Args:
            symbol: Stock symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            base_price: Base price (mid price)
            order_type: 'market' or 'limit'

        Returns:
            Execution price after slippage
        """
        current_date = self._current_date if self._current_date else datetime.now()

        # 1. Dynamic bid-ask spread based on liquidity and volatility
        dynamic_spread_bps = self._calculate_dynamic_spread(symbol, current_date, self.spread_bps)
        spread_cost = base_price * (dynamic_spread_bps / 10000.0)

        # 2. Get ACTUAL daily volume (not hardcoded 1M)
        avg_daily_volume = self._get_actual_daily_volume(symbol, current_date)
        participation_rate = quantity / avg_daily_volume

        # 3. Almgren-Chriss inspired market impact model
        # Temporary impact: I_temp = c * sigma * sqrt(participation_rate)
        # Permanent impact: I_perm = d * sigma * participation_rate
        volatility = self._get_stock_volatility(symbol, current_date)

        c_temp = 0.6  # Temporary impact coefficient
        d_perm = 0.15  # Permanent impact coefficient

        temporary_impact_pct = c_temp * volatility * np.sqrt(participation_rate)
        permanent_impact_pct = d_perm * volatility * participation_rate

        # Total impact (capped at 10%)
        total_impact_pct = min(temporary_impact_pct + permanent_impact_pct, 0.10)
        market_impact_cost = base_price * total_impact_pct

        # 4. Market orders pay spread + impact; limit orders pay reduced impact
        if order_type == "market":
            total_slippage = spread_cost + market_impact_cost
        else:  # limit orders pay less (assuming they get filled at limit)
            total_slippage = market_impact_cost * 0.3  # 30% of market order slippage

        total_slippage *= self.execution_profile.slippage_multiplier

        # 5. Apply slippage direction
        if side == "buy":
            execution_price = base_price + total_slippage  # Buy at higher price
        else:  # sell
            execution_price = base_price - total_slippage  # Sell at lower price

        # Log significant slippage for analysis
        if participation_rate > 0.05:
            logger.debug(
                f"Market impact for {symbol}: {participation_rate:.1%} of ADV, "
                f"impact: {(execution_price/base_price - 1)*100:+.3f}%"
            )

        return execution_price

    def _simulate_partial_fill(self, quantity, symbol):
        """
        Simulate partial fills for large orders.

        Large orders may not fill completely, especially in backtest.
        This prevents unrealistic fills of huge positions.

        Args:
            quantity: Requested quantity
            symbol: Stock symbol

        Returns:
            Actual filled quantity (may be less than requested)
        """
        if not self.enable_partial_fills:
            return quantity

        # Use ACTUAL daily volume (not hardcoded 1M)
        current_date = self._current_date if self._current_date else datetime.now()
        avg_daily_volume = self._get_actual_daily_volume(symbol, current_date)

        # If order is >10% of daily volume, may not fill completely
        participation_rate = quantity / avg_daily_volume

        if participation_rate > 0.10:  # Order is >10% of daily volume
            # Fill rate decreases as participation rate increases
            # At 10% participation: ~95% fill
            # At 50% participation: ~70% fill
            # At 100%+ participation: ~50% fill
            if participation_rate >= 1.0:
                fill_rate = 0.5 + (self._rng.random() * 0.15)
            elif participation_rate >= 0.5:
                fill_rate = 0.65 + (self._rng.random() * 0.15)
            elif participation_rate >= 0.2:
                fill_rate = 0.75 + (self._rng.random() * 0.15)
            else:  # 10-20%
                fill_rate = 0.85 + (self._rng.random() * 0.10)

            fill_rate *= self.execution_profile.partial_fill_multiplier
            fill_rate = max(0.05, min(fill_rate, 1.0))
            filled_qty = int(quantity * fill_rate)
            logger.warning(
                f"Partial fill for {symbol}: {filled_qty}/{quantity} "
                f"({fill_rate:.1%}) - order is {participation_rate:.1%} of ADV"
            )
            return max(filled_qty, 1)  # Fill at least 1 share

        return quantity

    def _sample_execution_latency_ms(self) -> int:
        """Sample simulated venue/network latency."""
        return int(
            self._rng.integers(
                self.execution_profile.min_latency_ms,
                self.execution_profile.max_latency_ms + 1,
            )
        )

    def place_order(self, symbol, quantity, side, price=None, order_type="market"):
        """
        Place an order with realistic slippage and partial fills.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            price: Limit price (optional, for limit orders)
            order_type: 'market' or 'limit'

        Returns:
            Order dict with execution details
        """
        current_date = self._current_date if self._current_date else datetime.now()
        latency_ms = self._sample_execution_latency_ms()

        # Get base price (mid price)
        base_price = price if price else self.get_price(symbol, current_date)

        if self._rng.random() < self.execution_profile.reject_probability:
            order = {
                "id": len(self.orders) + 1,
                "run_id": self.run_id,
                "symbol": symbol,
                "quantity": quantity,
                "filled_qty": 0,
                "side": side,
                "price": base_price,
                "filled_avg_price": None,
                "type": order_type,
                "status": "rejected",
                "created_at": current_date,
                "filled_at": None,
                "slippage_bps": 0.0,
                "latency_ms": latency_ms,
                "execution_profile": self.execution_profile.name,
                "rejection_reason": "simulated_liquidity_reject",
            }
            self.orders.append(order)
            return order

        # Apply slippage to get realistic execution price
        execution_price = self._calculate_slippage(symbol, quantity, side, base_price, order_type)

        # Simulate partial fills for large orders
        filled_quantity = self._simulate_partial_fill(quantity, symbol)

        order = {
            "id": len(self.orders) + 1,
            "run_id": self.run_id,
            "symbol": symbol,
            "quantity": quantity,
            "filled_qty": filled_quantity,  # Actual filled amount
            "side": side,
            "price": base_price,  # Requested price
            "filled_avg_price": execution_price,  # Actual execution price (with slippage)
            "type": order_type,
            "status": "filled" if filled_quantity == quantity else "partially_filled",
            "created_at": current_date,
            "filled_at": current_date,
            "slippage_bps": (
                abs((execution_price - base_price) / base_price) * 10000 if base_price else 0.0
            ),
            "latency_ms": latency_ms,
            "execution_profile": self.execution_profile.name,
        }

        self.orders.append(order)

        # Update positions and cash using FILLED quantity and EXECUTION price
        cost = filled_quantity * execution_price

        if side == "buy":
            self.balance -= cost
            if symbol in self.positions:
                existing_qty = self.positions[symbol]["quantity"]
                existing_entry = self.positions[symbol]["entry_price"]
                new_qty = existing_qty + filled_quantity

                if existing_qty < 0:
                    # Covering a short position: avoid averaging long entry across a zero-cross.
                    if new_qty < 0:
                        self.positions[symbol]["quantity"] = new_qty
                    elif new_qty == 0:
                        del self.positions[symbol]
                    else:
                        self.positions[symbol] = {
                            "symbol": symbol,
                            "quantity": new_qty,
                            "entry_price": execution_price,
                        }
                else:
                    # Long-only path with safe average-price update.
                    if new_qty > 0:
                        prev_cost = existing_entry * existing_qty
                        self.positions[symbol]["quantity"] = new_qty
                        self.positions[symbol]["entry_price"] = (prev_cost + cost) / new_qty
                    elif new_qty == 0:
                        del self.positions[symbol]
                    else:
                        # Defensive fallback: do not keep negative quantity in long-only path.
                        del self.positions[symbol]
            elif filled_quantity > 0:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": filled_quantity,
                    "entry_price": execution_price,
                }
        else:  # sell
            self.balance += cost
            if symbol in self.positions:
                existing_qty = self.positions[symbol]["quantity"]
                new_qty = existing_qty - filled_quantity

                if existing_qty <= 0:
                    # Existing short position and additional sell keeps/increases short.
                    if new_qty < 0:
                        existing_abs = abs(existing_qty)
                        added_abs = filled_quantity
                        total_abs = existing_abs + added_abs
                        if total_abs > 0:
                            avg_entry = (
                                (self.positions[symbol]["entry_price"] * existing_abs)
                                + (execution_price * added_abs)
                            ) / total_abs
                            self.positions[symbol]["entry_price"] = avg_entry
                        self.positions[symbol]["quantity"] = new_qty
                    elif new_qty == 0:
                        del self.positions[symbol]
                    else:
                        # Defensive fallback for unexpected sign flip in sell path.
                        del self.positions[symbol]
                else:
                    self.positions[symbol]["quantity"] = new_qty
                    if new_qty <= 0:
                        del self.positions[symbol]

        # Record the trade with actual execution details
        self.trades.append(
            {
                "id": len(self.trades) + 1,
                "run_id": self.run_id,
                "symbol": symbol,
                "quantity": filled_quantity,
                "side": side,
                "price": execution_price,
                "slippage": execution_price - base_price,
                "timestamp": current_date,
            }
        )

        return order
