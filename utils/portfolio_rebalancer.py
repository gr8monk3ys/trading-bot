#!/usr/bin/env python3
"""
Portfolio Rebalancer

Automatically rebalances portfolio to maintain target allocations.

Rebalancing Benefits:
1. Maintains risk profile (prevents over-concentration)
2. Enforces discipline (sell winners, buy losers)
3. Mean reversion benefit (buy low, sell high)
4. Reduces volatility

Example usage:
    # Initialize rebalancer
    rebalancer = PortfolioRebalancer(
        broker=broker,
        target_allocations={'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25},
        rebalance_threshold=0.05,  # Rebalance when drift > 5%
        rebalance_frequency='daily'
    )

    # Check if rebalancing needed
    if await rebalancer.needs_rebalancing():
        orders = await rebalancer.generate_rebalance_orders()
        await rebalancer.execute_rebalancing(orders)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine.

    Supports multiple rebalancing strategies:
    - Equal weight: All positions same size
    - Target weight: Custom allocation per symbol
    - Threshold: Only rebalance when drift exceeds threshold
    - Periodic: Rebalance on schedule (daily, weekly, monthly)
    """

    def __init__(
        self,
        broker,
        target_allocations: Optional[Dict[str, float]] = None,
        rebalance_threshold: float = 0.05,
        rebalance_frequency: str = "weekly",
        equal_weight_symbols: Optional[List[str]] = None,
        min_trade_size: float = 100.0,  # Minimum dollar value to trade
        dry_run: bool = False,
    ):
        """
        Initialize portfolio rebalancer.

        Args:
            broker: Broker instance
            target_allocations: Dict of symbol -> target weight (must sum to 1.0)
                               e.g., {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.50}
            rebalance_threshold: Rebalance when position drifts > this amount (0.05 = 5%)
            rebalance_frequency: 'daily', 'weekly', 'monthly'
            equal_weight_symbols: If provided, use equal weighting across these symbols
            min_trade_size: Minimum trade size in dollars (avoid tiny trades)
            dry_run: If True, log actions but don't execute orders
        """
        self.broker = broker
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_frequency = rebalance_frequency
        self.min_trade_size = min_trade_size
        self.dry_run = dry_run

        # Set up target allocations
        if equal_weight_symbols:
            # Equal weight: divide portfolio equally
            n = len(equal_weight_symbols)
            self.target_allocations = dict.fromkeys(equal_weight_symbols, 1.0 / n)
        elif target_allocations:
            # Validate allocations sum to 1.0
            total = sum(target_allocations.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Target allocations must sum to 1.0, got {total}")
            self.target_allocations = target_allocations
        else:
            raise ValueError("Must provide either target_allocations or equal_weight_symbols")

        # Tracking
        self.last_rebalance = None
        self.rebalance_history = []

        logger.info("Portfolio rebalancer initialized:")
        logger.info(f"  Target allocations: {self.target_allocations}")
        logger.info(f"  Threshold: {self.rebalance_threshold:.1%}")
        logger.info(f"  Frequency: {self.rebalance_frequency}")
        logger.info(f"  Dry run: {self.dry_run}")

    async def get_current_allocations(self) -> Dict[str, float]:
        """
        Get current portfolio allocations.

        Returns:
            Dict of symbol -> current weight in portfolio
        """
        try:
            # Get account value
            account = await self.broker.get_account()
            total_value = float(account.equity)

            if total_value == 0:
                logger.warning("Portfolio value is zero")
                return {}

            # Get current positions
            positions = await self.broker.get_positions()

            # Calculate current allocations
            current_allocations = {}
            for symbol in self.target_allocations.keys():
                position = next((p for p in positions if p.symbol == symbol), None)

                if position:
                    position_value = float(position.market_value)
                    weight = position_value / total_value
                    current_allocations[symbol] = weight
                else:
                    current_allocations[symbol] = 0.0

            return current_allocations

        except Exception as e:
            logger.error(f"Error getting current allocations: {e}")
            return {}

    async def calculate_drift(self) -> Dict[str, float]:
        """
        Calculate how much each position has drifted from target.

        Returns:
            Dict of symbol -> drift amount (negative = underweight, positive = overweight)
        """
        current = await self.get_current_allocations()
        target = self.target_allocations

        drift = {}
        for symbol in target.keys():
            current_weight = current.get(symbol, 0.0)
            target_weight = target[symbol]
            drift[symbol] = current_weight - target_weight

        return drift

    async def needs_rebalancing(self) -> bool:
        """
        Check if portfolio needs rebalancing.

        Returns:
            True if rebalancing is needed based on threshold and frequency
        """
        # Check frequency
        if not await self._should_rebalance_by_schedule():
            return False

        # Check drift threshold
        drift = await self.calculate_drift()

        # Rebalance if any position has drifted beyond threshold
        max_drift = max(abs(d) for d in drift.values())

        if max_drift > self.rebalance_threshold:
            logger.info(
                f"Rebalancing needed: max drift = {max_drift:.1%} (threshold = {self.rebalance_threshold:.1%})"
            )
            return True

        logger.debug(f"No rebalancing needed: max drift = {max_drift:.1%}")
        return False

    async def _should_rebalance_by_schedule(self) -> bool:
        """Check if enough time has passed since last rebalance."""
        if self.last_rebalance is None:
            return True

        now = datetime.now()
        time_since_rebalance = now - self.last_rebalance

        if self.rebalance_frequency == "daily":
            return time_since_rebalance >= timedelta(days=1)
        elif self.rebalance_frequency == "weekly":
            return time_since_rebalance >= timedelta(weeks=1)
        elif self.rebalance_frequency == "monthly":
            return time_since_rebalance >= timedelta(days=30)
        else:
            return True

    async def generate_rebalance_orders(self) -> List[Dict]:
        """
        Generate orders to rebalance portfolio to target allocations.

        Returns:
            List of order dicts with symbol, side, quantity, and reason
        """
        try:
            # Get current state
            account = await self.broker.get_account()
            total_value = float(account.equity)
            current_allocations = await self.get_current_allocations()
            drift = await self.calculate_drift()

            # Get current prices
            positions = await self.broker.get_positions()
            prices = {}
            for symbol in self.target_allocations.keys():
                position = next((p for p in positions if p.symbol == symbol), None)
                if position:
                    prices[symbol] = float(position.current_price)
                else:
                    # Need to fetch price if no position
                    try:
                        quote = await self.broker.get_quote(symbol)
                        prices[symbol] = float(quote.ask)
                    except Exception as e:
                        logger.error(f"Could not get price for {symbol}: {e}")
                        prices[symbol] = None

            # Generate orders
            orders = []
            for symbol, target_weight in self.target_allocations.items():
                current_weight = current_allocations.get(symbol, 0.0)
                symbol_drift = drift[symbol]

                if prices.get(symbol) is None:
                    logger.warning(f"Skipping {symbol} - no price available")
                    continue

                # Calculate target dollar value for this symbol
                target_value = total_value * target_weight
                current_value = total_value * current_weight

                # Calculate adjustment needed
                adjustment_value = target_value - current_value

                # Skip if adjustment is too small
                if abs(adjustment_value) < self.min_trade_size:
                    logger.debug(
                        f"Skipping {symbol}: adjustment ${adjustment_value:.2f} < min ${self.min_trade_size:.2f}"
                    )
                    continue

                # Calculate quantity to trade
                quantity = abs(adjustment_value) / prices[symbol]

                # Determine side
                if adjustment_value > 0:
                    side = "buy"
                    action = "underweight"
                else:
                    side = "sell"
                    action = "overweight"

                orders.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": prices[symbol],
                        "value": abs(adjustment_value),
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "drift": symbol_drift,
                        "reason": f"{action} ({symbol_drift:+.1%} drift)",
                    }
                )

            # Sort by absolute drift (rebalance biggest drifts first)
            orders.sort(key=lambda x: abs(x["drift"]), reverse=True)

            return orders

        except Exception as e:
            logger.error(f"Error generating rebalance orders: {e}", exc_info=True)
            return []

    async def execute_rebalancing(self, orders: List[Dict]) -> Dict:
        """
        Execute rebalancing orders.

        Args:
            orders: List of order dicts from generate_rebalance_orders()

        Returns:
            Dict with summary of rebalancing results
        """
        try:
            if not orders:
                logger.info("No rebalancing orders to execute")
                return {"status": "no_action", "orders_executed": 0}

            logger.info("\n" + "=" * 80)
            logger.info("🔄 PORTFOLIO REBALANCING")
            logger.info("=" * 80)
            logger.info(f"Rebalancing {len(orders)} positions:")

            for order in orders:
                logger.info(f"\n  {order['symbol']}:")
                logger.info(
                    f"    Current: {order['current_weight']:.1%} → Target: {order['target_weight']:.1%}"
                )
                logger.info(f"    Drift: {order['drift']:+.1%}")
                logger.info(
                    f"    Action: {order['side'].upper()} {order['quantity']:.4f} shares (${order['value']:.2f})"
                )
                logger.info(f"    Reason: {order['reason']}")

            logger.info(f"{'='*80}\n")

            if self.dry_run:
                logger.info("DRY RUN MODE - Orders not executed")
                return {"status": "dry_run", "orders_generated": len(orders)}

            # Execute orders
            from brokers.order_builder import OrderBuilder

            executed = 0
            failed = 0

            for order in orders:
                try:
                    # Create market order
                    alpaca_order = (
                        OrderBuilder(order["symbol"], order["side"], order["quantity"])
                        .market()
                        .day()
                        .build()
                    )

                    result = await self.broker.submit_order_advanced(alpaca_order)

                    if result:
                        logger.info(
                            f"✅ Rebalance order executed: {order['side']} {order['quantity']:.4f} {order['symbol']}"
                        )
                        executed += 1
                    else:
                        logger.error(f"❌ Failed to execute: {order['symbol']}")
                        failed += 1

                except Exception as e:
                    logger.error(f"Error executing rebalance order for {order['symbol']}: {e}")
                    failed += 1

            # Update tracking
            self.last_rebalance = datetime.now()
            self.rebalance_history.append(
                {
                    "timestamp": self.last_rebalance,
                    "orders_executed": executed,
                    "orders_failed": failed,
                }
            )

            logger.info(f"\n✅ Rebalancing complete: {executed} orders executed, {failed} failed")

            return {
                "status": "success",
                "orders_executed": executed,
                "orders_failed": failed,
                "timestamp": self.last_rebalance,
            }

        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def get_rebalance_report(self) -> str:
        """
        Generate a rebalancing status report.

        Returns:
            Formatted string report
        """
        current = await self.get_current_allocations()
        drift = await self.calculate_drift()

        report = []
        report.append("\n" + "=" * 80)
        report.append("PORTFOLIO REBALANCING REPORT")
        report.append("=" * 80)

        report.append("\nTarget Allocations:")
        for symbol, target in self.target_allocations.items():
            current_weight = current.get(symbol, 0.0)
            symbol_drift = drift.get(symbol, 0.0)

            status = "✅" if abs(symbol_drift) < self.rebalance_threshold else "⚠️"

            report.append(
                f"  {status} {symbol}: "
                f"{current_weight:.1%} (target: {target:.1%}, "
                f"drift: {symbol_drift:+.1%})"
            )

        needs_rebal = await self.needs_rebalancing()
        report.append(f"\nRebalancing needed: {'YES ⚠️' if needs_rebal else 'NO ✅'}")

        if self.last_rebalance:
            report.append(f"Last rebalanced: {self.last_rebalance.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            report.append("Last rebalanced: Never")

        report.append("=" * 80)

        return "\n".join(report)
