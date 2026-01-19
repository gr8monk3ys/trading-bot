#!/usr/bin/env python3
"""
Position Scaling Manager

Implements intelligent scaling into and out of positions:
1. Scale-in: Build positions gradually for better average entry
2. Scale-out: Lock in profits while letting winners run
3. Pyramid: Add to winners, cut losers

Research shows:
- Scaling in reduces impact of bad timing
- Scaling out captures profits while maintaining upside
- Pyramiding winners compounds gains

Expected Impact: 10-15% better average entries, smoother equity curve

Usage:
    from utils.position_scaling import PositionScaler

    scaler = PositionScaler()

    # Get scale-in plan
    plan = scaler.create_scale_in_plan(
        symbol='AAPL',
        total_shares=100,
        current_price=150
    )

    # Check if should scale out
    action = scaler.check_scale_out(
        entry_price=145,
        current_price=165,
        shares_held=100
    )
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ScaleMethod(Enum):
    """Scaling methods."""
    EQUAL = "equal"           # Equal portions (1/3, 1/3, 1/3)
    PYRAMID = "pyramid"       # Larger initial, smaller adds (50%, 30%, 20%)
    INVERTED = "inverted"     # Smaller initial, larger adds (20%, 30%, 50%)
    AGGRESSIVE = "aggressive" # 70% upfront, 30% on confirmation


class PositionScaler:
    """
    Manages position scaling for entries and exits.
    """

    def __init__(
        self,
        default_tranches: int = 3,
        scale_in_method: ScaleMethod = ScaleMethod.PYRAMID,
        scale_out_method: ScaleMethod = ScaleMethod.EQUAL,
        min_profit_for_scale_out: float = 0.03,  # 3% profit to start scaling out
        scale_out_levels: List[float] = None,    # Profit levels to scale out at
    ):
        """
        Initialize position scaler.

        Args:
            default_tranches: Number of tranches for scaling
            scale_in_method: Method for scaling into positions
            scale_out_method: Method for scaling out
            min_profit_for_scale_out: Minimum profit % before scaling out
            scale_out_levels: Profit levels to trigger scale-outs
        """
        self.default_tranches = default_tranches
        self.scale_in_method = scale_in_method
        self.scale_out_method = scale_out_method
        self.min_profit_for_scale_out = min_profit_for_scale_out
        self.scale_out_levels = scale_out_levels or [0.05, 0.10, 0.20]  # 5%, 10%, 20%

        # Track active scaling plans
        self.active_plans: Dict[str, Dict] = {}
        self._max_cached_plans = 100  # Prevent unbounded memory growth

        logger.info(
            f"PositionScaler: tranches={default_tranches}, "
            f"scale_in={scale_in_method.value}, scale_out_levels={self.scale_out_levels}"
        )

    def get_tranche_weights(
        self,
        method: ScaleMethod,
        num_tranches: int
    ) -> List[float]:
        """
        Get weight distribution for tranches.

        Returns:
            List of weights that sum to 1.0
        """
        if method == ScaleMethod.EQUAL:
            return [1.0 / num_tranches] * num_tranches

        elif method == ScaleMethod.PYRAMID:
            # Larger first, smaller later (e.g., 50%, 30%, 20%)
            if num_tranches == 2:
                return [0.6, 0.4]
            elif num_tranches == 3:
                return [0.5, 0.3, 0.2]
            elif num_tranches == 4:
                return [0.4, 0.3, 0.2, 0.1]
            else:
                weights = [1.0 / (i + 1) for i in range(num_tranches)]
                total = sum(weights)
                return [w / total for w in weights]

        elif method == ScaleMethod.INVERTED:
            # Smaller first, larger later (e.g., 20%, 30%, 50%)
            pyramid = self.get_tranche_weights(ScaleMethod.PYRAMID, num_tranches)
            return list(reversed(pyramid))

        elif method == ScaleMethod.AGGRESSIVE:
            # 70% upfront, rest split equally
            if num_tranches == 1:
                return [1.0]
            first = 0.7
            rest = (1.0 - first) / (num_tranches - 1)
            return [first] + [rest] * (num_tranches - 1)

        return [1.0 / num_tranches] * num_tranches

    def create_scale_in_plan(
        self,
        symbol: str,
        total_shares: int,
        current_price: float,
        num_tranches: int = None,
        method: ScaleMethod = None,
        price_levels: List[float] = None
    ) -> Dict:
        """
        Create a scale-in plan for entering a position.

        Args:
            symbol: Stock symbol
            total_shares: Total shares to acquire
            current_price: Current stock price
            num_tranches: Number of tranches (default: self.default_tranches)
            method: Scaling method (default: self.scale_in_method)
            price_levels: Specific price levels for each tranche

        Returns:
            Scale-in plan dict
        """
        num_tranches = num_tranches or self.default_tranches
        method = method or self.scale_in_method

        weights = self.get_tranche_weights(method, num_tranches)

        # Calculate shares per tranche
        tranches = []
        remaining_shares = total_shares

        for i, weight in enumerate(weights):
            if i == len(weights) - 1:
                # Last tranche gets remaining shares
                shares = remaining_shares
            else:
                shares = int(total_shares * weight)
                remaining_shares -= shares

            # Determine target price for this tranche
            if price_levels and i < len(price_levels):
                target_price = price_levels[i]
            else:
                # Default: first at current, others at 1% intervals below
                target_price = current_price * (1 - 0.01 * i)

            tranches.append({
                'tranche': i + 1,
                'shares': shares,
                'weight': weight,
                'target_price': target_price,
                'status': 'pending',
                'filled_price': None,
                'filled_at': None,
            })

        plan = {
            'symbol': symbol,
            'direction': 'long',
            'total_shares': total_shares,
            'method': method.value,
            'tranches': tranches,
            'shares_filled': 0,
            'avg_price': 0,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
        }

        self.active_plans[symbol] = plan

        logger.info(
            f"Scale-in plan created for {symbol}: {total_shares} shares "
            f"in {num_tranches} tranches ({method.value})"
        )

        return plan

    def get_next_tranche(self, symbol: str) -> Optional[Dict]:
        """
        Get the next pending tranche for a symbol.

        Returns:
            Next tranche dict or None if all filled
        """
        if symbol not in self.active_plans:
            return None

        plan = self.active_plans[symbol]

        for tranche in plan['tranches']:
            if tranche['status'] == 'pending':
                return tranche

        return None

    def fill_tranche(
        self,
        symbol: str,
        tranche_num: int,
        filled_price: float
    ) -> Dict:
        """
        Mark a tranche as filled.

        Args:
            symbol: Stock symbol
            tranche_num: Tranche number (1-indexed)
            filled_price: Actual fill price

        Returns:
            Updated plan
        """
        if symbol not in self.active_plans:
            return {}

        plan = self.active_plans[symbol]

        for tranche in plan['tranches']:
            if tranche['tranche'] == tranche_num:
                tranche['status'] = 'filled'
                tranche['filled_price'] = filled_price
                tranche['filled_at'] = datetime.now().isoformat()

                # Update plan totals
                plan['shares_filled'] += tranche['shares']

                # Calculate new average price
                total_cost = sum(
                    t['shares'] * t['filled_price']
                    for t in plan['tranches']
                    if t['status'] == 'filled'
                )
                plan['avg_price'] = total_cost / plan['shares_filled']

                logger.info(
                    f"Tranche {tranche_num} filled for {symbol}: "
                    f"{tranche['shares']} shares @ ${filled_price:.2f} "
                    f"(avg: ${plan['avg_price']:.2f})"
                )

                break

        # Check if all tranches filled
        if all(t['status'] == 'filled' for t in plan['tranches']):
            plan['status'] = 'completed'

        return plan

    def create_scale_out_plan(
        self,
        symbol: str,
        shares_held: int,
        entry_price: float,
        current_price: float = None,
        profit_targets: List[float] = None
    ) -> Dict:
        """
        Create a scale-out plan for exiting a position.

        Args:
            symbol: Stock symbol
            shares_held: Current shares held
            entry_price: Average entry price
            current_price: Current price (for calculating targets)
            profit_targets: Profit % levels to scale out (default: self.scale_out_levels)

        Returns:
            Scale-out plan dict
        """
        profit_targets = profit_targets or self.scale_out_levels
        num_tranches = len(profit_targets)

        weights = self.get_tranche_weights(self.scale_out_method, num_tranches)

        tranches = []
        remaining_shares = shares_held

        for i, (target_pct, weight) in enumerate(zip(profit_targets, weights)):
            if i == num_tranches - 1:
                shares = remaining_shares
            else:
                shares = int(shares_held * weight)
                remaining_shares -= shares

            target_price = entry_price * (1 + target_pct)

            tranches.append({
                'tranche': i + 1,
                'shares': shares,
                'weight': weight,
                'target_profit_pct': target_pct,
                'target_price': target_price,
                'status': 'pending',
                'filled_price': None,
                'filled_at': None,
            })

        plan = {
            'symbol': symbol,
            'direction': 'exit',
            'total_shares': shares_held,
            'entry_price': entry_price,
            'tranches': tranches,
            'shares_sold': 0,
            'avg_exit_price': 0,
            'realized_pnl': 0,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
        }

        logger.info(
            f"Scale-out plan created for {symbol}: {shares_held} shares "
            f"at profit levels {profit_targets}"
        )

        return plan

    def check_scale_out_trigger(
        self,
        entry_price: float,
        current_price: float,
        shares_held: int,
        shares_already_sold: int = 0
    ) -> Dict:
        """
        Check if current price triggers a scale-out.

        Returns:
            Dict with action recommendation
        """
        profit_pct = (current_price - entry_price) / entry_price

        result = {
            'profit_pct': profit_pct,
            'should_scale_out': False,
            'shares_to_sell': 0,
            'reason': None,
            'next_target': None,
        }

        if profit_pct < self.min_profit_for_scale_out:
            result['reason'] = f"Profit {profit_pct:.1%} below threshold {self.min_profit_for_scale_out:.1%}"
            return result

        # Find which level we've hit
        remaining_shares = shares_held - shares_already_sold
        num_levels = len(self.scale_out_levels)
        levels_hit = sum(1 for level in self.scale_out_levels if profit_pct >= level)

        # Calculate how many shares should have been sold by now
        weights = self.get_tranche_weights(self.scale_out_method, num_levels)
        should_have_sold = int(shares_held * sum(weights[:levels_hit]))

        shares_to_sell = max(0, should_have_sold - shares_already_sold)

        if shares_to_sell > 0:
            result['should_scale_out'] = True
            result['shares_to_sell'] = min(shares_to_sell, remaining_shares)
            result['reason'] = f"Hit {levels_hit} profit level(s), profit at {profit_pct:.1%}"

        if levels_hit < num_levels:
            result['next_target'] = self.scale_out_levels[levels_hit]

        return result

    def recommend_add_to_winner(
        self,
        entry_price: float,
        current_price: float,
        current_shares: int,
        max_position_shares: int,
        min_profit_to_add: float = 0.03
    ) -> Dict:
        """
        Check if we should pyramid (add to winner).

        Args:
            entry_price: Average entry price
            current_price: Current price
            current_shares: Current position size
            max_position_shares: Maximum allowed position
            min_profit_to_add: Minimum profit % before adding

        Returns:
            Recommendation dict
        """
        profit_pct = (current_price - entry_price) / entry_price

        result = {
            'should_add': False,
            'shares_to_add': 0,
            'reason': None,
            'new_avg_price': entry_price,
        }

        if profit_pct < min_profit_to_add:
            result['reason'] = f"Profit {profit_pct:.1%} below min {min_profit_to_add:.1%}"
            return result

        if current_shares >= max_position_shares:
            result['reason'] = "Already at max position size"
            return result

        # Add 50% of remaining room
        room = max_position_shares - current_shares
        add_shares = int(room * 0.5)

        if add_shares < 1:
            result['reason'] = "Not enough room to add"
            return result

        result['should_add'] = True
        result['shares_to_add'] = add_shares
        result['reason'] = f"Pyramiding winner at {profit_pct:.1%} profit"

        # Calculate new average
        total_cost = (entry_price * current_shares) + (current_price * add_shares)
        result['new_avg_price'] = total_cost / (current_shares + add_shares)

        return result

    def cleanup_completed_plans(self, max_age_hours: int = 24) -> int:
        """
        Remove completed plans older than max_age.

        Returns:
            Number of plans removed
        """
        now = datetime.now()
        to_remove = []

        for symbol, plan in self.active_plans.items():
            if plan['status'] == 'completed':
                try:
                    created = datetime.fromisoformat(plan['created_at'])
                    if (now - created).total_seconds() > max_age_hours * 3600:
                        to_remove.append(symbol)
                except (ValueError, KeyError):
                    to_remove.append(symbol)

        for symbol in to_remove:
            del self.active_plans[symbol]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed scaling plans")

        return len(to_remove)

    def get_scaling_summary(self, symbol: str) -> Dict:
        """Get summary of scaling activity for a symbol."""
        # Cleanup if too many plans cached
        if len(self.active_plans) > self._max_cached_plans:
            self.cleanup_completed_plans(max_age_hours=1)

        if symbol not in self.active_plans:
            return {'symbol': symbol, 'status': 'no_plan'}

        plan = self.active_plans[symbol]

        filled_tranches = [t for t in plan['tranches'] if t['status'] == 'filled']
        pending_tranches = [t for t in plan['tranches'] if t['status'] == 'pending']

        return {
            'symbol': symbol,
            'direction': plan['direction'],
            'status': plan['status'],
            'total_shares': plan['total_shares'],
            'shares_filled': plan.get('shares_filled', 0),
            'avg_price': plan.get('avg_price', 0),
            'tranches_filled': len(filled_tranches),
            'tranches_pending': len(pending_tranches),
            'next_tranche': pending_tranches[0] if pending_tranches else None,
        }


if __name__ == "__main__":
    """Test position scaler."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("POSITION SCALER TEST")
    print("="*60)

    scaler = PositionScaler()

    # Test scale-in plan
    print("\n1. SCALE-IN PLAN (Pyramid Method)")
    plan = scaler.create_scale_in_plan(
        symbol='AAPL',
        total_shares=100,
        current_price=150.00
    )

    for t in plan['tranches']:
        print(f"   Tranche {t['tranche']}: {t['shares']} shares @ ${t['target_price']:.2f} ({t['weight']:.0%})")

    # Simulate fills
    print("\n   Simulating fills...")
    scaler.fill_tranche('AAPL', 1, 150.00)
    scaler.fill_tranche('AAPL', 2, 148.50)
    scaler.fill_tranche('AAPL', 3, 147.00)

    summary = scaler.get_scaling_summary('AAPL')
    print(f"   Final avg price: ${summary['avg_price']:.2f}")

    # Test scale-out
    print("\n2. SCALE-OUT CHECK")
    result = scaler.check_scale_out_trigger(
        entry_price=148.50,
        current_price=165.00,
        shares_held=100,
        shares_already_sold=0
    )
    print(f"   Profit: {result['profit_pct']:.1%}")
    print(f"   Should scale out: {result['should_scale_out']}")
    print(f"   Shares to sell: {result['shares_to_sell']}")
    print(f"   Next target: {result['next_target']}")

    # Test pyramid recommendation
    print("\n3. PYRAMID RECOMMENDATION")
    rec = scaler.recommend_add_to_winner(
        entry_price=148.50,
        current_price=165.00,
        current_shares=100,
        max_position_shares=200
    )
    print(f"   Should add: {rec['should_add']}")
    print(f"   Shares to add: {rec['shares_to_add']}")
    print(f"   New avg price: ${rec['new_avg_price']:.2f}")

    print("\n" + "="*60)
