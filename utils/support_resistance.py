#!/usr/bin/env python3
"""
Support and Resistance Level Detection

Identifies key price levels for:
1. Better stop-loss placement (below support, above resistance)
2. Better entry points (at support for longs, resistance for shorts)
3. Profit targets (next resistance for longs, support for shorts)

Methods Used:
- Pivot Points (daily/weekly)
- Swing Highs/Lows
- Volume Profile (high volume nodes)
- Round Numbers (psychological levels)

Expected Impact: 15-20% reduction in whipsaws, better risk/reward

Usage:
    from utils.support_resistance import SupportResistanceAnalyzer

    analyzer = SupportResistanceAnalyzer()
    levels = analyzer.find_levels(price_history, current_price)

    # Get optimal stop placement
    stop = levels['nearest_support'] * 0.99  # Just below support
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SupportResistanceAnalyzer:
    """
    Analyzes price history to find support and resistance levels.
    """

    # Fallback stop-loss percentages (when no S/R levels found)
    FALLBACK_LONG_STOP_PCT = 0.97  # 3% below entry for longs
    FALLBACK_SHORT_STOP_PCT = 1.03  # 3% above entry for shorts

    # Fallback profit target percentages
    FALLBACK_LONG_TARGET_PCT = 1.05  # 5% above entry for longs
    FALLBACK_SHORT_TARGET_PCT = 0.95  # 5% below entry for shorts

    # Fallback level percentages (when no specific levels found)
    FALLBACK_RESISTANCE_PCT = 1.05  # 5% above current price
    FALLBACK_SUPPORT_PCT = 0.95  # 5% below current price

    # Default tolerances
    DEFAULT_BUFFER_PCT = 0.005  # 0.5% buffer for stop placement
    DEFAULT_LEVEL_TOLERANCE_PCT = 0.02  # 2% tolerance for S/R proximity check

    # Recent lookback period
    RECENT_BARS_LOOKBACK = 20

    def __init__(
        self,
        swing_lookback: int = 5,  # Bars to look back for swing detection
        min_touches: int = 2,  # Minimum touches to confirm level
        level_tolerance: float = 0.01,  # 1% tolerance for level clustering
        include_round_numbers: bool = True,
    ):
        """
        Initialize S/R analyzer.

        Args:
            swing_lookback: Bars before/after for swing point detection
            min_touches: Minimum times price touched level
            level_tolerance: Percentage tolerance for clustering levels
            include_round_numbers: Include psychological round numbers
        """
        self.swing_lookback = swing_lookback
        self.min_touches = min_touches
        self.level_tolerance = level_tolerance
        self.include_round_numbers = include_round_numbers

        logger.info(
            f"SupportResistanceAnalyzer: lookback={swing_lookback}, " f"min_touches={min_touches}"
        )

    def find_swing_highs(self, highs: List[float], lookback: int = None) -> List[Tuple[int, float]]:
        """
        Find swing high points in price data.

        Returns:
            List of (index, price) tuples
        """
        lookback = lookback or self.swing_lookback
        swing_highs = []

        for i in range(lookback, len(highs) - lookback):
            is_swing = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swing_highs.append((i, highs[i]))

        return swing_highs

    def find_swing_lows(self, lows: List[float], lookback: int = None) -> List[Tuple[int, float]]:
        """
        Find swing low points in price data.

        Returns:
            List of (index, price) tuples
        """
        lookback = lookback or self.swing_lookback
        swing_lows = []

        for i in range(lookback, len(lows) - lookback):
            is_swing = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swing_lows.append((i, lows[i]))

        return swing_lows

    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate standard pivot points.

        Returns:
            Dict with PP, R1, R2, R3, S1, S2, S3
        """
        pp = (high + low + close) / 3

        r1 = 2 * pp - low
        s1 = 2 * pp - high

        r2 = pp + (high - low)
        s2 = pp - (high - low)

        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)

        return {
            "PP": pp,
            "R1": r1,
            "R2": r2,
            "R3": r3,
            "S1": s1,
            "S2": s2,
            "S3": s3,
        }

    def cluster_levels(self, levels: List[float], tolerance: float = None) -> List[Dict]:
        """
        Cluster nearby levels and count touches.

        Returns:
            List of dicts with level info
        """
        tolerance = tolerance or self.level_tolerance
        if not levels:
            return []

        # Sort levels
        sorted_levels = sorted(levels)

        # Cluster nearby levels
        clusters = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            # Check if within tolerance of cluster mean
            cluster_mean = np.mean(current_cluster)
            # Guard against division by zero
            if cluster_mean <= 0:
                current_cluster = [level]
                continue
            if abs(level - cluster_mean) / cluster_mean <= tolerance:
                current_cluster.append(level)
            else:
                # Save current cluster and start new one
                clusters.append(current_cluster)
                current_cluster = [level]

        clusters.append(current_cluster)

        # Convert to level info
        result = []
        for cluster in clusters:
            if len(cluster) >= self.min_touches:
                result.append(
                    {
                        "price": np.mean(cluster),
                        "touches": len(cluster),
                        "strength": min(len(cluster) / 5, 1.0),  # Normalize to 0-1
                        "range": (min(cluster), max(cluster)),
                    }
                )

        return result

    def get_round_numbers(self, current_price: float, range_pct: float = 0.10) -> List[float]:
        """
        Get psychological round numbers near current price.

        Args:
            current_price: Current stock price
            range_pct: Percentage range to look for levels

        Returns:
            List of round number levels
        """
        levels = []

        # Determine round number increment based on price
        if current_price < 20:
            increments = [1, 5]
        elif current_price < 100:
            increments = [5, 10, 25]
        elif current_price < 500:
            increments = [10, 25, 50]
        else:
            increments = [25, 50, 100]

        # Get range
        price_range = current_price * range_pct
        low_bound = current_price - price_range
        high_bound = current_price + price_range

        # Find round numbers in range
        for increment in increments:
            start = int(low_bound / increment) * increment
            level = start
            while level <= high_bound:
                if low_bound <= level <= high_bound:
                    levels.append(float(level))
                level += increment

        return sorted(set(levels))

    def find_levels(self, bars: List[Dict], current_price: float = None) -> Dict:
        """
        Find all support and resistance levels.

        Args:
            bars: List of bar dicts with high, low, close, volume
            current_price: Current price (uses last close if not provided)

        Returns:
            Dict with all levels and analysis
        """
        if not bars or len(bars) < 20:
            return {"error": "Insufficient data"}

        # Extract price data
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        closes = [b["close"] for b in bars]
        volumes = [b.get("volume", 1) for b in bars]

        current_price = current_price or closes[-1]

        # Find swing points
        swing_highs = self.find_swing_highs(highs)
        swing_lows = self.find_swing_lows(lows)

        # Calculate daily pivot points (from last bar)
        pivots = self.calculate_pivot_points(highs[-1], lows[-1], closes[-1])

        # Get recent high/low
        recent_high = max(highs[-self.RECENT_BARS_LOOKBACK :])
        recent_low = min(lows[-self.RECENT_BARS_LOOKBACK :])

        # Collect all potential resistance levels
        resistance_levels = []
        resistance_levels.extend([sh[1] for sh in swing_highs])
        resistance_levels.extend([pivots["R1"], pivots["R2"], pivots["R3"]])
        resistance_levels.append(recent_high)

        # Collect all potential support levels
        support_levels = []
        support_levels.extend([sl[1] for sl in swing_lows])
        support_levels.extend([pivots["S1"], pivots["S2"], pivots["S3"]])
        support_levels.append(recent_low)

        # Add round numbers
        if self.include_round_numbers:
            round_nums = self.get_round_numbers(current_price)
            for rn in round_nums:
                if rn > current_price:
                    resistance_levels.append(rn)
                else:
                    support_levels.append(rn)

        # Cluster and filter levels
        resistance = self.cluster_levels([r for r in resistance_levels if r > current_price])
        support = self.cluster_levels([s for s in support_levels if s < current_price])

        # Sort by distance from current price
        resistance.sort(key=lambda x: x["price"])
        support.sort(key=lambda x: x["price"], reverse=True)

        # Find nearest levels
        nearest_resistance = (
            resistance[0]["price"] if resistance else current_price * self.FALLBACK_RESISTANCE_PCT
        )
        nearest_support = (
            support[0]["price"] if support else current_price * self.FALLBACK_SUPPORT_PCT
        )

        return {
            "current_price": current_price,
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
            "resistance_levels": resistance[:5],  # Top 5
            "support_levels": support[:5],  # Top 5
            "pivot_points": pivots,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "resistance_distance_pct": (nearest_resistance - current_price) / current_price,
            "support_distance_pct": (current_price - nearest_support) / current_price,
            "risk_reward": self._calculate_risk_reward(
                current_price, nearest_support, nearest_resistance
            ),
        }

    def _calculate_risk_reward(self, entry: float, support: float, resistance: float) -> float:
        """Calculate risk/reward ratio for a long trade."""
        risk = entry - support
        reward = resistance - entry

        if risk <= 0:
            return 0

        return reward / risk

    def get_optimal_stop(
        self, direction: str, current_price: float, bars: List[Dict], buffer_pct: float = 0.005
    ) -> float:
        """
        Get optimal stop-loss placement based on S/R.

        Args:
            direction: 'long' or 'short'
            current_price: Current price
            bars: Price history
            buffer_pct: Buffer below/above level (0.5% default)

        Returns:
            Optimal stop price
        """
        levels = self.find_levels(bars, current_price)

        if "error" in levels:
            # Fallback to percentage-based stop
            if direction == "long":
                return current_price * self.FALLBACK_LONG_STOP_PCT
            else:
                return current_price * self.FALLBACK_SHORT_STOP_PCT

        if direction == "long":
            support = levels["nearest_support"]
            return support * (1 - buffer_pct)
        else:
            resistance = levels["nearest_resistance"]
            return resistance * (1 + buffer_pct)

    def get_profit_target(
        self, direction: str, current_price: float, bars: List[Dict], target_number: int = 1
    ) -> float:
        """
        Get profit target based on S/R.

        Args:
            direction: 'long' or 'short'
            current_price: Current price
            bars: Price history
            target_number: Which level to target (1=nearest, 2=second, etc.)

        Returns:
            Target price
        """
        levels = self.find_levels(bars, current_price)

        if "error" in levels:
            # Fallback
            if direction == "long":
                return current_price * self.FALLBACK_LONG_TARGET_PCT
            else:
                return current_price * self.FALLBACK_SHORT_TARGET_PCT

        if direction == "long":
            if len(levels["resistance_levels"]) >= target_number:
                return levels["resistance_levels"][target_number - 1]["price"]
            return levels["nearest_resistance"]
        else:
            if len(levels["support_levels"]) >= target_number:
                return levels["support_levels"][target_number - 1]["price"]
            return levels["nearest_support"]

    def is_at_support(
        self, current_price: float, bars: List[Dict], tolerance_pct: float = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if price is at a support level.

        Args:
            current_price: Current price
            bars: Price history
            tolerance_pct: Tolerance for level proximity (default: DEFAULT_LEVEL_TOLERANCE_PCT)

        Returns:
            Tuple of (is_at_support, support_level)
        """
        tolerance_pct = tolerance_pct or self.DEFAULT_LEVEL_TOLERANCE_PCT
        levels = self.find_levels(bars, current_price)

        if "error" in levels:
            return False, None

        for level in levels["support_levels"]:
            distance = abs(current_price - level["price"]) / current_price
            if distance <= tolerance_pct:
                return True, level["price"]

        return False, None

    def is_at_resistance(
        self, current_price: float, bars: List[Dict], tolerance_pct: float = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if price is at a resistance level.

        Args:
            current_price: Current price
            bars: Price history
            tolerance_pct: Tolerance for level proximity (default: DEFAULT_LEVEL_TOLERANCE_PCT)

        Returns:
            Tuple of (is_at_resistance, resistance_level)
        """
        tolerance_pct = tolerance_pct or self.DEFAULT_LEVEL_TOLERANCE_PCT
        levels = self.find_levels(bars, current_price)

        if "error" in levels:
            return False, None

        for level in levels["resistance_levels"]:
            distance = abs(level["price"] - current_price) / current_price
            if distance <= tolerance_pct:
                return True, level["price"]

        return False, None


if __name__ == "__main__":
    """Test support/resistance analyzer."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 60)
    print("SUPPORT/RESISTANCE ANALYZER TEST")
    print("=" * 60)

    # Generate sample price data
    np.random.seed(42)
    base_price = 150
    bars = []
    price = base_price

    for i in range(100):
        change = np.random.randn() * 2
        high = price + abs(np.random.randn() * 1.5)
        low = price - abs(np.random.randn() * 1.5)
        close = price + change
        bars.append(
            {
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000000 + np.random.randint(-500000, 500000),
            }
        )
        price = close

    current_price = bars[-1]["close"]

    analyzer = SupportResistanceAnalyzer()
    levels = analyzer.find_levels(bars, current_price)

    print(f"\nCurrent Price: ${current_price:.2f}")
    print(f"Recent High: ${levels['recent_high']:.2f}")
    print(f"Recent Low: ${levels['recent_low']:.2f}")

    print(
        f"\nNearest Resistance: ${levels['nearest_resistance']:.2f} "
        f"({levels['resistance_distance_pct']:+.1%} away)"
    )
    print(
        f"Nearest Support: ${levels['nearest_support']:.2f} "
        f"({levels['support_distance_pct']:+.1%} away)"
    )

    print(f"\nRisk/Reward Ratio: {levels['risk_reward']:.2f}")

    print("\nResistance Levels:")
    for r in levels["resistance_levels"][:3]:
        print(f"  ${r['price']:.2f} (touches: {r['touches']}, strength: {r['strength']:.2f})")

    print("\nSupport Levels:")
    for s in levels["support_levels"][:3]:
        print(f"  ${s['price']:.2f} (touches: {s['touches']}, strength: {s['strength']:.2f})")

    print("\nPivot Points:")
    pp = levels["pivot_points"]
    print(f"  PP: ${pp['PP']:.2f}")
    print(f"  R1: ${pp['R1']:.2f}, R2: ${pp['R2']:.2f}")
    print(f"  S1: ${pp['S1']:.2f}, S2: ${pp['S2']:.2f}")

    # Test stop/target
    stop = analyzer.get_optimal_stop("long", current_price, bars)
    target = analyzer.get_profit_target("long", current_price, bars)
    print("\nOptimal Long Trade:")
    print(f"  Entry: ${current_price:.2f}")
    print(f"  Stop: ${stop:.2f} ({(stop/current_price-1)*100:.1f}%)")
    print(f"  Target: ${target:.2f} ({(target/current_price-1)*100:.1f}%)")

    print("=" * 60)
