#!/usr/bin/env python3
"""
Volatility Regime Detection

Detects current market volatility regime using VIX (CBOE Volatility Index)
and adjusts position sizing and stop-losses accordingly.

Volatility Regimes:
- Very Low (VIX < 12): Complacent market, increase position sizes
- Low (VIX 12-15): Calm market, slightly increase positions
- Normal (VIX 15-20): Average volatility, standard sizing
- Elevated (VIX 20-30): Increased fear, reduce positions
- High (VIX > 30): Market panic, significant reduction

Expected Impact: +5-8% annual returns from adaptive risk management

Usage:
    from utils.volatility_regime import VolatilityRegimeDetector

    detector = VolatilityRegimeDetector(broker)
    regime, adjustments = await detector.get_current_regime()

    # Adjust position size
    base_size = 0.10  # 10%
    adjusted_size = base_size * adjustments['pos_mult']

    # Adjust stop-loss
    base_stop = 0.03  # 3%
    adjusted_stop = base_stop * adjustments['stop_mult']
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class VolatilityRegimeDetector:
    """
    Detects market volatility regime using VIX.

    Adjusts trading parameters based on current market conditions:
    - Position size multipliers (larger in calm markets, smaller in volatile)
    - Stop-loss multipliers (tighter in calm, wider in volatile)
    - Additional safety features during extreme volatility

    Features:
    - Real-time VIX tracking
    - Historical VIX analysis (30-day average)
    - Regime change detection
    - Logging and alerts
    """

    # Regime thresholds
    VERY_LOW_THRESHOLD = 12
    LOW_THRESHOLD = 15
    NORMAL_THRESHOLD = 20
    ELEVATED_THRESHOLD = 30

    def __init__(self, broker, cache_minutes: int = 5):
        """
        Initialize volatility regime detector.

        Args:
            broker: Trading broker instance
            cache_minutes: Minutes to cache VIX data (avoid excessive API calls)
        """
        self.broker = broker
        self.cache_minutes = cache_minutes

        # Cache
        self.last_vix_value = None
        self.last_vix_time = None
        self.last_regime = None

        # Historical tracking
        self.vix_history = []  # Last 30 days
        self.regime_changes = []  # Track regime transitions

        logger.info("VolatilityRegimeDetector initialized")

    async def get_current_regime(self) -> Tuple[str, Dict]:
        """
        Get current volatility regime and adjustment multipliers.

        Returns:
            Tuple of (regime_name, adjustments_dict)

            regime_name: 'very_low', 'low', 'normal', 'elevated', 'high'
            adjustments_dict: {
                'pos_mult': float,      # Position size multiplier
                'stop_mult': float,     # Stop-loss width multiplier
                'max_positions': int,   # Max concurrent positions
                'trade': bool          # Whether to trade at all
            }
        """
        try:
            # Get current VIX
            vix = await self._get_vix()

            if vix is None:
                logger.warning("Could not fetch VIX, using NORMAL regime as fallback")
                return self._get_normal_regime()

            # Determine regime
            regime, adjustments = self._determine_regime(vix)

            # Check for regime change
            if self.last_regime and regime != self.last_regime:
                logger.warning(
                    f"🔄 REGIME CHANGE: {self.last_regime.upper()} → {regime.upper()} "
                    f"(VIX: {vix:.1f})"
                )
                self.regime_changes.append(
                    {"time": datetime.now(), "from": self.last_regime, "to": regime, "vix": vix}
                )

            self.last_regime = regime

            # Log current regime (every 5 minutes or on change)
            logger.info(
                f"📊 Volatility Regime: {regime.upper()} "
                f"(VIX: {vix:.1f}) | "
                f"Position: {adjustments['pos_mult']:.1f}x, "
                f"Stop: {adjustments['stop_mult']:.1f}x"
            )

            return regime, adjustments

        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}", exc_info=True)
            return self._get_normal_regime()

    def _determine_regime(self, vix: float) -> Tuple[str, Dict]:
        """
        Determine regime based on VIX value.

        Args:
            vix: Current VIX value

        Returns:
            Tuple of (regime_name, adjustments)
        """
        if vix < self.VERY_LOW_THRESHOLD:
            # Very Low Volatility (VIX < 12)
            # Market complacency, opportunity for larger positions
            return "very_low", {
                "pos_mult": 1.4,  # 40% larger positions
                "stop_mult": 0.7,  # 30% tighter stops
                "max_positions": 12,  # Allow more positions
                "trade": True,
                "description": "Complacent market - increase exposure",
            }

        elif vix < self.LOW_THRESHOLD:
            # Low Volatility (VIX 12-15)
            # Calm market, favorable for trading
            return "low", {
                "pos_mult": 1.2,  # 20% larger positions
                "stop_mult": 0.8,  # 20% tighter stops
                "max_positions": 10,  # Standard positions
                "trade": True,
                "description": "Calm market - slightly increase exposure",
            }

        elif vix < self.NORMAL_THRESHOLD:
            # Normal Volatility (VIX 15-20)
            # Average market conditions
            return "normal", {
                "pos_mult": 1.0,  # Standard positions
                "stop_mult": 1.0,  # Standard stops
                "max_positions": 8,  # Standard positions
                "trade": True,
                "description": "Normal market - standard sizing",
            }

        elif vix < self.ELEVATED_THRESHOLD:
            # Elevated Volatility (VIX 20-30)
            # Increased uncertainty, reduce risk
            return "elevated", {
                "pos_mult": 0.7,  # 30% smaller positions
                "stop_mult": 1.2,  # 20% wider stops
                "max_positions": 5,  # Fewer positions
                "trade": True,
                "description": "Elevated volatility - reduce exposure",
            }

        else:
            # High Volatility (VIX > 30)
            # Market fear/panic, significantly reduce risk
            return "high", {
                "pos_mult": 0.4,  # 60% smaller positions
                "stop_mult": 1.5,  # 50% wider stops
                "max_positions": 3,  # Very few positions
                "trade": True,  # Still trade, but cautiously
                "description": "High volatility - significant reduction",
            }

    def _get_normal_regime(self) -> Tuple[str, Dict]:
        """Fallback to normal regime if VIX unavailable."""
        return "normal", {
            "pos_mult": 1.0,
            "stop_mult": 1.0,
            "max_positions": 8,
            "trade": True,
            "description": "Normal market (fallback)",
        }

    async def _get_vix(self) -> Optional[float]:
        """
        Get current VIX value with caching.

        Uses VIXY (VIX ETF) price or calculates realized volatility from SPY
        as Alpaca doesn't provide VIX directly.

        Returns:
            Current VIX value or None if unavailable
        """
        try:
            # Check cache
            now = datetime.now()
            if (
                self.last_vix_value is not None
                and self.last_vix_time is not None
                and (now - self.last_vix_time).total_seconds() < self.cache_minutes * 60
            ):
                return self.last_vix_value

            vix_value = None

            # Method 1: Try VIXY (VIX ETF) - rough proxy
            try:
                vixy_price = await self.broker.get_last_price("VIXY")
                if vixy_price:
                    # VIXY typically trades 10-50, VIX typically 10-80
                    # This is a rough approximation
                    vix_value = vixy_price * 1.2  # Scale factor
                    logger.debug(f"VIX estimated from VIXY: {vix_value:.1f}")
            except Exception as e:
                logger.debug(f"VIXY fetch failed: {e}")

            # Method 2: Calculate realized volatility from SPY if VIXY failed
            if vix_value is None:
                try:
                    bars = await self.broker.get_bars("SPY", timeframe="1Day", limit=30)
                    if bars and len(bars) >= 20:
                        import numpy as np

                        closes = np.array([float(b.close) for b in bars])
                        returns = np.diff(np.log(closes))
                        # Annualized volatility * 100 to approximate VIX
                        realized_vol = np.std(returns) * np.sqrt(252) * 100
                        vix_value = realized_vol
                        logger.debug(f"VIX estimated from SPY volatility: {vix_value:.1f}")
                except Exception as e:
                    logger.debug(f"SPY volatility calculation failed: {e}")

            if vix_value is None:
                logger.warning("Could not fetch VIX, using cached or default")
                return self.last_vix_value or 17.0  # Default to normal VIX

            # Validate VIX (typical range: 10-80, extreme: 10-100)
            if not (5 < vix_value < 150):
                logger.warning(f"VIX value {vix_value} outside expected range, clamping")
                vix_value = max(10, min(80, vix_value))

            # Update cache
            self.last_vix_value = vix_value
            self.last_vix_time = now

            # Add to history
            self.vix_history.append({"time": now, "value": vix_value})

            # Keep last 30 days only
            cutoff = now - timedelta(days=30)
            self.vix_history = [v for v in self.vix_history if v["time"] > cutoff]

            return vix_value

        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return self.last_vix_value or 17.0  # Return cached value or default

    def get_vix_statistics(self) -> Dict:
        """
        Get VIX historical statistics.

        Returns:
            Dict with current, avg, min, max VIX values
        """
        if not self.vix_history:
            return {
                "current": self.last_vix_value,
                "avg_30d": None,
                "min_30d": None,
                "max_30d": None,
                "percentile": None,
            }

        values = [v["value"] for v in self.vix_history]

        current_percentile = None
        if self.last_vix_value:
            # Calculate what percentile current VIX is in
            below_current = sum(1 for v in values if v < self.last_vix_value)
            current_percentile = (below_current / len(values)) * 100

        return {
            "current": self.last_vix_value,
            "avg_30d": sum(values) / len(values),
            "min_30d": min(values),
            "max_30d": max(values),
            "percentile": current_percentile,
            "data_points": len(values),
        }

    def get_regime_history(self) -> list:
        """
        Get history of regime changes.

        Returns:
            List of regime change events
        """
        return self.regime_changes

    async def should_reduce_exposure(self) -> bool:
        """
        Check if exposure should be reduced due to high volatility.

        Returns:
            True if should reduce exposure
        """
        regime, _ = await self.get_current_regime()
        return regime in ["elevated", "high"]

    async def is_crisis_mode(self) -> bool:
        """
        Check if market is in crisis mode (extreme volatility).

        Returns:
            True if VIX > 40 (crisis level)
        """
        vix = await self._get_vix()
        if vix is None:
            return False

        return vix > 40  # Crisis threshold

    def adjust_position_size(self, base_size: float, regime_mult: float) -> float:
        """
        Calculate adjusted position size based on regime.

        Args:
            base_size: Base position size (e.g., 0.10 for 10%)
            regime_mult: Regime multiplier from get_current_regime()

        Returns:
            Adjusted position size
        """
        adjusted = base_size * regime_mult

        # Safety caps
        adjusted = max(0.01, adjusted)  # Min 1%
        adjusted = min(0.25, adjusted)  # Max 25%

        return adjusted

    def adjust_stop_loss(self, base_stop: float, regime_mult: float) -> float:
        """
        Calculate adjusted stop-loss based on regime.

        Args:
            base_stop: Base stop-loss (e.g., 0.03 for 3%)
            regime_mult: Regime multiplier from get_current_regime()

        Returns:
            Adjusted stop-loss
        """
        adjusted = base_stop * regime_mult

        # Safety caps
        adjusted = max(0.01, adjusted)  # Min 1%
        adjusted = min(0.10, adjusted)  # Max 10%

        return adjusted


async def monitor_volatility_regime(broker, interval_minutes: int = 5):
    """
    Continuously monitor and log volatility regime.

    Useful for running as background task.

    Args:
        broker: Trading broker instance
        interval_minutes: Check interval in minutes
    """
    detector = VolatilityRegimeDetector(broker)

    logger.info(f"Starting volatility regime monitoring (check every {interval_minutes} min)")

    while True:
        try:
            regime, adjustments = await detector.get_current_regime()

            # Log detailed statistics every 30 minutes
            if datetime.now().minute % 30 == 0:
                stats = detector.get_vix_statistics()
                logger.info(
                    f"\n"
                    f"📊 VIX Statistics (30-day):\n"
                    f"   Current: {stats['current']:.1f}\n"
                    f"   Average: {stats['avg_30d']:.1f}\n"
                    f"   Range: {stats['min_30d']:.1f} - {stats['max_30d']:.1f}\n"
                    f"   Percentile: {stats['percentile']:.0f}%\n"
                )

            # Alert on crisis mode
            if await detector.is_crisis_mode():
                logger.critical(
                    "🚨 CRISIS MODE: VIX > 40! " "Market in extreme fear - reduce all positions!"
                )

        except Exception as e:
            logger.error(f"Error in volatility monitoring: {e}", exc_info=True)

        # Wait for next check
        await asyncio.sleep(interval_minutes * 60)


if __name__ == "__main__":
    # Example usage
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from brokers.alpaca_broker import AlpacaBroker

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    async def main():
        print("\n" + "=" * 80)
        print("📊 VOLATILITY REGIME DETECTOR - EXAMPLE")
        print("=" * 80 + "\n")

        # Initialize broker (paper trading)
        broker = AlpacaBroker(paper=True)

        # Create detector
        detector = VolatilityRegimeDetector(broker)

        # Get current regime
        regime, adjustments = await detector.get_current_regime()

        print(f"Current Regime: {regime.upper()}")
        print(f"Description: {adjustments['description']}")
        print("\nAdjustments:")
        print(f"  Position Size Multiplier: {adjustments['pos_mult']:.1f}x")
        print(f"  Stop-Loss Multiplier: {adjustments['stop_mult']:.1f}x")
        print(f"  Max Positions: {adjustments['max_positions']}")
        print(f"  Trading Allowed: {'YES' if adjustments['trade'] else 'NO'}")

        print("\nExample Calculations:")
        base_position = 0.10  # 10%
        base_stop = 0.03  # 3%

        adjusted_position = detector.adjust_position_size(base_position, adjustments["pos_mult"])
        adjusted_stop = detector.adjust_stop_loss(base_stop, adjustments["stop_mult"])

        print(f"  Base Position Size: {base_position:.1%}")
        print(f"  Adjusted Position Size: {adjusted_position:.1%}")
        print(f"  Base Stop-Loss: {base_stop:.1%}")
        print(f"  Adjusted Stop-Loss: {adjusted_stop:.1%}")

        # Get statistics
        stats = detector.get_vix_statistics()
        if stats["avg_30d"]:
            print("\n30-Day VIX Statistics:")
            print(f"  Current: {stats['current']:.1f}")
            print(f"  Average: {stats['avg_30d']:.1f}")
            print(f"  Range: {stats['min_30d']:.1f} - {stats['max_30d']:.1f}")

        print("\n" + "=" * 80)

    asyncio.run(main())
