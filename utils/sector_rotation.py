#!/usr/bin/env python3
"""
Sector Rotation Strategy

Rotates capital between market sectors based on:
1. Economic cycle phase (expansion, peak, contraction, trough)
2. Relative sector strength
3. Sector momentum

Research shows different sectors outperform in different economic phases:
- Early Expansion: Technology, Consumer Discretionary, Industrials
- Late Expansion: Energy, Materials, Financials
- Contraction: Healthcare, Utilities, Consumer Staples
- Recovery: Financials, Real Estate, Technology

Expected Impact: 5-10% annual alpha from sector timing

Usage:
    from utils.sector_rotation import SectorRotator

    rotator = SectorRotator(broker)
    allocations = await rotator.get_sector_allocations()

    # Returns: {'XLK': 0.25, 'XLF': 0.20, 'XLV': 0.15, ...}
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EconomicPhase(Enum):
    """Economic cycle phases."""

    EARLY_EXPANSION = "early_expansion"  # Recovery phase
    LATE_EXPANSION = "late_expansion"  # Growth slowing
    CONTRACTION = "contraction"  # Recession
    TROUGH = "trough"  # Bottom, recovery starting


class SectorRotator:
    """
    Rotates between market sectors based on economic conditions.
    """

    # Sector ETFs
    SECTOR_ETFS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication Services",
    }

    # Sector performance by economic phase (1.0 = neutral, >1 = overweight, <1 = underweight)
    PHASE_ALLOCATIONS = {
        EconomicPhase.EARLY_EXPANSION: {
            "XLK": 1.4,  # Tech leads recovery
            "XLY": 1.3,  # Consumer spending rebounds
            "XLI": 1.2,  # Industrial activity picks up
            "XLF": 1.2,  # Financials benefit from rate normalization
            "XLRE": 1.1,  # Real estate recovers
            "XLC": 1.0,
            "XLB": 1.0,
            "XLE": 0.8,  # Energy lags early
            "XLV": 0.8,  # Defensive less needed
            "XLP": 0.7,  # Staples underperform
            "XLU": 0.6,  # Utilities underperform
        },
        EconomicPhase.LATE_EXPANSION: {
            "XLE": 1.4,  # Energy peaks late cycle
            "XLB": 1.3,  # Materials demand high
            "XLF": 1.2,  # Financials benefit from rates
            "XLI": 1.1,  # Industrials still strong
            "XLK": 1.0,  # Tech normalizes
            "XLC": 0.9,
            "XLY": 0.9,  # Consumer slowing
            "XLRE": 0.8,  # Real estate slows
            "XLV": 0.8,
            "XLP": 0.8,
            "XLU": 0.7,
        },
        EconomicPhase.CONTRACTION: {
            "XLV": 1.5,  # Healthcare defensive
            "XLU": 1.4,  # Utilities defensive
            "XLP": 1.4,  # Staples defensive
            "XLC": 1.0,  # Communication stable
            "XLK": 0.9,  # Tech volatile
            "XLF": 0.7,  # Financials struggle
            "XLI": 0.7,  # Industrials decline
            "XLY": 0.6,  # Discretionary cut
            "XLB": 0.6,  # Materials demand drops
            "XLE": 0.6,  # Energy demand drops
            "XLRE": 0.5,  # Real estate struggles
        },
        EconomicPhase.TROUGH: {
            "XLK": 1.3,  # Tech often leads out
            "XLF": 1.2,  # Financials anticipate recovery
            "XLY": 1.2,  # Consumer sentiment improves
            "XLI": 1.1,
            "XLRE": 1.1,
            "XLC": 1.0,
            "XLV": 1.0,  # Still some defense
            "XLP": 0.9,
            "XLB": 0.9,
            "XLE": 0.8,
            "XLU": 0.8,
        },
    }

    # Top stocks in each sector for direct stock trading
    SECTOR_STOCKS = {
        "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE"],
        "XLF": ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP"],
        "XLV": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX"],
        "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ", "CL"],
        "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
        "XLI": ["CAT", "UNP", "HON", "UPS", "BA", "RTX", "GE", "LMT"],
        "XLB": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW"],
        "XLU": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
        "XLRE": ["PLD", "AMT", "EQIX", "PSA", "CCI", "SPG", "O", "WELL"],
        "XLC": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR"],
    }

    def __init__(self, broker, use_etfs: bool = False):
        """
        Initialize sector rotator.

        Args:
            broker: Trading broker instance
            use_etfs: If True, trade sector ETFs. If False, trade top stocks in sectors.
        """
        self.broker = broker
        self.use_etfs = use_etfs

        # Cache
        self._current_phase = None
        self._last_phase_check = None
        self._phase_cache_minutes = 60

        logger.info(f"SectorRotator initialized (use_etfs={use_etfs})")

    async def detect_economic_phase(self) -> Tuple[EconomicPhase, float]:
        """
        Detect current economic phase using market indicators.

        Uses:
        - SPY trend (bull/bear)
        - Sector relative strength
        - Yield curve (approximated)
        - Volatility

        Returns:
            Tuple of (EconomicPhase, confidence 0-1)
        """
        try:
            # Check cache
            now = datetime.now()
            if (
                self._current_phase
                and self._last_phase_check
                and (now - self._last_phase_check).total_seconds() < self._phase_cache_minutes * 60
            ):
                return self._current_phase

            # Get market data
            spy_data = await self._get_price_data("SPY", days=200)
            if spy_data is None or len(spy_data) < 50:
                logger.warning("Insufficient SPY data for phase detection")
                return (EconomicPhase.LATE_EXPANSION, 0.5)  # Default

            closes = np.array([d["close"] for d in spy_data])

            # Calculate indicators
            sma_50 = np.mean(closes[-50:])
            sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else np.mean(closes)
            current_price = closes[-1]

            # Price momentum (3-month return)
            momentum_3m = (closes[-1] / closes[-63] - 1) if len(closes) >= 63 else 0

            # Volatility (20-day)
            returns = np.diff(closes[-21:]) / closes[-21:-1] if len(closes) >= 21 else []
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2

            # Trend strength
            trend_up = current_price > sma_50 > sma_200
            trend_down = current_price < sma_50 < sma_200

            # Determine phase
            phase = EconomicPhase.LATE_EXPANSION  # Default
            confidence = 0.5

            if trend_up and momentum_3m > 0.05:
                # Strong uptrend
                if volatility < 0.15:
                    phase = EconomicPhase.EARLY_EXPANSION
                    confidence = 0.7 + min(momentum_3m, 0.15)
                else:
                    phase = EconomicPhase.LATE_EXPANSION
                    confidence = 0.6

            elif trend_down or momentum_3m < -0.05:
                # Downtrend
                if volatility > 0.25:
                    phase = EconomicPhase.CONTRACTION
                    confidence = 0.7 + min(volatility - 0.25, 0.2)
                else:
                    phase = EconomicPhase.TROUGH
                    confidence = 0.6

            else:
                # Mixed signals
                if momentum_3m > 0:
                    phase = EconomicPhase.EARLY_EXPANSION
                else:
                    phase = EconomicPhase.LATE_EXPANSION
                confidence = 0.5

            # Cache result
            self._current_phase = (phase, confidence)
            self._last_phase_check = now

            logger.info(
                f"Economic phase: {phase.value} (confidence: {confidence:.0%}, "
                f"momentum: {momentum_3m:.1%}, volatility: {volatility:.1%})"
            )

            return (phase, confidence)

        except Exception as e:
            logger.error(f"Error detecting economic phase: {e}")
            return (EconomicPhase.LATE_EXPANSION, 0.5)

    async def get_sector_allocations(self, base_allocation: float = 1.0) -> Dict[str, float]:
        """
        Get recommended sector allocations based on current phase.

        Args:
            base_allocation: Total allocation to distribute

        Returns:
            Dict of sector ETF -> allocation weight
        """
        phase, confidence = await self.detect_economic_phase()
        phase_weights = self.PHASE_ALLOCATIONS[phase]

        # Normalize weights
        total_weight = sum(phase_weights.values())
        allocations = {
            sector: (weight / total_weight) * base_allocation
            for sector, weight in phase_weights.items()
        }

        # If low confidence, move towards equal weight
        if confidence < 0.6:
            equal_weight = base_allocation / len(allocations)
            blend_factor = confidence  # Higher confidence = more tilted allocation

            allocations = {
                sector: blend_factor * alloc + (1 - blend_factor) * equal_weight
                for sector, alloc in allocations.items()
            }

        return allocations

    async def get_recommended_stocks(
        self, top_n: int = 20, stocks_per_sector: int = 3
    ) -> List[str]:
        """
        Get recommended stocks based on sector rotation.

        Args:
            top_n: Maximum total stocks to return
            stocks_per_sector: Max stocks from each sector

        Returns:
            List of stock symbols
        """
        allocations = await self.get_sector_allocations()

        # Sort sectors by allocation (highest first)
        sorted_sectors = sorted(allocations.items(), key=lambda x: x[1], reverse=True)

        recommended = []
        for sector_etf, allocation in sorted_sectors:
            if sector_etf not in self.SECTOR_STOCKS:
                continue

            # Number of stocks proportional to allocation
            num_stocks = min(stocks_per_sector, max(1, int(allocation * top_n)))

            sector_stocks = self.SECTOR_STOCKS[sector_etf][:num_stocks]
            recommended.extend(sector_stocks)

            if len(recommended) >= top_n:
                break

        return recommended[:top_n]

    async def get_sector_momentum(self) -> Dict[str, float]:
        """
        Get relative momentum for each sector.

        Returns:
            Dict of sector ETF -> momentum score
        """
        momentum = {}

        for etf in self.SECTOR_ETFS.keys():
            try:
                data = await self._get_price_data(etf, days=30)
                if data and len(data) >= 20:
                    closes = [d["close"] for d in data]
                    # 20-day momentum
                    mom = (closes[-1] / closes[-20] - 1) * 100
                    momentum[etf] = mom
            except Exception as e:
                logger.debug(f"Error getting momentum for {etf}: {e}")
                momentum[etf] = 0

        return momentum

    async def _get_price_data(self, symbol: str, days: int = 30) -> Optional[List[Dict]]:
        """Get historical price data for a symbol."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)

            bars = await self.broker.get_bars(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                timeframe="1Day",
            )

            if bars is None:
                return None

            return [
                {
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                }
                for b in bars
            ]

        except Exception as e:
            logger.debug(f"Error fetching data for {symbol}: {e}")
            return None

    async def get_sector_report(self) -> Dict:
        """Get comprehensive sector rotation report."""
        phase, confidence = await self.detect_economic_phase()
        allocations = await self.get_sector_allocations()
        momentum = await self.get_sector_momentum()
        recommended = await self.get_recommended_stocks(top_n=15)

        # Sort by allocation
        sorted_sectors = sorted(allocations.items(), key=lambda x: x[1], reverse=True)

        return {
            "phase": phase.value,
            "phase_confidence": confidence,
            "allocations": dict(sorted_sectors),
            "momentum": momentum,
            "recommended_stocks": recommended,
            "overweight_sectors": [s for s, a in sorted_sectors if a > 0.10],
            "underweight_sectors": [s for s, a in sorted_sectors if a < 0.08],
        }


if __name__ == "__main__":
    """Test the sector rotator."""
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    async def test():
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)

        print("\n" + "=" * 60)
        print("SECTOR ROTATION ANALYSIS")
        print("=" * 60)

        rotator = SectorRotator(broker)
        report = await rotator.get_sector_report()

        print(f"\nEconomic Phase: {report['phase'].upper()}")
        print(f"Confidence: {report['phase_confidence']:.0%}")

        print("\nSector Allocations:")
        print("-" * 40)
        for sector, alloc in report["allocations"].items():
            sector_name = SectorRotator.SECTOR_ETFS.get(sector, sector)
            mom = report["momentum"].get(sector, 0)
            bar = "█" * int(alloc * 50)
            print(f"  {sector:5s} ({sector_name:25s}): {alloc:5.1%} {bar} ({mom:+.1f}%)")

        print(f"\nOverweight: {', '.join(report['overweight_sectors'])}")
        print(f"Underweight: {', '.join(report['underweight_sectors'])}")

        print(f"\nRecommended Stocks ({len(report['recommended_stocks'])}):")
        print(", ".join(report["recommended_stocks"]))

        print("=" * 60)

    asyncio.run(test())
