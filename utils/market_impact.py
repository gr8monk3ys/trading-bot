"""
Market Impact Model - Almgren-Chriss Implementation

Realistic transaction cost modeling for backtesting and execution optimization.

The Almgren-Chriss model (2001) decomposes market impact into:
1. **Permanent Impact**: Price change that persists (information revelation)
   - Proportional to order size relative to daily volume
   - Formula: permanent = γ * σ * (v/V)^α

2. **Temporary Impact**: Price concession during execution (liquidity demand)
   - Depends on execution urgency (order size / time horizon)
   - Formula: temporary = η * σ * (v/V/T)^β

3. **Spread Cost**: Bid-ask spread (constant per trade)

Why this matters:
- Fixed slippage (0.1%) underestimates impact on large orders by 5-10x
- A $1M order in a $10M daily volume stock costs ~2-3%, not 0.1%
- Proper modeling prevents false confidence in backtests

Usage:
    model = AlmgrenChrissModel()

    # Single trade impact
    impact = model.calculate_impact(
        order_value=100000,
        daily_volume_usd=5000000,
        volatility=0.25,
        execution_time_fraction=0.1  # 10% of day
    )

    # Expected cost
    print(f"Expected slippage: {impact.total_cost_bps} bps")
    print(f"Permanent: {impact.permanent_impact_bps} bps")
    print(f"Temporary: {impact.temporary_impact_bps} bps")

Research sources:
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Kissell & Glantz (2003): "Optimal Trading Strategies"
- Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Market microstructure conditions affecting impact."""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"  # VIX > 25
    LOW_LIQUIDITY = "low_liquidity"  # Volume < 50% of average
    MARKET_OPEN = "market_open"  # First 30 minutes
    MARKET_CLOSE = "market_close"  # Last 30 minutes
    EARNINGS = "earnings"  # Near earnings announcement


@dataclass
class MarketImpactResult:
    """Result of market impact calculation."""

    # Impact components (in basis points)
    permanent_impact_bps: float
    temporary_impact_bps: float
    spread_cost_bps: float
    total_cost_bps: float

    # In dollar terms
    permanent_impact_usd: float
    temporary_impact_usd: float
    spread_cost_usd: float
    total_cost_usd: float

    # Order characteristics
    order_value: float
    participation_rate: float  # Order as % of daily volume
    execution_time_hours: float

    # Market context
    daily_volume_usd: float
    volatility: float
    condition: MarketCondition

    # Effective price
    effective_price: float  # Price after impact
    slippage_pct: float  # As percentage for compatibility

    timestamp: datetime

    def __repr__(self) -> str:
        return (
            f"MarketImpact(total={self.total_cost_bps:.1f}bps, "
            f"perm={self.permanent_impact_bps:.1f}bps, "
            f"temp={self.temporary_impact_bps:.1f}bps, "
            f"participation={self.participation_rate:.1%})"
        )


class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model.

    Model parameters are calibrated to empirical data from US equity markets.
    Different parameter sets for different market cap tiers.

    Key insight: Impact scales non-linearly with order size.
    A 1% participation order costs ~5 bps, but 10% costs ~50 bps (not 50).
    """

    # Almgren-Chriss parameters (empirically calibrated)
    # Large cap (>$10B market cap)
    LARGE_CAP_PARAMS = {
        "gamma": 0.314,  # Permanent impact coefficient
        "eta": 0.142,    # Temporary impact coefficient
        "alpha": 0.891,  # Permanent impact exponent
        "beta": 0.600,   # Temporary impact exponent
        "spread_bps": 3.0,  # Typical spread for large caps
    }

    # Mid cap ($2B - $10B)
    MID_CAP_PARAMS = {
        "gamma": 0.512,
        "eta": 0.215,
        "alpha": 0.842,
        "beta": 0.650,
        "spread_bps": 8.0,
    }

    # Small cap (<$2B)
    SMALL_CAP_PARAMS = {
        "gamma": 0.823,
        "eta": 0.328,
        "alpha": 0.764,
        "beta": 0.720,
        "spread_bps": 20.0,
    }

    # Time-of-day adjustments (market microstructure)
    TIME_ADJUSTMENTS = {
        MarketCondition.MARKET_OPEN: 1.5,   # 50% higher impact at open
        MarketCondition.MARKET_CLOSE: 1.3,  # 30% higher at close
        MarketCondition.HIGH_VOLATILITY: 1.4,
        MarketCondition.LOW_LIQUIDITY: 2.0,  # Double impact in low liquidity
        MarketCondition.EARNINGS: 1.6,
        MarketCondition.NORMAL: 1.0,
    }

    def __init__(
        self,
        default_spread_bps: float = 5.0,
        use_time_adjustments: bool = True,
        min_impact_bps: float = 1.0,
        max_impact_bps: float = 500.0,  # 5% max
    ):
        """
        Initialize market impact model.

        Args:
            default_spread_bps: Default bid-ask spread in basis points
            use_time_adjustments: Apply time-of-day adjustments
            min_impact_bps: Minimum impact floor
            max_impact_bps: Maximum impact cap
        """
        self.default_spread_bps = default_spread_bps
        self.use_time_adjustments = use_time_adjustments
        self.min_impact_bps = min_impact_bps
        self.max_impact_bps = max_impact_bps

    def calculate_impact(
        self,
        order_value: float,
        daily_volume_usd: float,
        price: float,
        volatility: float = 0.25,
        execution_time_hours: float = 1.0,
        market_cap: Optional[float] = None,
        condition: MarketCondition = MarketCondition.NORMAL,
        side: str = "buy",
    ) -> MarketImpactResult:
        """
        Calculate expected market impact for an order.

        Args:
            order_value: Order size in dollars
            daily_volume_usd: Average daily volume in dollars
            price: Current stock price
            volatility: Annualized volatility (e.g., 0.25 = 25%)
            execution_time_hours: Expected execution duration
            market_cap: Market capitalization for parameter selection
            condition: Current market condition
            side: 'buy' or 'sell'

        Returns:
            MarketImpactResult with detailed breakdown
        """
        # Select parameters based on market cap
        params = self._select_parameters(market_cap, daily_volume_usd)

        # Participation rate (order as fraction of daily volume)
        if daily_volume_usd <= 0:
            daily_volume_usd = order_value * 10  # Assume 10% participation if unknown

        participation = order_value / daily_volume_usd
        participation = min(participation, 1.0)  # Cap at 100% of daily volume

        # Execution rate (fraction of day)
        trading_hours = 6.5  # NYSE trading hours
        execution_fraction = min(execution_time_hours / trading_hours, 1.0)
        execution_fraction = max(execution_fraction, 0.01)  # Minimum 1% of day

        # Daily volatility (annualized -> daily)
        daily_vol = volatility / np.sqrt(252)

        # Permanent impact: γ * σ * (v/V)^α
        # Represents information revealed by trading
        permanent_impact = (
            params["gamma"] *
            daily_vol *
            (participation ** params["alpha"]) *
            10000  # Convert to basis points
        )

        # Temporary impact: η * σ * (v/(V*T))^β
        # Represents liquidity demand premium
        execution_rate = participation / execution_fraction
        temporary_impact = (
            params["eta"] *
            daily_vol *
            (execution_rate ** params["beta"]) *
            10000  # Convert to basis points
        )

        # Spread cost
        spread_bps = params.get("spread_bps", self.default_spread_bps)

        # Apply time-of-day adjustment
        if self.use_time_adjustments:
            adjustment = self.TIME_ADJUSTMENTS.get(condition, 1.0)
            temporary_impact *= adjustment

        # Total impact (permanent is one-way, temporary is also one-way)
        total_impact_bps = permanent_impact + temporary_impact + spread_bps / 2

        # Apply floors and caps
        total_impact_bps = np.clip(
            total_impact_bps,
            self.min_impact_bps,
            self.max_impact_bps
        )

        # Calculate dollar amounts
        permanent_usd = order_value * permanent_impact / 10000
        temporary_usd = order_value * temporary_impact / 10000
        spread_usd = order_value * spread_bps / 2 / 10000
        total_usd = order_value * total_impact_bps / 10000

        # Effective price after impact
        if side == "buy":
            effective_price = price * (1 + total_impact_bps / 10000)
        else:
            effective_price = price * (1 - total_impact_bps / 10000)

        slippage_pct = total_impact_bps / 10000

        result = MarketImpactResult(
            permanent_impact_bps=float(permanent_impact),
            temporary_impact_bps=float(temporary_impact),
            spread_cost_bps=float(spread_bps / 2),
            total_cost_bps=float(total_impact_bps),
            permanent_impact_usd=float(permanent_usd),
            temporary_impact_usd=float(temporary_usd),
            spread_cost_usd=float(spread_usd),
            total_cost_usd=float(total_usd),
            order_value=order_value,
            participation_rate=float(participation),
            execution_time_hours=execution_time_hours,
            daily_volume_usd=daily_volume_usd,
            volatility=volatility,
            condition=condition,
            effective_price=float(effective_price),
            slippage_pct=float(slippage_pct),
            timestamp=datetime.now(),
        )

        logger.debug(
            f"Market impact: {order_value/1000:.0f}K order -> "
            f"{total_impact_bps:.1f}bps (perm={permanent_impact:.1f}, "
            f"temp={temporary_impact:.1f}, spread={spread_bps/2:.1f}), "
            f"participation={participation:.1%}"
        )

        return result

    def _select_parameters(
        self, market_cap: Optional[float], daily_volume: float
    ) -> Dict[str, float]:
        """Select model parameters based on stock characteristics."""
        if market_cap is not None:
            if market_cap >= 10e9:
                return self.LARGE_CAP_PARAMS
            elif market_cap >= 2e9:
                return self.MID_CAP_PARAMS
            else:
                return self.SMALL_CAP_PARAMS

        # Infer from volume if market cap unknown
        # Rough heuristic: large caps trade > $100M/day
        if daily_volume >= 100e6:
            return self.LARGE_CAP_PARAMS
        elif daily_volume >= 20e6:
            return self.MID_CAP_PARAMS
        else:
            return self.SMALL_CAP_PARAMS

    def estimate_optimal_execution_time(
        self,
        order_value: float,
        daily_volume_usd: float,
        volatility: float = 0.25,
        urgency: float = 0.5,
    ) -> float:
        """
        Estimate optimal execution time horizon.

        Balances:
        - Faster execution = higher temporary impact
        - Slower execution = more volatility risk

        Args:
            order_value: Order size in dollars
            daily_volume_usd: Average daily volume
            volatility: Annualized volatility
            urgency: 0 = patient, 1 = urgent

        Returns:
            Recommended execution time in hours
        """
        participation = order_value / max(daily_volume_usd, 1)

        # Base time proportional to participation rate
        # 1% participation -> ~0.5 hour, 10% -> ~2.5 hours
        base_hours = 0.5 + participation * 20

        # Adjust for urgency
        urgency_factor = 1.0 - (urgency * 0.7)  # Urgent = shorter time

        # Adjust for volatility (high vol = execute faster to reduce risk)
        vol_factor = 0.25 / max(volatility, 0.1)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)

        optimal_hours = base_hours * urgency_factor * vol_factor

        # Clip to reasonable bounds
        return float(np.clip(optimal_hours, 0.1, 6.5))  # 6 minutes to full day


class VolumeCurveModel:
    """
    Intraday volume curve for execution scheduling.

    US equity markets follow a U-shaped volume pattern:
    - High volume at open (9:30-10:30)
    - Low volume mid-day (12:00-14:00)
    - High volume at close (15:00-16:00)

    VWAP and TWAP algorithms use this for optimal scheduling.
    """

    # Typical US equity intraday volume profile (cumulative %)
    # Indexed by 30-minute intervals from 9:30
    VOLUME_PROFILE = {
        0: 0.12,   # 9:30-10:00
        1: 0.08,   # 10:00-10:30
        2: 0.06,   # 10:30-11:00
        3: 0.06,   # 11:00-11:30
        4: 0.05,   # 11:30-12:00
        5: 0.05,   # 12:00-12:30
        6: 0.05,   # 12:30-13:00
        7: 0.05,   # 13:00-13:30
        8: 0.06,   # 13:30-14:00
        9: 0.07,   # 14:00-14:30
        10: 0.08,  # 14:30-15:00
        11: 0.10,  # 15:00-15:30
        12: 0.17,  # 15:30-16:00
    }

    @classmethod
    def get_volume_fraction(cls, start_time: time, end_time: time) -> float:
        """
        Get fraction of daily volume expected between two times.

        Args:
            start_time: Start of execution window
            end_time: End of execution window

        Returns:
            Fraction of daily volume (0.0 - 1.0)
        """
        market_open = time(9, 30)
        market_close = time(16, 0)

        # Clip to market hours
        if start_time < market_open:
            start_time = market_open
        if end_time > market_close:
            end_time = market_close

        if start_time >= end_time:
            return 0.0

        # Calculate volume between times
        start_minutes = (start_time.hour - 9) * 60 + start_time.minute - 30
        end_minutes = (end_time.hour - 9) * 60 + end_time.minute - 30

        # Sum volume fractions
        total_volume = 0.0
        for interval in range(13):
            interval_start = interval * 30
            interval_end = (interval + 1) * 30

            # Check overlap
            overlap_start = max(start_minutes, interval_start)
            overlap_end = min(end_minutes, interval_end)

            if overlap_end > overlap_start:
                fraction = (overlap_end - overlap_start) / 30
                total_volume += cls.VOLUME_PROFILE[interval] * fraction

        return total_volume

    @classmethod
    def get_market_condition(cls, current_time: time) -> MarketCondition:
        """Determine market condition based on time of day."""
        if current_time < time(10, 0):
            return MarketCondition.MARKET_OPEN
        elif current_time >= time(15, 30):
            return MarketCondition.MARKET_CLOSE
        else:
            return MarketCondition.NORMAL


class ExecutionCostTracker:
    """
    Tracks and analyzes actual execution costs vs predicted.

    Used for:
    - Model calibration (update parameters based on realized costs)
    - Performance attribution (how much did execution cost us?)
    - Execution quality analysis (are we beating VWAP?)
    """

    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self._daily_stats: Dict[str, Dict[str, float]] = {}

    def record_execution(
        self,
        symbol: str,
        order_value: float,
        predicted_impact: MarketImpactResult,
        actual_slippage_pct: float,
        fill_price: float,
        arrival_price: float,
        vwap: Optional[float] = None,
    ):
        """
        Record an executed trade for analysis.

        Args:
            symbol: Stock symbol
            order_value: Order value in dollars
            predicted_impact: Predicted impact from model
            actual_slippage_pct: Actual slippage observed
            fill_price: Average fill price
            arrival_price: Price when order was submitted
            vwap: Volume-weighted average price during execution
        """
        actual_cost_bps = actual_slippage_pct * 10000
        predicted_cost_bps = predicted_impact.total_cost_bps

        record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "order_value": order_value,
            "predicted_cost_bps": predicted_cost_bps,
            "actual_cost_bps": actual_cost_bps,
            "fill_price": fill_price,
            "arrival_price": arrival_price,
            "vwap": vwap,
            "participation_rate": predicted_impact.participation_rate,
            "daily_volume": predicted_impact.daily_volume_usd,
            "volatility": predicted_impact.volatility,
        }

        # VWAP slippage (if available)
        if vwap:
            record["vwap_slippage_bps"] = ((fill_price / vwap) - 1) * 10000

        # Prediction error
        record["prediction_error_bps"] = actual_cost_bps - predicted_cost_bps

        self.execution_history.append(record)

        logger.debug(
            f"Execution recorded: {symbol} ${order_value/1000:.0f}K, "
            f"predicted={predicted_cost_bps:.1f}bps, actual={actual_cost_bps:.1f}bps"
        )

    def get_execution_quality_report(self) -> Dict[str, Any]:
        """
        Generate execution quality report.

        Returns:
            Dict with execution quality metrics
        """
        if not self.execution_history:
            return {"error": "No executions recorded"}

        n = len(self.execution_history)

        predicted = [e["predicted_cost_bps"] for e in self.execution_history]
        actual = [e["actual_cost_bps"] for e in self.execution_history]
        errors = [e["prediction_error_bps"] for e in self.execution_history]

        # VWAP performance
        vwap_slippages = [
            e.get("vwap_slippage_bps", 0)
            for e in self.execution_history
            if e.get("vwap") is not None
        ]

        report = {
            "total_executions": n,
            "total_volume_usd": sum(e["order_value"] for e in self.execution_history),
            "predicted_cost": {
                "mean_bps": float(np.mean(predicted)),
                "median_bps": float(np.median(predicted)),
                "std_bps": float(np.std(predicted)),
            },
            "actual_cost": {
                "mean_bps": float(np.mean(actual)),
                "median_bps": float(np.median(actual)),
                "std_bps": float(np.std(actual)),
            },
            "prediction_accuracy": {
                "mean_error_bps": float(np.mean(errors)),
                "rmse_bps": float(np.sqrt(np.mean(np.array(errors) ** 2))),
                "correlation": float(np.corrcoef(predicted, actual)[0, 1]) if n > 1 else 0,
            },
        }

        if vwap_slippages:
            report["vwap_performance"] = {
                "mean_vs_vwap_bps": float(np.mean(vwap_slippages)),
                "beat_vwap_pct": sum(1 for v in vwap_slippages if v < 0) / len(vwap_slippages),
            }

        return report

    def get_model_calibration_adjustments(self) -> Dict[str, float]:
        """
        Calculate parameter adjustments based on realized execution.

        Returns:
            Dict with multipliers for model parameters
        """
        if len(self.execution_history) < 20:
            return {"gamma_mult": 1.0, "eta_mult": 1.0}  # Need more data

        # Regress actual vs predicted to find adjustment
        predicted = np.array([e["predicted_cost_bps"] for e in self.execution_history])
        actual = np.array([e["actual_cost_bps"] for e in self.execution_history])

        # Simple scaling factor
        mean_ratio = np.mean(actual) / np.mean(predicted) if np.mean(predicted) > 0 else 1.0

        # Clamp to reasonable range
        mean_ratio = np.clip(mean_ratio, 0.5, 2.0)

        return {
            "gamma_mult": float(mean_ratio),
            "eta_mult": float(mean_ratio),
            "sample_size": len(self.execution_history),
        }
