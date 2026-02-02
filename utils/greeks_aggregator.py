"""
Greeks Aggregator - Portfolio-Level Options Risk Monitoring

Aggregates option Greeks across all positions for risk control:
- Delta: Directional exposure (target: |net delta| < 100 SPY-equivalent)
- Gamma: Curvature risk (target: |gamma| < 5% of portfolio per 1% move)
- Vega: Volatility exposure (target: |vega| < 2% of portfolio per 1 vol point)
- Theta: Time decay (informational, not typically limited)

Usage:
    aggregator = GreeksAggregator(
        max_delta_pct=0.50,  # 50% of portfolio in directional exposure
        max_gamma_pct=0.05,  # 5% P&L change per 1% underlying move
        max_vega_pct=0.02,   # 2% P&L change per 1 vol point
    )

    result = aggregator.aggregate(
        positions=options_positions,
        underlying_prices={"SPY": 450, "AAPL": 180},
        portfolio_value=100000,
    )

    if not result.within_limits:
        print(f"Greeks violations: {result.violations}")

References:
- Option Greeks primer: Hull's Options, Futures, and Other Derivatives
- Portfolio Greeks management: Taleb's Dynamic Hedging
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GreeksViolation(Enum):
    """Types of Greeks limit violations."""
    DELTA_TOO_HIGH = "delta_too_high"
    GAMMA_TOO_HIGH = "gamma_too_high"
    VEGA_TOO_HIGH = "vega_too_high"
    CONCENTRATION = "concentration"  # Too much Greeks from single underlying


@dataclass
class PositionGreeks:
    """Greeks for a single options position."""
    symbol: str  # OCC symbol (e.g., AAPL230120C00150000)
    underlying: str  # Underlying symbol (e.g., AAPL)
    quantity: int  # Number of contracts (positive = long, negative = short)
    multiplier: int = 100  # Contract multiplier (usually 100)

    # Per-contract Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Option details
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: str = "call"  # "call" or "put"

    @property
    def position_delta(self) -> float:
        """Total delta for this position (contracts * multiplier * delta)."""
        return self.quantity * self.multiplier * self.delta

    @property
    def position_gamma(self) -> float:
        """Total gamma for this position."""
        return self.quantity * self.multiplier * self.gamma

    @property
    def position_theta(self) -> float:
        """Total theta for this position (daily decay in $)."""
        return self.quantity * self.multiplier * self.theta

    @property
    def position_vega(self) -> float:
        """Total vega for this position ($ P&L per 1 vol point)."""
        return self.quantity * self.multiplier * self.vega


@dataclass
class AggregatedGreeks:
    """Portfolio-level aggregated Greeks."""
    # Raw Greeks (in $ terms)
    net_delta: float = 0.0  # $ equivalent delta
    net_gamma: float = 0.0  # $ P&L per 1% move
    net_theta: float = 0.0  # $ daily decay
    net_vega: float = 0.0   # $ per 1 vol point

    # Greeks by underlying
    delta_by_underlying: Dict[str, float] = field(default_factory=dict)
    gamma_by_underlying: Dict[str, float] = field(default_factory=dict)
    vega_by_underlying: Dict[str, float] = field(default_factory=dict)

    # Normalized (as % of portfolio)
    delta_pct: float = 0.0
    gamma_pct: float = 0.0  # P&L impact per 1% underlying move
    vega_pct: float = 0.0   # P&L impact per 1 vol point

    # Risk metrics
    dollar_gamma: float = 0.0  # $ P&L for 1% move in all underlyings
    dollar_vega: float = 0.0   # $ P&L for 1 point vol change

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_theta": self.net_theta,
            "net_vega": self.net_vega,
            "delta_by_underlying": self.delta_by_underlying,
            "gamma_by_underlying": self.gamma_by_underlying,
            "vega_by_underlying": self.vega_by_underlying,
            "delta_pct": self.delta_pct,
            "gamma_pct": self.gamma_pct,
            "vega_pct": self.vega_pct,
            "dollar_gamma": self.dollar_gamma,
            "dollar_vega": self.dollar_vega,
        }


@dataclass
class GreeksLimitResult:
    """Result of Greeks limit check."""
    within_limits: bool
    portfolio_value: float
    greeks: AggregatedGreeks
    violations: List[Tuple[str, GreeksViolation, float, float]]  # (desc, type, value, limit)
    position_count: int
    underlying_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "within_limits": self.within_limits,
            "portfolio_value": self.portfolio_value,
            "greeks": self.greeks.to_dict(),
            "violations": [
                {"description": d, "type": t.value, "value": v, "limit": l}
                for d, t, v, l in self.violations
            ],
            "position_count": self.position_count,
            "underlying_count": self.underlying_count,
            "timestamp": self.timestamp.isoformat(),
        }


class GreeksAggregator:
    """
    Aggregate and monitor portfolio-level Greeks.

    Greeks Definitions:
    - Delta: Change in option value per $1 change in underlying
    - Gamma: Change in delta per $1 change in underlying
    - Theta: Daily time decay (negative for long options)
    - Vega: Change in option value per 1% change in implied volatility

    Risk Interpretation:
    - Net Delta of 100 = equivalent to owning 100 shares
    - Dollar Gamma = P&L change for a 1% move in the underlying
    - Dollar Vega = P&L change for a 1 point change in IV
    """

    # Default limits (conservative institutional standards)
    DEFAULT_MAX_DELTA_PCT = 0.50  # 50% of portfolio in directional exposure
    DEFAULT_MAX_GAMMA_PCT = 0.05  # 5% P&L swing per 1% underlying move
    DEFAULT_MAX_VEGA_PCT = 0.02   # 2% P&L swing per 1 vol point
    DEFAULT_MAX_UNDERLYING_PCT = 0.30  # Max 30% Greeks from single underlying

    def __init__(
        self,
        max_delta_pct: float = DEFAULT_MAX_DELTA_PCT,
        max_gamma_pct: float = DEFAULT_MAX_GAMMA_PCT,
        max_vega_pct: float = DEFAULT_MAX_VEGA_PCT,
        max_underlying_pct: float = DEFAULT_MAX_UNDERLYING_PCT,
    ):
        """
        Initialize Greeks aggregator with limits.

        Args:
            max_delta_pct: Max absolute delta as % of portfolio
            max_gamma_pct: Max gamma P&L impact per 1% underlying move
            max_vega_pct: Max vega P&L impact per 1 vol point change
            max_underlying_pct: Max Greeks concentration in single underlying
        """
        self.max_delta_pct = max_delta_pct
        self.max_gamma_pct = max_gamma_pct
        self.max_vega_pct = max_vega_pct
        self.max_underlying_pct = max_underlying_pct

        self._history: List[GreeksLimitResult] = []

    def aggregate(
        self,
        positions: List[PositionGreeks],
        underlying_prices: Dict[str, float],
        portfolio_value: float,
    ) -> GreeksLimitResult:
        """
        Aggregate Greeks across all options positions and check limits.

        Args:
            positions: List of PositionGreeks for each option position
            underlying_prices: Current prices for each underlying
            portfolio_value: Total portfolio value in $

        Returns:
            GreeksLimitResult with aggregated Greeks and limit check
        """
        if not positions or portfolio_value <= 0:
            return self._empty_result(portfolio_value)

        # Initialize accumulators
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0

        delta_by_underlying: Dict[str, float] = {}
        gamma_by_underlying: Dict[str, float] = {}
        vega_by_underlying: Dict[str, float] = {}

        underlyings = set()

        for pos in positions:
            underlying = pos.underlying
            underlyings.add(underlying)
            price = underlying_prices.get(underlying, 100.0)  # Default price if not provided

            # Position delta in $ terms
            pos_delta_dollars = pos.position_delta * price
            net_delta += pos_delta_dollars
            delta_by_underlying[underlying] = delta_by_underlying.get(underlying, 0) + pos_delta_dollars

            # Position gamma: $ P&L per 1% move
            # Gamma P&L ≈ 0.5 * gamma * (price * 0.01)^2 * multiplier * quantity
            # Simplified: dollar gamma = gamma * price^2 * 0.01 * multiplier * quantity
            pos_gamma_dollars = pos.position_gamma * price * price * 0.01
            net_gamma += pos_gamma_dollars
            gamma_by_underlying[underlying] = gamma_by_underlying.get(underlying, 0) + pos_gamma_dollars

            # Position vega: $ P&L per 1 vol point
            # Vega is typically quoted per 1% vol change, so multiply by 100 for 1 point
            pos_vega_dollars = pos.position_vega
            net_vega += pos_vega_dollars
            vega_by_underlying[underlying] = vega_by_underlying.get(underlying, 0) + pos_vega_dollars

            # Theta in $ terms (already in $ per day)
            net_theta += pos.position_theta

        # Calculate normalized metrics
        delta_pct = net_delta / portfolio_value if portfolio_value > 0 else 0
        gamma_pct = net_gamma / portfolio_value if portfolio_value > 0 else 0
        vega_pct = net_vega / portfolio_value if portfolio_value > 0 else 0

        greeks = AggregatedGreeks(
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            delta_by_underlying=delta_by_underlying,
            gamma_by_underlying=gamma_by_underlying,
            vega_by_underlying=vega_by_underlying,
            delta_pct=delta_pct,
            gamma_pct=gamma_pct,
            vega_pct=vega_pct,
            dollar_gamma=net_gamma,
            dollar_vega=net_vega,
        )

        # Check limits
        violations = []

        # Delta limit
        if abs(delta_pct) > self.max_delta_pct:
            violations.append((
                f"Net delta {delta_pct:.1%} exceeds limit {self.max_delta_pct:.1%}",
                GreeksViolation.DELTA_TOO_HIGH,
                abs(delta_pct),
                self.max_delta_pct,
            ))

        # Gamma limit
        if abs(gamma_pct) > self.max_gamma_pct:
            violations.append((
                f"Net gamma {gamma_pct:.1%} exceeds limit {self.max_gamma_pct:.1%}",
                GreeksViolation.GAMMA_TOO_HIGH,
                abs(gamma_pct),
                self.max_gamma_pct,
            ))

        # Vega limit
        if abs(vega_pct) > self.max_vega_pct:
            violations.append((
                f"Net vega {vega_pct:.1%} exceeds limit {self.max_vega_pct:.1%}",
                GreeksViolation.VEGA_TOO_HIGH,
                abs(vega_pct),
                self.max_vega_pct,
            ))

        # Concentration limits (per underlying)
        for underlying, u_delta in delta_by_underlying.items():
            u_delta_pct = abs(u_delta) / portfolio_value if portfolio_value > 0 else 0
            if u_delta_pct > self.max_underlying_pct:
                violations.append((
                    f"{underlying} delta {u_delta_pct:.1%} exceeds concentration limit",
                    GreeksViolation.CONCENTRATION,
                    u_delta_pct,
                    self.max_underlying_pct,
                ))

        result = GreeksLimitResult(
            within_limits=len(violations) == 0,
            portfolio_value=portfolio_value,
            greeks=greeks,
            violations=violations,
            position_count=len(positions),
            underlying_count=len(underlyings),
        )

        # Track history
        self._history.append(result)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return result

    def aggregate_from_broker_positions(
        self,
        options_positions: List[Dict[str, Any]],
        underlying_prices: Dict[str, float],
        portfolio_value: float,
    ) -> GreeksLimitResult:
        """
        Aggregate from broker-style position dictionaries.

        Args:
            options_positions: List of position dicts with symbol, qty, delta, gamma, etc.
            underlying_prices: Current underlying prices
            portfolio_value: Portfolio value

        Returns:
            GreeksLimitResult
        """
        positions = []

        for pos in options_positions:
            # Parse OCC symbol to get underlying
            symbol = pos.get("symbol", "")
            underlying = self._parse_underlying(symbol)

            position = PositionGreeks(
                symbol=symbol,
                underlying=underlying,
                quantity=pos.get("qty", pos.get("quantity", 0)),
                multiplier=pos.get("multiplier", 100),
                delta=pos.get("delta", 0),
                gamma=pos.get("gamma", 0),
                theta=pos.get("theta", 0),
                vega=pos.get("vega", 0),
                strike=pos.get("strike"),
                option_type=pos.get("option_type", "call"),
            )
            positions.append(position)

        return self.aggregate(positions, underlying_prices, portfolio_value)

    def _parse_underlying(self, occ_symbol: str) -> str:
        """Parse underlying symbol from OCC option symbol."""
        # OCC format: AAPL230120C00150000
        # First 1-6 chars are the underlying
        if not occ_symbol:
            return "UNKNOWN"

        # Find where the date starts (6 digits)
        for i in range(1, min(7, len(occ_symbol))):
            if occ_symbol[i:i+6].isdigit():
                return occ_symbol[:i]

        return occ_symbol[:min(6, len(occ_symbol))]

    def _empty_result(self, portfolio_value: float) -> GreeksLimitResult:
        """Return empty result for edge cases."""
        return GreeksLimitResult(
            within_limits=True,
            portfolio_value=portfolio_value,
            greeks=AggregatedGreeks(),
            violations=[],
            position_count=0,
            underlying_count=0,
        )

    def get_hedge_recommendations(self, result: GreeksLimitResult) -> Dict[str, Any]:
        """
        Get hedge recommendations to bring Greeks within limits.

        Args:
            result: Current GreeksLimitResult

        Returns:
            Dictionary with hedge recommendations
        """
        recommendations = {
            "delta_hedge": None,
            "gamma_hedge": None,
            "vega_hedge": None,
            "summary": [],
        }

        if result.within_limits:
            recommendations["summary"].append("All Greeks within limits - no hedging needed")
            return recommendations

        greeks = result.greeks
        portfolio_value = result.portfolio_value

        for desc, violation_type, value, limit in result.violations:
            if violation_type == GreeksViolation.DELTA_TOO_HIGH:
                # Delta hedge with shares or futures
                excess_delta = greeks.net_delta * (1 - limit / value) * np.sign(greeks.net_delta)
                recommendations["delta_hedge"] = {
                    "action": "sell" if greeks.net_delta > 0 else "buy",
                    "instrument": "SPY shares or ES futures",
                    "amount": f"~{abs(excess_delta):,.0f} delta equivalent",
                    "details": f"Current delta: ${greeks.net_delta:,.0f}, Target: ${greeks.net_delta - excess_delta:,.0f}",
                }
                recommendations["summary"].append(
                    f"Reduce delta by ${abs(excess_delta):,.0f} via stock/futures"
                )

            elif violation_type == GreeksViolation.GAMMA_TOO_HIGH:
                # Gamma hedge with options spreads
                excess_gamma = greeks.net_gamma * (1 - limit / value)
                recommendations["gamma_hedge"] = {
                    "action": "sell" if greeks.net_gamma > 0 else "buy",
                    "instrument": "ATM straddles/strangles",
                    "amount": f"~${abs(excess_gamma):,.0f} gamma reduction needed",
                    "details": "Consider selling premium to reduce long gamma or buying to reduce short gamma",
                }
                recommendations["summary"].append(
                    f"Reduce gamma exposure via options spreads"
                )

            elif violation_type == GreeksViolation.VEGA_TOO_HIGH:
                # Vega hedge with VIX products or calendar spreads
                excess_vega = greeks.net_vega * (1 - limit / value)
                recommendations["vega_hedge"] = {
                    "action": "sell" if greeks.net_vega > 0 else "buy",
                    "instrument": "VIX futures or calendar spreads",
                    "amount": f"~${abs(excess_vega):,.0f} vega reduction needed",
                    "details": "Consider VIX products for broad vol hedge or calendars for single-name",
                }
                recommendations["summary"].append(
                    f"Reduce vega exposure via VIX or calendar spreads"
                )

        return recommendations

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from history."""
        if not self._history:
            return {"error": "No history available"}

        return {
            "history_length": len(self._history),
            "pct_within_limits": sum(1 for r in self._history if r.within_limits) / len(self._history),
            "avg_delta_pct": np.mean([abs(r.greeks.delta_pct) for r in self._history]),
            "avg_gamma_pct": np.mean([abs(r.greeks.gamma_pct) for r in self._history]),
            "avg_vega_pct": np.mean([abs(r.greeks.vega_pct) for r in self._history]),
            "max_delta_pct": max(abs(r.greeks.delta_pct) for r in self._history),
            "max_gamma_pct": max(abs(r.greeks.gamma_pct) for r in self._history),
            "max_vega_pct": max(abs(r.greeks.vega_pct) for r in self._history),
        }


class RealTimeGreeksMonitor:
    """
    Real-time Greeks monitoring with alerts.

    Integrates with trading loop for continuous monitoring.
    """

    def __init__(
        self,
        aggregator: GreeksAggregator,
        alert_callback: Optional[callable] = None,
        check_interval_seconds: int = 60,  # 1 minute
    ):
        """
        Initialize real-time monitor.

        Args:
            aggregator: GreeksAggregator instance
            alert_callback: Function to call on violations
            check_interval_seconds: Minimum time between checks
        """
        self.aggregator = aggregator
        self.alert_callback = alert_callback
        self.check_interval = check_interval_seconds

        self._last_check: Optional[datetime] = None
        self._last_result: Optional[GreeksLimitResult] = None
        self._alert_cooldown: Dict[str, datetime] = {}

    def should_check(self) -> bool:
        """Check if enough time has passed."""
        if self._last_check is None:
            return True
        elapsed = (datetime.now() - self._last_check).total_seconds()
        return elapsed >= self.check_interval

    def check_and_alert(
        self,
        positions: List[PositionGreeks],
        underlying_prices: Dict[str, float],
        portfolio_value: float,
        force: bool = False,
    ) -> Optional[GreeksLimitResult]:
        """Check Greeks and alert on violations."""
        if not force and not self.should_check():
            return None

        result = self.aggregator.aggregate(positions, underlying_prices, portfolio_value)
        self._last_check = datetime.now()
        self._last_result = result

        if not result.within_limits and self.alert_callback:
            for desc, vtype, value, limit in result.violations:
                alert_key = vtype.value

                # Cooldown check (30 min for Greeks alerts)
                if alert_key in self._alert_cooldown:
                    elapsed = (datetime.now() - self._alert_cooldown[alert_key]).total_seconds()
                    if elapsed < 1800:
                        continue

                try:
                    self.alert_callback(desc, vtype, value, limit)
                    self._alert_cooldown[alert_key] = datetime.now()
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return result

    @property
    def last_result(self) -> Optional[GreeksLimitResult]:
        """Get most recent result."""
        return self._last_result


def print_greeks_report(result: GreeksLimitResult):
    """Print formatted Greeks report to console."""
    print("\n" + "=" * 60)
    print("PORTFOLIO GREEKS REPORT")
    print("=" * 60)
    print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Portfolio Value: ${result.portfolio_value:,.0f}")
    print(f"Positions: {result.position_count} options on {result.underlying_count} underlyings")
    print(f"Within Limits: {'✓ Yes' if result.within_limits else '✗ No'}")

    g = result.greeks
    print("\n--- Aggregated Greeks ---")
    print(f"{'Greek':<12} {'Value ($)':>15} {'% of Portfolio':>15}")
    print("-" * 45)
    print(f"{'Delta':<12} {g.net_delta:>15,.0f} {g.delta_pct:>14.1%}")
    print(f"{'Gamma':<12} {g.net_gamma:>15,.0f} {g.gamma_pct:>14.1%}")
    print(f"{'Theta':<12} {g.net_theta:>15,.0f} {'N/A':>15}")
    print(f"{'Vega':<12} {g.net_vega:>15,.0f} {g.vega_pct:>14.1%}")

    if g.delta_by_underlying:
        print("\n--- Delta by Underlying ---")
        sorted_delta = sorted(g.delta_by_underlying.items(), key=lambda x: abs(x[1]), reverse=True)
        for underlying, delta in sorted_delta[:5]:
            pct = delta / result.portfolio_value if result.portfolio_value > 0 else 0
            print(f"  {underlying:<8} ${delta:>12,.0f} ({pct:>6.1%})")

    if result.violations:
        print("\n--- Violations ---")
        for desc, vtype, value, limit in result.violations:
            print(f"  ✗ {desc}")

    print("=" * 60 + "\n")
