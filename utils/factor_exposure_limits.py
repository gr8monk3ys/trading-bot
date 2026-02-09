"""
Factor Exposure Limits - Portfolio-Level Factor Risk Control

Ensures no single factor dominates portfolio risk:
- Calculates factor exposures from positions and factor loadings
- Monitors factor risk contribution (target: no factor > 15% of risk)
- Provides limits and warnings when exposures exceed thresholds
- Suggests position adjustments to bring within limits

Usage:
    limiter = FactorExposureLimiter(max_factor_risk_pct=0.15)

    # Check exposures
    result = limiter.check_exposures(positions, factor_loadings, covariance_matrix)

    if not result.within_limits:
        print(f"Factors exceeding limits: {result.violations}")
        print(f"Suggested adjustments: {result.suggested_adjustments}")

References:
- Barra risk model factor exposure limits
- MSCI factor model documentation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FactorExposureViolation(Enum):
    """Types of factor exposure violations."""
    SINGLE_FACTOR_TOO_HIGH = "single_factor_too_high"
    CONCENTRATION_TOO_HIGH = "concentration_too_high"
    CORRELATION_CLUSTER = "correlation_cluster"
    MOMENTUM_TILT = "momentum_tilt"
    SECTOR_TILT = "sector_tilt"


@dataclass
class FactorExposure:
    """Factor exposure for a single factor."""
    factor_name: str
    raw_exposure: float  # Sum of position weights * factor loadings
    risk_contribution: float  # Contribution to portfolio variance
    risk_pct: float  # Percentage of total portfolio risk
    marginal_risk: float  # Marginal contribution to risk
    within_limit: bool
    limit: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExposureLimitResult:
    """Result of factor exposure limit check."""
    within_limits: bool
    total_portfolio_risk: float  # Portfolio volatility
    factor_exposures: Dict[str, FactorExposure]
    violations: List[Tuple[str, FactorExposureViolation, float]]  # (factor, type, value)
    concentration_hhi: float  # Herfindahl index of factor risk contributions
    suggested_adjustments: Dict[str, float]  # symbol -> adjustment multiplier
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "within_limits": self.within_limits,
            "total_portfolio_risk": self.total_portfolio_risk,
            "factor_exposures": {
                k: {
                    "raw_exposure": v.raw_exposure,
                    "risk_contribution": v.risk_contribution,
                    "risk_pct": v.risk_pct,
                    "within_limit": v.within_limit,
                    "limit": v.limit,
                }
                for k, v in self.factor_exposures.items()
            },
            "violations": [
                {"factor": f, "type": t.value, "value": v}
                for f, t, v in self.violations
            ],
            "concentration_hhi": self.concentration_hhi,
            "suggested_adjustments": self.suggested_adjustments,
            "timestamp": self.timestamp.isoformat(),
        }


class FactorExposureLimiter:
    """
    Monitor and limit factor exposures in a portfolio.

    Risk Contribution Calculation:
    For a portfolio with factor model: r_p = sum(beta_i * f_i) + epsilon

    Risk contribution of factor i = beta_i * Cov(f_i, r_p) / Var(r_p)
                                  = beta_i * sum_j(beta_j * Cov(f_i, f_j)) / Var(r_p)

    This decomposes total portfolio variance into factor contributions.
    """

    # Default limits (institutional standard)
    DEFAULT_MAX_FACTOR_RISK_PCT = 0.15  # No single factor > 15% of risk
    DEFAULT_MAX_CONCENTRATION_HHI = 0.20  # HHI < 0.20 (diversified)
    DEFAULT_MIN_FACTORS = 3  # At least 3 factors with significant exposure

    def __init__(
        self,
        max_factor_risk_pct: float = DEFAULT_MAX_FACTOR_RISK_PCT,
        max_concentration_hhi: float = DEFAULT_MAX_CONCENTRATION_HHI,
        min_significant_factors: int = DEFAULT_MIN_FACTORS,
        factor_correlation_threshold: float = 0.70,
    ):
        """
        Initialize factor exposure limiter.

        Args:
            max_factor_risk_pct: Maximum risk contribution from any single factor
            max_concentration_hhi: Maximum Herfindahl index for factor risk
            min_significant_factors: Minimum number of factors with >5% exposure
            factor_correlation_threshold: Threshold for correlated factor warnings
        """
        self.max_factor_risk_pct = max_factor_risk_pct
        self.max_concentration_hhi = max_concentration_hhi
        self.min_significant_factors = min_significant_factors
        self.factor_correlation_threshold = factor_correlation_threshold

        # History for tracking
        self._exposure_history: List[ExposureLimitResult] = []

    def check_exposures(
        self,
        positions: Dict[str, float],  # symbol -> weight (sum = 1)
        factor_loadings: Dict[str, Dict[str, float]],  # factor -> {symbol: loading}
        factor_covariance: Optional[np.ndarray] = None,  # Factor covariance matrix
        factor_names: Optional[List[str]] = None,  # Factor names (ordered)
        factor_volatilities: Optional[Dict[str, float]] = None,  # Factor vols
    ) -> ExposureLimitResult:
        """
        Check if portfolio factor exposures are within limits.

        Args:
            positions: Portfolio weights by symbol (should sum to 1)
            factor_loadings: Factor loadings for each symbol
            factor_covariance: Factor covariance matrix (optional, uses identity if not provided)
            factor_names: Ordered factor names for covariance matrix
            factor_volatilities: Factor volatilities (optional, uses 0.20 default)

        Returns:
            ExposureLimitResult with exposure analysis
        """
        if not positions or not factor_loadings:
            logger.warning("Empty positions or factor loadings")
            return self._empty_result()

        # Normalize weights
        total_weight = sum(abs(w) for w in positions.values())
        if total_weight == 0:
            return self._empty_result()

        weights = {s: w / total_weight for s, w in positions.items()}

        # Get factor list
        factors = factor_names or list(factor_loadings.keys())
        n_factors = len(factors)

        if n_factors == 0:
            return self._empty_result()

        # Calculate portfolio factor exposures
        # beta_p_i = sum_j(w_j * beta_j_i) for each factor i
        portfolio_exposures = {}
        for factor in factors:
            loadings = factor_loadings.get(factor, {})
            exposure = sum(
                weights.get(symbol, 0) * loadings.get(symbol, 0)
                for symbol in weights
            )
            portfolio_exposures[factor] = exposure

        # Build factor covariance matrix
        if factor_covariance is None:
            # Use diagonal matrix with default volatilities
            default_vol = 0.20
            vols = [
                factor_volatilities.get(f, default_vol) if factor_volatilities else default_vol
                for f in factors
            ]
            factor_covariance = np.diag([v ** 2 for v in vols])

        # Ensure covariance matrix is valid
        factor_covariance = np.atleast_2d(factor_covariance)
        if factor_covariance.shape[0] != n_factors:
            logger.warning(f"Covariance matrix size mismatch: {factor_covariance.shape} vs {n_factors} factors")
            factor_covariance = np.eye(n_factors) * 0.04  # 20% vol assumption

        # Calculate portfolio variance from factors
        # Var(r_p) = beta' * Cov(F) * beta
        beta_vector = np.array([portfolio_exposures.get(f, 0) for f in factors])
        portfolio_variance = float(beta_vector @ factor_covariance @ beta_vector)
        portfolio_risk = np.sqrt(max(portfolio_variance, 1e-10))

        # Calculate risk contribution of each factor
        # RC_i = beta_i * (Cov(F) * beta)_i / Var(r_p)
        factor_risk_contribution = factor_covariance @ beta_vector

        factor_exposures = {}
        violations = []

        for i, factor in enumerate(factors):
            beta_i = beta_vector[i]

            # Risk contribution (may be negative for hedging factors)
            risk_contrib = beta_i * factor_risk_contribution[i]
            risk_pct = risk_contrib / portfolio_variance if portfolio_variance > 0 else 0

            # Marginal risk contribution
            marginal_risk = factor_risk_contribution[i] / portfolio_risk if portfolio_risk > 0 else 0

            within_limit = abs(risk_pct) <= self.max_factor_risk_pct

            factor_exposures[factor] = FactorExposure(
                factor_name=factor,
                raw_exposure=beta_i,
                risk_contribution=risk_contrib,
                risk_pct=risk_pct,
                marginal_risk=marginal_risk,
                within_limit=within_limit,
                limit=self.max_factor_risk_pct,
            )

            if not within_limit:
                violations.append((
                    factor,
                    FactorExposureViolation.SINGLE_FACTOR_TOO_HIGH,
                    abs(risk_pct),
                ))

        # Calculate concentration (Herfindahl index)
        risk_pcts = [abs(fe.risk_pct) for fe in factor_exposures.values()]
        total_abs_risk = sum(risk_pcts)

        if total_abs_risk > 0:
            normalized_pcts = [r / total_abs_risk for r in risk_pcts]
            concentration_hhi = sum(p ** 2 for p in normalized_pcts)
        else:
            concentration_hhi = 1.0  # Fully concentrated (no factors)

        if concentration_hhi > self.max_concentration_hhi:
            violations.append((
                "portfolio",
                FactorExposureViolation.CONCENTRATION_TOO_HIGH,
                concentration_hhi,
            ))

        # Check for correlated factor clusters
        self._check_correlation_clusters(
            factors, factor_covariance, portfolio_exposures, violations
        )

        # Generate suggested adjustments for violations
        suggested_adjustments = self._suggest_adjustments(
            positions, factor_loadings, factors, violations, portfolio_exposures
        )

        within_limits = len(violations) == 0

        result = ExposureLimitResult(
            within_limits=within_limits,
            total_portfolio_risk=portfolio_risk,
            factor_exposures=factor_exposures,
            violations=violations,
            concentration_hhi=concentration_hhi,
            suggested_adjustments=suggested_adjustments,
        )

        # Track history
        self._exposure_history.append(result)
        if len(self._exposure_history) > 100:
            self._exposure_history = self._exposure_history[-100:]

        return result

    def _check_correlation_clusters(
        self,
        factors: List[str],
        covariance: np.ndarray,
        exposures: Dict[str, float],
        violations: List[Tuple[str, FactorExposureViolation, float]],
    ):
        """Check for highly correlated factor clusters with significant exposure."""
        n = len(factors)
        if n < 2:
            return

        # Convert to correlation matrix
        std_devs = np.sqrt(np.diag(covariance))
        std_devs = np.where(std_devs == 0, 1, std_devs)  # Avoid division by zero
        corr_matrix = covariance / np.outer(std_devs, std_devs)

        # Find highly correlated pairs with significant exposure
        threshold = self.factor_correlation_threshold
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > threshold:
                    exp_i = abs(exposures.get(factors[i], 0))
                    exp_j = abs(exposures.get(factors[j], 0))
                    # Both factors must have significant exposure
                    if exp_i > 0.05 and exp_j > 0.05:
                        exp_i + exp_j
                        violations.append((
                            f"{factors[i]}+{factors[j]}",
                            FactorExposureViolation.CORRELATION_CLUSTER,
                            abs(corr_matrix[i, j]),
                        ))

    def _suggest_adjustments(
        self,
        positions: Dict[str, float],
        factor_loadings: Dict[str, Dict[str, float]],
        factors: List[str],
        violations: List[Tuple[str, FactorExposureViolation, float]],
        portfolio_exposures: Dict[str, float],
    ) -> Dict[str, float]:
        """Suggest position adjustments to bring exposures within limits."""
        adjustments = {}

        if not violations:
            return adjustments

        # Find factors that need reduction
        factors_to_reduce = [
            (f, v) for f, vtype, v in violations
            if vtype == FactorExposureViolation.SINGLE_FACTOR_TOO_HIGH
        ]

        if not factors_to_reduce:
            return adjustments

        for factor, excess_pct in factors_to_reduce:
            loadings = factor_loadings.get(factor, {})
            exposure = portfolio_exposures.get(factor, 0)

            if abs(exposure) < 1e-10:
                continue

            # Target exposure to reduce to limit
            target_exposure = exposure * (self.max_factor_risk_pct / excess_pct) * 0.95  # 5% buffer

            # Find positions contributing most to this factor
            contributors = []
            for symbol, weight in positions.items():
                loading = loadings.get(symbol, 0)
                contribution = weight * loading
                if abs(contribution) > 1e-10 and np.sign(contribution) == np.sign(exposure):
                    contributors.append((symbol, contribution, loading, weight))

            if not contributors:
                continue

            # Sort by absolute contribution
            contributors.sort(key=lambda x: abs(x[1]), reverse=True)

            # Calculate reduction needed
            reduction_needed = exposure - target_exposure
            reduction_applied = 0

            for symbol, contribution, loading, weight in contributors:
                if abs(reduction_applied) >= abs(reduction_needed):
                    break

                # How much to reduce this position
                max_reduce = min(abs(contribution), abs(reduction_needed - reduction_applied))

                # Calculate adjustment multiplier
                if abs(weight * loading) > 1e-10:
                    adjustment = 1 - (max_reduce / abs(contribution))
                    adjustment = max(0.5, min(1.0, adjustment))  # Limit to 50% reduction

                    # Aggregate if symbol already has adjustment
                    if symbol in adjustments:
                        adjustments[symbol] = min(adjustments[symbol], adjustment)
                    else:
                        adjustments[symbol] = adjustment

                    reduction_applied += max_reduce

        return adjustments

    def _empty_result(self) -> ExposureLimitResult:
        """Return empty result for edge cases."""
        return ExposureLimitResult(
            within_limits=True,
            total_portfolio_risk=0.0,
            factor_exposures={},
            violations=[],
            concentration_hhi=0.0,
            suggested_adjustments={},
        )

    def get_exposure_summary(self) -> Dict[str, Any]:
        """Get summary of current and historical exposures."""
        if not self._exposure_history:
            return {"error": "No exposure history available"}

        latest = self._exposure_history[-1]

        # Calculate average violations over history
        violation_counts = {}
        for result in self._exposure_history:
            for factor, vtype, _ in result.violations:
                key = f"{factor}:{vtype.value}"
                violation_counts[key] = violation_counts.get(key, 0) + 1

        # Most common violations
        sorted_violations = sorted(
            violation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "latest": latest.to_dict(),
            "history_length": len(self._exposure_history),
            "avg_portfolio_risk": np.mean([r.total_portfolio_risk for r in self._exposure_history]),
            "pct_within_limits": sum(1 for r in self._exposure_history if r.within_limits) / len(self._exposure_history),
            "avg_concentration_hhi": np.mean([r.concentration_hhi for r in self._exposure_history]),
            "most_common_violations": sorted_violations,
        }

    def clear_history(self):
        """Clear exposure history."""
        self._exposure_history = []


class RealTimeFactorMonitor:
    """
    Real-time factor exposure monitoring with alerts.

    Integrates with trading loop to continuously monitor exposures.
    """

    def __init__(
        self,
        limiter: FactorExposureLimiter,
        alert_callback: Optional[callable] = None,
        check_interval_seconds: int = 300,  # 5 minutes
    ):
        """
        Initialize real-time monitor.

        Args:
            limiter: FactorExposureLimiter instance
            alert_callback: Function to call on violations (factor, violation_type, value)
            check_interval_seconds: Minimum time between checks
        """
        self.limiter = limiter
        self.alert_callback = alert_callback
        self.check_interval_seconds = check_interval_seconds

        self._last_check: Optional[datetime] = None
        self._last_result: Optional[ExposureLimitResult] = None
        self._violation_cooldown: Dict[str, datetime] = {}  # Avoid alert spam

    def should_check(self) -> bool:
        """Check if enough time has passed for new exposure check."""
        if self._last_check is None:
            return True

        elapsed = (datetime.now() - self._last_check).total_seconds()
        return elapsed >= self.check_interval_seconds

    def check_and_alert(
        self,
        positions: Dict[str, float],
        factor_loadings: Dict[str, Dict[str, float]],
        factor_covariance: Optional[np.ndarray] = None,
        factor_names: Optional[List[str]] = None,
        force: bool = False,
    ) -> Optional[ExposureLimitResult]:
        """
        Check exposures and send alerts if violations found.

        Args:
            positions: Current portfolio positions
            factor_loadings: Factor loadings
            factor_covariance: Factor covariance matrix
            factor_names: Factor names
            force: Force check even if interval not elapsed

        Returns:
            ExposureLimitResult if check was performed, None otherwise
        """
        if not force and not self.should_check():
            return None

        result = self.limiter.check_exposures(
            positions, factor_loadings, factor_covariance, factor_names
        )

        self._last_check = datetime.now()
        self._last_result = result

        # Send alerts for new violations
        if not result.within_limits and self.alert_callback:
            for factor, vtype, value in result.violations:
                alert_key = f"{factor}:{vtype.value}"

                # Check cooldown (don't spam same alert within 1 hour)
                if alert_key in self._violation_cooldown:
                    elapsed = (datetime.now() - self._violation_cooldown[alert_key]).total_seconds()
                    if elapsed < 3600:  # 1 hour cooldown
                        continue

                try:
                    self.alert_callback(factor, vtype, value)
                    self._violation_cooldown[alert_key] = datetime.now()
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return result

    @property
    def last_result(self) -> Optional[ExposureLimitResult]:
        """Get most recent exposure check result."""
        return self._last_result


def print_exposure_report(result: ExposureLimitResult):
    """Print formatted exposure report to console."""
    print("\n" + "=" * 60)
    print("FACTOR EXPOSURE REPORT")
    print("=" * 60)
    print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Portfolio Risk: {result.total_portfolio_risk:.2%}")
    print(f"Within Limits: {'✓ Yes' if result.within_limits else '✗ No'}")
    print(f"Concentration (HHI): {result.concentration_hhi:.3f}")

    print("\n--- Factor Exposures ---")
    print(f"{'Factor':<25} {'Exposure':>10} {'Risk %':>10} {'Status':>10}")
    print("-" * 60)

    # Sort by absolute risk contribution
    sorted_factors = sorted(
        result.factor_exposures.items(),
        key=lambda x: abs(x[1].risk_pct),
        reverse=True
    )

    for factor, exposure in sorted_factors:
        status = "✓" if exposure.within_limit else "✗ BREACH"
        print(
            f"{factor:<25} {exposure.raw_exposure:>10.3f} "
            f"{exposure.risk_pct:>9.1%} {status:>10}"
        )

    if result.violations:
        print("\n--- Violations ---")
        for factor, vtype, value in result.violations:
            print(f"  • {factor}: {vtype.value} ({value:.2%})")

    if result.suggested_adjustments:
        print("\n--- Suggested Adjustments ---")
        for symbol, adjustment in result.suggested_adjustments.items():
            reduction = (1 - adjustment) * 100
            print(f"  • {symbol}: Reduce position by {reduction:.1f}%")

    print("=" * 60 + "\n")
