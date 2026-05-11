"""
Factor Attribution Analysis

Decomposes portfolio returns into:
1. Factor contributions (how much each factor added/subtracted)
2. Stock selection alpha (residual after factor adjustment)
3. Factor timing (changes in factor exposure over time)

This is CRITICAL for institutional-grade analysis:
- Understand WHERE returns come from
- Identify true alpha vs factor beta
- Detect style drift
- Evaluate manager skill vs luck
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from strategies.factor_models import CompositeScore, FactorType

logger = logging.getLogger(__name__)


@dataclass
class FactorReturn:
    """Return of a single factor over a period."""

    factor: FactorType
    period_return: float  # Factor portfolio return
    t_statistic: float
    p_value: float
    sharpe_ratio: float
    n_observations: int


@dataclass
class AttributionResult:
    """Attribution analysis result for a portfolio."""

    period_start: datetime
    period_end: datetime
    total_return: float  # Portfolio total return

    # Factor contributions
    factor_contributions: Dict[FactorType, float]  # How much each factor added
    total_factor_return: float  # Sum of factor contributions

    # Alpha
    alpha: float  # Residual after factor adjustment (TRUE skill)
    alpha_t_stat: float
    alpha_p_value: float
    is_alpha_significant: bool

    # Exposures
    factor_exposures: Dict[FactorType, float]  # Average exposures
    r_squared: float  # How much variance explained by factors

    # Stock selection
    selection_return: float  # Return from stock selection within factors
    timing_return: float  # Return from factor timing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": f"{self.period_start.date()} to {self.period_end.date()}",
            "total_return": f"{self.total_return:.2%}",
            "factor_contributions": {
                ft.value: f"{c:.2%}" for ft, c in self.factor_contributions.items()
            },
            "total_factor_return": f"{self.total_factor_return:.2%}",
            "alpha": f"{self.alpha:.2%}",
            "alpha_t_stat": f"{self.alpha_t_stat:.2f}",
            "alpha_significant": self.is_alpha_significant,
            "r_squared": f"{self.r_squared:.1%}",
            "selection_return": f"{self.selection_return:.2%}",
            "timing_return": f"{self.timing_return:.2%}",
        }


class FactorAttributor:
    """
    Performs factor-based return attribution.

    Attribution Types:
    1. Brinson Attribution: Allocation + Selection effects
    2. Risk Factor Attribution: Factor exposure x Factor return
    3. Regression Attribution: Alpha and factor betas via regression

    Usage:
        attributor = FactorAttributor()

        # Build factor return history
        attributor.add_factor_returns(date, factor_returns_dict)

        # Attribute portfolio returns
        result = attributor.attribute(
            portfolio_returns,
            portfolio_exposures,
            start_date,
            end_date
        )
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_observations: int = 30,
    ):
        """
        Initialize the attributor.

        Args:
            significance_level: Alpha for statistical tests
            min_observations: Minimum observations for valid analysis
        """
        self.significance_level = significance_level
        self.min_observations = min_observations

        # Historical data
        self._factor_returns: Dict[datetime, Dict[FactorType, float]] = {}
        self._portfolio_returns: List[Tuple[datetime, float]] = []
        self._portfolio_exposures: Dict[datetime, Dict[FactorType, float]] = {}

    @staticmethod
    def _safe_ttest_1samp(samples: np.ndarray, popmean: float = 0.0) -> tuple[float, float]:
        """
        Run one-sample t-test with deterministic handling for degenerate samples.

        Avoids scipy runtime warnings when all observations are effectively identical.
        """
        values = np.asarray(samples, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            return 0.0, 1.0

        sample_mean = float(np.mean(values))
        sample_std = float(np.std(values))
        if sample_std <= 1e-12:
            if np.isclose(sample_mean, popmean, atol=1e-12):
                return 0.0, 1.0
            t_stat = np.inf if sample_mean > popmean else -np.inf
            return float(t_stat), 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_stat, p_value = stats.ttest_1samp(values, popmean)
        if not np.isfinite(t_stat):
            t_stat = 0.0
        if not np.isfinite(p_value):
            p_value = 1.0
        return float(t_stat), float(p_value)

    @staticmethod
    def _safe_ttest_ind(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:
        """
        Run two-sample t-test with deterministic handling for degenerate samples.

        Avoids scipy runtime warnings when both samples have near-zero variance.
        """
        a = np.asarray(sample_a, dtype=float)
        b = np.asarray(sample_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if a.size < 2 or b.size < 2:
            return 0.0, 1.0

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        std_a = float(np.std(a))
        std_b = float(np.std(b))

        if std_a <= 1e-12 and std_b <= 1e-12:
            if np.isclose(mean_a, mean_b, atol=1e-12):
                return 0.0, 1.0
            t_stat = np.inf if mean_a > mean_b else -np.inf
            return float(t_stat), 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        if not np.isfinite(t_stat):
            t_stat = 0.0
        if not np.isfinite(p_value):
            p_value = 1.0
        return float(t_stat), float(p_value)

    def add_factor_returns(
        self,
        date: datetime,
        factor_returns: Dict[FactorType, float],
    ):
        """
        Add daily factor returns.

        Args:
            date: Date of returns
            factor_returns: Dict of FactorType -> daily return
        """
        self._factor_returns[date] = factor_returns

    def add_portfolio_observation(
        self,
        date: datetime,
        portfolio_return: float,
        factor_exposures: Dict[FactorType, float],
    ):
        """
        Add portfolio return and exposures for a date.

        Args:
            date: Date
            portfolio_return: Portfolio return
            factor_exposures: Portfolio's factor exposures
        """
        self._portfolio_returns.append((date, portfolio_return))
        self._portfolio_exposures[date] = factor_exposures

    def attribute(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Optional[AttributionResult]:
        """
        Perform return attribution for the period.

        Uses regression-based attribution:
        r_portfolio = alpha + sum(beta_i * r_factor_i) + epsilon

        Returns:
            AttributionResult or None if insufficient data
        """
        # Filter to date range
        returns = [
            (d, r)
            for d, r in self._portfolio_returns
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        ]

        if len(returns) < self.min_observations:
            logger.warning(f"Insufficient observations: {len(returns)} < {self.min_observations}")
            return None

        dates = [d for d, _ in returns]
        portfolio_rets = np.array([r for _, r in returns])

        # Build factor return matrix
        factor_types = list(FactorType)
        factor_matrix = []

        for date in dates:
            if date not in self._factor_returns:
                # Use zero for missing factor returns
                factor_matrix.append([0.0] * len(factor_types))
            else:
                factor_matrix.append(
                    [self._factor_returns[date].get(ft, 0.0) for ft in factor_types]
                )

        factor_matrix = np.array(factor_matrix)

        # Regression: portfolio = alpha + betas * factors
        X = np.column_stack([np.ones(len(dates)), factor_matrix])
        y = portfolio_rets

        try:
            # OLS regression
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            alpha = coeffs[0]
            betas = coeffs[1:]

            # Calculate predicted returns
            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # T-statistics for alpha
            n = len(y)
            p = len(coeffs)
            mse = ss_res / (n - p) if n > p else 0
            se_alpha = np.sqrt(mse / n) if mse > 0 else 0
            t_stat = alpha / se_alpha if se_alpha > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - p)) if n > p else 1.0

        except Exception as e:
            logger.warning(f"Regression failed: {e}")
            alpha = 0.0
            betas = np.zeros(len(factor_types))
            r_squared = 0.0
            t_stat = 0.0
            p_value = 1.0

        # Factor contributions
        factor_contributions = {}
        avg_factor_exposures = {}

        for i, ft in enumerate(factor_types):
            # Average exposure to this factor
            exposures = [self._portfolio_exposures.get(d, {}).get(ft, betas[i]) for d in dates]
            avg_exposure = np.mean(exposures)
            avg_factor_exposures[ft] = avg_exposure

            # Contribution = exposure * factor return
            factor_ret = np.mean(factor_matrix[:, i]) if len(factor_matrix) > 0 else 0
            factor_contributions[ft] = avg_exposure * factor_ret * len(dates)

        total_factor_return = sum(factor_contributions.values())

        # Selection and timing decomposition
        selection_return = self._calculate_selection_return(
            dates, portfolio_rets, factor_matrix, betas
        )
        timing_return = self._calculate_timing_return(dates, factor_matrix, betas)

        # Total return
        total_return = np.sum(portfolio_rets)

        return AttributionResult(
            period_start=min(dates),
            period_end=max(dates),
            total_return=total_return,
            factor_contributions=factor_contributions,
            total_factor_return=total_factor_return,
            alpha=alpha * len(dates),  # Annualize
            alpha_t_stat=t_stat,
            alpha_p_value=p_value,
            is_alpha_significant=p_value < self.significance_level,
            factor_exposures=avg_factor_exposures,
            r_squared=r_squared,
            selection_return=selection_return,
            timing_return=timing_return,
        )

    def _calculate_selection_return(
        self,
        dates: List[datetime],
        portfolio_returns: np.ndarray,
        factor_matrix: np.ndarray,
        betas: np.ndarray,
    ) -> float:
        """
        Calculate stock selection return.

        Selection = How well you picked stocks within each factor exposure.
        """
        # Predicted return from factors
        predicted = factor_matrix @ betas

        # Selection is residual return correlated with factors
        residuals = portfolio_returns - predicted

        # If residuals consistently positive, good selection
        return float(np.mean(residuals) * len(dates))

    def _calculate_timing_return(
        self,
        dates: List[datetime],
        factor_matrix: np.ndarray,
        betas: np.ndarray,
    ) -> float:
        """
        Calculate factor timing return.

        Timing = Did exposure increases coincide with factor gains?
        """
        if len(dates) < 2:
            return 0.0

        timing_return = 0.0

        for i, ft in enumerate(FactorType):
            exposures = [self._portfolio_exposures.get(d, {}).get(ft, betas[i]) for d in dates]

            # Exposure changes
            exp_changes = np.diff(exposures)

            # Subsequent factor returns
            factor_rets = factor_matrix[1:, i]

            # Timing = correlation of exposure change with future return
            if len(exp_changes) > 0 and len(factor_rets) > 0:
                timing_return += float(np.sum(exp_changes * factor_rets))

        return timing_return

    def calculate_factor_returns(
        self,
        price_data: pd.DataFrame,
        factor_scores: Dict[str, CompositeScore],
        n_quantiles: int = 5,
    ) -> Dict[FactorType, float]:
        """
        Calculate factor returns from long-short portfolios.

        For each factor:
        - Long top quintile, short bottom quintile
        - Return = (top quintile return) - (bottom quintile return)

        Args:
            price_data: Price DataFrame with symbols as columns
            factor_scores: Factor scores for each symbol
            n_quantiles: Number of quantiles (default 5 for quintiles)

        Returns:
            Dictionary of factor -> return
        """
        returns = price_data.pct_change().iloc[-1]

        factor_returns = {}

        for factor_type in FactorType:
            # Get factor-specific scores
            factor_z_scores = {}
            for symbol, score in factor_scores.items():
                if factor_type in score.factor_scores:
                    factor_z_scores[symbol] = score.factor_scores[factor_type].z_score

            if len(factor_z_scores) < n_quantiles * 2:
                factor_returns[factor_type] = 0.0
                continue

            # Sort by factor score
            sorted_symbols = sorted(factor_z_scores.items(), key=lambda x: x[1], reverse=True)

            # Top and bottom quintile
            n_per_quantile = len(sorted_symbols) // n_quantiles
            top_symbols = [s for s, _ in sorted_symbols[:n_per_quantile]]
            bottom_symbols = [s for s, _ in sorted_symbols[-n_per_quantile:]]

            # Calculate returns
            top_return = np.mean([returns.get(s, 0.0) for s in top_symbols if s in returns.index])
            bottom_return = np.mean(
                [returns.get(s, 0.0) for s in bottom_symbols if s in returns.index]
            )

            factor_returns[factor_type] = top_return - bottom_return

        return factor_returns

    def get_factor_report(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive factor performance report.

        Returns:
            Dictionary with factor statistics
        """
        # Filter factor returns to date range
        factor_data = {
            d: rets
            for d, rets in self._factor_returns.items()
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        }

        if not factor_data:
            return {"error": "No factor data in date range"}

        report = {}

        for factor_type in FactorType:
            returns = [rets.get(factor_type, 0.0) for rets in factor_data.values()]

            if not returns:
                continue

            returns = np.array(returns)
            cumulative = np.sum(returns)
            volatility = np.std(returns) * np.sqrt(252)
            sharpe = (np.mean(returns) * 252) / volatility if volatility > 0 else 0

            # T-test vs zero with stable handling for low-variance samples.
            t_stat, p_value = self._safe_ttest_1samp(returns, 0.0)

            report[factor_type.value] = {
                "cumulative_return": f"{cumulative:.2%}",
                "annualized_vol": f"{volatility:.2%}",
                "sharpe_ratio": f"{sharpe:.2f}",
                "t_statistic": f"{t_stat:.2f}",
                "p_value": f"{p_value:.4f}",
                "significant": p_value < self.significance_level,
                "n_observations": len(returns),
            }

        return report

    def detect_style_drift(
        self,
        window_days: int = 63,  # Quarterly
    ) -> Dict[str, Any]:
        """
        Detect if portfolio factor exposures are drifting.

        Style drift occurs when exposures change significantly over time.

        Args:
            window_days: Window for comparison

        Returns:
            Dictionary with drift analysis
        """
        if len(self._portfolio_exposures) < window_days * 2:
            return {"error": "Insufficient data for drift analysis"}

        dates = sorted(self._portfolio_exposures.keys())

        # Recent vs historical exposures
        recent_dates = dates[-window_days:]
        historical_dates = dates[-2 * window_days : -window_days]

        drift_results = {}

        for factor_type in FactorType:
            recent_exp = [self._portfolio_exposures[d].get(factor_type, 0.0) for d in recent_dates]
            historical_exp = [
                self._portfolio_exposures[d].get(factor_type, 0.0) for d in historical_dates
            ]

            recent_mean = np.mean(recent_exp)
            historical_mean = np.mean(historical_exp)
            change = recent_mean - historical_mean

            # T-test for significant difference with stable degenerate-sample handling.
            t_stat, p_value = self._safe_ttest_ind(recent_exp, historical_exp)

            drift_results[factor_type.value] = {
                "recent_exposure": f"{recent_mean:.2f}",
                "historical_exposure": f"{historical_mean:.2f}",
                "change": f"{change:+.2f}",
                "is_significant": p_value < 0.05,
                "p_value": f"{p_value:.4f}",
            }

        # Overall drift score
        significant_drifts = sum(1 for r in drift_results.values() if r["is_significant"])
        total_factors = len(drift_results)

        return {
            "factor_drift": drift_results,
            "significant_drifts": significant_drifts,
            "total_factors": total_factors,
            "drift_score": significant_drifts / total_factors if total_factors > 0 else 0,
            "alert": significant_drifts >= 2,  # Alert if 2+ factors drifting
        }


def create_attribution_report(
    portfolio_returns: pd.Series,
    factor_scores_history: Dict[datetime, Dict[str, CompositeScore]],
    price_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Convenience function to create full attribution report.

    Args:
        portfolio_returns: Series of portfolio daily returns
        factor_scores_history: Historical factor scores
        price_data: Historical price data

    Returns:
        Complete attribution report
    """
    attributor = FactorAttributor()

    # Calculate factor returns for each date
    for date in portfolio_returns.index:
        if date in factor_scores_history:
            factor_rets = attributor.calculate_factor_returns(
                price_data.loc[:date], factor_scores_history[date]
            )
            attributor.add_factor_returns(date, factor_rets)

        # Add portfolio observation
        attributor.add_portfolio_observation(
            date, portfolio_returns.loc[date], {}  # Would need exposure history
        )

    # Run attribution
    result = attributor.attribute()

    if result is None:
        return {"error": "Attribution failed - insufficient data"}

    # Add factor report
    factor_report = attributor.get_factor_report()
    drift_analysis = attributor.detect_style_drift()

    return {
        "attribution": result.to_dict(),
        "factor_performance": factor_report,
        "style_drift": drift_analysis,
    }
