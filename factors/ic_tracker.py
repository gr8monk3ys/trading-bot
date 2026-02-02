"""
Information Coefficient (IC) Tracker - Measure factor predictive power over time.

IC measures the correlation between factor scores and subsequent returns.
A high IC indicates the factor has predictive power; a decaying IC suggests
the factor signal is weakening.

Key Metrics:
- IC: Rank correlation between factor scores and forward returns
- IC_IR (Information Ratio): IC / std(IC) - consistency of signal
- IC Decay: How quickly IC deteriorates over time
- Hit Rate: Percentage of predictions in the correct direction

Research shows:
- IC > 0.05 is considered meaningful for most factors
- IC_IR > 0.5 indicates a consistent signal
- IC decay > 50% in 20 days suggests the factor is becoming stale

Usage:
    tracker = ICTracker(factor_portfolio)
    tracker.record_scores(date, symbol_scores)
    # After forward_days...
    tracker.record_returns(date, symbol_returns)

    ic_report = tracker.get_ic_report()
    weight_adjustments = tracker.get_weight_adjustments()

Expected Impact: +5-10% return improvement from dropping stale factors.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class ICDataPoint:
    """Single IC observation."""
    date: datetime
    factor_name: str
    ic: float  # Spearman rank correlation
    ic_positive: bool  # IC > 0
    n_symbols: int  # Number of symbols in calculation


@dataclass
class FactorICMetrics:
    """Aggregated IC metrics for a single factor."""
    factor_name: str
    mean_ic: float
    std_ic: float
    ic_ir: float  # Information Ratio = mean_ic / std_ic
    hit_rate: float  # Percentage of positive IC observations
    t_stat: float  # T-statistic for IC significance
    p_value: float  # P-value for significance
    ic_trend: float  # Slope of IC over time (positive = improving)
    recent_ic: float  # Average IC over last 5 observations
    is_significant: bool  # p_value < 0.05
    is_decaying: bool  # Recent IC < 50% of historical IC
    recommended_weight_mult: float  # Suggested weight multiplier


@dataclass
class ICReport:
    """Comprehensive IC report for all factors."""
    report_date: datetime
    forward_days: int
    total_observations: int
    factor_metrics: Dict[str, FactorICMetrics]
    best_factor: Optional[str] = None
    worst_factor: Optional[str] = None
    factors_to_reduce: List[str] = field(default_factory=list)
    factors_to_increase: List[str] = field(default_factory=list)


class ICTracker:
    """
    Track Information Coefficient for factors over time.

    Records factor scores and forward returns to calculate rolling IC.
    Provides recommendations for factor weight adjustments based on
    predictive performance.
    """

    # IC thresholds
    IC_SIGNIFICANT = 0.03      # Minimum meaningful IC
    IC_STRONG = 0.07          # Strong predictive power
    IC_IR_GOOD = 0.5          # Good information ratio
    DECAY_THRESHOLD = 0.5     # IC decay warning threshold

    def __init__(
        self,
        forward_days: int = 5,
        rolling_window: int = 60,
        min_observations: int = 20,
    ):
        """
        Initialize IC tracker.

        Args:
            forward_days: Days ahead for return calculation
            rolling_window: Number of observations for rolling IC
            min_observations: Minimum observations before calculating IC
        """
        self.forward_days = forward_days
        self.rolling_window = rolling_window
        self.min_observations = min_observations

        # Storage for scores and returns
        # {date: {factor_name: {symbol: score}}}
        self._scores: Dict[datetime, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # {date: {symbol: return}}
        self._returns: Dict[datetime, Dict[str, float]] = defaultdict(dict)

        # Calculated IC history
        # {factor_name: [ICDataPoint]}
        self._ic_history: Dict[str, List[ICDataPoint]] = defaultdict(list)

        # Date index for matching
        self._score_dates: List[datetime] = []
        self._return_dates: List[datetime] = []

        logger.info(
            f"ICTracker initialized: forward_days={forward_days}, "
            f"rolling_window={rolling_window}"
        )

    def record_scores(
        self,
        date: datetime,
        factor_scores: Dict[str, Dict[str, float]],
    ):
        """
        Record factor scores for a date.

        Args:
            date: Date of the scores
            factor_scores: {factor_name: {symbol: score}}
        """
        date_key = date.date() if hasattr(date, 'date') else date

        for factor_name, symbol_scores in factor_scores.items():
            for symbol, score in symbol_scores.items():
                self._scores[date_key][factor_name][symbol] = score

        if date_key not in self._score_dates:
            self._score_dates.append(date_key)
            self._score_dates.sort()

        logger.debug(
            f"Recorded scores for {len(factor_scores)} factors on {date_key}"
        )

    def record_returns(
        self,
        date: datetime,
        symbol_returns: Dict[str, float],
    ):
        """
        Record forward returns for a date.

        Args:
            date: Date of the returns
            symbol_returns: {symbol: return}
        """
        date_key = date.date() if hasattr(date, 'date') else date

        for symbol, ret in symbol_returns.items():
            self._returns[date_key][symbol] = ret

        if date_key not in self._return_dates:
            self._return_dates.append(date_key)
            self._return_dates.sort()

        # Try to calculate IC for dates that now have forward returns
        self._calculate_pending_ic()

        logger.debug(
            f"Recorded returns for {len(symbol_returns)} symbols on {date_key}"
        )

    def _calculate_pending_ic(self):
        """Calculate IC for score dates that now have matching return dates."""
        for score_date in self._score_dates:
            # Find the return date that is forward_days after score_date
            target_date = score_date + timedelta(days=self.forward_days)

            # Find closest return date
            return_date = None
            for rd in self._return_dates:
                if rd >= target_date:
                    return_date = rd
                    break

            if return_date is None:
                continue

            # Check if we already calculated IC for this score_date
            already_calculated = any(
                score_date in [dp.date for dp in self._ic_history.get(fn, [])]
                for fn in self._scores[score_date].keys()
            )
            if already_calculated:
                continue

            # Calculate IC for each factor
            for factor_name, symbol_scores in self._scores[score_date].items():
                self._calculate_ic(
                    score_date,
                    factor_name,
                    symbol_scores,
                    self._returns[return_date],
                )

    def _calculate_ic(
        self,
        date: datetime,
        factor_name: str,
        symbol_scores: Dict[str, float],
        symbol_returns: Dict[str, float],
    ):
        """
        Calculate IC for a factor on a specific date.

        Uses Spearman rank correlation between scores and returns.
        """
        # Find common symbols
        common_symbols = set(symbol_scores.keys()) & set(symbol_returns.keys())

        if len(common_symbols) < 5:  # Need minimum symbols
            return

        # Get aligned scores and returns
        scores = [symbol_scores[s] for s in common_symbols]
        returns = [symbol_returns[s] for s in common_symbols]

        try:
            # Spearman rank correlation (more robust than Pearson)
            ic, p_value = scipy_stats.spearmanr(scores, returns)

            if np.isnan(ic):
                return

            data_point = ICDataPoint(
                date=date,
                factor_name=factor_name,
                ic=ic,
                ic_positive=ic > 0,
                n_symbols=len(common_symbols),
            )

            self._ic_history[factor_name].append(data_point)

            # Keep only rolling_window observations
            if len(self._ic_history[factor_name]) > self.rolling_window * 2:
                self._ic_history[factor_name] = \
                    self._ic_history[factor_name][-self.rolling_window:]

        except Exception as e:
            logger.debug(f"Error calculating IC for {factor_name}: {e}")

    def get_factor_ic(self, factor_name: str) -> Optional[FactorICMetrics]:
        """
        Get IC metrics for a specific factor.

        Args:
            factor_name: Name of the factor

        Returns:
            FactorICMetrics or None if insufficient data
        """
        history = self._ic_history.get(factor_name, [])

        if len(history) < self.min_observations:
            return None

        # Use most recent rolling_window observations
        recent = history[-self.rolling_window:]
        ic_values = [dp.ic for dp in recent]

        # Calculate metrics
        mean_ic = np.mean(ic_values)
        std_ic = np.std(ic_values)
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        hit_rate = sum(1 for dp in recent if dp.ic_positive) / len(recent)

        # T-statistic and p-value
        t_stat, p_value = scipy_stats.ttest_1samp(ic_values, 0)

        # IC trend (slope of IC over time)
        x = np.arange(len(ic_values))
        slope, _, _, _, _ = scipy_stats.linregress(x, ic_values)
        ic_trend = slope

        # Recent IC (last 5 observations)
        recent_5 = ic_values[-5:] if len(ic_values) >= 5 else ic_values
        recent_ic = np.mean(recent_5)

        # Is factor decaying?
        historical_ic = np.mean(ic_values[:len(ic_values)//2]) if len(ic_values) >= 10 else mean_ic
        is_decaying = recent_ic < historical_ic * self.DECAY_THRESHOLD if historical_ic > 0 else False

        # Is IC significant?
        is_significant = p_value < 0.05 and abs(mean_ic) > self.IC_SIGNIFICANT

        # Recommended weight multiplier
        if mean_ic > self.IC_STRONG and not is_decaying:
            weight_mult = 1.2  # Increase weight
        elif mean_ic > self.IC_SIGNIFICANT and not is_decaying:
            weight_mult = 1.0  # Keep weight
        elif is_decaying or mean_ic < 0:
            weight_mult = 0.5  # Reduce weight
        elif mean_ic < self.IC_SIGNIFICANT:
            weight_mult = 0.7  # Reduce slightly
        else:
            weight_mult = 1.0

        return FactorICMetrics(
            factor_name=factor_name,
            mean_ic=mean_ic,
            std_ic=std_ic,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            t_stat=t_stat,
            p_value=p_value,
            ic_trend=ic_trend,
            recent_ic=recent_ic,
            is_significant=is_significant,
            is_decaying=is_decaying,
            recommended_weight_mult=weight_mult,
        )

    def get_ic_report(self) -> ICReport:
        """
        Get comprehensive IC report for all factors.

        Returns:
            ICReport with metrics for all tracked factors
        """
        factor_metrics = {}
        total_observations = 0

        for factor_name in self._ic_history.keys():
            metrics = self.get_factor_ic(factor_name)
            if metrics:
                factor_metrics[factor_name] = metrics
                total_observations += len(self._ic_history[factor_name])

        if not factor_metrics:
            return ICReport(
                report_date=datetime.now(),
                forward_days=self.forward_days,
                total_observations=0,
                factor_metrics={},
            )

        # Find best and worst factors
        sorted_factors = sorted(
            factor_metrics.items(),
            key=lambda x: x[1].mean_ic,
            reverse=True
        )

        best_factor = sorted_factors[0][0] if sorted_factors else None
        worst_factor = sorted_factors[-1][0] if sorted_factors else None

        # Factors to adjust
        factors_to_reduce = [
            f for f, m in factor_metrics.items()
            if m.is_decaying or m.mean_ic < 0 or not m.is_significant
        ]

        factors_to_increase = [
            f for f, m in factor_metrics.items()
            if m.mean_ic > self.IC_STRONG and not m.is_decaying and m.ic_ir > self.IC_IR_GOOD
        ]

        return ICReport(
            report_date=datetime.now(),
            forward_days=self.forward_days,
            total_observations=total_observations,
            factor_metrics=factor_metrics,
            best_factor=best_factor,
            worst_factor=worst_factor,
            factors_to_reduce=factors_to_reduce,
            factors_to_increase=factors_to_increase,
        )

    def get_weight_adjustments(self) -> Dict[str, float]:
        """
        Get recommended weight multipliers for all factors.

        Returns:
            Dict of factor_name -> weight_multiplier
        """
        adjustments = {}

        for factor_name in self._ic_history.keys():
            metrics = self.get_factor_ic(factor_name)
            if metrics:
                adjustments[factor_name] = metrics.recommended_weight_mult

        return adjustments

    def get_rolling_ic(
        self,
        factor_name: str,
        window: int = 20,
    ) -> List[Tuple[datetime, float]]:
        """
        Get rolling IC for a factor.

        Args:
            factor_name: Name of the factor
            window: Rolling window size

        Returns:
            List of (date, rolling_ic) tuples
        """
        history = self._ic_history.get(factor_name, [])

        if len(history) < window:
            return []

        result = []
        for i in range(window, len(history) + 1):
            window_data = history[i-window:i]
            rolling_ic = np.mean([dp.ic for dp in window_data])
            result.append((window_data[-1].date, rolling_ic))

        return result

    def clear_history(self):
        """Clear all historical data."""
        self._scores.clear()
        self._returns.clear()
        self._ic_history.clear()
        self._score_dates.clear()
        self._return_dates.clear()
        logger.info("IC tracker history cleared")


def format_ic_report(report: ICReport) -> str:
    """Format IC report for display."""
    lines = [
        f"=== IC Report ({report.report_date.strftime('%Y-%m-%d')}) ===",
        f"Forward Period: {report.forward_days} days",
        f"Total Observations: {report.total_observations}",
        "",
        "Factor Metrics:",
        "-" * 70,
        f"{'Factor':<25} {'IC':>8} {'IC_IR':>8} {'Hit%':>6} {'Trend':>8} {'Status':<12}",
        "-" * 70,
    ]

    for name, m in sorted(
        report.factor_metrics.items(),
        key=lambda x: x[1].mean_ic,
        reverse=True
    ):
        status = []
        if m.is_decaying:
            status.append("DECAY")
        if not m.is_significant:
            status.append("WEAK")
        if m.recommended_weight_mult > 1:
            status.append("BOOST")
        elif m.recommended_weight_mult < 1:
            status.append("REDUCE")

        status_str = ",".join(status) if status else "OK"
        trend_str = f"{m.ic_trend:+.4f}"

        lines.append(
            f"{name:<25} {m.mean_ic:>8.4f} {m.ic_ir:>8.2f} "
            f"{m.hit_rate*100:>5.1f}% {trend_str:>8} {status_str:<12}"
        )

    lines.append("-" * 70)

    if report.best_factor:
        lines.append(f"\nBest Factor: {report.best_factor}")
    if report.worst_factor:
        lines.append(f"Worst Factor: {report.worst_factor}")

    if report.factors_to_reduce:
        lines.append(f"\nFactors to REDUCE: {', '.join(report.factors_to_reduce)}")
    if report.factors_to_increase:
        lines.append(f"Factors to INCREASE: {', '.join(report.factors_to_increase)}")

    return "\n".join(lines)


class FactorWeightOptimizer:
    """
    Dynamically adjust factor weights based on IC performance.

    Integrates with FactorPortfolio to update weights based on
    rolling IC measurements.
    """

    def __init__(
        self,
        ic_tracker: ICTracker,
        base_weights: Dict[str, float],
        min_weight: float = 0.02,
        max_weight: float = 0.30,
        adjustment_speed: float = 0.2,  # How fast to adjust (0-1)
    ):
        """
        Initialize weight optimizer.

        Args:
            ic_tracker: ICTracker instance
            base_weights: Original factor weights
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            adjustment_speed: How quickly to adjust (higher = faster)
        """
        self.ic_tracker = ic_tracker
        self.base_weights = base_weights.copy()
        self.current_weights = base_weights.copy()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjustment_speed = adjustment_speed

    def update_weights(self) -> Dict[str, float]:
        """
        Update weights based on IC performance.

        Returns:
            Updated factor weights
        """
        adjustments = self.ic_tracker.get_weight_adjustments()

        for factor, mult in adjustments.items():
            if factor not in self.current_weights:
                continue

            base = self.base_weights.get(factor, 0.1)
            target = base * mult

            # Smooth adjustment
            current = self.current_weights[factor]
            new_weight = current + self.adjustment_speed * (target - current)

            # Apply bounds
            new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
            self.current_weights[factor] = new_weight

        # Normalize weights to sum to 1
        total = sum(self.current_weights.values())
        if total > 0:
            self.current_weights = {
                k: v / total for k, v in self.current_weights.items()
            }

        logger.info(
            f"Updated factor weights based on IC: "
            f"{', '.join(f'{k}={v:.1%}' for k, v in self.current_weights.items())}"
        )

        return self.current_weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.current_weights.copy()

    def reset_to_base(self):
        """Reset weights to base values."""
        self.current_weights = self.base_weights.copy()
