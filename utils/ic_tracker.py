"""
Information Coefficient (IC) Tracker

Tracks the correlation between strategy signals and forward returns over time.
IC is a key metric for evaluating signal quality and detecting alpha decay.

Information Coefficient (IC):
- Measures how well your signal predicts future returns
- Ranges from -1 to +1 (positive = predictive, negative = inverse predictive)
- IC > 0.05 is considered good for most strategies
- IC decay indicates alpha erosion

This module provides:
1. Rolling IC calculation
2. IC decay detection
3. IC t-statistics for significance
4. IC by market regime
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ICObservation:
    """Single IC observation (signal + realized return)."""

    timestamp: datetime
    symbol: str
    signal_value: float  # Predicted direction/magnitude
    forward_return: float  # Actual return over horizon
    market_regime: Optional[str] = None


@dataclass
class ICMetrics:
    """Aggregated IC metrics for a period."""

    period_start: datetime
    period_end: datetime
    ic_value: float  # Pearson correlation
    ic_rank: float  # Spearman rank correlation
    t_statistic: float
    p_value: float
    n_observations: int
    hit_rate: float  # % of correct direction predictions
    is_significant: bool


class ICTracker:
    """
    Tracks Information Coefficient over time for alpha decay detection.

    Usage:
        tracker = ICTracker(lookback_days=90, decay_threshold=0.5)

        # Record signals and their outcomes
        tracker.record_signal("AAPL", signal=0.8, timestamp=signal_time)
        # ... later when return is known ...
        tracker.record_return("AAPL", signal_time, actual_return=0.02)

        # Check for decay
        metrics = tracker.calculate_ic()
        if tracker.check_decay():
            print("Alpha is decaying!")
    """

    def __init__(
        self,
        lookback_days: int = 90,
        min_observations: int = 30,
        decay_threshold: float = 0.5,
        significance_level: float = 0.05,
        ic_warning_threshold: float = 0.03,
        ic_critical_threshold: float = 0.01,
    ):
        """
        Initialize the IC tracker.

        Args:
            lookback_days: Days of history to use for IC calculation
            min_observations: Minimum observations for valid IC
            decay_threshold: IC ratio below which decay is flagged
            significance_level: Alpha for statistical significance
            ic_warning_threshold: Absolute IC below which warning is issued
            ic_critical_threshold: Absolute IC below which critical alert is issued
        """
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.decay_threshold = decay_threshold
        self.significance_level = significance_level
        self.ic_warning_threshold = ic_warning_threshold
        self.ic_critical_threshold = ic_critical_threshold

        # Storage
        self._pending_signals: Dict[Tuple[str, datetime], float] = {}  # Awaiting returns
        self._observations: deque = deque(maxlen=10000)  # IC observations
        self._ic_history: deque = deque(maxlen=52)  # Weekly IC values
        self._baseline_ic: Optional[float] = None

    def record_signal(
        self,
        symbol: str,
        signal: float,
        timestamp: Optional[datetime] = None,
        market_regime: Optional[str] = None,
    ):
        """
        Record a signal prediction.

        Args:
            symbol: Stock symbol
            signal: Signal value (positive = bullish, negative = bearish)
            timestamp: Signal timestamp (default: now)
            market_regime: Optional market regime label
        """
        timestamp = timestamp or datetime.now()
        key = (symbol, timestamp)
        self._pending_signals[key] = (signal, market_regime)

    def record_return(
        self,
        symbol: str,
        signal_timestamp: datetime,
        forward_return: float,
    ):
        """
        Record the realized return for a previous signal.

        Args:
            symbol: Stock symbol
            signal_timestamp: Timestamp of the original signal
            forward_return: Realized return over the forecast horizon
        """
        key = (symbol, signal_timestamp)

        if key not in self._pending_signals:
            logger.warning(f"No pending signal for {symbol} at {signal_timestamp}")
            return

        signal, market_regime = self._pending_signals.pop(key)

        observation = ICObservation(
            timestamp=signal_timestamp,
            symbol=symbol,
            signal_value=signal,
            forward_return=forward_return,
            market_regime=market_regime,
        )

        self._observations.append(observation)

    def record_observation(
        self,
        symbol: str,
        signal: float,
        forward_return: float,
        timestamp: Optional[datetime] = None,
        market_regime: Optional[str] = None,
    ):
        """
        Record a complete observation (signal + return) in one call.

        Use this when you have both signal and return at the same time
        (e.g., in backtesting).

        Args:
            symbol: Stock symbol
            signal: Signal value
            forward_return: Realized forward return
            timestamp: Observation timestamp
            market_regime: Optional market regime label
        """
        timestamp = timestamp or datetime.now()

        observation = ICObservation(
            timestamp=timestamp,
            symbol=symbol,
            signal_value=signal,
            forward_return=forward_return,
            market_regime=market_regime,
        )

        self._observations.append(observation)

    def calculate_ic(
        self,
        lookback_days: Optional[int] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[ICMetrics]:
        """
        Calculate Information Coefficient for recent observations.

        Args:
            lookback_days: Days to look back (default: self.lookback_days)
            market_regime: Filter to specific regime (optional)

        Returns:
            ICMetrics or None if insufficient data
        """
        lookback = lookback_days or self.lookback_days
        cutoff = datetime.now() - timedelta(days=lookback)

        # Filter observations
        observations = [
            o for o in self._observations
            if o.timestamp >= cutoff
            and (market_regime is None or o.market_regime == market_regime)
        ]

        if len(observations) < self.min_observations:
            logger.debug(
                f"Insufficient observations for IC: {len(observations)} < {self.min_observations}"
            )
            return None

        signals = np.array([o.signal_value for o in observations])
        returns = np.array([o.forward_return for o in observations])

        # Pearson correlation (standard IC)
        if np.std(signals) == 0 or np.std(returns) == 0:
            ic_value = 0.0
            t_stat = 0.0
            p_value = 1.0
        else:
            ic_value, p_value = stats.pearsonr(signals, returns)
            # T-statistic for IC
            n = len(signals)
            t_stat = ic_value * np.sqrt(n - 2) / np.sqrt(1 - ic_value**2)

        # Spearman rank correlation (robust to outliers)
        ic_rank, _ = stats.spearmanr(signals, returns)

        # Hit rate (% correct direction)
        correct_direction = np.sum(np.sign(signals) == np.sign(returns))
        hit_rate = correct_direction / len(signals)

        metrics = ICMetrics(
            period_start=min(o.timestamp for o in observations),
            period_end=max(o.timestamp for o in observations),
            ic_value=ic_value,
            ic_rank=ic_rank,
            t_statistic=t_stat,
            p_value=p_value,
            n_observations=len(observations),
            hit_rate=hit_rate,
            is_significant=p_value < self.significance_level,
        )

        # Record in history
        self._ic_history.append((datetime.now(), ic_value))

        return metrics

    def set_baseline_ic(self, ic: float):
        """
        Set baseline IC from backtest/training.

        Args:
            ic: Baseline IC value
        """
        self._baseline_ic = ic
        logger.info(f"IC baseline set to {ic:.4f}")

    def check_decay(self) -> Dict[str, Any]:
        """
        Check for IC decay relative to baseline.

        Returns:
            Dictionary with decay analysis
        """
        current_metrics = self.calculate_ic()

        result = {
            "has_decay": False,
            "alert_level": "none",
            "current_ic": None,
            "baseline_ic": self._baseline_ic,
            "ic_ratio": None,
            "message": "",
            "recommendation": "",
        }

        if current_metrics is None:
            result["message"] = "Insufficient data for IC calculation"
            return result

        result["current_ic"] = current_metrics.ic_value

        # Check absolute IC thresholds
        if abs(current_metrics.ic_value) < self.ic_critical_threshold:
            result["has_decay"] = True
            result["alert_level"] = "critical"
            result["message"] = (
                f"IC ({current_metrics.ic_value:.4f}) is below critical threshold "
                f"({self.ic_critical_threshold}). Signal has no predictive power."
            )
            result["recommendation"] = "HALT: Signal is effectively random"
            return result

        if abs(current_metrics.ic_value) < self.ic_warning_threshold:
            result["has_decay"] = True
            result["alert_level"] = "warning"
            result["message"] = (
                f"IC ({current_metrics.ic_value:.4f}) is below warning threshold "
                f"({self.ic_warning_threshold}). Signal quality is degrading."
            )
            result["recommendation"] = "Investigate signal quality and consider retraining"

        # Check decay vs baseline
        if self._baseline_ic and self._baseline_ic != 0:
            ic_ratio = current_metrics.ic_value / self._baseline_ic
            result["ic_ratio"] = ic_ratio

            if ic_ratio < self.decay_threshold:
                result["has_decay"] = True
                if result["alert_level"] != "critical":
                    result["alert_level"] = "warning"
                result["message"] = (
                    f"IC has decayed to {ic_ratio:.0%} of baseline "
                    f"(current: {current_metrics.ic_value:.4f}, baseline: {self._baseline_ic:.4f})"
                )
                result["recommendation"] = "Consider retraining with recent data"

        # Check for negative IC (inverted signal)
        if current_metrics.ic_value < -0.02:
            result["has_decay"] = True
            result["alert_level"] = "critical"
            result["message"] = (
                f"NEGATIVE IC ({current_metrics.ic_value:.4f})! "
                "Signal is inversely predictive - consider inverting or halting."
            )
            result["recommendation"] = "HALT: Signal is backwards"

        return result

    def get_ic_trend(self) -> Dict[str, Any]:
        """
        Analyze IC trend over time.

        Returns:
            Dictionary with trend analysis
        """
        if len(self._ic_history) < 4:
            return {"error": "Insufficient history for trend analysis"}

        timestamps, ics = zip(*self._ic_history, strict=False)
        ics = np.array(ics)

        # Linear regression for trend
        x = np.arange(len(ics))
        slope, intercept = np.polyfit(x, ics, 1)

        # Recent vs historical
        recent_ic = np.mean(ics[-4:]) if len(ics) >= 4 else np.mean(ics)
        historical_ic = np.mean(ics[:-4]) if len(ics) > 4 else recent_ic

        return {
            "n_periods": len(ics),
            "current_ic": ics[-1],
            "recent_avg_ic": recent_ic,
            "historical_avg_ic": historical_ic,
            "trend_slope": slope,
            "trend_direction": (
                "IMPROVING" if slope > 0.001 else
                "STABLE" if slope > -0.001 else
                "DECLINING"
            ),
            "ic_volatility": np.std(ics),
            "min_ic": np.min(ics),
            "max_ic": np.max(ics),
        }

    def get_ic_by_regime(self) -> Dict[str, ICMetrics]:
        """
        Calculate IC separately for each market regime.

        Returns:
            Dictionary mapping regime name to ICMetrics
        """
        regimes = {o.market_regime for o in self._observations if o.market_regime}

        results = {}
        for regime in regimes:
            metrics = self.calculate_ic(market_regime=regime)
            if metrics:
                results[regime] = metrics

        return results

    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive IC report.

        Returns:
            Dictionary with full IC analysis
        """
        current = self.calculate_ic()
        trend = self.get_ic_trend()
        decay_check = self.check_decay()
        by_regime = self.get_ic_by_regime()

        return {
            "current_metrics": (
                {
                    "ic": current.ic_value,
                    "ic_rank": current.ic_rank,
                    "t_stat": current.t_statistic,
                    "p_value": current.p_value,
                    "hit_rate": current.hit_rate,
                    "is_significant": current.is_significant,
                    "n_observations": current.n_observations,
                }
                if current else None
            ),
            "baseline_ic": self._baseline_ic,
            "trend": trend,
            "decay_analysis": decay_check,
            "by_regime": {
                regime: {
                    "ic": m.ic_value,
                    "n_observations": m.n_observations,
                    "is_significant": m.is_significant,
                }
                for regime, m in by_regime.items()
            },
            "total_observations": len(self._observations),
            "pending_signals": len(self._pending_signals),
        }


def calculate_ic_from_signals(
    signals: np.ndarray,
    forward_returns: np.ndarray,
    method: str = "pearson",
) -> Tuple[float, float, bool]:
    """
    Calculate IC from arrays of signals and returns.

    Convenience function for backtesting.

    Args:
        signals: Array of signal values
        forward_returns: Array of forward returns
        method: 'pearson' or 'spearman'

    Returns:
        Tuple of (IC value, p-value, is_significant at 5%)
    """
    if len(signals) != len(forward_returns):
        raise ValueError("Signals and returns must have same length")

    if len(signals) < 10:
        return 0.0, 1.0, False

    if method == "spearman":
        ic, p_value = stats.spearmanr(signals, forward_returns)
    else:
        ic, p_value = stats.pearsonr(signals, forward_returns)

    return ic, p_value, p_value < 0.05
