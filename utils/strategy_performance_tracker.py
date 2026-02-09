"""
Strategy Performance Tracker

Tracks rolling performance of individual strategies to dynamically adjust weights.

Features:
- Records signal outcomes for each strategy
- Calculates rolling hit rate and profit factor
- Adjusts weights based on recent accuracy
- Decay factor for older observations

Usage:
    tracker = StrategyPerformanceTracker()

    # Record outcomes
    tracker.record_signal_outcome("MomentumStrategy", symbol, predicted, actual, pnl)

    # Get adaptive weights
    weights = tracker.get_adaptive_weights(["MomentumStrategy", "MeanReversionStrategy"])
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalOutcome:
    """Record of a strategy signal and its outcome."""

    strategy_name: str
    symbol: str
    timestamp: datetime
    predicted_direction: str  # 'buy', 'sell', 'neutral'
    actual_direction: str  # What actually happened
    was_correct: bool
    pnl: float  # Realized P&L if trade was taken
    confidence: float  # Strategy's confidence in signal
    regime: Optional[str] = None  # Market regime at time of signal


class StrategyPerformanceTracker:
    """
    Tracks strategy performance for adaptive weight adjustment.
    """

    def __init__(
        self,
        lookback_trades: int = 100,
        min_trades_for_adjustment: int = 20,
        decay_half_life_days: float = 30,
        base_weight: float = 1.0,
    ):
        """
        Initialize performance tracker.

        Args:
            lookback_trades: Number of recent trades to consider
            min_trades_for_adjustment: Minimum trades before adjusting weights
            decay_half_life_days: Half-life for exponential decay
            base_weight: Default weight for new strategies
        """
        self.lookback_trades = lookback_trades
        self.min_trades = min_trades_for_adjustment
        self.decay_half_life = decay_half_life_days
        self.base_weight = base_weight

        # Outcome history per strategy
        self._outcomes: Dict[str, Deque[SignalOutcome]] = {}

        # Performance cache
        self._performance_cache: Dict[str, Dict[str, float]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

    def record_signal_outcome(
        self,
        strategy_name: str,
        symbol: str,
        predicted: str,
        actual: str,
        pnl: float,
        confidence: float = 1.0,
        regime: Optional[str] = None,
    ):
        """
        Record the outcome of a strategy signal.

        Args:
            strategy_name: Name of the strategy
            symbol: Stock symbol
            predicted: Predicted direction ('buy', 'sell', 'neutral')
            actual: Actual price movement direction
            pnl: Realized P&L (can be 0 if trade wasn't taken)
            confidence: Strategy's confidence (0-1)
            regime: Current market regime
        """
        if strategy_name not in self._outcomes:
            self._outcomes[strategy_name] = deque(maxlen=self.lookback_trades)

        # Determine if prediction was correct
        was_correct = (
            (predicted == "buy" and actual == "up")
            or (predicted == "sell" and actual == "down")
            or (predicted == "neutral" and actual == "flat")
        )

        outcome = SignalOutcome(
            strategy_name=strategy_name,
            symbol=symbol,
            timestamp=datetime.now(),
            predicted_direction=predicted,
            actual_direction=actual,
            was_correct=was_correct,
            pnl=pnl,
            confidence=confidence,
            regime=regime,
        )

        self._outcomes[strategy_name].append(outcome)

        # Invalidate cache
        if strategy_name in self._performance_cache:
            del self._performance_cache[strategy_name]

        logger.debug(
            f"Recorded {strategy_name} outcome: {predicted} -> {actual}, "
            f"correct={was_correct}, pnl=${pnl:.2f}"
        )

    def get_strategy_performance(
        self, strategy_name: str
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a strategy.

        Returns:
            Dict with hit_rate, profit_factor, avg_pnl, sharpe_ratio
        """
        # Check cache
        if (
            strategy_name in self._performance_cache
            and strategy_name in self._cache_time
            and (datetime.now() - self._cache_time[strategy_name]).total_seconds()
            < self._cache_ttl_seconds
        ):
            return self._performance_cache[strategy_name]

        if strategy_name not in self._outcomes:
            return self._default_performance()

        outcomes = list(self._outcomes[strategy_name])
        if len(outcomes) < self.min_trades:
            return self._default_performance()

        # Calculate decay weights
        now = datetime.now()
        weights = []
        for outcome in outcomes:
            days_old = (now - outcome.timestamp).total_seconds() / 86400
            weight = 0.5 ** (days_old / self.decay_half_life)
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Hit rate (weighted)
        correct_mask = np.array([o.was_correct for o in outcomes])
        hit_rate = np.sum(correct_mask * weights)

        # P&L metrics
        pnls = np.array([o.pnl for o in outcomes])
        avg_pnl = np.sum(pnls * weights)

        # Profit factor
        gains = pnls[pnls > 0]
        losses = np.abs(pnls[pnls < 0])

        if len(gains) > 0 and len(losses) > 0:
            profit_factor = gains.sum() / losses.sum()
        elif len(gains) > 0:
            profit_factor = 3.0  # No losses = excellent
        else:
            profit_factor = 0.5  # No gains = poor

        # Sharpe-like ratio (signal-to-noise)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = avg_pnl / np.std(pnls)
        else:
            sharpe_ratio = 0

        performance = {
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "sharpe_ratio": sharpe_ratio,
            "trade_count": len(outcomes),
            "confidence_correlation": self._calculate_confidence_correlation(outcomes),
        }

        # Cache result
        self._performance_cache[strategy_name] = performance
        self._cache_time[strategy_name] = datetime.now()

        return performance

    def _default_performance(self) -> Dict[str, float]:
        """Return default performance metrics for new strategies."""
        return {
            "hit_rate": 0.5,
            "profit_factor": 1.0,
            "avg_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "trade_count": 0,
            "confidence_correlation": 0.5,
        }

    def _calculate_confidence_correlation(
        self, outcomes: List[SignalOutcome]
    ) -> float:
        """Calculate correlation between confidence and accuracy."""
        if len(outcomes) < 10:
            return 0.5

        confidences = [o.confidence for o in outcomes]
        correct = [1.0 if o.was_correct else 0.0 for o in outcomes]

        try:
            correlation = np.corrcoef(confidences, correct)[0, 1]
            if np.isnan(correlation):
                return 0.5
            return (correlation + 1) / 2  # Scale to 0-1
        except Exception:
            return 0.5

    def get_adaptive_weights(
        self,
        strategy_names: List[str],
        min_weight: float = 0.1,
        max_weight: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on recent performance.

        Args:
            strategy_names: List of strategy names
            min_weight: Minimum weight per strategy
            max_weight: Maximum weight per strategy

        Returns:
            Dict of strategy_name -> weight (sum = 1.0)
        """
        raw_weights = {}

        for name in strategy_names:
            performance = self.get_strategy_performance(name)

            # Base score on hit rate and profit factor
            hit_score = performance["hit_rate"]  # 0-1
            pf_score = min(2.0, performance["profit_factor"]) / 2  # Capped at 1

            # Combined score
            score = 0.6 * hit_score + 0.4 * pf_score

            # Boost if confidence correlates with accuracy
            conf_corr = performance["confidence_correlation"]
            if conf_corr > 0.6:
                score *= 1.1
            elif conf_corr < 0.4:
                score *= 0.9

            raw_weights[name] = max(0.1, score)

        # Normalize to sum = 1.0
        total = sum(raw_weights.values())
        weights = {k: v / total for k, v in raw_weights.items()}

        # Apply min/max constraints
        constrained_weights = {}
        for name, weight in weights.items():
            constrained_weights[name] = max(min_weight, min(max_weight, weight))

        # Renormalize after constraints
        total = sum(constrained_weights.values())
        final_weights = {k: v / total for k, v in constrained_weights.items()}

        logger.debug(
            f"Adaptive weights: {', '.join(f'{k}={v:.2f}' for k, v in final_weights.items())}"
        )

        return final_weights

    def get_regime_adjusted_weights(
        self,
        strategy_names: List[str],
        current_regime: str,
    ) -> Dict[str, float]:
        """
        Get weights adjusted for current market regime.

        Args:
            strategy_names: List of strategy names
            current_regime: Current market regime (e.g., 'bull', 'bear', 'sideways')

        Returns:
            Dict of strategy_name -> weight
        """
        # Get base adaptive weights
        weights = self.get_adaptive_weights(strategy_names)

        # Calculate regime-specific performance
        for name in strategy_names:
            if name not in self._outcomes:
                continue

            outcomes = list(self._outcomes[name])
            regime_outcomes = [o for o in outcomes if o.regime == current_regime]

            if len(regime_outcomes) >= 10:
                correct_in_regime = sum(1 for o in regime_outcomes if o.was_correct)
                regime_hit_rate = correct_in_regime / len(regime_outcomes)

                # Adjust weight based on regime performance
                if regime_hit_rate > 0.6:
                    weights[name] *= 1.2
                elif regime_hit_rate < 0.4:
                    weights[name] *= 0.8

        # Renormalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for all tracked strategies."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "strategies": {},
        }

        for name in self._outcomes.keys():
            perf = self.get_strategy_performance(name)
            report["strategies"][name] = perf

        # Calculate overall metrics
        if report["strategies"]:
            all_hit_rates = [s["hit_rate"] for s in report["strategies"].values()]
            all_pf = [s["profit_factor"] for s in report["strategies"].values()]

            report["summary"] = {
                "avg_hit_rate": np.mean(all_hit_rates),
                "avg_profit_factor": np.mean(all_pf),
                "best_strategy": max(
                    report["strategies"].items(),
                    key=lambda x: x[1]["hit_rate"],
                )[0],
                "total_trades": sum(
                    s["trade_count"] for s in report["strategies"].values()
                ),
            }

        return report

    def clear_history(self, strategy_name: Optional[str] = None):
        """Clear outcome history for a strategy or all strategies."""
        if strategy_name:
            if strategy_name in self._outcomes:
                self._outcomes[strategy_name].clear()
            if strategy_name in self._performance_cache:
                del self._performance_cache[strategy_name]
        else:
            self._outcomes.clear()
            self._performance_cache.clear()
