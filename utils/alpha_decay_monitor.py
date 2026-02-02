"""
Alpha Decay Monitor

CRITICAL FOR PRODUCTION SAFETY: Tracks out-of-sample performance degradation
to detect when your trading edge is disappearing.

Alpha decay is the natural erosion of a strategy's edge over time due to:
1. Market regime changes
2. Increased competition (others trading similar strategies)
3. Structural market changes
4. Model staleness

This module provides:
1. Real-time tracking of OOS performance vs IS performance
2. Automatic alerts when decay exceeds thresholds
3. Retraining triggers
4. Historical decay analysis
"""

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels for alpha decay."""

    NONE = "none"
    WARNING = "warning"  # Performance degraded but still positive
    CRITICAL = "critical"  # Performance below threshold
    RETRAIN_URGENTLY = "retrain_urgently"  # Model is stale, retrain immediately
    HALT_TRADING = "halt_trading"  # Stop trading until reviewed


@dataclass
class DecayMetrics:
    """Metrics for tracking alpha decay."""

    timestamp: datetime
    strategy_name: str

    # In-sample metrics (from training/backtest)
    is_sharpe: float
    is_return: float
    is_win_rate: float

    # Out-of-sample metrics (live/paper trading)
    oos_sharpe: float
    oos_return: float
    oos_win_rate: float

    # Derived metrics
    sharpe_ratio: float  # OOS / IS ratio
    return_ratio: float  # OOS / IS ratio
    win_rate_ratio: float  # OOS / IS ratio

    # Period info
    oos_period_days: int
    trade_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "is_sharpe": self.is_sharpe,
            "is_return": self.is_return,
            "is_win_rate": self.is_win_rate,
            "oos_sharpe": self.oos_sharpe,
            "oos_return": self.oos_return,
            "oos_win_rate": self.oos_win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "return_ratio": self.return_ratio,
            "win_rate_ratio": self.win_rate_ratio,
            "oos_period_days": self.oos_period_days,
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecayMetrics":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            strategy_name=data["strategy_name"],
            is_sharpe=data["is_sharpe"],
            is_return=data["is_return"],
            is_win_rate=data["is_win_rate"],
            oos_sharpe=data["oos_sharpe"],
            oos_return=data["oos_return"],
            oos_win_rate=data["oos_win_rate"],
            sharpe_ratio=data["sharpe_ratio"],
            return_ratio=data["return_ratio"],
            win_rate_ratio=data["win_rate_ratio"],
            oos_period_days=data["oos_period_days"],
            trade_count=data["trade_count"],
        )


@dataclass
class DecayAlert:
    """Alert generated when alpha decay is detected."""

    timestamp: datetime
    strategy_name: str
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    threshold: float
    message: str
    recommended_action: str


class AlphaDecayMonitor:
    """
    Monitors out-of-sample performance degradation for trading strategies.

    CRITICAL: Deploy this in production to detect when your edge is disappearing.

    Usage:
        monitor = AlphaDecayMonitor(
            retraining_threshold=0.5,  # Alert if OOS < 50% of IS
            alert_callback=send_discord_alert
        )

        # Record in-sample performance from backtest
        monitor.set_baseline(
            strategy_name="MomentumStrategy",
            is_sharpe=2.0,
            is_return=0.15,
            is_win_rate=0.55
        )

        # Periodically update with live performance
        alert = monitor.update(
            strategy_name="MomentumStrategy",
            oos_sharpe=1.5,
            oos_return=0.08,
            oos_win_rate=0.52,
            trade_count=25
        )

        if alert.alert_level == AlertLevel.RETRAIN_URGENTLY:
            await retrain_model()
    """

    def __init__(
        self,
        retraining_threshold: float = 0.5,
        warning_threshold: float = 0.7,
        critical_threshold: float = 0.3,
        min_trades_for_alert: int = 20,
        staleness_days: int = 30,
        history_months: int = 12,
        persistence_path: Optional[str] = None,
        alert_callback: Optional[Callable[[DecayAlert], None]] = None,
    ):
        """
        Initialize the alpha decay monitor.

        Args:
            retraining_threshold: OOS/IS ratio below which retraining is recommended (default 0.5)
            warning_threshold: Ratio below which warning is issued (default 0.7)
            critical_threshold: Ratio below which trading should halt (default 0.3)
            min_trades_for_alert: Minimum trades before alerts are valid (default 20)
            staleness_days: Days before model is considered stale (default 30)
            history_months: Months of history to retain (default 12)
            persistence_path: Path to save/load state (default None - no persistence)
            alert_callback: Function to call when alert is generated
        """
        self.retraining_threshold = retraining_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_trades_for_alert = min_trades_for_alert
        self.staleness_days = staleness_days
        self.history_months = history_months
        self.persistence_path = persistence_path
        self.alert_callback = alert_callback

        # Per-strategy tracking
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._metrics_history: Dict[str, deque] = {}
        self._last_validation: Dict[str, datetime] = {}
        self._cumulative_trades: Dict[str, int] = {}
        self._alerts: List[DecayAlert] = []

        # Load persisted state if available
        if persistence_path:
            self._load_state()

    def set_baseline(
        self,
        strategy_name: str,
        is_sharpe: float,
        is_return: float,
        is_win_rate: float,
        validation_date: Optional[datetime] = None,
    ):
        """
        Set in-sample baseline metrics for a strategy.

        Call this after each backtest/validation run.

        Args:
            strategy_name: Name of the strategy
            is_sharpe: In-sample Sharpe ratio
            is_return: In-sample total return
            is_win_rate: In-sample win rate
            validation_date: Date of validation (default: now)
        """
        self._baselines[strategy_name] = {
            "is_sharpe": is_sharpe,
            "is_return": is_return,
            "is_win_rate": is_win_rate,
        }
        self._last_validation[strategy_name] = validation_date or datetime.now()
        self._cumulative_trades[strategy_name] = 0

        if strategy_name not in self._metrics_history:
            self._metrics_history[strategy_name] = deque(
                maxlen=self.history_months * 4  # ~weekly samples for 12 months
            )

        logger.info(
            f"Alpha decay baseline set for {strategy_name}: "
            f"Sharpe={is_sharpe:.2f}, Return={is_return:.2%}, WinRate={is_win_rate:.1%}"
        )

        self._save_state()

    def update(
        self,
        strategy_name: str,
        oos_sharpe: float,
        oos_return: float,
        oos_win_rate: float,
        trade_count: int,
        oos_period_days: int = 30,
    ) -> Optional[DecayAlert]:
        """
        Update with out-of-sample performance and check for decay.

        Call this periodically (e.g., weekly or monthly) with live performance.

        Args:
            strategy_name: Name of the strategy
            oos_sharpe: Out-of-sample Sharpe ratio
            oos_return: Out-of-sample total return
            oos_win_rate: Out-of-sample win rate
            trade_count: Number of trades in OOS period
            oos_period_days: Length of OOS period in days

        Returns:
            DecayAlert if decay detected, None otherwise
        """
        if strategy_name not in self._baselines:
            logger.warning(f"No baseline set for {strategy_name}. Call set_baseline first.")
            return None

        baseline = self._baselines[strategy_name]
        self._cumulative_trades[strategy_name] = (
            self._cumulative_trades.get(strategy_name, 0) + trade_count
        )

        # Calculate ratios (avoid division by zero)
        def safe_ratio(oos: float, is_val: float) -> float:
            if is_val == 0:
                return 0.0 if oos <= 0 else float("inf")
            return oos / is_val

        sharpe_ratio = safe_ratio(oos_sharpe, baseline["is_sharpe"])
        return_ratio = safe_ratio(oos_return, baseline["is_return"])
        win_rate_ratio = safe_ratio(oos_win_rate, baseline["is_win_rate"])

        # Record metrics
        metrics = DecayMetrics(
            timestamp=datetime.now(),
            strategy_name=strategy_name,
            is_sharpe=baseline["is_sharpe"],
            is_return=baseline["is_return"],
            is_win_rate=baseline["is_win_rate"],
            oos_sharpe=oos_sharpe,
            oos_return=oos_return,
            oos_win_rate=oos_win_rate,
            sharpe_ratio=sharpe_ratio,
            return_ratio=return_ratio,
            win_rate_ratio=win_rate_ratio,
            oos_period_days=oos_period_days,
            trade_count=trade_count,
        )

        self._metrics_history[strategy_name].append(metrics)
        self._save_state()

        # Check for decay and generate alert
        alert = self._check_decay(strategy_name, metrics)

        if alert:
            self._alerts.append(alert)
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return alert

    def _check_decay(
        self, strategy_name: str, metrics: DecayMetrics
    ) -> Optional[DecayAlert]:
        """Check for alpha decay and generate alert if needed."""
        cumulative_trades = self._cumulative_trades.get(strategy_name, 0)

        # Don't alert if insufficient trades
        if cumulative_trades < self.min_trades_for_alert:
            logger.debug(
                f"{strategy_name}: {cumulative_trades} trades < {self.min_trades_for_alert} minimum"
            )
            return None

        # Check staleness first
        last_validation = self._last_validation.get(strategy_name)
        if last_validation:
            days_since_validation = (datetime.now() - last_validation).days
            if days_since_validation > self.staleness_days:
                return DecayAlert(
                    timestamp=datetime.now(),
                    strategy_name=strategy_name,
                    alert_level=AlertLevel.RETRAIN_URGENTLY,
                    metric_name="model_staleness",
                    current_value=days_since_validation,
                    threshold=self.staleness_days,
                    message=(
                        f"Model hasn't been validated in {days_since_validation} days "
                        f"(threshold: {self.staleness_days})"
                    ),
                    recommended_action="Retrain and revalidate the model immediately",
                )

        # Check Sharpe ratio decay (primary metric)
        if metrics.sharpe_ratio <= self.critical_threshold:
            return DecayAlert(
                timestamp=datetime.now(),
                strategy_name=strategy_name,
                alert_level=AlertLevel.HALT_TRADING,
                metric_name="sharpe_ratio",
                current_value=metrics.sharpe_ratio,
                threshold=self.critical_threshold,
                message=(
                    f"CRITICAL: OOS Sharpe ({metrics.oos_sharpe:.2f}) is only "
                    f"{metrics.sharpe_ratio:.0%} of IS Sharpe ({metrics.is_sharpe:.2f})"
                ),
                recommended_action="HALT TRADING and investigate. Model may be broken.",
            )

        if metrics.sharpe_ratio <= self.retraining_threshold:
            return DecayAlert(
                timestamp=datetime.now(),
                strategy_name=strategy_name,
                alert_level=AlertLevel.RETRAIN_URGENTLY,
                metric_name="sharpe_ratio",
                current_value=metrics.sharpe_ratio,
                threshold=self.retraining_threshold,
                message=(
                    f"OOS Sharpe ({metrics.oos_sharpe:.2f}) is only "
                    f"{metrics.sharpe_ratio:.0%} of IS Sharpe ({metrics.is_sharpe:.2f})"
                ),
                recommended_action="Retrain the model with recent data",
            )

        if metrics.sharpe_ratio <= self.warning_threshold:
            return DecayAlert(
                timestamp=datetime.now(),
                strategy_name=strategy_name,
                alert_level=AlertLevel.WARNING,
                metric_name="sharpe_ratio",
                current_value=metrics.sharpe_ratio,
                threshold=self.warning_threshold,
                message=(
                    f"Warning: OOS Sharpe ({metrics.oos_sharpe:.2f}) is "
                    f"{metrics.sharpe_ratio:.0%} of IS Sharpe ({metrics.is_sharpe:.2f})"
                ),
                recommended_action="Monitor closely. Consider reducing position sizes.",
            )

        # Check for negative OOS returns when IS was positive
        if metrics.oos_return < 0 and metrics.is_return > 0:
            return DecayAlert(
                timestamp=datetime.now(),
                strategy_name=strategy_name,
                alert_level=AlertLevel.CRITICAL,
                metric_name="return",
                current_value=metrics.oos_return,
                threshold=0.0,
                message=(
                    f"OOS return is NEGATIVE ({metrics.oos_return:.2%}) "
                    f"while IS return was positive ({metrics.is_return:.2%})"
                ),
                recommended_action="Investigate immediately. Strategy may be inverted.",
            )

        return None

    def get_decay_report(self, strategy_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive decay report for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with decay analysis
        """
        if strategy_name not in self._metrics_history:
            return {"error": f"No history for {strategy_name}"}

        history = list(self._metrics_history[strategy_name])
        if not history:
            return {"error": f"No metrics recorded for {strategy_name}"}

        # Calculate trends
        sharpe_ratios = [m.sharpe_ratio for m in history]
        return_ratios = [m.return_ratio for m in history]

        # Linear regression for trend
        if len(sharpe_ratios) >= 3:
            x = np.arange(len(sharpe_ratios))
            sharpe_trend = np.polyfit(x, sharpe_ratios, 1)[0]  # Slope
        else:
            sharpe_trend = 0

        baseline = self._baselines.get(strategy_name, {})
        last_validation = self._last_validation.get(strategy_name)
        cumulative_trades = self._cumulative_trades.get(strategy_name, 0)

        latest = history[-1]

        return {
            "strategy_name": strategy_name,
            "baseline": baseline,
            "last_validation": last_validation.isoformat() if last_validation else None,
            "days_since_validation": (
                (datetime.now() - last_validation).days if last_validation else None
            ),
            "cumulative_trades": cumulative_trades,
            "latest_metrics": latest.to_dict(),
            "history_length": len(history),
            "sharpe_ratio_trend": sharpe_trend,
            "trend_interpretation": (
                "IMPROVING" if sharpe_trend > 0.01 else
                "STABLE" if sharpe_trend > -0.01 else
                "DECLINING"
            ),
            "average_sharpe_ratio": np.mean(sharpe_ratios),
            "min_sharpe_ratio": min(sharpe_ratios),
            "max_sharpe_ratio": max(sharpe_ratios),
            "current_status": self._get_status(strategy_name),
            "recent_alerts": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "level": a.alert_level.value,
                    "message": a.message,
                }
                for a in self._alerts[-5:]
                if a.strategy_name == strategy_name
            ],
        }

    def _get_status(self, strategy_name: str) -> str:
        """Get current status for a strategy."""
        if strategy_name not in self._metrics_history:
            return "NO_DATA"

        history = list(self._metrics_history[strategy_name])
        if not history:
            return "NO_DATA"

        latest = history[-1]

        if latest.sharpe_ratio <= self.critical_threshold:
            return "CRITICAL"
        elif latest.sharpe_ratio <= self.retraining_threshold:
            return "NEEDS_RETRAINING"
        elif latest.sharpe_ratio <= self.warning_threshold:
            return "WARNING"
        else:
            return "HEALTHY"

    def get_all_strategies_status(self) -> Dict[str, str]:
        """Get status for all monitored strategies."""
        return {name: self._get_status(name) for name in self._baselines}

    def should_retrain(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Check if a strategy should be retrained.

        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        status = self._get_status(strategy_name)

        if status == "NO_DATA":
            return False, "No performance data available"
        elif status == "CRITICAL":
            return True, "Performance critically degraded"
        elif status == "NEEDS_RETRAINING":
            return True, "Performance below retraining threshold"

        # Check staleness
        last_validation = self._last_validation.get(strategy_name)
        if last_validation:
            days_since = (datetime.now() - last_validation).days
            if days_since > self.staleness_days:
                return True, f"Model is stale ({days_since} days since validation)"

        return False, "Strategy is healthy"

    def _save_state(self):
        """Save state to persistence path."""
        if not self.persistence_path:
            return

        try:
            state = {
                "baselines": self._baselines,
                "last_validation": {
                    k: v.isoformat() for k, v in self._last_validation.items()
                },
                "cumulative_trades": self._cumulative_trades,
                "metrics_history": {
                    k: [m.to_dict() for m in v]
                    for k, v in self._metrics_history.items()
                },
            }

            Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alpha decay state: {e}")

    def _load_state(self):
        """Load state from persistence path."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path, "r") as f:
                state = json.load(f)

            self._baselines = state.get("baselines", {})
            self._last_validation = {
                k: datetime.fromisoformat(v)
                for k, v in state.get("last_validation", {}).items()
            }
            self._cumulative_trades = state.get("cumulative_trades", {})

            for strategy_name, history in state.get("metrics_history", {}).items():
                self._metrics_history[strategy_name] = deque(
                    [DecayMetrics.from_dict(m) for m in history],
                    maxlen=self.history_months * 4,
                )

            logger.info(
                f"Loaded alpha decay state: {len(self._baselines)} strategies"
            )

        except Exception as e:
            logger.error(f"Failed to load alpha decay state: {e}")


def create_notifier_callback(notifier) -> Callable[[DecayAlert], None]:
    """
    Create an alert callback that sends alerts via the Notifier.

    Args:
        notifier: Notifier instance from utils/notifier.py

    Returns:
        Callback function for AlphaDecayMonitor
    """
    def callback(alert: DecayAlert):
        level_emoji = {
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.RETRAIN_URGENTLY: "🔄",
            AlertLevel.HALT_TRADING: "🛑",
        }

        # Map alert levels to notifier levels
        level_map = {
            AlertLevel.WARNING: "warning",
            AlertLevel.CRITICAL: "error",
            AlertLevel.RETRAIN_URGENTLY: "error",
            AlertLevel.HALT_TRADING: "error",
            AlertLevel.NONE: "info",
        }

        emoji = level_emoji.get(alert.alert_level, "ℹ️")
        title = f"{emoji} Alpha Decay Alert: {alert.strategy_name}"

        message = f"""
**{alert.alert_level.value.upper()}**

{alert.message}

**Metric:** {alert.metric_name}
**Current Value:** {alert.current_value:.3f}
**Threshold:** {alert.threshold:.3f}

**Recommended Action:**
{alert.recommended_action}
"""

        notifier_level = level_map.get(alert.alert_level, "warning")

        # Use notifier's async notify_alert method
        asyncio.create_task(notifier.notify_alert(title, message, notifier_level))

    return callback
