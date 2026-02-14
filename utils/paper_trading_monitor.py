"""
Paper Trading Monitor - Track and validate paper trading performance.

CRITICAL: Use this to validate strategy before live trading!

This module provides:
1. Daily performance tracking and reporting
2. Comparison of paper results vs backtest expectations
3. Slippage, fill rate, and signal accuracy tracking
4. Go-live readiness assessment

Usage:
    from utils.paper_trading_monitor import PaperTradingMonitor

    monitor = PaperTradingMonitor(strategy_name="MomentumStrategy")

    # Record trades as they happen
    await monitor.record_trade(trade_data)

    # Generate daily report
    report = await monitor.daily_report()

    # Check if ready for live trading
    if await monitor.is_go_live_ready():
        print("Safe to go live!")
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single paper trade."""

    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    expected_price: float
    actual_price: float
    slippage_pct: float
    timestamp: datetime
    signal_strength: float = 0  # 0-1 confidence of signal
    order_type: str = "market"
    fill_time_ms: int = 0
    pnl: float = 0
    strategy_name: str = ""


@dataclass
class DailyStats:
    """Daily performance statistics."""

    date: date
    trades_count: int = 0
    signals_generated: int = 0
    signals_executed: int = 0
    pnl: float = 0
    slippage_total: float = 0
    avg_slippage_pct: float = 0
    fill_rate: float = 0  # signals_executed / signals_generated
    equity: float = 0
    drawdown: float = 0
    sharpe_estimate: float = 0


@dataclass
class MonitoringState:
    """Persistent state for paper trading monitoring."""

    strategy_name: str
    start_date: datetime
    initial_equity: float
    current_equity: float
    peak_equity: float
    total_trades: int = 0
    total_pnl: float = 0
    max_drawdown: float = 0
    trades: List[Dict] = field(default_factory=list)
    daily_stats: List[Dict] = field(default_factory=list)
    backtest_sharpe: Optional[float] = None
    backtest_return: Optional[float] = None


class PaperTradingMonitor:
    """
    Monitor and validate paper trading performance.

    Tracks trades, compares to backtest expectations, and determines
    when it's safe to go live.
    """

    # Go-live criteria
    MIN_PAPER_DAYS = 30  # Minimum days of paper trading
    MIN_PAPER_TRADES = 30  # Minimum trades in paper trading
    MAX_SLIPPAGE_DEVIATION = 0.5  # Max deviation from expected slippage
    MAX_RETURN_DEVIATION = 0.3  # Max deviation from backtest return
    MAX_DRAWDOWN = 0.15  # Maximum acceptable drawdown
    MIN_CORRELATION = 0.6  # Minimum correlation with backtest

    def __init__(
        self,
        strategy_name: str,
        initial_equity: float = 100000,
        data_dir: str = "paper_trading_data",
        backtest_sharpe: Optional[float] = None,
        backtest_return: Optional[float] = None,
    ):
        """
        Initialize the paper trading monitor.

        Args:
            strategy_name: Name of the strategy being monitored
            initial_equity: Starting capital
            data_dir: Directory to store monitoring data
            backtest_sharpe: Expected Sharpe from backtest (for comparison)
            backtest_return: Expected return from backtest (for comparison)
        """
        self.strategy_name = strategy_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.data_dir / f"{strategy_name}_state.json"

        # Load or initialize state
        self.state = self._load_state() or MonitoringState(
            strategy_name=strategy_name,
            start_date=datetime.now(),
            initial_equity=initial_equity,
            current_equity=initial_equity,
            peak_equity=initial_equity,
            backtest_sharpe=backtest_sharpe,
            backtest_return=backtest_return,
        )

        # In-memory tracking
        self._today_trades: List[TradeRecord] = []
        self._signals_today = 0

        logger.info(
            f"PaperTradingMonitor initialized for {strategy_name}: "
            f"equity=${self.state.current_equity:,.2f}, "
            f"trades={self.state.total_trades}"
        )

    def _load_state(self) -> Optional[MonitoringState]:
        """Load state from file."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            data["start_date"] = datetime.fromisoformat(data["start_date"])

            return MonitoringState(**data)

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def _save_state(self):
        """Save state to file."""
        try:
            data = asdict(self.state)
            # Convert datetime to string
            data["start_date"] = self.state.start_date.isoformat()

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def record_signal(self, symbol: str, signal: str, strength: float = 0.5):
        """
        Record a signal that was generated (may or may not execute).

        Args:
            symbol: Stock symbol
            signal: Signal type ("buy", "sell", "neutral")
            strength: Signal strength (0-1)
        """
        if signal != "neutral":
            self._signals_today += 1

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        expected_price: float,
        actual_price: float,
        pnl: float = 0,
        signal_strength: float = 0.5,
        order_type: str = "market",
        fill_time_ms: int = 0,
    ):
        """
        Record a completed paper trade.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            expected_price: Price expected when signal generated
            actual_price: Actual fill price
            pnl: Profit/loss from this trade (if closing position)
            signal_strength: Confidence of the signal (0-1)
            order_type: Order type used
            fill_time_ms: Time to fill in milliseconds
        """
        # Calculate slippage
        slippage_pct = abs(actual_price - expected_price) / expected_price

        trade = TradeRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_pct=slippage_pct,
            timestamp=datetime.now(),
            signal_strength=signal_strength,
            order_type=order_type,
            fill_time_ms=fill_time_ms,
            pnl=pnl,
            strategy_name=self.strategy_name,
        )

        self._today_trades.append(trade)
        self.state.trades.append(asdict(trade))
        self.state.total_trades += 1
        self.state.total_pnl += pnl

        # Update equity
        self.state.current_equity += pnl
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity

        # Update drawdown
        current_dd = (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity
        if current_dd > self.state.max_drawdown:
            self.state.max_drawdown = current_dd

        # Save state
        self._save_state()

        logger.info(
            f"Trade recorded: {side.upper()} {quantity} {symbol} @ ${actual_price:.2f} "
            f"(slippage: {slippage_pct:.2%}, PnL: ${pnl:+.2f})"
        )

    async def daily_report(self) -> Dict[str, Any]:
        """
        Generate daily performance report.

        Returns:
            Dictionary with daily stats and comparisons
        """
        today = date.today()

        # Calculate today's stats
        today_pnl = sum(t.pnl for t in self._today_trades)
        today_slippage = (
            np.mean([t.slippage_pct for t in self._today_trades]) if self._today_trades else 0
        )
        fill_rate = len(self._today_trades) / self._signals_today if self._signals_today > 0 else 0

        # Calculate drawdown
        current_dd = (
            (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity
            if self.state.peak_equity > 0
            else 0
        )

        # Estimate Sharpe from paper trading so far
        if len(self.state.trades) >= 10:
            returns = [t.get("pnl", 0) for t in self.state.trades]
            if np.std(returns) > 0:
                sharpe_estimate = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_estimate = 0
        else:
            sharpe_estimate = 0

        daily_stat = DailyStats(
            date=today,
            trades_count=len(self._today_trades),
            signals_generated=self._signals_today,
            signals_executed=len(self._today_trades),
            pnl=today_pnl,
            slippage_total=sum(t.slippage_pct for t in self._today_trades),
            avg_slippage_pct=today_slippage,
            fill_rate=fill_rate,
            equity=self.state.current_equity,
            drawdown=current_dd,
            sharpe_estimate=sharpe_estimate,
        )

        # Save daily stats
        self.state.daily_stats.append(asdict(daily_stat))
        self._save_state()

        # Generate report
        report = {
            "date": today.isoformat(),
            "strategy": self.strategy_name,
            "today": {
                "trades": len(self._today_trades),
                "signals": self._signals_today,
                "fill_rate": f"{fill_rate:.1%}",
                "pnl": f"${today_pnl:+,.2f}",
                "avg_slippage": f"{today_slippage:.3%}",
            },
            "cumulative": {
                "days_trading": (datetime.now() - self.state.start_date).days,
                "total_trades": self.state.total_trades,
                "total_pnl": f"${self.state.total_pnl:+,.2f}",
                "current_equity": f"${self.state.current_equity:,.2f}",
                "total_return": f"{(self.state.current_equity / self.state.initial_equity - 1):.2%}",
                "max_drawdown": f"{self.state.max_drawdown:.2%}",
                "sharpe_estimate": f"{sharpe_estimate:.2f}",
            },
            "backtest_comparison": {},
            "warnings": [],
            "go_live_ready": False,
        }

        # Compare to backtest expectations
        if self.state.backtest_sharpe:
            sharpe_deviation = abs(sharpe_estimate - self.state.backtest_sharpe) / abs(
                self.state.backtest_sharpe
            )
            report["backtest_comparison"]["sharpe_deviation"] = f"{sharpe_deviation:.1%}"

            if sharpe_deviation > 0.5:
                report["warnings"].append(
                    f"Sharpe ratio differs significantly from backtest "
                    f"(paper: {sharpe_estimate:.2f}, backtest: {self.state.backtest_sharpe:.2f})"
                )

        if self.state.backtest_return:
            paper_return = self.state.current_equity / self.state.initial_equity - 1
            # Annualize for comparison
            days = max(1, (datetime.now() - self.state.start_date).days)
            annualized_return = (1 + paper_return) ** (365 / days) - 1

            return_deviation = (
                abs(annualized_return - self.state.backtest_return)
                / abs(self.state.backtest_return)
                if self.state.backtest_return != 0
                else 0
            )
            report["backtest_comparison"]["return_deviation"] = f"{return_deviation:.1%}"

            if return_deviation > 0.3:
                report["warnings"].append(
                    f"Returns differ significantly from backtest expectations "
                    f"(paper: {annualized_return:.1%}, backtest: {self.state.backtest_return:.1%})"
                )

        # Check drawdown warning
        if self.state.max_drawdown > 0.10:
            report["warnings"].append(
                f"Drawdown of {self.state.max_drawdown:.1%} approaching limit"
            )

        # Check go-live readiness
        go_live_result = await self.is_go_live_ready()
        report["go_live_ready"] = go_live_result["ready"]
        if not go_live_result["ready"]:
            report["go_live_blockers"] = go_live_result["blockers"]

        # Reset daily counters
        self._today_trades = []
        self._signals_today = 0

        return report

    async def is_go_live_ready(self) -> Dict[str, Any]:
        """
        Check if strategy is ready for live trading.

        Returns:
            Dictionary with ready status and any blockers
        """
        result = {
            "ready": True,
            "blockers": [],
            "warnings": [],
            "metrics": {},
        }

        # 1. Minimum days of paper trading
        days_trading = (datetime.now() - self.state.start_date).days
        result["metrics"]["days_trading"] = days_trading

        if days_trading < self.MIN_PAPER_DAYS:
            result["ready"] = False
            result["blockers"].append(
                f"Insufficient paper trading time: {days_trading} days "
                f"(need {self.MIN_PAPER_DAYS}+)"
            )

        # 2. Minimum trades
        result["metrics"]["total_trades"] = self.state.total_trades

        if self.state.total_trades < self.MIN_PAPER_TRADES:
            result["ready"] = False
            result["blockers"].append(
                f"Insufficient trades: {self.state.total_trades} "
                f"(need {self.MIN_PAPER_TRADES}+)"
            )

        # 3. Positive returns
        total_return = self.state.current_equity / self.state.initial_equity - 1
        result["metrics"]["total_return"] = total_return

        if total_return <= 0:
            result["ready"] = False
            result["blockers"].append(f"Paper trading is not profitable: {total_return:.1%} return")

        # 4. Maximum drawdown
        result["metrics"]["max_drawdown"] = self.state.max_drawdown

        if self.state.max_drawdown > self.MAX_DRAWDOWN:
            result["ready"] = False
            result["blockers"].append(
                f"Drawdown too high: {self.state.max_drawdown:.1%} "
                f"(max {self.MAX_DRAWDOWN:.1%})"
            )

        # 5. Check backtest correlation (if backtest data available)
        if self.state.backtest_return and len(self.state.trades) >= 20:
            # Annualize paper return
            days = max(1, days_trading)
            annualized_return = (1 + total_return) ** (365 / days) - 1

            return_deviation = (
                abs(annualized_return - self.state.backtest_return)
                / abs(self.state.backtest_return)
                if self.state.backtest_return != 0
                else 0
            )
            result["metrics"]["return_deviation"] = return_deviation

            if return_deviation > self.MAX_RETURN_DEVIATION:
                result["ready"] = False
                result["blockers"].append(
                    f"Paper trading deviates too much from backtest: "
                    f"{return_deviation:.0%} deviation (max {self.MAX_RETURN_DEVIATION:.0%})"
                )

        # 6. Check slippage
        if self.state.trades:
            avg_slippage = np.mean([t.get("slippage_pct", 0) for t in self.state.trades])
            result["metrics"]["avg_slippage"] = avg_slippage

            # Warn if slippage is high but don't block
            if avg_slippage > 0.005:  # 0.5%
                result["warnings"].append(f"Average slippage is high: {avg_slippage:.2%}")

        # 7. No circuit breaker triggers (check for large single-day losses)
        if self.state.daily_stats:
            max_daily_loss = min(s.get("pnl", 0) for s in self.state.daily_stats)
            if max_daily_loss < -0.03 * self.state.initial_equity:  # 3% loss
                result["warnings"].append(f"Had a large daily loss: ${max_daily_loss:,.2f}")

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of paper trading performance."""
        days_trading = (datetime.now() - self.state.start_date).days

        return {
            "strategy_name": self.strategy_name,
            "start_date": self.state.start_date.isoformat(),
            "days_trading": days_trading,
            "initial_equity": self.state.initial_equity,
            "current_equity": self.state.current_equity,
            "total_return": (self.state.current_equity / self.state.initial_equity - 1),
            "total_trades": self.state.total_trades,
            "total_pnl": self.state.total_pnl,
            "max_drawdown": self.state.max_drawdown,
            "avg_trade_pnl": (
                self.state.total_pnl / self.state.total_trades if self.state.total_trades > 0 else 0
            ),
        }

    def reset(self, initial_equity: float = 100000):
        """Reset monitoring state (start fresh)."""
        self.state = MonitoringState(
            strategy_name=self.strategy_name,
            start_date=datetime.now(),
            initial_equity=initial_equity,
            current_equity=initial_equity,
            peak_equity=initial_equity,
            backtest_sharpe=self.state.backtest_sharpe,
            backtest_return=self.state.backtest_return,
        )
        self._today_trades = []
        self._signals_today = 0
        self._save_state()
        logger.info(f"Paper trading monitor reset for {self.strategy_name}")
