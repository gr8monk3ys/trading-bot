#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing

The Kelly Criterion is a mathematical formula for optimal position sizing that
maximizes long-term growth while managing risk.

Kelly Formula:
    f* = (bp - q) / b

Where:
    f* = fraction of capital to bet (optimal position size)
    b = odds received (reward/risk ratio or average_win/average_loss)
    p = probability of winning (win rate)
    q = probability of losing (1 - p)

Example:
    Win rate = 60% (p = 0.6)
    Average win = $300, Average loss = $100 (b = 3.0)
    Kelly = (3.0 * 0.6 - 0.4) / 3.0 = 0.467 = 46.7% of capital

Safety: Most traders use "Half Kelly" (divide by 2) or "Quarter Kelly" (divide by 4)
because full Kelly can be aggressive and lead to large drawdowns.

Usage:
    # Initialize Kelly calculator
    kelly = KellyCriterion(
        trades_history=past_trades,
        kelly_fraction=0.5  # Half Kelly for safety
    )

    # Get optimal position size
    position_size = kelly.calculate_position_size(
        current_capital=100000,
        expected_win_rate=0.55,
        avg_win_loss_ratio=2.0
    )
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade for Kelly calculation."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    is_winner: bool


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator.

    Features:
    - Calculates optimal position size based on historical performance
    - Supports multiple Kelly fractions (full, half, quarter)
    - Tracks trade history automatically
    - Adapts to changing win rates and profit factors
    - Includes safety limits and warnings
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,  # Half Kelly by default (conservative)
        min_trades_required: int = 30,  # Minimum trades before using Kelly
        max_position_size: float = 0.20,  # Never exceed 20% of capital
        min_position_size: float = 0.01,  # Minimum 1% position
        lookback_trades: int = 50,  # Use last N trades for calculation
    ):
        """
        Initialize Kelly Criterion calculator.

        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter, 0.5 = half, 1.0 = full)
            min_trades_required: Minimum trades before trusting Kelly calculation
            max_position_size: Maximum position size cap (safety limit)
            min_position_size: Minimum position size floor
            lookback_trades: Number of recent trades to analyze
        """
        self.kelly_fraction = kelly_fraction
        self.min_trades_required = min_trades_required
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.lookback_trades = lookback_trades

        # Trade history
        self.trades: List[Trade] = []

        # Performance metrics
        self.win_rate = None
        self.avg_win = None
        self.avg_loss = None
        self.profit_factor = None

        logger.info("Kelly Criterion initialized:")
        logger.info(f"  Kelly fraction: {kelly_fraction} ({self._get_kelly_name()})")
        logger.info(f"  Max position size: {max_position_size:.1%}")
        logger.info(f"  Min trades required: {min_trades_required}")

    def _get_kelly_name(self) -> str:
        """Get friendly name for Kelly fraction."""
        if self.kelly_fraction >= 0.9:
            return "Full Kelly (Aggressive)"
        elif self.kelly_fraction >= 0.45:
            return "Half Kelly (Moderate)"
        elif self.kelly_fraction >= 0.20:
            return "Quarter Kelly (Conservative)"
        else:
            return "Mini Kelly (Very Conservative)"

    def add_trade(self, trade: Trade):
        """
        Add a completed trade to history.

        Args:
            trade: Trade object with entry/exit details
        """
        self.trades.append(trade)

        # Update performance metrics
        self._update_performance_metrics()

        logger.debug(f"Trade added: {trade.symbol} P/L: {trade.pnl_pct:+.2f}%")

    def add_trade_from_position(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str = "long",
    ):
        """
        Add a trade from position details.

        Args:
            symbol: Stock symbol
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of shares
            side: 'long' or 'short'
        """
        # Calculate P/L
        if side == "long":
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price

        trade = Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_winner=pnl > 0,
        )

        self.add_trade(trade)

    def _update_performance_metrics(self):
        """Update win rate and profit factor from trade history."""
        if not self.trades:
            return

        # Use most recent trades
        recent_trades = self.trades[-self.lookback_trades :]

        # Calculate win rate
        winners = [t for t in recent_trades if t.is_winner]
        self.win_rate = len(winners) / len(recent_trades)

        # Calculate average win and loss
        if winners:
            self.avg_win = np.mean([t.pnl_pct for t in winners])
        else:
            self.avg_win = 0.0

        losers = [t for t in recent_trades if not t.is_winner]
        if losers:
            self.avg_loss = abs(np.mean([t.pnl_pct for t in losers]))
        else:
            self.avg_loss = 0.01  # Avoid division by zero

        # Calculate profit factor (avg win / avg loss)
        if self.avg_loss > 0:
            self.profit_factor = self.avg_win / self.avg_loss
        else:
            self.profit_factor = 0.0

    def calculate_kelly_fraction(
        self, win_rate: Optional[float] = None, profit_factor: Optional[float] = None
    ) -> float:
        """
        Calculate optimal Kelly fraction.

        Args:
            win_rate: Win rate (0.0 to 1.0). If None, uses historical win rate
            profit_factor: Average win / average loss. If None, uses historical

        Returns:
            Kelly fraction (can be negative if edge is negative)
        """
        # Use provided values or fall back to historical
        p = win_rate if win_rate is not None else self.win_rate
        b = profit_factor if profit_factor is not None else self.profit_factor

        if p is None or b is None:
            logger.warning("Insufficient data for Kelly calculation")
            return self.min_position_size

        # Kelly formula: f* = (bp - q) / b
        # Where q = 1 - p
        q = 1 - p

        if b == 0:
            return 0.0

        kelly = (b * p - q) / b

        # Apply Kelly fraction (half Kelly, quarter Kelly, etc.)
        adjusted_kelly = kelly * self.kelly_fraction

        return adjusted_kelly

    def calculate_position_size(
        self,
        current_capital: float,
        win_rate: Optional[float] = None,
        profit_factor: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            current_capital: Current account value
            win_rate: Expected win rate (uses historical if None)
            profit_factor: Expected profit factor (uses historical if None)
            current_price: Current price (for share calculation)

        Returns:
            Tuple of (position_value, position_fraction)
        """
        # Check if we have enough trade history
        if len(self.trades) < self.min_trades_required:
            logger.warning(
                f"Insufficient trade history ({len(self.trades)}/{self.min_trades_required}). "
                f"Using minimum position size."
            )
            position_fraction = self.min_position_size
        else:
            # Calculate Kelly fraction
            kelly = self.calculate_kelly_fraction(win_rate, profit_factor)

            # Apply safety limits
            if kelly < 0:
                logger.warning(f"Negative Kelly ({kelly:.2%}) - No edge detected!")
                position_fraction = 0.0
            elif kelly > self.max_position_size:
                logger.warning(
                    f"Kelly position size ({kelly:.1%}) exceeds max ({self.max_position_size:.1%}). "
                    f"Capping at max."
                )
                position_fraction = self.max_position_size
            elif kelly < self.min_position_size:
                position_fraction = self.min_position_size
            else:
                position_fraction = kelly

        # Calculate dollar value
        position_value = current_capital * position_fraction

        # Calculate shares if price provided
        shares = None
        if current_price and current_price > 0:
            shares = position_value / current_price

        logger.info("Kelly position sizing:")
        logger.info(f"  Win rate: {(win_rate or self.win_rate or 0):.1%}")
        logger.info(f"  Profit factor: {(profit_factor or self.profit_factor or 0):.2f}")
        logger.info(f"  Kelly fraction: {position_fraction:.1%}")
        logger.info(f"  Position value: ${position_value:,.2f}")
        if shares:
            logger.info(f"  Shares: {shares:.4f}")

        return position_value, position_fraction

    def get_performance_summary(self) -> Dict:
        """
        Get current performance summary.

        Returns:
            Dict with performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "kelly_fraction": 0.0,
            }

        kelly = self.calculate_kelly_fraction()

        return {
            "total_trades": len(self.trades),
            "recent_trades": min(len(self.trades), self.lookback_trades),
            "win_rate": self.win_rate or 0.0,
            "avg_win": self.avg_win or 0.0,
            "avg_loss": self.avg_loss or 0.0,
            "profit_factor": self.profit_factor or 0.0,
            "kelly_fraction": kelly,
            "recommended_position": kelly * self.kelly_fraction,
            "min_trades_met": len(self.trades) >= self.min_trades_required,
        }

    def get_recommended_sizes_table(self, capital_levels: List[float]) -> str:
        """
        Generate a table of recommended position sizes for different capital levels.

        Args:
            capital_levels: List of capital amounts to calculate for

        Returns:
            Formatted string table
        """
        kelly = self.calculate_kelly_fraction()

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("KELLY CRITERION POSITION SIZING RECOMMENDATIONS")
        lines.append("=" * 70)
        lines.append(f"Strategy: {self._get_kelly_name()}")
        lines.append(f"Win Rate: {(self.win_rate or 0):.1%}")
        lines.append(f"Profit Factor: {(self.profit_factor or 0):.2f}")
        lines.append(f"Kelly Fraction: {kelly:.1%}")
        lines.append("")
        lines.append(f"{'Capital':<15} {'Kelly %':<12} {'Position Size':<15} {'Max Loss (2%)':<15}")
        lines.append("-" * 70)

        for capital in capital_levels:
            position_value, position_fraction = self.calculate_position_size(capital)
            max_loss = position_value * 0.02  # Assume 2% stop loss

            lines.append(
                f"${capital:>12,}  "
                f"{position_fraction:>6.1%}     "
                f"${position_value:>12,.2f}  "
                f"${max_loss:>12,.2f}"
            )

        lines.append("=" * 70 + "\n")

        return "\n".join(lines)
