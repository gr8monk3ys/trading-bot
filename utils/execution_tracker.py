"""
Execution Quality Tracker - Analyze and track trade execution performance.

Measures the quality of trade execution by comparing expected vs actual slippage,
tracking fill rates, and providing analytics to improve execution over time.

Key Metrics:
- Implementation Shortfall: Cost vs arrival price
- Slippage: Actual vs expected
- Fill Rate: Percentage of orders filled
- Market Impact: Price movement caused by order

Usage:
    from utils.execution_tracker import ExecutionTracker

    tracker = ExecutionTracker(database)

    # Record an execution
    await tracker.record_execution(
        symbol="AAPL",
        expected_price=150.00,
        actual_price=150.05,
        expected_slippage=0.0005,
        quantity=100,
        side="buy",
    )

    # Get execution report
    report = await tracker.get_execution_report(days=30)
    print(f"Avg slippage: {report['avg_actual_slippage']:.2%}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a single trade execution."""

    id: str
    symbol: str
    side: str
    quantity: int
    expected_price: float
    actual_price: float
    expected_slippage: float
    actual_slippage: float
    slippage_delta: float  # actual - expected (negative = better than expected)
    timestamp: datetime
    order_type: str = "market"
    execution_algo: str = "direct"  # 'direct', 'twap', 'vwap'
    venue: str = "alpaca"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """Aggregated execution quality report."""

    period_start: datetime
    period_end: datetime
    total_executions: int
    total_volume: int
    total_notional: float

    # Slippage metrics
    avg_expected_slippage: float
    avg_actual_slippage: float
    avg_slippage_delta: float  # How much better/worse than expected
    slippage_std: float
    worst_slippage: float
    best_slippage: float

    # Performance vs benchmark
    total_slippage_cost: float  # Total $ cost from slippage
    slippage_savings: float  # $ saved vs expected (negative = overpaid)

    # Execution quality scores
    fill_rate: float  # % of orders fully filled
    avg_fill_time_seconds: float
    execution_score: float  # 0-100 overall quality score

    # Breakdown by category
    by_side: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_algo: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Worst executions (for review)
    worst_executions: List[Dict[str, Any]] = field(default_factory=list)


class ExecutionTracker:
    """
    Tracks and analyzes trade execution quality.

    Provides insights into:
    - Whether executions are better or worse than expected
    - Which symbols/sides have higher slippage
    - Trends in execution quality over time
    - Cost savings from execution algorithms
    """

    def __init__(
        self,
        database=None,
        max_memory_records: int = 10000,
    ):
        """
        Initialize execution tracker.

        Args:
            database: Optional TradingDatabase for persistence
            max_memory_records: Max records to keep in memory (if no database)
        """
        self.database = database
        self.max_memory_records = max_memory_records

        # In-memory storage (used if no database)
        self._records: List[ExecutionRecord] = []
        self._record_counter = 0

    async def record_execution(
        self,
        symbol: str,
        expected_price: float,
        actual_price: float,
        expected_slippage: float,
        quantity: int,
        side: str,
        order_type: str = "market",
        execution_algo: str = "direct",
        venue: str = "alpaca",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionRecord:
        """
        Record a trade execution for analysis.

        Args:
            symbol: Stock symbol
            expected_price: Price expected at order submission
            actual_price: Actual fill price
            expected_slippage: Expected slippage as decimal (e.g., 0.001 = 0.1%)
            quantity: Number of shares
            side: 'buy' or 'sell'
            order_type: Order type ('market', 'limit', etc.)
            execution_algo: Algorithm used ('direct', 'twap', 'vwap')
            venue: Execution venue
            metadata: Additional metadata

        Returns:
            ExecutionRecord
        """
        # Calculate actual slippage
        if side.lower() == "buy":
            actual_slippage = (actual_price - expected_price) / expected_price
        else:
            actual_slippage = (expected_price - actual_price) / expected_price

        slippage_delta = actual_slippage - expected_slippage

        self._record_counter += 1
        record = ExecutionRecord(
            id=f"exec_{self._record_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            expected_price=expected_price,
            actual_price=actual_price,
            expected_slippage=expected_slippage,
            actual_slippage=actual_slippage,
            slippage_delta=slippage_delta,
            timestamp=datetime.now(),
            order_type=order_type,
            execution_algo=execution_algo,
            venue=venue,
            metadata=metadata or {},
        )

        # Store record
        if self.database:
            await self._save_to_database(record)
        else:
            self._records.append(record)
            # Trim if needed
            if len(self._records) > self.max_memory_records:
                self._records = self._records[-self.max_memory_records :]

        # Log significant deviations
        if abs(slippage_delta) > 0.005:  # > 50 bps deviation
            direction = "higher" if slippage_delta > 0 else "lower"
            logger.warning(
                f"Significant slippage deviation for {symbol}: "
                f"expected {expected_slippage:.2%}, actual {actual_slippage:.2%} "
                f"({direction} by {abs(slippage_delta):.2%})"
            )
        else:
            logger.debug(
                f"Execution recorded: {symbol} {side} {quantity} @ ${actual_price:.2f} "
                f"(slippage: {actual_slippage:.2%})"
            )

        return record

    async def _save_to_database(self, record: ExecutionRecord):
        """Save execution record to database."""
        try:
            await self.database.insert_execution(
                {
                    "id": record.id,
                    "symbol": record.symbol,
                    "side": record.side,
                    "quantity": record.quantity,
                    "expected_price": record.expected_price,
                    "actual_price": record.actual_price,
                    "expected_slippage": record.expected_slippage,
                    "actual_slippage": record.actual_slippage,
                    "slippage_delta": record.slippage_delta,
                    "timestamp": record.timestamp.isoformat(),
                    "order_type": record.order_type,
                    "execution_algo": record.execution_algo,
                    "venue": record.venue,
                    "metadata": record.metadata,
                }
            )
        except Exception as e:
            logger.error(f"Failed to save execution to database: {e}")
            # Fallback to memory
            self._records.append(record)

    async def _get_records(
        self,
        days: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> List[ExecutionRecord]:
        """Get execution records with optional filters."""
        if self.database:
            try:
                records = await self.database.get_executions(days=days, symbol=symbol)
                return [
                    ExecutionRecord(
                        id=r.get("id", ""),
                        symbol=r["symbol"],
                        side=r["side"],
                        quantity=r["quantity"],
                        expected_price=r["expected_price"],
                        actual_price=r["actual_price"],
                        expected_slippage=r["expected_slippage"],
                        actual_slippage=r["actual_slippage"],
                        slippage_delta=r["slippage_delta"],
                        timestamp=(
                            datetime.fromisoformat(r["timestamp"])
                            if isinstance(r["timestamp"], str)
                            else r["timestamp"]
                        ),
                        order_type=r.get("order_type", "market"),
                        execution_algo=r.get("execution_algo", "direct"),
                        venue=r.get("venue", "alpaca"),
                        metadata=r.get("metadata", {}),
                    )
                    for r in records
                ]
            except Exception as e:
                logger.warning(f"Database query failed, using memory: {e}")

        # Use in-memory records
        records = self._records

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            records = [r for r in records if r.timestamp >= cutoff]

        if symbol:
            records = [r for r in records if r.symbol == symbol]

        return records

    async def get_execution_report(
        self,
        days: int = 30,
        symbol: Optional[str] = None,
    ) -> ExecutionReport:
        """
        Generate comprehensive execution quality report.

        Args:
            days: Number of days to analyze
            symbol: Optional symbol filter

        Returns:
            ExecutionReport with aggregated metrics
        """
        records = await self._get_records(days=days, symbol=symbol)

        if not records:
            return ExecutionReport(
                period_start=datetime.now() - timedelta(days=days),
                period_end=datetime.now(),
                total_executions=0,
                total_volume=0,
                total_notional=0,
                avg_expected_slippage=0,
                avg_actual_slippage=0,
                avg_slippage_delta=0,
                slippage_std=0,
                worst_slippage=0,
                best_slippage=0,
                total_slippage_cost=0,
                slippage_savings=0,
                fill_rate=0,
                avg_fill_time_seconds=0,
                execution_score=0,
            )

        # Basic aggregations
        total_volume = sum(r.quantity for r in records)
        total_notional = sum(r.quantity * r.actual_price for r in records)

        expected_slippages = [r.expected_slippage for r in records]
        actual_slippages = [r.actual_slippage for r in records]
        slippage_deltas = [r.slippage_delta for r in records]

        # Calculate slippage costs
        slippage_costs = []
        for r in records:
            cost = r.actual_slippage * r.quantity * r.expected_price
            slippage_costs.append(cost)

        expected_costs = []
        for r in records:
            cost = r.expected_slippage * r.quantity * r.expected_price
            expected_costs.append(cost)

        total_slippage_cost = sum(slippage_costs)
        slippage_savings = sum(expected_costs) - total_slippage_cost

        # Breakdown by side
        by_side = {}
        for side in ["buy", "sell"]:
            side_records = [r for r in records if r.side == side]
            if side_records:
                by_side[side] = {
                    "count": len(side_records),
                    "avg_slippage": np.mean([r.actual_slippage for r in side_records]),
                    "avg_delta": np.mean([r.slippage_delta for r in side_records]),
                }

        # Breakdown by symbol
        by_symbol = {}
        symbols = {r.symbol for r in records}
        for sym in symbols:
            sym_records = [r for r in records if r.symbol == sym]
            if sym_records:
                by_symbol[sym] = {
                    "count": len(sym_records),
                    "avg_slippage": np.mean([r.actual_slippage for r in sym_records]),
                    "total_volume": sum(r.quantity for r in sym_records),
                }

        # Breakdown by algorithm
        by_algo = {}
        algos = {r.execution_algo for r in records}
        for algo in algos:
            algo_records = [r for r in records if r.execution_algo == algo]
            if algo_records:
                by_algo[algo] = {
                    "count": len(algo_records),
                    "avg_slippage": np.mean([r.actual_slippage for r in algo_records]),
                    "avg_delta": np.mean([r.slippage_delta for r in algo_records]),
                }

        # Worst executions (highest slippage delta)
        sorted_records = sorted(records, key=lambda r: r.slippage_delta, reverse=True)
        worst_executions = [
            {
                "symbol": r.symbol,
                "side": r.side,
                "quantity": r.quantity,
                "expected_slippage": r.expected_slippage,
                "actual_slippage": r.actual_slippage,
                "slippage_delta": r.slippage_delta,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in sorted_records[:5]
        ]

        # Calculate execution quality score (0-100)
        # Based on: slippage vs expected, consistency, worst case
        avg_delta = np.mean(slippage_deltas)
        std_delta = np.std(slippage_deltas) if len(slippage_deltas) > 1 else 0

        # Score components:
        # - 40% for avg slippage vs expected (< 0 = good, > 0 = bad)
        # - 30% for consistency (low std = good)
        # - 30% for worst case (small worst case = good)

        delta_score = max(0, 100 - abs(avg_delta) * 10000)  # 1bp = 1 point
        consistency_score = max(0, 100 - std_delta * 20000)
        worst_case_score = max(0, 100 - max(slippage_deltas) * 5000)

        execution_score = 0.4 * delta_score + 0.3 * consistency_score + 0.3 * worst_case_score

        return ExecutionReport(
            period_start=min(r.timestamp for r in records),
            period_end=max(r.timestamp for r in records),
            total_executions=len(records),
            total_volume=total_volume,
            total_notional=total_notional,
            avg_expected_slippage=np.mean(expected_slippages),
            avg_actual_slippage=np.mean(actual_slippages),
            avg_slippage_delta=avg_delta,
            slippage_std=std_delta,
            worst_slippage=max(actual_slippages),
            best_slippage=min(actual_slippages),
            total_slippage_cost=total_slippage_cost,
            slippage_savings=slippage_savings,
            fill_rate=1.0,  # Would need partial fill data
            avg_fill_time_seconds=0,  # Would need timing data
            execution_score=execution_score,
            by_side=by_side,
            by_symbol=by_symbol,
            by_algo=by_algo,
            worst_executions=worst_executions,
        )

    async def get_symbol_stats(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get execution statistics for a specific symbol.

        Args:
            symbol: Stock symbol
            days: Analysis period

        Returns:
            Dict with symbol-specific stats
        """
        records = await self._get_records(days=days, symbol=symbol)

        if not records:
            return {
                "symbol": symbol,
                "executions": 0,
                "message": "No execution data",
            }

        return {
            "symbol": symbol,
            "period_days": days,
            "total_executions": len(records),
            "total_volume": sum(r.quantity for r in records),
            "total_notional": sum(r.quantity * r.actual_price for r in records),
            "avg_actual_slippage": np.mean([r.actual_slippage for r in records]),
            "avg_expected_slippage": np.mean([r.expected_slippage for r in records]),
            "slippage_vs_expected": np.mean([r.slippage_delta for r in records]),
            "buy_count": sum(1 for r in records if r.side == "buy"),
            "sell_count": sum(1 for r in records if r.side == "sell"),
            "best_execution": min(r.actual_slippage for r in records),
            "worst_execution": max(r.actual_slippage for r in records),
        }

    async def get_algorithm_comparison(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Compare execution quality across different algorithms.

        Args:
            days: Analysis period

        Returns:
            Dict of algorithm -> performance stats
        """
        records = await self._get_records(days=days)

        if not records:
            return {}

        algos = {r.execution_algo for r in records}
        comparison = {}

        for algo in algos:
            algo_records = [r for r in records if r.execution_algo == algo]

            avg_slippage = np.mean([r.actual_slippage for r in algo_records])
            avg_delta = np.mean([r.slippage_delta for r in algo_records])

            comparison[algo] = {
                "executions": len(algo_records),
                "avg_slippage_pct": avg_slippage,
                "avg_slippage_bps": avg_slippage * 10000,
                "avg_delta_pct": avg_delta,
                "avg_delta_bps": avg_delta * 10000,
                "total_volume": sum(r.quantity for r in algo_records),
                "recommendation": "good" if avg_delta <= 0 else "review",
            }

        return comparison

    async def get_daily_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get daily execution summary for trend analysis.

        Args:
            days: Number of days

        Returns:
            List of daily summaries
        """
        records = await self._get_records(days=days)

        if not records:
            return []

        # Group by date
        by_date: Dict[str, List[ExecutionRecord]] = {}
        for r in records:
            date_key = r.timestamp.strftime("%Y-%m-%d")
            if date_key not in by_date:
                by_date[date_key] = []
            by_date[date_key].append(r)

        summaries = []
        for date, day_records in sorted(by_date.items()):
            summaries.append(
                {
                    "date": date,
                    "executions": len(day_records),
                    "volume": sum(r.quantity for r in day_records),
                    "avg_slippage": np.mean([r.actual_slippage for r in day_records]),
                    "avg_delta": np.mean([r.slippage_delta for r in day_records]),
                    "total_cost": sum(
                        r.actual_slippage * r.quantity * r.expected_price for r in day_records
                    ),
                }
            )

        return summaries

    def clear_records(self):
        """Clear all in-memory records."""
        self._records.clear()
        logger.info("Execution tracker records cleared")


def format_execution_report(report: ExecutionReport) -> str:
    """Format execution report for display."""
    lines = [
        "=" * 60,
        "EXECUTION QUALITY REPORT",
        "=" * 60,
        f"Period: {report.period_start.date()} to {report.period_end.date()}",
        f"Executions: {report.total_executions:,}",
        f"Volume: {report.total_volume:,} shares",
        f"Notional: ${report.total_notional:,.2f}",
        "",
        "SLIPPAGE METRICS:",
        f"  Expected Avg: {report.avg_expected_slippage:.2%} ({report.avg_expected_slippage * 10000:.1f} bps)",
        f"  Actual Avg:   {report.avg_actual_slippage:.2%} ({report.avg_actual_slippage * 10000:.1f} bps)",
        f"  Delta:        {report.avg_slippage_delta:+.2%} ({report.avg_slippage_delta * 10000:+.1f} bps)",
        f"  Std Dev:      {report.slippage_std:.2%}",
        f"  Best:         {report.best_slippage:.2%}",
        f"  Worst:        {report.worst_slippage:.2%}",
        "",
        "COST ANALYSIS:",
        f"  Total Slippage Cost: ${report.total_slippage_cost:,.2f}",
        f"  Savings vs Expected: ${report.slippage_savings:+,.2f}",
        "",
        f"EXECUTION SCORE: {report.execution_score:.1f}/100",
        "",
    ]

    if report.by_algo:
        lines.append("BY ALGORITHM:")
        for algo, stats in report.by_algo.items():
            lines.append(
                f"  {algo}: {stats['count']} trades, "
                f"{stats['avg_slippage']:.2%} slippage, "
                f"{stats['avg_delta']:+.2%} vs expected"
            )

    if report.worst_executions:
        lines.append("")
        lines.append("WORST EXECUTIONS:")
        for i, worst in enumerate(report.worst_executions[:3], 1):
            lines.append(
                f"  {i}. {worst['symbol']} {worst['side']}: "
                f"{worst['slippage_delta']:+.2%} vs expected"
            )

    lines.append("=" * 60)

    return "\n".join(lines)
