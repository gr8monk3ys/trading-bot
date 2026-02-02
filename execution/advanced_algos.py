"""
Advanced Execution Algorithms

Provides institutional-grade execution algorithms:
1. Implementation Shortfall: Minimize slippage from arrival price
2. POV (Percent of Volume): Match market participation rate
3. Adaptive TWAP/VWAP: Dynamic scheduling with market conditions
4. SWEEP: Aggressive liquidity capture across venues

Why these algos matter:
- Naive execution costs 50-200 bps on large orders
- IS optimization balances urgency vs market impact
- POV algorithms reduce information leakage
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AlgoState(Enum):
    """Execution algorithm state."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Urgency(Enum):
    """Execution urgency level."""
    LOW = "low"           # Patient, maximize price
    MEDIUM = "medium"     # Balanced
    HIGH = "high"         # Faster, accept impact
    CRITICAL = "critical" # Immediate, ignore impact


@dataclass
class ExecutionSlice:
    """A single slice of the parent order."""
    slice_id: int
    target_quantity: int
    target_start_time: datetime
    target_end_time: datetime

    # Execution state
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fill_count: int = 0
    status: str = "pending"

    # Analytics
    vwap: float = 0.0
    market_volume: int = 0
    participation_rate: float = 0.0

    @property
    def remaining(self) -> int:
        return self.target_quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        if self.target_quantity == 0:
            return 0
        return self.filled_quantity / self.target_quantity


@dataclass
class AlgoOrder:
    """Parent order for algorithmic execution."""
    algo_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: int
    algo_type: str

    # Timing
    start_time: datetime
    end_time: datetime

    # Parameters
    params: Dict[str, Any] = field(default_factory=dict)

    # State
    state: AlgoState = AlgoState.PENDING
    slices: List[ExecutionSlice] = field(default_factory=list)

    # Benchmark prices
    arrival_price: float = 0.0
    decision_price: float = 0.0

    # Results
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    total_fees: float = 0.0

    @property
    def remaining(self) -> int:
        return self.total_quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        if self.total_quantity == 0:
            return 0
        return self.filled_quantity / self.total_quantity

    @property
    def slippage_bps(self) -> float:
        """Implementation shortfall from arrival price."""
        if self.arrival_price <= 0 or self.avg_fill_price <= 0:
            return 0

        if self.side == "buy":
            return (self.avg_fill_price - self.arrival_price) / self.arrival_price * 10000
        else:
            return (self.arrival_price - self.avg_fill_price) / self.arrival_price * 10000


@dataclass
class MarketSnapshot:
    """Current market state for algo decisions."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    spread_bps: float
    volume_today: int
    volatility: float
    adv: int  # Average daily volume


@dataclass
class AlgoMetrics:
    """Performance metrics for completed algo."""
    algo_id: str
    symbol: str

    # Execution
    total_quantity: int
    filled_quantity: int
    avg_fill_price: float

    # Slippage
    arrival_price: float
    vwap_price: float
    implementation_shortfall_bps: float
    vs_vwap_bps: float

    # Market impact
    realized_impact_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float

    # Timing
    duration_seconds: int
    participation_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algo_id": self.algo_id,
            "symbol": self.symbol,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "vs_vwap_bps": self.vs_vwap_bps,
            "participation_rate": self.participation_rate,
        }


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    def __init__(self, name: str):
        self.name = name
        self._current_order: Optional[AlgoOrder] = None

    @abstractmethod
    def create_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
    ) -> List[ExecutionSlice]:
        """Create execution schedule."""
        pass

    @abstractmethod
    def adjust_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
        current_slice: int,
    ) -> List[ExecutionSlice]:
        """Dynamically adjust schedule based on market conditions."""
        pass

    @abstractmethod
    def get_slice_quantity(
        self,
        slice_: ExecutionSlice,
        market: MarketSnapshot,
    ) -> int:
        """Get quantity to execute in current slice."""
        pass


class ImplementationShortfall(ExecutionAlgorithm):
    """
    Implementation Shortfall Algorithm.

    Minimizes deviation from arrival price by balancing:
    - Urgency: Higher urgency = faster execution, more impact
    - Market impact: Square-root model with temporary/permanent components
    - Timing risk: Price drift while waiting

    Based on Almgren-Chriss optimal execution framework.
    """

    def __init__(
        self,
        urgency: Urgency = Urgency.MEDIUM,
        risk_aversion: float = 1e-6,
        temp_impact_coef: float = 0.1,
        perm_impact_coef: float = 0.01,
    ):
        super().__init__("implementation_shortfall")
        self.urgency = urgency
        self.risk_aversion = risk_aversion
        self.temp_impact_coef = temp_impact_coef
        self.perm_impact_coef = perm_impact_coef

    def _calculate_optimal_trajectory(
        self,
        total_quantity: int,
        num_periods: int,
        volatility: float,
        adv: int,
    ) -> List[float]:
        """
        Calculate optimal execution trajectory using Almgren-Chriss.

        Returns list of cumulative execution percentages.
        """
        if num_periods <= 0:
            return [1.0]

        # Simplified Almgren-Chriss
        # kappa = sqrt(lambda * sigma^2 / eta)
        # where lambda = risk aversion, sigma = volatility, eta = temporary impact

        participation = total_quantity / (adv * num_periods / 390)  # Intraday periods

        # Adjust based on urgency
        urgency_factor = {
            Urgency.LOW: 0.5,
            Urgency.MEDIUM: 1.0,
            Urgency.HIGH: 2.0,
            Urgency.CRITICAL: 10.0,
        }[self.urgency]

        kappa = np.sqrt(self.risk_aversion * volatility**2 / self.temp_impact_coef)
        kappa *= urgency_factor

        # Generate trajectory
        trajectory = []
        for i in range(num_periods):
            t = (i + 1) / num_periods
            # Exponential decay trajectory
            if kappa > 0:
                x = (1 - np.exp(-kappa * t)) / (1 - np.exp(-kappa))
            else:
                x = t  # Linear if kappa = 0
            trajectory.append(min(1.0, x))

        return trajectory

    def create_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
    ) -> List[ExecutionSlice]:
        """Create IS-optimal execution schedule."""
        duration = (order.end_time - order.start_time).total_seconds()
        num_slices = max(1, int(duration / 60))  # 1-minute slices

        # Get optimal trajectory
        trajectory = self._calculate_optimal_trajectory(
            order.total_quantity,
            num_slices,
            market.volatility,
            market.adv,
        )

        slices = []
        slice_duration = duration / num_slices
        prev_pct = 0.0

        for i in range(num_slices):
            target_pct = trajectory[i]
            slice_pct = target_pct - prev_pct
            slice_qty = int(order.total_quantity * slice_pct)

            # Handle rounding on last slice
            if i == num_slices - 1:
                slice_qty = order.total_quantity - sum(s.target_quantity for s in slices)

            slices.append(ExecutionSlice(
                slice_id=i,
                target_quantity=slice_qty,
                target_start_time=order.start_time + timedelta(seconds=i * slice_duration),
                target_end_time=order.start_time + timedelta(seconds=(i + 1) * slice_duration),
            ))
            prev_pct = target_pct

        return slices

    def adjust_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
        current_slice: int,
    ) -> List[ExecutionSlice]:
        """Adjust schedule based on realized fills and market conditions."""
        if current_slice >= len(order.slices):
            return order.slices

        # Check if we're behind schedule
        expected_filled = sum(s.target_quantity for s in order.slices[:current_slice + 1])
        actual_filled = order.filled_quantity
        shortfall = expected_filled - actual_filled

        if shortfall <= 0:
            return order.slices  # On track

        # Distribute shortfall across remaining slices
        remaining_slices = order.slices[current_slice + 1:]
        if not remaining_slices:
            # Increase current slice
            order.slices[current_slice].target_quantity += shortfall
        else:
            # Spread across remaining
            per_slice = shortfall // len(remaining_slices)
            for s in remaining_slices:
                s.target_quantity += per_slice

        return order.slices

    def get_slice_quantity(
        self,
        slice_: ExecutionSlice,
        market: MarketSnapshot,
    ) -> int:
        """Get quantity considering market conditions."""
        base_qty = slice_.remaining

        # Adjust based on spread
        if market.spread_bps > 20:  # Wide spread
            return int(base_qty * 0.7)  # Reduce size

        # Adjust based on volatility
        if market.volatility > 0.03:  # High vol
            return int(base_qty * 0.8)

        return base_qty


class POVAlgorithm(ExecutionAlgorithm):
    """
    Percent of Volume (POV) Algorithm.

    Executes at a target percentage of market volume to:
    - Minimize market impact by matching market flow
    - Reduce information leakage
    - Provide natural camouflage

    Commonly used rates: 5%, 10%, 15%, 20%
    """

    def __init__(
        self,
        target_pov: float = 0.10,  # 10% of volume
        min_pov: float = 0.05,
        max_pov: float = 0.25,
        volume_lookback_minutes: int = 5,
    ):
        super().__init__("pov")
        self.target_pov = target_pov
        self.min_pov = min_pov
        self.max_pov = max_pov
        self.volume_lookback_minutes = volume_lookback_minutes

        # State
        self._volume_history: List[Tuple[datetime, int]] = []

    def create_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
    ) -> List[ExecutionSlice]:
        """Create POV schedule (indicative, adjusts in real-time)."""
        duration = (order.end_time - order.start_time).total_seconds()
        num_slices = max(1, int(duration / 60))  # 1-minute slices

        # Estimate volume per slice
        trading_minutes = 390  # Regular session
        avg_volume_per_minute = market.adv / trading_minutes
        expected_slice_volume = avg_volume_per_minute * 1  # 1 minute slices

        slices = []
        slice_duration = duration / num_slices

        for i in range(num_slices):
            # Target = POV * expected market volume
            slice_qty = int(expected_slice_volume * self.target_pov)

            slices.append(ExecutionSlice(
                slice_id=i,
                target_quantity=slice_qty,
                target_start_time=order.start_time + timedelta(seconds=i * slice_duration),
                target_end_time=order.start_time + timedelta(seconds=(i + 1) * slice_duration),
            ))

        # Adjust to match total quantity
        total_scheduled = sum(s.target_quantity for s in slices)
        if total_scheduled > 0:
            scale = order.total_quantity / total_scheduled
            for s in slices:
                s.target_quantity = int(s.target_quantity * scale)

        # Fix rounding
        diff = order.total_quantity - sum(s.target_quantity for s in slices)
        if diff > 0 and slices:
            slices[-1].target_quantity += diff

        return slices

    def adjust_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
        current_slice: int,
    ) -> List[ExecutionSlice]:
        """Adjust based on actual market volume."""
        if current_slice >= len(order.slices):
            return order.slices

        current = order.slices[current_slice]

        # Calculate actual participation rate
        if current.market_volume > 0:
            actual_pov = current.filled_quantity / current.market_volume
        else:
            actual_pov = 0

        # Adjust future slices if too fast/slow
        remaining_slices = order.slices[current_slice + 1:]

        if actual_pov > self.max_pov:
            # Slow down
            for s in remaining_slices:
                s.target_quantity = int(s.target_quantity * 0.8)
        elif actual_pov < self.min_pov and order.remaining > 0:
            # Speed up
            for s in remaining_slices:
                s.target_quantity = int(s.target_quantity * 1.2)

        return order.slices

    def get_slice_quantity(
        self,
        slice_: ExecutionSlice,
        market: MarketSnapshot,
    ) -> int:
        """Get quantity based on recent market volume."""
        # Estimate recent volume rate
        recent_volume = self._get_recent_volume()

        if recent_volume > 0:
            # Match target POV
            target_qty = int(recent_volume * self.target_pov)
            return min(target_qty, slice_.remaining)

        return slice_.remaining

    def _get_recent_volume(self) -> int:
        """Get volume over lookback period."""
        if not self._volume_history:
            return 0

        cutoff = datetime.now() - timedelta(minutes=self.volume_lookback_minutes)
        recent = [v for t, v in self._volume_history if t >= cutoff]
        return sum(recent)

    def update_volume(self, timestamp: datetime, volume: int) -> None:
        """Update volume history."""
        self._volume_history.append((timestamp, volume))

        # Trim old entries
        cutoff = datetime.now() - timedelta(minutes=self.volume_lookback_minutes * 2)
        self._volume_history = [
            (t, v) for t, v in self._volume_history if t >= cutoff
        ]


class AdaptiveTWAP(ExecutionAlgorithm):
    """
    Adaptive Time-Weighted Average Price Algorithm.

    Improves on basic TWAP by:
    - Adjusting slice sizes based on market conditions
    - Avoiding execution during wide spreads
    - Accelerating when price is favorable
    """

    def __init__(
        self,
        spread_threshold_bps: float = 10.0,
        volatility_threshold: float = 0.02,
        price_advantage_threshold_bps: float = 5.0,
    ):
        super().__init__("adaptive_twap")
        self.spread_threshold_bps = spread_threshold_bps
        self.volatility_threshold = volatility_threshold
        self.price_advantage_threshold_bps = price_advantage_threshold_bps

    def create_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
    ) -> List[ExecutionSlice]:
        """Create uniform TWAP schedule."""
        duration = (order.end_time - order.start_time).total_seconds()
        num_slices = max(1, int(duration / 60))

        slice_qty = order.total_quantity // num_slices
        slice_duration = duration / num_slices

        slices = []
        for i in range(num_slices):
            qty = slice_qty
            if i == num_slices - 1:
                qty = order.total_quantity - sum(s.target_quantity for s in slices)

            slices.append(ExecutionSlice(
                slice_id=i,
                target_quantity=qty,
                target_start_time=order.start_time + timedelta(seconds=i * slice_duration),
                target_end_time=order.start_time + timedelta(seconds=(i + 1) * slice_duration),
            ))

        return slices

    def adjust_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
        current_slice: int,
    ) -> List[ExecutionSlice]:
        """Adjust based on market conditions."""
        if current_slice >= len(order.slices):
            return order.slices

        # Wide spread: defer to later
        if market.spread_bps > self.spread_threshold_bps:
            current = order.slices[current_slice]
            remaining = order.slices[current_slice + 1:]

            if remaining:
                deferred = current.target_quantity // 2
                current.target_quantity -= deferred

                # Distribute to future slices
                per_slice = deferred // len(remaining)
                for s in remaining:
                    s.target_quantity += per_slice

        return order.slices

    def get_slice_quantity(
        self,
        slice_: ExecutionSlice,
        market: MarketSnapshot,
    ) -> int:
        """Get quantity with adaptive adjustments."""
        base_qty = slice_.remaining

        # Reduce if spread is wide
        if market.spread_bps > self.spread_threshold_bps:
            return int(base_qty * 0.5)

        # Reduce if volatile
        if market.volatility > self.volatility_threshold:
            return int(base_qty * 0.7)

        return base_qty


class AdaptiveVWAP(ExecutionAlgorithm):
    """
    Adaptive Volume-Weighted Average Price Algorithm.

    Improves on basic VWAP by:
    - Using historical intraday volume patterns
    - Adjusting to real-time volume deviations
    - Pausing during unusual market conditions
    """

    # Typical intraday volume pattern (% of daily volume per 30-min bucket)
    INTRADAY_PATTERN = [
        0.08,  # 9:30-10:00 (high open)
        0.06,  # 10:00-10:30
        0.05,  # 10:30-11:00
        0.05,  # 11:00-11:30
        0.04,  # 11:30-12:00
        0.04,  # 12:00-12:30 (lunch lull)
        0.04,  # 12:30-13:00
        0.05,  # 13:00-13:30
        0.06,  # 13:30-14:00
        0.07,  # 14:00-14:30
        0.08,  # 14:30-15:00
        0.10,  # 15:00-15:30
        0.12,  # 15:30-16:00 (high close)
        0.16,  # Extended close
    ]

    def __init__(
        self,
        volume_pattern: Optional[List[float]] = None,
        adaptive_factor: float = 0.3,
    ):
        super().__init__("adaptive_vwap")
        self.volume_pattern = volume_pattern or self.INTRADAY_PATTERN
        self.adaptive_factor = adaptive_factor

    def _get_volume_weight(self, time: datetime) -> float:
        """Get expected volume weight for time of day."""
        market_open = time.replace(hour=9, minute=30, second=0)
        minutes_since_open = (time - market_open).total_seconds() / 60

        bucket = int(minutes_since_open / 30)
        bucket = max(0, min(bucket, len(self.volume_pattern) - 1))

        return self.volume_pattern[bucket]

    def create_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
    ) -> List[ExecutionSlice]:
        """Create VWAP schedule based on historical volume pattern."""
        duration = (order.end_time - order.start_time).total_seconds()
        num_slices = max(1, int(duration / 60))
        slice_duration = duration / num_slices

        slices = []
        total_weight = 0

        # First pass: calculate weights
        for i in range(num_slices):
            slice_time = order.start_time + timedelta(seconds=i * slice_duration)
            weight = self._get_volume_weight(slice_time)
            total_weight += weight

        # Second pass: allocate quantities
        for i in range(num_slices):
            slice_time = order.start_time + timedelta(seconds=i * slice_duration)
            weight = self._get_volume_weight(slice_time)

            if total_weight > 0:
                slice_pct = weight / total_weight
            else:
                slice_pct = 1.0 / num_slices

            qty = int(order.total_quantity * slice_pct)

            slices.append(ExecutionSlice(
                slice_id=i,
                target_quantity=qty,
                target_start_time=order.start_time + timedelta(seconds=i * slice_duration),
                target_end_time=order.start_time + timedelta(seconds=(i + 1) * slice_duration),
            ))

        # Fix rounding
        diff = order.total_quantity - sum(s.target_quantity for s in slices)
        if slices:
            slices[-1].target_quantity += diff

        return slices

    def adjust_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
        current_slice: int,
    ) -> List[ExecutionSlice]:
        """Adjust based on actual vs expected volume."""
        if current_slice >= len(order.slices):
            return order.slices

        current = order.slices[current_slice]

        # Compare actual vs expected volume
        expected_volume = market.adv * self._get_volume_weight(datetime.now())

        if current.market_volume > 0 and expected_volume > 0:
            volume_ratio = current.market_volume / expected_volume

            # Adjust remaining slices
            adjustment = 1.0 + (volume_ratio - 1.0) * self.adaptive_factor

            for s in order.slices[current_slice + 1:]:
                s.target_quantity = int(s.target_quantity * adjustment)

        return order.slices

    def get_slice_quantity(
        self,
        slice_: ExecutionSlice,
        market: MarketSnapshot,
    ) -> int:
        """Get quantity with volume-adaptive adjustments."""
        return slice_.remaining


class SweepAlgorithm(ExecutionAlgorithm):
    """
    Liquidity Sweep Algorithm.

    Aggressively sweeps liquidity across multiple venues to:
    - Capture immediate liquidity before it moves
    - Minimize information leakage from sequential execution
    - Execute large orders quickly

    Best for urgent orders where speed > price.
    """

    def __init__(
        self,
        max_venues: int = 5,
        min_fill_pct: float = 0.8,
    ):
        super().__init__("sweep")
        self.max_venues = max_venues
        self.min_fill_pct = min_fill_pct

    def create_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
    ) -> List[ExecutionSlice]:
        """Create single-slice sweep schedule."""
        # Sweep executes immediately in one slice
        return [ExecutionSlice(
            slice_id=0,
            target_quantity=order.total_quantity,
            target_start_time=order.start_time,
            target_end_time=order.start_time + timedelta(seconds=1),
        )]

    def adjust_schedule(
        self,
        order: AlgoOrder,
        market: MarketSnapshot,
        current_slice: int,
    ) -> List[ExecutionSlice]:
        """No adjustment for sweep - execute immediately."""
        return order.slices

    def get_slice_quantity(
        self,
        slice_: ExecutionSlice,
        market: MarketSnapshot,
    ) -> int:
        """Execute full quantity."""
        return slice_.remaining


class AlgorithmicExecutor:
    """
    Orchestrates execution of algorithmic orders.

    Features:
    - Algorithm selection and management
    - Real-time schedule adjustment
    - Performance tracking
    - Risk controls
    """

    def __init__(
        self,
        order_sender: Optional[Callable] = None,
        market_data_fn: Optional[Callable] = None,
    ):
        self.order_sender = order_sender
        self.market_data_fn = market_data_fn

        self._algorithms: Dict[str, ExecutionAlgorithm] = {
            "is": ImplementationShortfall(),
            "pov": POVAlgorithm(),
            "twap": AdaptiveTWAP(),
            "vwap": AdaptiveVWAP(),
            "sweep": SweepAlgorithm(),
        }

        self._active_orders: Dict[str, AlgoOrder] = {}
        self._completed_orders: List[AlgoOrder] = []
        self._running = False

    def add_algorithm(self, name: str, algo: ExecutionAlgorithm) -> None:
        """Register an execution algorithm."""
        self._algorithms[name] = algo

    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        algo_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> AlgoOrder:
        """Submit an algorithmic order."""
        if algo_type not in self._algorithms:
            raise ValueError(f"Unknown algorithm: {algo_type}")

        start_time = start_time or datetime.now()
        end_time = end_time or (start_time + timedelta(minutes=30))

        algo_id = f"{algo_type}_{symbol}_{datetime.now().timestamp()}"

        order = AlgoOrder(
            algo_id=algo_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algo_type=algo_type,
            start_time=start_time,
            end_time=end_time,
            params=params or {},
        )

        # Get arrival price
        if self.market_data_fn:
            market = await self.market_data_fn(symbol)
            order.arrival_price = market.mid
            order.decision_price = market.mid

            # Create schedule
            algo = self._algorithms[algo_type]
            order.slices = algo.create_schedule(order, market)

        self._active_orders[algo_id] = order
        order.state = AlgoState.RUNNING

        logger.info(f"Started algo order: {algo_id} for {quantity} {symbol}")
        return order

    async def run_order(self, order: AlgoOrder) -> AlgoMetrics:
        """Execute an algo order (typically run in background)."""
        algo = self._algorithms[order.algo_type]
        current_slice = 0

        while order.state == AlgoState.RUNNING and order.remaining > 0:
            # Check if order window has ended
            if datetime.now() >= order.end_time:
                order.state = AlgoState.COMPLETED
                break

            # Get current slice
            if current_slice >= len(order.slices):
                order.state = AlgoState.COMPLETED
                break

            slice_ = order.slices[current_slice]

            # Wait for slice start time
            if datetime.now() < slice_.target_start_time:
                await asyncio.sleep(0.1)
                continue

            # Get market data
            if self.market_data_fn:
                market = await self.market_data_fn(order.symbol)

                # Adjust schedule
                order.slices = algo.adjust_schedule(order, market, current_slice)

                # Get quantity for this slice
                qty = algo.get_slice_quantity(slice_, market)

                # Send order (if sender configured)
                if self.order_sender and qty > 0:
                    try:
                        result = await self.order_sender(
                            symbol=order.symbol,
                            side=order.side,
                            quantity=qty,
                        )
                        # Update fills (simplified)
                        slice_.filled_quantity += qty
                        order.filled_quantity += qty

                    except Exception as e:
                        logger.error(f"Order failed: {e}")

            # Check if slice is complete
            if datetime.now() >= slice_.target_end_time or slice_.remaining <= 0:
                slice_.status = "completed"
                current_slice += 1

            await asyncio.sleep(0.1)

        # Calculate final metrics
        metrics = self._calculate_metrics(order)
        self._completed_orders.append(order)
        del self._active_orders[order.algo_id]

        return metrics

    def _calculate_metrics(self, order: AlgoOrder) -> AlgoMetrics:
        """Calculate execution metrics."""
        # Calculate VWAP from slices
        if order.filled_quantity > 0 and order.slices:
            weighted_price = sum(
                s.avg_fill_price * s.filled_quantity
                for s in order.slices if s.filled_quantity > 0
            )
            vwap = weighted_price / order.filled_quantity
        else:
            vwap = order.arrival_price

        # Implementation shortfall
        is_bps = order.slippage_bps

        # VWAP deviation
        if vwap > 0 and order.avg_fill_price > 0:
            if order.side == "buy":
                vs_vwap = (order.avg_fill_price - vwap) / vwap * 10000
            else:
                vs_vwap = (vwap - order.avg_fill_price) / vwap * 10000
        else:
            vs_vwap = 0

        # Participation rate
        total_market_vol = sum(s.market_volume for s in order.slices)
        if total_market_vol > 0:
            participation = order.filled_quantity / total_market_vol
        else:
            participation = 0

        return AlgoMetrics(
            algo_id=order.algo_id,
            symbol=order.symbol,
            total_quantity=order.total_quantity,
            filled_quantity=order.filled_quantity,
            avg_fill_price=order.avg_fill_price,
            arrival_price=order.arrival_price,
            vwap_price=vwap,
            implementation_shortfall_bps=is_bps,
            vs_vwap_bps=vs_vwap,
            realized_impact_bps=is_bps,  # Simplified
            temporary_impact_bps=is_bps * 0.7,
            permanent_impact_bps=is_bps * 0.3,
            duration_seconds=int((datetime.now() - order.start_time).total_seconds()),
            participation_rate=participation,
        )

    def cancel_order(self, algo_id: str) -> bool:
        """Cancel an active algo order."""
        if algo_id in self._active_orders:
            self._active_orders[algo_id].state = AlgoState.CANCELLED
            return True
        return False

    def get_active_orders(self) -> Dict[str, AlgoOrder]:
        """Get all active orders."""
        return self._active_orders.copy()

    def get_order_status(self, algo_id: str) -> Optional[AlgoOrder]:
        """Get status of an order."""
        return self._active_orders.get(algo_id)


def create_algo_executor(
    algo_type: str = "is",
    **algo_params,
) -> Tuple[AlgorithmicExecutor, ExecutionAlgorithm]:
    """
    Factory function to create executor with specific algorithm.

    Args:
        algo_type: Algorithm type ('is', 'pov', 'twap', 'vwap', 'sweep')
        **algo_params: Parameters for the algorithm

    Returns:
        Tuple of (executor, algorithm)
    """
    algo_classes = {
        "is": ImplementationShortfall,
        "pov": POVAlgorithm,
        "twap": AdaptiveTWAP,
        "vwap": AdaptiveVWAP,
        "sweep": SweepAlgorithm,
    }

    if algo_type not in algo_classes:
        raise ValueError(f"Unknown algorithm: {algo_type}")

    algo = algo_classes[algo_type](**algo_params)
    executor = AlgorithmicExecutor()
    executor.add_algorithm(algo_type, algo)

    return executor, algo


def print_algo_metrics(metrics: AlgoMetrics) -> None:
    """Print formatted algo execution metrics."""
    print("\n" + "=" * 60)
    print(f"ALGORITHM EXECUTION REPORT: {metrics.algo_id}")
    print("=" * 60)

    print(f"\n{'Execution Summary':-^40}")
    print(f"  Symbol: {metrics.symbol}")
    print(f"  Filled: {metrics.filled_quantity:,} / {metrics.total_quantity:,}")
    print(f"  Avg Price: ${metrics.avg_fill_price:.4f}")
    print(f"  Duration: {metrics.duration_seconds}s")

    print(f"\n{'Performance vs Benchmarks':-^40}")
    print(f"  Arrival Price: ${metrics.arrival_price:.4f}")
    print(f"  VWAP: ${metrics.vwap_price:.4f}")
    print(f"  Implementation Shortfall: {metrics.implementation_shortfall_bps:.2f} bps")
    print(f"  vs VWAP: {metrics.vs_vwap_bps:.2f} bps")

    print(f"\n{'Market Impact':-^40}")
    print(f"  Realized Impact: {metrics.realized_impact_bps:.2f} bps")
    print(f"  Temporary: {metrics.temporary_impact_bps:.2f} bps")
    print(f"  Permanent: {metrics.permanent_impact_bps:.2f} bps")
    print(f"  Participation Rate: {metrics.participation_rate:.2%}")

    print("=" * 60 + "\n")
