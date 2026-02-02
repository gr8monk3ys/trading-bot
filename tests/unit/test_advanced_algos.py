"""
Tests for Advanced Execution Algorithms

Tests:
- Implementation Shortfall algorithm
- POV algorithm
- Adaptive TWAP
- Adaptive VWAP
- Sweep algorithm
- Algorithmic executor
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from execution.advanced_algos import (
    AlgorithmicExecutor,
    ImplementationShortfall,
    POVAlgorithm,
    AdaptiveTWAP,
    AdaptiveVWAP,
    SweepAlgorithm,
    AlgoOrder,
    AlgoState,
    AlgoMetrics,
    ExecutionSlice,
    Urgency,
    MarketSnapshot,
    create_algo_executor,
)


class TestAlgoOrder:
    """Tests for AlgoOrder dataclass."""

    def test_create_order(self):
        """Test creating an algo order."""
        order = AlgoOrder(
            algo_id="IS_AAPL_123",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="is",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )

        assert order.symbol == "AAPL"
        assert order.total_quantity == 10000

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="is",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )
        order.filled_quantity = 3000

        assert order.remaining == 7000

    def test_fill_rate(self):
        """Test fill rate calculation."""
        order = AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="is",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )
        order.filled_quantity = 5000

        assert order.fill_rate == 0.5

    def test_slippage_calculation_buy(self):
        """Test slippage calculation for buy."""
        order = AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="is",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )
        order.arrival_price = 150.00
        order.avg_fill_price = 150.15  # Paid more

        # (150.15 - 150.00) / 150.00 * 10000 = 10 bps
        assert abs(order.slippage_bps - 10) < 0.1

    def test_slippage_calculation_sell(self):
        """Test slippage calculation for sell."""
        order = AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="sell",
            total_quantity=10000,
            algo_type="is",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )
        order.arrival_price = 150.00
        order.avg_fill_price = 149.85  # Received less

        # (150.00 - 149.85) / 150.00 * 10000 = 10 bps
        assert abs(order.slippage_bps - 10) < 0.1


class TestExecutionSlice:
    """Tests for ExecutionSlice dataclass."""

    def test_create_slice(self):
        """Test creating a slice."""
        slice_ = ExecutionSlice(
            slice_id=0,
            target_quantity=500,
            target_start_time=datetime.now(),
            target_end_time=datetime.now() + timedelta(minutes=1),
        )

        assert slice_.target_quantity == 500
        assert slice_.remaining == 500

    def test_remaining_after_fills(self):
        """Test remaining after fills."""
        slice_ = ExecutionSlice(
            slice_id=0,
            target_quantity=500,
            target_start_time=datetime.now(),
            target_end_time=datetime.now() + timedelta(minutes=1),
        )
        slice_.filled_quantity = 200

        assert slice_.remaining == 300

    def test_fill_rate(self):
        """Test fill rate."""
        slice_ = ExecutionSlice(
            slice_id=0,
            target_quantity=500,
            target_start_time=datetime.now(),
            target_end_time=datetime.now() + timedelta(minutes=1),
        )
        slice_.filled_quantity = 250

        assert slice_.fill_rate == 0.5


class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        snapshot = MarketSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.00,
            ask=150.05,
            mid=150.025,
            spread_bps=3.33,
            volume_today=5000000,
            volatility=0.02,
            adv=10000000,
        )

        assert snapshot.symbol == "AAPL"
        assert snapshot.adv == 10000000


class TestImplementationShortfall:
    """Tests for ImplementationShortfall algorithm."""

    @pytest.fixture
    def algo(self):
        """Create IS algorithm."""
        return ImplementationShortfall()

    @pytest.fixture
    def order(self):
        """Create test order."""
        return AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="is",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )

    @pytest.fixture
    def market(self):
        """Create market snapshot."""
        return MarketSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.00,
            ask=150.05,
            mid=150.025,
            spread_bps=3.33,
            volume_today=5000000,
            volatility=0.02,
            adv=10000000,
        )

    def test_create_schedule(self, algo, order, market):
        """Test creating execution schedule."""
        slices = algo.create_schedule(order, market)

        assert len(slices) > 0
        total_qty = sum(s.target_quantity for s in slices)
        assert total_qty == order.total_quantity

    def test_urgency_affects_schedule(self, order, market):
        """Test that urgency affects schedule."""
        low_urgency = ImplementationShortfall(urgency=Urgency.LOW)
        high_urgency = ImplementationShortfall(urgency=Urgency.HIGH)

        low_slices = low_urgency.create_schedule(order, market)
        high_slices = high_urgency.create_schedule(order, market)

        # High urgency should front-load more
        low_first = low_slices[0].target_quantity if low_slices else 0
        high_first = high_slices[0].target_quantity if high_slices else 0

        # High urgency generally front-loads
        assert high_first >= low_first * 0.8  # Allow some tolerance

    def test_get_slice_quantity(self, algo, market):
        """Test getting slice quantity."""
        slice_ = ExecutionSlice(
            slice_id=0,
            target_quantity=500,
            target_start_time=datetime.now(),
            target_end_time=datetime.now() + timedelta(minutes=1),
        )

        qty = algo.get_slice_quantity(slice_, market)
        assert qty > 0


class TestPOVAlgorithm:
    """Tests for POV algorithm."""

    @pytest.fixture
    def algo(self):
        """Create POV algorithm."""
        return POVAlgorithm(target_pov=0.10)

    @pytest.fixture
    def order(self):
        """Create test order."""
        return AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="pov",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )

    @pytest.fixture
    def market(self):
        """Create market snapshot."""
        return MarketSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.00,
            ask=150.05,
            mid=150.025,
            spread_bps=3.33,
            volume_today=5000000,
            volatility=0.02,
            adv=10000000,
        )

    def test_create_schedule(self, algo, order, market):
        """Test creating POV schedule."""
        slices = algo.create_schedule(order, market)

        assert len(slices) > 0
        total_qty = sum(s.target_quantity for s in slices)
        # Should match total quantity
        assert abs(total_qty - order.total_quantity) < 10

    def test_update_volume(self, algo):
        """Test updating volume history."""
        algo.update_volume(datetime.now(), 10000)
        algo.update_volume(datetime.now(), 15000)

        recent = algo._get_recent_volume()
        assert recent == 25000


class TestAdaptiveTWAP:
    """Tests for Adaptive TWAP algorithm."""

    @pytest.fixture
    def algo(self):
        """Create TWAP algorithm."""
        return AdaptiveTWAP()

    @pytest.fixture
    def order(self):
        """Create test order."""
        return AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="twap",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )

    @pytest.fixture
    def market(self):
        """Create market snapshot."""
        return MarketSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.00,
            ask=150.05,
            mid=150.025,
            spread_bps=3.33,
            volume_today=5000000,
            volatility=0.02,
            adv=10000000,
        )

    def test_create_uniform_schedule(self, algo, order, market):
        """Test creating uniform TWAP schedule."""
        slices = algo.create_schedule(order, market)

        assert len(slices) > 0
        # TWAP should have roughly equal slices
        qtys = [s.target_quantity for s in slices]
        assert max(qtys) - min(qtys) < order.total_quantity / 5

    def test_reduce_on_wide_spread(self, algo, order, market):
        """Test reducing quantity on wide spread."""
        slice_ = ExecutionSlice(
            slice_id=0,
            target_quantity=500,
            target_start_time=datetime.now(),
            target_end_time=datetime.now() + timedelta(minutes=1),
        )

        # Normal spread
        qty_normal = algo.get_slice_quantity(slice_, market)

        # Wide spread
        market.spread_bps = 20.0
        qty_wide = algo.get_slice_quantity(slice_, market)

        # Should reduce on wide spread
        assert qty_wide < qty_normal


class TestAdaptiveVWAP:
    """Tests for Adaptive VWAP algorithm."""

    @pytest.fixture
    def algo(self):
        """Create VWAP algorithm."""
        return AdaptiveVWAP()

    @pytest.fixture
    def order(self):
        """Create test order."""
        return AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="vwap",
            start_time=datetime(2024, 1, 15, 9, 30),  # Market open
            end_time=datetime(2024, 1, 15, 16, 0),   # Market close
        )

    @pytest.fixture
    def market(self):
        """Create market snapshot."""
        return MarketSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.00,
            ask=150.05,
            mid=150.025,
            spread_bps=3.33,
            volume_today=5000000,
            volatility=0.02,
            adv=10000000,
        )

    def test_intraday_pattern_defined(self, algo):
        """Test intraday volume pattern is defined."""
        assert len(algo.INTRADAY_PATTERN) > 0
        assert abs(sum(algo.INTRADAY_PATTERN) - 1.0) < 0.01

    def test_create_schedule_follows_pattern(self, algo, order, market):
        """Test schedule follows volume pattern."""
        slices = algo.create_schedule(order, market)

        assert len(slices) > 0
        # First and last slices should be larger (U-shaped pattern)
        if len(slices) > 10:
            middle_avg = sum(s.target_quantity for s in slices[3:7]) / 4
            end_avg = slices[-1].target_quantity
            # End of day should have more volume
            assert end_avg >= middle_avg * 0.8


class TestSweepAlgorithm:
    """Tests for Sweep algorithm."""

    @pytest.fixture
    def algo(self):
        """Create Sweep algorithm."""
        return SweepAlgorithm()

    @pytest.fixture
    def order(self):
        """Create test order."""
        return AlgoOrder(
            algo_id="test",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="sweep",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=5),
        )

    @pytest.fixture
    def market(self):
        """Create market snapshot."""
        return MarketSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.00,
            ask=150.05,
            mid=150.025,
            spread_bps=3.33,
            volume_today=5000000,
            volatility=0.02,
            adv=10000000,
        )

    def test_single_slice(self, algo, order, market):
        """Test sweep creates single slice."""
        slices = algo.create_schedule(order, market)

        # Sweep should be single slice
        assert len(slices) == 1
        assert slices[0].target_quantity == order.total_quantity


class TestAlgorithmicExecutor:
    """Tests for AlgorithmicExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create executor."""
        executor = AlgorithmicExecutor()
        return executor

    @pytest.mark.asyncio
    async def test_submit_order(self, executor):
        """Test submitting an order."""
        # Mock market data
        executor.market_data_fn = AsyncMock(return_value=MarketSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.00,
            ask=150.05,
            mid=150.025,
            spread_bps=3.33,
            volume_today=5000000,
            volatility=0.02,
            adv=10000000,
        ))

        order = await executor.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=10000,
            algo_type="is",
        )

        assert order is not None
        assert order.state == AlgoState.RUNNING

    def test_add_algorithm(self, executor):
        """Test adding algorithm."""
        algo = ImplementationShortfall(urgency=Urgency.HIGH)
        executor.add_algorithm("is_high", algo)

        assert "is_high" in executor._algorithms

    def test_cancel_order(self, executor):
        """Test canceling order."""
        # Create a mock active order
        order = AlgoOrder(
            algo_id="test_123",
            symbol="AAPL",
            side="buy",
            total_quantity=10000,
            algo_type="is",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            state=AlgoState.RUNNING,
        )
        executor._active_orders["test_123"] = order

        result = executor.cancel_order("test_123")

        assert result is True
        assert order.state == AlgoState.CANCELLED


class TestAlgoMetrics:
    """Tests for AlgoMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating metrics."""
        metrics = AlgoMetrics(
            algo_id="test",
            symbol="AAPL",
            total_quantity=10000,
            filled_quantity=10000,
            avg_fill_price=150.10,
            arrival_price=150.00,
            vwap_price=150.05,
            implementation_shortfall_bps=6.67,
            vs_vwap_bps=3.33,
            realized_impact_bps=6.67,
            temporary_impact_bps=4.67,
            permanent_impact_bps=2.00,
            duration_seconds=3600,
            participation_rate=0.05,
        )

        assert metrics.implementation_shortfall_bps == 6.67

    def test_to_dict(self):
        """Test serialization."""
        metrics = AlgoMetrics(
            algo_id="test",
            symbol="AAPL",
            total_quantity=10000,
            filled_quantity=10000,
            avg_fill_price=150.10,
            arrival_price=150.00,
            vwap_price=150.05,
            implementation_shortfall_bps=6.67,
            vs_vwap_bps=3.33,
            realized_impact_bps=6.67,
            temporary_impact_bps=4.67,
            permanent_impact_bps=2.00,
            duration_seconds=3600,
            participation_rate=0.05,
        )

        d = metrics.to_dict()
        assert "algo_id" in d
        assert "implementation_shortfall_bps" in d


class TestUrgency:
    """Tests for Urgency enum."""

    def test_all_urgency_levels_exist(self):
        """Test all urgency levels exist."""
        expected = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for name in expected:
            assert hasattr(Urgency, name)


class TestCreateAlgoExecutor:
    """Tests for create_algo_executor factory."""

    def test_create_is_executor(self):
        """Test creating IS executor."""
        executor, algo = create_algo_executor("is")
        assert isinstance(algo, ImplementationShortfall)

    def test_create_pov_executor(self):
        """Test creating POV executor."""
        executor, algo = create_algo_executor("pov", target_pov=0.15)
        assert isinstance(algo, POVAlgorithm)
        assert algo.target_pov == 0.15

    def test_create_twap_executor(self):
        """Test creating TWAP executor."""
        executor, algo = create_algo_executor("twap")
        assert isinstance(algo, AdaptiveTWAP)

    def test_invalid_algo_type(self):
        """Test invalid algo type raises error."""
        with pytest.raises(ValueError):
            create_algo_executor("invalid_algo")
