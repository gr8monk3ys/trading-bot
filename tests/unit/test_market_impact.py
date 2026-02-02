"""
Tests for Market Impact Model (Almgren-Chriss)

Tests:
- Impact calculation with different order sizes
- Volume-based scaling
- Market cap tier selection
- Time-of-day adjustments
- Volume curve model
- Execution cost tracking
"""

import numpy as np
import pytest
from datetime import datetime, time
from unittest.mock import MagicMock

from utils.market_impact import (
    AlmgrenChrissModel,
    MarketImpactResult,
    MarketCondition,
    VolumeCurveModel,
    ExecutionCostTracker,
)


class TestAlmgrenChrissModel:
    """Tests for AlmgrenChrissModel class."""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return AlmgrenChrissModel()

    def test_small_order_low_impact(self, model):
        """Test that small orders have low market impact."""
        result = model.calculate_impact(
            order_value=10_000,           # $10K order
            daily_volume_usd=100_000_000, # $100M daily volume
            price=100.0,
            volatility=0.25,
        )

        assert result is not None
        assert isinstance(result, MarketImpactResult)
        # Small order in liquid stock should have < 10 bps impact
        assert result.total_cost_bps < 10

    def test_large_order_high_impact(self, model):
        """Test that large orders have higher market impact."""
        result = model.calculate_impact(
            order_value=1_000_000,      # $1M order
            daily_volume_usd=5_000_000, # $5M daily volume (20% participation)
            price=100.0,
            volatility=0.30,
        )

        assert result is not None
        # Large order in illiquid stock should have significant impact
        assert result.total_cost_bps > 50
        assert result.participation_rate == pytest.approx(0.20, rel=0.01)

    def test_impact_scales_with_participation(self, model):
        """Test that impact increases with participation rate."""
        results = []
        for participation in [0.01, 0.05, 0.10, 0.20]:
            order_value = 100_000_000 * participation  # Varying order size
            result = model.calculate_impact(
                order_value=order_value,
                daily_volume_usd=100_000_000,
                price=100.0,
                volatility=0.25,
            )
            results.append(result.total_cost_bps)

        # Impact should be monotonically increasing
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_impact_scales_with_volatility(self, model):
        """Test that impact increases with volatility."""
        low_vol = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            price=100.0,
            volatility=0.15,  # Low vol
        )

        high_vol = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            price=100.0,
            volatility=0.50,  # High vol
        )

        assert high_vol.total_cost_bps > low_vol.total_cost_bps

    def test_market_cap_tier_selection(self, model):
        """Test that appropriate parameters are selected by market cap."""
        # Large cap (should use LARGE_CAP_PARAMS)
        large_cap = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=500_000_000,  # High volume indicates large cap
            price=100.0,
            volatility=0.25,
            market_cap=50_000_000_000,  # $50B market cap
        )

        # Small cap (should use SMALL_CAP_PARAMS)
        small_cap = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=1_000_000,  # Low volume
            price=100.0,
            volatility=0.25,
            market_cap=500_000_000,  # $500M market cap
        )

        # Same order size should have more impact in small cap
        assert small_cap.total_cost_bps > large_cap.total_cost_bps

    def test_time_adjustments(self, model):
        """Test that time-of-day adjustments affect impact."""
        normal = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            price=100.0,
            volatility=0.25,
            condition=MarketCondition.NORMAL,
        )

        market_open = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            price=100.0,
            volatility=0.25,
            condition=MarketCondition.MARKET_OPEN,
        )

        # Market open should have higher impact
        assert market_open.total_cost_bps > normal.total_cost_bps

    def test_impact_components_positive(self, model):
        """Test that all impact components are non-negative."""
        result = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            price=100.0,
            volatility=0.25,
        )

        assert result.permanent_impact_bps >= 0
        assert result.temporary_impact_bps >= 0
        assert result.spread_cost_bps >= 0
        assert result.total_cost_bps >= 0

    def test_effective_price_buy(self, model):
        """Test effective price calculation for buy orders."""
        result = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            price=100.0,
            volatility=0.25,
            side="buy",
        )

        # Buy price should be higher than market price
        assert result.effective_price > 100.0

    def test_effective_price_sell(self, model):
        """Test effective price calculation for sell orders."""
        result = model.calculate_impact(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            price=100.0,
            volatility=0.25,
            side="sell",
        )

        # Sell price should be lower than market price
        assert result.effective_price < 100.0

    def test_impact_bounds(self, model):
        """Test that impact is clamped to min/max bounds."""
        # Very small order
        tiny = model.calculate_impact(
            order_value=100,
            daily_volume_usd=1_000_000_000,
            price=100.0,
            volatility=0.10,
        )
        assert tiny.total_cost_bps >= model.min_impact_bps

        # Very large order
        huge = model.calculate_impact(
            order_value=10_000_000,
            daily_volume_usd=1_000_000,
            price=100.0,
            volatility=1.0,
        )
        assert huge.total_cost_bps <= model.max_impact_bps

    def test_optimal_execution_time(self, model):
        """Test optimal execution time estimation."""
        # Small order should have shorter optimal time
        small_time = model.estimate_optimal_execution_time(
            order_value=10_000,
            daily_volume_usd=100_000_000,
            volatility=0.25,
        )

        # Large order should have longer optimal time
        large_time = model.estimate_optimal_execution_time(
            order_value=1_000_000,
            daily_volume_usd=10_000_000,
            volatility=0.25,
        )

        assert large_time > small_time

    def test_urgency_affects_execution_time(self, model):
        """Test that urgency reduces optimal execution time."""
        patient = model.estimate_optimal_execution_time(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            volatility=0.25,
            urgency=0.1,  # Patient
        )

        urgent = model.estimate_optimal_execution_time(
            order_value=100_000,
            daily_volume_usd=10_000_000,
            volatility=0.25,
            urgency=0.9,  # Urgent
        )

        assert urgent < patient


class TestVolumeCurveModel:
    """Tests for VolumeCurveModel class."""

    def test_volume_profile_sums_to_one(self):
        """Test that volume profile sums to approximately 1."""
        total = sum(VolumeCurveModel.VOLUME_PROFILE.values())
        assert 0.99 <= total <= 1.01

    def test_get_volume_fraction_full_day(self):
        """Test volume fraction for full trading day."""
        fraction = VolumeCurveModel.get_volume_fraction(
            time(9, 30),
            time(16, 0),
        )
        assert 0.99 <= fraction <= 1.01

    def test_get_volume_fraction_morning(self):
        """Test volume fraction for morning session."""
        fraction = VolumeCurveModel.get_volume_fraction(
            time(9, 30),
            time(12, 0),
        )
        # Morning (first 2.5 hours) should be significant fraction
        assert 0.30 <= fraction <= 0.50

    def test_get_volume_fraction_closing(self):
        """Test volume fraction for closing hour."""
        fraction = VolumeCurveModel.get_volume_fraction(
            time(15, 0),
            time(16, 0),
        )
        # Last hour typically has 25-30% of volume
        assert 0.20 <= fraction <= 0.35

    def test_get_volume_fraction_outside_hours(self):
        """Test volume fraction returns 0 outside market hours."""
        fraction = VolumeCurveModel.get_volume_fraction(
            time(8, 0),  # Before market
            time(9, 0),
        )
        assert fraction == 0.0

    def test_get_market_condition_open(self):
        """Test market condition detection at open."""
        condition = VolumeCurveModel.get_market_condition(time(9, 45))
        assert condition == MarketCondition.MARKET_OPEN

    def test_get_market_condition_close(self):
        """Test market condition detection at close."""
        condition = VolumeCurveModel.get_market_condition(time(15, 45))
        assert condition == MarketCondition.MARKET_CLOSE

    def test_get_market_condition_normal(self):
        """Test market condition detection during normal hours."""
        condition = VolumeCurveModel.get_market_condition(time(12, 0))
        assert condition == MarketCondition.NORMAL


class TestExecutionCostTracker:
    """Tests for ExecutionCostTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker instance."""
        return ExecutionCostTracker()

    @pytest.fixture
    def sample_impact(self):
        """Create a sample impact result."""
        return MarketImpactResult(
            permanent_impact_bps=5.0,
            temporary_impact_bps=10.0,
            spread_cost_bps=2.0,
            total_cost_bps=17.0,
            permanent_impact_usd=50.0,
            temporary_impact_usd=100.0,
            spread_cost_usd=20.0,
            total_cost_usd=170.0,
            order_value=100_000,
            participation_rate=0.05,
            execution_time_hours=1.0,
            daily_volume_usd=2_000_000,
            volatility=0.25,
            condition=MarketCondition.NORMAL,
            effective_price=100.17,
            slippage_pct=0.0017,
            timestamp=datetime.now(),
        )

    def test_record_execution(self, tracker, sample_impact):
        """Test recording an execution."""
        tracker.record_execution(
            symbol="AAPL",
            order_value=100_000,
            predicted_impact=sample_impact,
            actual_slippage_pct=0.0020,  # 20 bps actual
            fill_price=100.20,
            arrival_price=100.00,
        )

        assert len(tracker.execution_history) == 1

    def test_execution_quality_report(self, tracker, sample_impact):
        """Test execution quality report generation."""
        # Record multiple executions
        for i in range(10):
            actual_slip = 0.0015 + np.random.randn() * 0.0005
            tracker.record_execution(
                symbol=f"SYM{i}",
                order_value=100_000,
                predicted_impact=sample_impact,
                actual_slippage_pct=actual_slip,
                fill_price=100.0 * (1 + actual_slip),
                arrival_price=100.0,
            )

        report = tracker.get_execution_quality_report()

        assert "total_executions" in report
        assert report["total_executions"] == 10
        assert "predicted_cost" in report
        assert "actual_cost" in report
        assert "prediction_accuracy" in report

    def test_vwap_tracking(self, tracker, sample_impact):
        """Test VWAP performance tracking."""
        tracker.record_execution(
            symbol="AAPL",
            order_value=100_000,
            predicted_impact=sample_impact,
            actual_slippage_pct=0.0020,
            fill_price=100.20,
            arrival_price=100.00,
            vwap=100.15,  # We beat VWAP
        )

        report = tracker.get_execution_quality_report()

        assert "vwap_performance" in report
        assert report["vwap_performance"]["beat_vwap_pct"] == 0.0  # We didn't beat VWAP

    def test_model_calibration_adjustments(self, tracker, sample_impact):
        """Test model calibration calculation."""
        # Record many executions with consistent bias
        for i in range(30):
            tracker.record_execution(
                symbol=f"SYM{i}",
                order_value=100_000,
                predicted_impact=sample_impact,
                actual_slippage_pct=0.0025,  # Actual higher than predicted
                fill_price=100.25,
                arrival_price=100.0,
            )

        adjustments = tracker.get_model_calibration_adjustments()

        assert "gamma_mult" in adjustments
        assert "eta_mult" in adjustments
        # Model should suggest increasing multipliers since actual > predicted
        assert adjustments["gamma_mult"] > 1.0

    def test_empty_tracker_report(self, tracker):
        """Test report with no executions."""
        report = tracker.get_execution_quality_report()
        assert "error" in report


class TestMarketImpactResultDataclass:
    """Tests for MarketImpactResult dataclass."""

    def test_repr(self):
        """Test string representation."""
        result = MarketImpactResult(
            permanent_impact_bps=5.0,
            temporary_impact_bps=10.0,
            spread_cost_bps=2.0,
            total_cost_bps=17.0,
            permanent_impact_usd=50.0,
            temporary_impact_usd=100.0,
            spread_cost_usd=20.0,
            total_cost_usd=170.0,
            order_value=100_000,
            participation_rate=0.05,
            execution_time_hours=1.0,
            daily_volume_usd=2_000_000,
            volatility=0.25,
            condition=MarketCondition.NORMAL,
            effective_price=100.17,
            slippage_pct=0.0017,
            timestamp=datetime.now(),
        )

        repr_str = repr(result)
        assert "17.0bps" in repr_str
        assert "perm=5.0bps" in repr_str
        assert "participation=5.0%" in repr_str
