#!/usr/bin/env python3
"""
Unit tests for utils/position_scaling.py

Tests PositionScaler class for:
- Tranche weight calculations
- Scale-in plan creation
- Scale-out plan creation
- Pyramid recommendations
- Plan management
"""

from datetime import datetime, timedelta

import pytest

from utils.position_scaling import PositionScaler, ScaleMethod

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def scaler():
    """Default position scaler."""
    return PositionScaler()


@pytest.fixture
def scaler_custom():
    """Custom position scaler with different settings."""
    return PositionScaler(
        default_tranches=4,
        scale_in_method=ScaleMethod.EQUAL,
        scale_out_method=ScaleMethod.PYRAMID,
        min_profit_for_scale_out=0.05,
        scale_out_levels=[0.10, 0.20, 0.30],
    )


# ============================================================================
# ScaleMethod Enum Tests
# ============================================================================


class TestScaleMethod:
    """Test ScaleMethod enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert ScaleMethod.EQUAL.value == "equal"
        assert ScaleMethod.PYRAMID.value == "pyramid"
        assert ScaleMethod.INVERTED.value == "inverted"
        assert ScaleMethod.AGGRESSIVE.value == "aggressive"

    def test_enum_count(self):
        """Test number of scale methods."""
        methods = list(ScaleMethod)
        assert len(methods) == 4


# ============================================================================
# PositionScaler Initialization Tests
# ============================================================================


class TestPositionScalerInit:
    """Test PositionScaler initialization."""

    def test_default_init(self, scaler):
        """Test default initialization values."""
        assert scaler.default_tranches == 3
        assert scaler.scale_in_method == ScaleMethod.PYRAMID
        assert scaler.scale_out_method == ScaleMethod.EQUAL
        assert scaler.min_profit_for_scale_out == 0.03
        assert scaler.scale_out_levels == [0.05, 0.10, 0.20]
        assert scaler.active_plans == {}

    def test_custom_init(self, scaler_custom):
        """Test custom initialization values."""
        assert scaler_custom.default_tranches == 4
        assert scaler_custom.scale_in_method == ScaleMethod.EQUAL
        assert scaler_custom.scale_out_method == ScaleMethod.PYRAMID
        assert scaler_custom.min_profit_for_scale_out == 0.05
        assert scaler_custom.scale_out_levels == [0.10, 0.20, 0.30]


# ============================================================================
# Tranche Weights Tests
# ============================================================================


class TestGetTrancheWeights:
    """Test get_tranche_weights method."""

    def test_equal_weights_2_tranches(self, scaler):
        """Test equal weights with 2 tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.EQUAL, 2)
        assert len(weights) == 2
        assert weights == [0.5, 0.5]
        assert sum(weights) == pytest.approx(1.0)

    def test_equal_weights_3_tranches(self, scaler):
        """Test equal weights with 3 tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.EQUAL, 3)
        assert len(weights) == 3
        assert all(abs(w - 1/3) < 0.01 for w in weights)
        assert sum(weights) == pytest.approx(1.0)

    def test_pyramid_weights_2_tranches(self, scaler):
        """Test pyramid weights with 2 tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.PYRAMID, 2)
        assert weights == [0.6, 0.4]
        assert weights[0] > weights[1]  # First larger

    def test_pyramid_weights_3_tranches(self, scaler):
        """Test pyramid weights with 3 tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.PYRAMID, 3)
        assert weights == [0.5, 0.3, 0.2]
        assert weights[0] > weights[1] > weights[2]  # Decreasing

    def test_pyramid_weights_4_tranches(self, scaler):
        """Test pyramid weights with 4 tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.PYRAMID, 4)
        assert weights == [0.4, 0.3, 0.2, 0.1]

    def test_pyramid_weights_5_tranches(self, scaler):
        """Test pyramid weights with 5+ tranches uses formula."""
        weights = scaler.get_tranche_weights(ScaleMethod.PYRAMID, 5)
        assert len(weights) == 5
        assert sum(weights) == pytest.approx(1.0)
        # Should be decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_inverted_weights(self, scaler):
        """Test inverted weights (reversed pyramid)."""
        weights = scaler.get_tranche_weights(ScaleMethod.INVERTED, 3)
        assert weights == [0.2, 0.3, 0.5]
        assert weights[0] < weights[1] < weights[2]  # Increasing

    def test_aggressive_weights_1_tranche(self, scaler):
        """Test aggressive weights with 1 tranche."""
        weights = scaler.get_tranche_weights(ScaleMethod.AGGRESSIVE, 1)
        assert weights == [1.0]

    def test_aggressive_weights_2_tranches(self, scaler):
        """Test aggressive weights with 2 tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.AGGRESSIVE, 2)
        assert weights[0] == pytest.approx(0.7)
        assert weights[1] == pytest.approx(0.3)

    def test_aggressive_weights_3_tranches(self, scaler):
        """Test aggressive weights with 3 tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.AGGRESSIVE, 3)
        assert weights[0] == pytest.approx(0.7)
        assert weights[1] == pytest.approx(0.15)
        assert weights[2] == pytest.approx(0.15)

    def test_weights_always_sum_to_one(self, scaler):
        """Test that weights always sum to 1.0."""
        for method in ScaleMethod:
            for num_tranches in range(1, 6):
                weights = scaler.get_tranche_weights(method, num_tranches)
                assert sum(weights) == pytest.approx(1.0, rel=0.01)


# ============================================================================
# Scale-In Plan Tests
# ============================================================================


class TestCreateScaleInPlan:
    """Test create_scale_in_plan method."""

    def test_basic_plan_creation(self, scaler):
        """Test basic plan creation."""
        plan = scaler.create_scale_in_plan("AAPL", 100, 150.0)

        assert plan["symbol"] == "AAPL"
        assert plan["direction"] == "long"
        assert plan["total_shares"] == 100
        assert plan["method"] == "pyramid"
        assert plan["status"] == "active"
        assert len(plan["tranches"]) == 3

    def test_plan_stored_in_active_plans(self, scaler):
        """Test plan is stored in active_plans."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        assert "AAPL" in scaler.active_plans

    def test_shares_add_up_correctly(self, scaler):
        """Test total shares in tranches equals requested."""
        plan = scaler.create_scale_in_plan("AAPL", 100, 150.0)
        total = sum(t["shares"] for t in plan["tranches"])
        assert total == 100

    def test_custom_num_tranches(self, scaler):
        """Test with custom number of tranches."""
        plan = scaler.create_scale_in_plan("AAPL", 100, 150.0, num_tranches=5)
        assert len(plan["tranches"]) == 5

    def test_custom_method(self, scaler):
        """Test with custom scaling method."""
        plan = scaler.create_scale_in_plan(
            "AAPL", 100, 150.0, method=ScaleMethod.EQUAL
        )
        assert plan["method"] == "equal"

    def test_custom_price_levels(self, scaler):
        """Test with custom price levels."""
        price_levels = [150.0, 145.0, 140.0]
        plan = scaler.create_scale_in_plan(
            "AAPL", 100, 150.0, price_levels=price_levels
        )
        for i, tranche in enumerate(plan["tranches"]):
            assert tranche["target_price"] == price_levels[i]

    def test_default_price_levels(self, scaler):
        """Test default price levels (1% intervals below current)."""
        plan = scaler.create_scale_in_plan("AAPL", 100, 150.0)
        assert plan["tranches"][0]["target_price"] == 150.0
        assert plan["tranches"][1]["target_price"] == pytest.approx(148.5, rel=0.01)
        assert plan["tranches"][2]["target_price"] == pytest.approx(147.0, rel=0.01)

    def test_all_tranches_pending(self, scaler):
        """Test all tranches start as pending."""
        plan = scaler.create_scale_in_plan("AAPL", 100, 150.0)
        for tranche in plan["tranches"]:
            assert tranche["status"] == "pending"

    def test_initial_shares_filled_zero(self, scaler):
        """Test initial shares_filled is zero."""
        plan = scaler.create_scale_in_plan("AAPL", 100, 150.0)
        assert plan["shares_filled"] == 0


# ============================================================================
# Get Next Tranche Tests
# ============================================================================


class TestGetNextTranche:
    """Test get_next_tranche method."""

    def test_returns_first_pending(self, scaler):
        """Test returns first pending tranche."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        tranche = scaler.get_next_tranche("AAPL")
        assert tranche is not None
        assert tranche["tranche"] == 1

    def test_returns_none_for_unknown_symbol(self, scaler):
        """Test returns None for unknown symbol."""
        tranche = scaler.get_next_tranche("UNKNOWN")
        assert tranche is None

    def test_returns_next_after_fill(self, scaler):
        """Test returns next pending after fill."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)

        tranche = scaler.get_next_tranche("AAPL")
        assert tranche["tranche"] == 2

    def test_returns_none_when_all_filled(self, scaler):
        """Test returns None when all tranches filled."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)
        scaler.fill_tranche("AAPL", 2, 148.5)
        scaler.fill_tranche("AAPL", 3, 147.0)

        tranche = scaler.get_next_tranche("AAPL")
        assert tranche is None


# ============================================================================
# Fill Tranche Tests
# ============================================================================


class TestFillTranche:
    """Test fill_tranche method."""

    def test_basic_fill(self, scaler):
        """Test basic tranche fill."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        plan = scaler.fill_tranche("AAPL", 1, 150.50)

        assert plan["tranches"][0]["status"] == "filled"
        assert plan["tranches"][0]["filled_price"] == 150.50
        assert plan["tranches"][0]["filled_at"] is not None

    def test_shares_filled_updated(self, scaler):
        """Test shares_filled is updated."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        plan = scaler.fill_tranche("AAPL", 1, 150.50)

        assert plan["shares_filled"] == plan["tranches"][0]["shares"]

    def test_avg_price_calculated(self, scaler):
        """Test average price is calculated correctly."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)
        plan = scaler.fill_tranche("AAPL", 2, 148.0)

        # Calculate expected average
        t1_shares = plan["tranches"][0]["shares"]
        t2_shares = plan["tranches"][1]["shares"]
        expected_avg = (t1_shares * 150.0 + t2_shares * 148.0) / (t1_shares + t2_shares)

        assert plan["avg_price"] == pytest.approx(expected_avg, rel=0.01)

    def test_plan_completed_when_all_filled(self, scaler):
        """Test plan status becomes completed when all filled."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)
        scaler.fill_tranche("AAPL", 2, 148.5)
        plan = scaler.fill_tranche("AAPL", 3, 147.0)

        assert plan["status"] == "completed"

    def test_returns_empty_for_unknown_symbol(self, scaler):
        """Test returns empty dict for unknown symbol."""
        result = scaler.fill_tranche("UNKNOWN", 1, 150.0)
        assert result == {}


# ============================================================================
# Scale-Out Plan Tests
# ============================================================================


class TestCreateScaleOutPlan:
    """Test create_scale_out_plan method."""

    def test_basic_scale_out_plan(self, scaler):
        """Test basic scale-out plan creation."""
        plan = scaler.create_scale_out_plan("AAPL", 100, 145.0)

        assert plan["symbol"] == "AAPL"
        assert plan["direction"] == "exit"
        assert plan["total_shares"] == 100
        assert plan["entry_price"] == 145.0

    def test_profit_target_prices(self, scaler):
        """Test profit target prices are calculated correctly."""
        plan = scaler.create_scale_out_plan("AAPL", 100, 100.0)

        # Default levels are 5%, 10%, 20%
        assert plan["tranches"][0]["target_price"] == pytest.approx(105.0, rel=0.01)
        assert plan["tranches"][1]["target_price"] == pytest.approx(110.0, rel=0.01)
        assert plan["tranches"][2]["target_price"] == pytest.approx(120.0, rel=0.01)

    def test_custom_profit_targets(self, scaler):
        """Test with custom profit targets."""
        plan = scaler.create_scale_out_plan(
            "AAPL", 100, 100.0, profit_targets=[0.10, 0.25, 0.50]
        )

        assert plan["tranches"][0]["target_price"] == pytest.approx(110.0)
        assert plan["tranches"][1]["target_price"] == pytest.approx(125.0)
        assert plan["tranches"][2]["target_price"] == pytest.approx(150.0)

    def test_shares_sum_correctly(self, scaler):
        """Test shares in tranches sum to total."""
        plan = scaler.create_scale_out_plan("AAPL", 100, 145.0)
        total = sum(t["shares"] for t in plan["tranches"])
        assert total == 100


# ============================================================================
# Check Scale-Out Trigger Tests
# ============================================================================


class TestCheckScaleOutTrigger:
    """Test check_scale_out_trigger method."""

    def test_no_trigger_below_min_profit(self, scaler):
        """Test no scale-out when below minimum profit."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=102.0, shares_held=100
        )
        assert not result["should_scale_out"]
        assert "below threshold" in result["reason"]

    def test_trigger_at_first_level(self, scaler):
        """Test scale-out triggered at first profit level."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=106.0, shares_held=100
        )
        # Default first level is 5%
        assert result["should_scale_out"]
        assert result["shares_to_sell"] > 0

    def test_trigger_at_second_level(self, scaler):
        """Test scale-out triggered at second profit level."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=111.0, shares_held=100
        )
        # Should have hit 5% and 10% levels
        assert result["should_scale_out"]

    def test_already_sold_reduces_shares(self, scaler):
        """Test already sold shares reduces recommendation."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0,
            current_price=106.0,
            shares_held=100,
            shares_already_sold=30,
        )
        # Already sold some, should reduce recommendation
        assert result["shares_to_sell"] >= 0

    def test_next_target_returned(self, scaler):
        """Test next target is returned."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=106.0, shares_held=100
        )
        # Next target after 5% should be 10%
        assert result["next_target"] == 0.10

    def test_no_next_target_at_max_level(self, scaler):
        """Test no next target when at max level."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=125.0, shares_held=100
        )
        # Above all levels (5%, 10%, 20%)
        assert result["next_target"] is None

    def test_profit_pct_calculated(self, scaler):
        """Test profit percentage is calculated."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=115.0, shares_held=100
        )
        assert result["profit_pct"] == pytest.approx(0.15, rel=0.01)


# ============================================================================
# Recommend Add To Winner Tests
# ============================================================================


class TestRecommendAddToWinner:
    """Test recommend_add_to_winner method."""

    def test_no_add_below_min_profit(self, scaler):
        """Test no recommendation when below min profit."""
        result = scaler.recommend_add_to_winner(
            entry_price=100.0,
            current_price=101.0,
            current_shares=100,
            max_position_shares=200,
        )
        assert not result["should_add"]
        assert "below min" in result["reason"]

    def test_no_add_at_max_position(self, scaler):
        """Test no recommendation when at max position."""
        result = scaler.recommend_add_to_winner(
            entry_price=100.0,
            current_price=110.0,
            current_shares=200,
            max_position_shares=200,
        )
        assert not result["should_add"]
        assert "max position" in result["reason"]

    def test_add_when_profitable(self, scaler):
        """Test recommendation when profitable and room available."""
        result = scaler.recommend_add_to_winner(
            entry_price=100.0,
            current_price=110.0,
            current_shares=100,
            max_position_shares=200,
        )
        assert result["should_add"]
        assert result["shares_to_add"] > 0

    def test_adds_half_of_remaining_room(self, scaler):
        """Test adds approximately 50% of remaining room."""
        result = scaler.recommend_add_to_winner(
            entry_price=100.0,
            current_price=110.0,
            current_shares=100,
            max_position_shares=200,
        )
        # Room is 100, should add ~50
        assert result["shares_to_add"] == 50

    def test_new_avg_price_calculated(self, scaler):
        """Test new average price is calculated."""
        result = scaler.recommend_add_to_winner(
            entry_price=100.0,
            current_price=110.0,
            current_shares=100,
            max_position_shares=200,
        )
        # New avg = (100*100 + 110*50) / 150 = 103.33
        expected = (100 * 100 + 110 * 50) / 150
        assert result["new_avg_price"] == pytest.approx(expected, rel=0.01)

    def test_custom_min_profit(self, scaler):
        """Test with custom minimum profit threshold."""
        result = scaler.recommend_add_to_winner(
            entry_price=100.0,
            current_price=104.0,
            current_shares=100,
            max_position_shares=200,
            min_profit_to_add=0.05,  # Requires 5%
        )
        # Only 4% profit, below 5% threshold
        assert not result["should_add"]


# ============================================================================
# Cleanup Completed Plans Tests
# ============================================================================


class TestCleanupCompletedPlans:
    """Test cleanup_completed_plans method."""

    def test_removes_old_completed_plans(self, scaler):
        """Test removes completed plans older than max_age."""
        # Create and complete a plan
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)
        scaler.fill_tranche("AAPL", 2, 148.5)
        scaler.fill_tranche("AAPL", 3, 147.0)

        # Make the plan old
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        scaler.active_plans["AAPL"]["created_at"] = old_time

        removed = scaler.cleanup_completed_plans(max_age_hours=24)

        assert removed == 1
        assert "AAPL" not in scaler.active_plans

    def test_keeps_recent_completed_plans(self, scaler):
        """Test keeps recent completed plans."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)
        scaler.fill_tranche("AAPL", 2, 148.5)
        scaler.fill_tranche("AAPL", 3, 147.0)

        removed = scaler.cleanup_completed_plans(max_age_hours=24)

        assert removed == 0
        assert "AAPL" in scaler.active_plans

    def test_keeps_active_plans(self, scaler):
        """Test keeps active (incomplete) plans."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)

        # Make it old but still active
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        scaler.active_plans["AAPL"]["created_at"] = old_time

        removed = scaler.cleanup_completed_plans(max_age_hours=24)

        assert removed == 0
        assert "AAPL" in scaler.active_plans

    def test_handles_invalid_date(self, scaler):
        """Test handles invalid date in plan."""
        scaler.active_plans["TEST"] = {
            "status": "completed",
            "created_at": "invalid-date",
        }

        removed = scaler.cleanup_completed_plans()

        assert removed == 1


# ============================================================================
# Get Scaling Summary Tests
# ============================================================================


class TestGetScalingSummary:
    """Test get_scaling_summary method."""

    def test_no_plan_status(self, scaler):
        """Test returns no_plan status for unknown symbol."""
        summary = scaler.get_scaling_summary("UNKNOWN")
        assert summary["status"] == "no_plan"

    def test_active_plan_summary(self, scaler):
        """Test summary for active plan."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)

        summary = scaler.get_scaling_summary("AAPL")

        assert summary["symbol"] == "AAPL"
        assert summary["status"] == "active"
        assert summary["tranches_filled"] == 1
        assert summary["tranches_pending"] == 2
        assert summary["next_tranche"] is not None

    def test_completed_plan_summary(self, scaler):
        """Test summary for completed plan."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.fill_tranche("AAPL", 1, 150.0)
        scaler.fill_tranche("AAPL", 2, 148.5)
        scaler.fill_tranche("AAPL", 3, 147.0)

        summary = scaler.get_scaling_summary("AAPL")

        assert summary["status"] == "completed"
        assert summary["tranches_filled"] == 3
        assert summary["tranches_pending"] == 0
        assert summary["next_tranche"] is None

    def test_triggers_cleanup_when_too_many_plans(self, scaler):
        """Test cleanup is triggered when too many plans."""
        scaler._max_cached_plans = 5

        # Create many plans
        for i in range(10):
            scaler.create_scale_in_plan(f"SYM{i}", 100, 150.0)

        # Complete some
        for i in range(5):
            for j in range(1, 4):
                scaler.fill_tranche(f"SYM{i}", j, 150.0)
            # Make them old
            scaler.active_plans[f"SYM{i}"]["created_at"] = (
                datetime.now() - timedelta(hours=2)
            ).isoformat()

        # Getting summary should trigger cleanup
        scaler.get_scaling_summary("SYM9")

        # Should have cleaned up some completed plans
        assert len(scaler.active_plans) < 10


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_share_plan(self, scaler):
        """Test plan with single share."""
        plan = scaler.create_scale_in_plan("AAPL", 1, 150.0, num_tranches=1)
        assert plan["tranches"][0]["shares"] == 1

    def test_small_shares_distributed(self, scaler):
        """Test small number of shares distributed correctly."""
        plan = scaler.create_scale_in_plan("AAPL", 5, 150.0, num_tranches=3)
        total = sum(t["shares"] for t in plan["tranches"])
        assert total == 5

    def test_zero_profit(self, scaler):
        """Test scale-out check at zero profit."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=100.0, shares_held=100
        )
        assert not result["should_scale_out"]
        assert result["profit_pct"] == 0

    def test_negative_profit(self, scaler):
        """Test scale-out check with negative profit."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=90.0, shares_held=100
        )
        assert not result["should_scale_out"]
        assert result["profit_pct"] < 0

    def test_large_profit(self, scaler):
        """Test scale-out check with large profit."""
        result = scaler.check_scale_out_trigger(
            entry_price=100.0, current_price=200.0, shares_held=100
        )
        assert result["should_scale_out"]
        assert result["profit_pct"] == 1.0

    def test_pyramid_with_large_num_tranches(self, scaler):
        """Test pyramid weights with large number of tranches."""
        weights = scaler.get_tranche_weights(ScaleMethod.PYRAMID, 10)
        assert len(weights) == 10
        assert sum(weights) == pytest.approx(1.0, rel=0.01)

    def test_fill_wrong_tranche_number(self, scaler):
        """Test filling non-existent tranche number."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        plan = scaler.fill_tranche("AAPL", 99, 150.0)  # Invalid tranche number

        # Should not crash, plan unchanged
        assert plan["shares_filled"] == 0

    def test_multiple_symbols(self, scaler):
        """Test managing multiple symbols."""
        scaler.create_scale_in_plan("AAPL", 100, 150.0)
        scaler.create_scale_in_plan("MSFT", 50, 300.0)
        scaler.create_scale_in_plan("GOOGL", 25, 2500.0)

        assert len(scaler.active_plans) == 3
        assert "AAPL" in scaler.active_plans
        assert "MSFT" in scaler.active_plans
        assert "GOOGL" in scaler.active_plans
