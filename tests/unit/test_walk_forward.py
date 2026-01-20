#!/usr/bin/env python3
"""
Unit tests for Walk-Forward Validation Engine.

Tests cover:
1. WalkForwardResult dataclass
2. WalkForwardValidator initialization
3. Time split creation
4. Validation with mock backtest functions
5. Result aggregation
6. Overfitting detection
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Mock config before importing the module
sys.modules["config"] = MagicMock(
    BACKTEST_PARAMS={
        "TRAIN_RATIO": 0.7,
        "N_SPLITS": 5,
        "MIN_TRAIN_DAYS": 30,
        "OVERFITTING_RATIO_THRESHOLD": 2.0,
        "SLIPPAGE_PCT": 0.004,
    }
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.walk_forward import WalkForwardResult, WalkForwardValidator


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def validator():
    """Create a basic WalkForwardValidator instance."""
    return WalkForwardValidator()


@pytest.fixture
def validator_custom():
    """Create a WalkForwardValidator with custom parameters."""
    return WalkForwardValidator(train_ratio=0.8, n_splits=3, min_train_days=20, gap_days=5)


@pytest.fixture
def sample_result():
    """Create a sample WalkForwardResult."""
    return WalkForwardResult(
        fold_num=1,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 6, 30),
        test_start=datetime(2024, 7, 1),
        test_end=datetime(2024, 9, 30),
        is_return=0.15,
        is_sharpe=1.5,
        is_trades=50,
        is_win_rate=0.55,
        oos_return=0.08,
        oos_sharpe=1.0,
        oos_trades=20,
        oos_win_rate=0.52,
        overfitting_ratio=1.875,
        degradation=0.467,
    )


# =============================================================================
# TEST WALKFORWARDRESULT DATACLASS
# =============================================================================


class TestWalkForwardResult:
    """Test WalkForwardResult dataclass."""

    def test_result_creation(self, sample_result):
        """Test creating a WalkForwardResult."""
        assert sample_result.fold_num == 1
        assert sample_result.is_return == 0.15
        assert sample_result.oos_return == 0.08
        assert sample_result.overfitting_ratio == 1.875

    def test_result_attributes(self, sample_result):
        """Test all attributes are accessible."""
        assert hasattr(sample_result, "train_start")
        assert hasattr(sample_result, "train_end")
        assert hasattr(sample_result, "test_start")
        assert hasattr(sample_result, "test_end")
        assert hasattr(sample_result, "is_sharpe")
        assert hasattr(sample_result, "oos_sharpe")
        assert hasattr(sample_result, "is_win_rate")
        assert hasattr(sample_result, "oos_win_rate")
        assert hasattr(sample_result, "degradation")


# =============================================================================
# TEST WALKFORWARDVALIDATOR INITIALIZATION
# =============================================================================


class TestValidatorInit:
    """Test WalkForwardValidator initialization."""

    def test_default_initialization(self, validator):
        """Test default initialization values."""
        assert validator.train_ratio == 0.7
        assert validator.n_splits == 5
        assert validator.min_train_days == 30
        assert validator.gap_days == 0
        assert validator.results == []

    def test_custom_initialization(self, validator_custom):
        """Test custom initialization values."""
        assert validator_custom.train_ratio == 0.8
        assert validator_custom.n_splits == 3
        assert validator_custom.min_train_days == 20
        assert validator_custom.gap_days == 5

    def test_overfitting_threshold(self, validator):
        """Test overfitting threshold is set from config."""
        assert validator.overfitting_threshold == 2.0


# =============================================================================
# TEST TIME SPLIT CREATION
# =============================================================================


class TestTimeSplits:
    """Test time split creation."""

    def test_create_time_splits_basic(self, validator):
        """Test creating basic time splits."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        splits = validator.create_time_splits(start, end)

        assert len(splits) > 0
        for train_start, train_end, test_start, test_end in splits:
            assert train_start < train_end
            assert test_start < test_end
            assert train_end <= test_start

    def test_create_time_splits_respects_gap(self, validator_custom):
        """Test that splits respect gap_days parameter."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        splits = validator_custom.create_time_splits(start, end)

        for train_start, train_end, test_start, test_end in splits:
            gap = (test_start - train_end).days
            assert gap >= validator_custom.gap_days

    def test_create_time_splits_too_short_raises_error(self, validator):
        """Test that short date range raises error."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 30)  # Less than 2 * min_train_days

        with pytest.raises(ValueError, match="Date range too short"):
            validator.create_time_splits(start, end)

    def test_create_time_splits_skips_short_test_periods(self):
        """Test that splits with short test periods are skipped."""
        validator = WalkForwardValidator(n_splits=10, min_train_days=10)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)

        splits = validator.create_time_splits(start, end)

        # Should have fewer splits than n_splits if some are too short
        for _, _, test_start, test_end in splits:
            assert (test_end - test_start).days >= 5

    def test_expanding_window(self, validator):
        """Test that training uses expanding window."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        splits = validator.create_time_splits(start, end)

        # All folds should start from the same date (expanding window)
        for train_start, _, _, _ in splits:
            assert train_start == start


# =============================================================================
# TEST VALIDATION
# =============================================================================


class TestValidation:
    """Test the validate method."""

    @pytest.mark.asyncio
    async def test_validate_runs_backtests(self, validator):
        """Test that validate calls backtest function for each fold."""
        call_count = 0

        async def mock_backtest(symbols, start, end, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"total_return": 0.05, "sharpe_ratio": 1.0, "num_trades": 10, "win_rate": 0.5}

        validator.n_splits = 3  # Reduce for faster test
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        # Each fold runs 2 backtests (IS and OOS)
        assert call_count >= 6

    @pytest.mark.asyncio
    async def test_validate_returns_aggregated_results(self, validator):
        """Test that validate returns proper aggregated results."""

        async def mock_backtest(symbols, start, end, **kwargs):
            return {"total_return": 0.10, "sharpe_ratio": 1.2, "num_trades": 15, "win_rate": 0.55}

        validator.n_splits = 2
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        assert "passes_validation" in result
        assert "n_folds" in result
        assert "is_avg_return" in result
        assert "oos_avg_return" in result
        assert "avg_overfit_ratio" in result
        assert "fold_results" in result

    @pytest.mark.asyncio
    async def test_validate_detects_overfitting(self, validator):
        """Test that validate detects overfitting."""
        call_num = [0]

        async def mock_backtest(symbols, start, end, **kwargs):
            call_num[0] += 1
            # IS has great returns, OOS has poor returns
            if call_num[0] % 2 == 1:  # IS backtest
                return {"total_return": 0.50, "sharpe_ratio": 2.0, "num_trades": 30, "win_rate": 0.7}
            else:  # OOS backtest
                return {"total_return": -0.05, "sharpe_ratio": -0.5, "num_trades": 10, "win_rate": 0.3}

        validator.n_splits = 2
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        assert result["passes_validation"] == False
        assert result["oos_avg_return"] < 0

    @pytest.mark.asyncio
    async def test_validate_empty_splits_raises_error(self, validator):
        """Test that validate raises error if no valid splits."""
        async def mock_backtest(symbols, start, end, **kwargs):
            return {}

        # Create very restrictive conditions that result in no valid splits
        validator.n_splits = 100
        validator.min_train_days = 100

        with pytest.raises(ValueError, match="Could not create valid train/test splits|Date range too short"):
            await validator.validate(
                mock_backtest, ["AAPL"], "2024-01-01", "2024-03-01"
            )

    @pytest.mark.asyncio
    async def test_validate_passes_kwargs_to_backtest(self, validator):
        """Test that validate passes kwargs to backtest function."""
        received_kwargs = {}

        async def mock_backtest(symbols, start, end, **kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            return {"total_return": 0.05, "sharpe_ratio": 1.0, "num_trades": 10, "win_rate": 0.5}

        validator.n_splits = 1
        await validator.validate(
            mock_backtest,
            ["AAPL"],
            "2024-01-01",
            "2024-12-31",
            custom_param="test_value",
        )

        assert received_kwargs.get("custom_param") == "test_value"


# =============================================================================
# TEST RESULT AGGREGATION
# =============================================================================


class TestResultAggregation:
    """Test result aggregation."""

    def test_aggregate_empty_results(self, validator):
        """Test aggregating empty results returns empty dict."""
        result = validator._aggregate_results()
        assert result == {}

    def test_aggregate_single_result(self, validator, sample_result):
        """Test aggregating single result."""
        validator.results = [sample_result]
        result = validator._aggregate_results()

        assert result["n_folds"] == 1
        assert result["is_avg_return"] == sample_result.is_return
        assert result["oos_avg_return"] == sample_result.oos_return

    def test_aggregate_multiple_results(self, validator):
        """Test aggregating multiple results."""
        results = [
            WalkForwardResult(
                fold_num=i,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10 + i * 0.02,
                is_sharpe=1.0 + i * 0.1,
                is_trades=20 + i * 5,
                is_win_rate=0.5,
                oos_return=0.05 + i * 0.01,
                oos_sharpe=0.8 + i * 0.1,
                oos_trades=10 + i * 2,
                oos_win_rate=0.48,
                overfitting_ratio=1.5,
                degradation=0.3,
            )
            for i in range(3)
        ]

        validator.results = results
        result = validator._aggregate_results()

        assert result["n_folds"] == 3
        assert result["is_avg_return"] == pytest.approx(np.mean([r.is_return for r in results]))
        assert result["oos_avg_return"] == pytest.approx(np.mean([r.oos_return for r in results]))

    def test_aggregate_calculates_consistency(self, validator):
        """Test consistency calculation."""
        results = [
            WalkForwardResult(
                fold_num=1,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=20,
                is_win_rate=0.5,
                oos_return=0.05,  # Positive
                oos_sharpe=0.8,
                oos_trades=10,
                oos_win_rate=0.48,
                overfitting_ratio=1.5,
                degradation=0.3,
            ),
            WalkForwardResult(
                fold_num=2,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=20,
                is_win_rate=0.5,
                oos_return=-0.02,  # Negative
                oos_sharpe=-0.3,
                oos_trades=10,
                oos_win_rate=0.4,
                overfitting_ratio=3.0,
                degradation=0.6,
            ),
        ]

        validator.results = results
        result = validator._aggregate_results()

        assert result["oos_consistency"] == 0.5  # 1 out of 2 folds profitable

    def test_aggregate_counts_overfit_folds(self, validator):
        """Test counting overfit folds."""
        results = [
            WalkForwardResult(
                fold_num=1,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=20,
                is_win_rate=0.5,
                oos_return=0.05,
                oos_sharpe=0.8,
                oos_trades=10,
                oos_win_rate=0.48,
                overfitting_ratio=1.5,  # Below threshold
                degradation=0.3,
            ),
            WalkForwardResult(
                fold_num=2,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=20,
                is_win_rate=0.5,
                oos_return=0.02,
                oos_sharpe=0.5,
                oos_trades=10,
                oos_win_rate=0.45,
                overfitting_ratio=3.0,  # Above threshold
                degradation=0.6,
            ),
        ]

        validator.results = results
        result = validator._aggregate_results()

        assert result["overfit_folds"] == 1

    def test_aggregate_handles_inf_overfitting_ratio(self, validator):
        """Test handling of infinite overfitting ratios."""
        results = [
            WalkForwardResult(
                fold_num=1,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=20,
                is_win_rate=0.5,
                oos_return=0.05,
                oos_sharpe=0.8,
                oos_trades=10,
                oos_win_rate=0.48,
                overfitting_ratio=float("inf"),  # Infinite ratio
                degradation=0.3,
            ),
        ]

        validator.results = results
        result = validator._aggregate_results()

        # Should use threshold instead of inf
        assert result["avg_overfit_ratio"] == validator.overfitting_threshold


# =============================================================================
# TEST OVERFITTING CALCULATION
# =============================================================================


class TestOverfittingCalculation:
    """Test overfitting ratio and degradation calculations."""

    @pytest.mark.asyncio
    async def test_both_positive_returns(self, validator):
        """Test overfitting calculation when both returns are positive."""
        async def mock_backtest(symbols, start, end, **kwargs):
            # Simulate based on call order
            return {"total_return": 0.20, "sharpe_ratio": 1.5, "num_trades": 20, "win_rate": 0.6}

        validator.n_splits = 1
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        # When IS = OOS, ratio should be 1.0
        assert result["avg_overfit_ratio"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_is_positive_oos_negative(self, validator):
        """Test overfitting calculation when IS positive, OOS negative."""
        call_num = [0]

        async def mock_backtest(symbols, start, end, **kwargs):
            call_num[0] += 1
            if call_num[0] % 2 == 1:  # IS
                return {"total_return": 0.15, "sharpe_ratio": 1.2, "num_trades": 20, "win_rate": 0.55}
            else:  # OOS
                return {"total_return": -0.05, "sharpe_ratio": -0.5, "num_trades": 10, "win_rate": 0.4}

        validator.n_splits = 1
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        # Should indicate severe overfitting
        assert result["avg_overfit_ratio"] > validator.overfitting_threshold

    @pytest.mark.asyncio
    async def test_both_negative_returns(self, validator):
        """Test overfitting calculation when both returns are negative."""
        async def mock_backtest(symbols, start, end, **kwargs):
            return {"total_return": -0.05, "sharpe_ratio": -0.5, "num_trades": 10, "win_rate": 0.4}

        validator.n_splits = 1
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        # Both negative = neutral (ratio 1.0)
        assert result["avg_overfit_ratio"] == pytest.approx(1.0)


# =============================================================================
# TEST VALIDATION PASS/FAIL CRITERIA
# =============================================================================


class TestValidationCriteria:
    """Test validation pass/fail criteria."""

    @pytest.mark.asyncio
    async def test_passes_all_criteria(self, validator):
        """Test validation passes when all criteria met."""
        async def mock_backtest(symbols, start, end, **kwargs):
            return {"total_return": 0.08, "sharpe_ratio": 1.0, "num_trades": 15, "win_rate": 0.55}

        validator.n_splits = 2
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        assert result["passes_validation"] == True

    @pytest.mark.asyncio
    async def test_fails_negative_oos_returns(self, validator):
        """Test validation fails with negative OOS returns."""
        call_num = [0]

        async def mock_backtest(symbols, start, end, **kwargs):
            call_num[0] += 1
            if call_num[0] % 2 == 1:  # IS
                return {"total_return": 0.10, "sharpe_ratio": 1.0, "num_trades": 20, "win_rate": 0.5}
            else:  # OOS
                return {"total_return": -0.10, "sharpe_ratio": -0.8, "num_trades": 10, "win_rate": 0.35}

        validator.n_splits = 2
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        assert result["passes_validation"] == False
        assert result["oos_avg_return"] < 0

    @pytest.mark.asyncio
    async def test_fails_high_overfit_ratio(self, validator):
        """Test validation fails with high overfitting ratio."""
        call_num = [0]

        async def mock_backtest(symbols, start, end, **kwargs):
            call_num[0] += 1
            if call_num[0] % 2 == 1:  # IS
                return {"total_return": 0.50, "sharpe_ratio": 2.5, "num_trades": 30, "win_rate": 0.7}
            else:  # OOS - much worse
                return {"total_return": 0.02, "sharpe_ratio": 0.3, "num_trades": 10, "win_rate": 0.45}

        validator.n_splits = 2
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        # High IS return vs low OOS return should indicate overfitting
        assert result["avg_overfit_ratio"] > 2.0

    @pytest.mark.asyncio
    async def test_fails_low_consistency(self, validator):
        """Test validation fails with low consistency."""
        call_num = [0]

        async def mock_backtest(symbols, start, end, **kwargs):
            call_num[0] += 1
            fold = (call_num[0] - 1) // 2
            if call_num[0] % 2 == 1:  # IS
                return {"total_return": 0.10, "sharpe_ratio": 1.0, "num_trades": 20, "win_rate": 0.5}
            else:  # OOS - alternates positive/negative
                if fold == 0:
                    return {"total_return": -0.10, "sharpe_ratio": -0.8, "num_trades": 10, "win_rate": 0.3}
                else:
                    return {"total_return": -0.05, "sharpe_ratio": -0.5, "num_trades": 8, "win_rate": 0.35}

        validator.n_splits = 2
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        # All OOS returns are negative = 0% consistency
        assert result["oos_consistency"] == 0.0
        assert result["passes_validation"] == False


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_single_fold(self, validator):
        """Test validation with single fold."""
        async def mock_backtest(symbols, start, end, **kwargs):
            return {"total_return": 0.10, "sharpe_ratio": 1.2, "num_trades": 15, "win_rate": 0.55}

        validator.n_splits = 1
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        assert result["n_folds"] == 1

    @pytest.mark.asyncio
    async def test_zero_trades(self, validator):
        """Test handling zero trades."""
        async def mock_backtest(symbols, start, end, **kwargs):
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "num_trades": 0, "win_rate": 0.0}

        validator.n_splits = 1
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        assert result["oos_total_trades"] == 0

    @pytest.mark.asyncio
    async def test_missing_backtest_fields(self, validator):
        """Test handling backtest results with missing fields."""
        async def mock_backtest(symbols, start, end, **kwargs):
            return {}  # Empty results

        validator.n_splits = 1
        result = await validator.validate(
            mock_backtest, ["AAPL"], "2024-01-01", "2024-12-31"
        )

        # Should use default values (0) for missing fields
        assert result["oos_avg_return"] == 0


# =============================================================================
# TEST RUN_WALK_FORWARD_VALIDATION FUNCTION
# =============================================================================


class TestRunWalkForwardValidation:
    """Test the convenience function."""

    @pytest.mark.asyncio
    async def test_function_imports_and_runs(self):
        """Test that run_walk_forward_validation can be imported."""
        from engine.walk_forward import run_walk_forward_validation

        # Just test that it's callable
        assert callable(run_walk_forward_validation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
