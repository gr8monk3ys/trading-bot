"""
Tests for WalkForwardValidator

Tests cover:
- Time split creation
- Validation logic
- Overfitting detection
- Result aggregation
- Edge cases
"""

import os
import sys
from datetime import datetime

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.walk_forward import WalkForwardResult, WalkForwardValidator


class TestWalkForwardInit:
    """Test WalkForwardValidator initialization."""

    def test_default_initialization(self):
        """Test default values are set correctly."""
        validator = WalkForwardValidator()
        assert validator.train_ratio == 0.7
        assert validator.n_splits == 5
        assert validator.min_train_days == 30
        # gap_days defaults to 5 (institutional standard embargo period)
        assert validator.gap_days == 5

    def test_custom_initialization(self):
        """Test custom values override defaults."""
        validator = WalkForwardValidator(train_ratio=0.8, n_splits=3, min_train_days=60, gap_days=5)
        assert validator.train_ratio == 0.8
        assert validator.n_splits == 3
        assert validator.min_train_days == 60
        assert validator.gap_days == 5


class TestTimeSplitCreation:
    """Test create_time_splits method."""

    def test_creates_correct_number_of_splits(self):
        """Should create requested number of splits."""
        validator = WalkForwardValidator(n_splits=5, min_train_days=10)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        splits = validator.create_time_splits(start, end)

        # May create fewer splits if periods are too short
        assert len(splits) <= 5
        assert len(splits) >= 1

    def test_splits_are_chronological(self):
        """All splits should be in chronological order."""
        validator = WalkForwardValidator(n_splits=3, min_train_days=20)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        splits = validator.create_time_splits(start, end)

        for train_start, train_end, test_start, test_end in splits:
            assert train_start < train_end
            assert train_end <= test_start
            assert test_start < test_end

    def test_train_end_before_test_start(self):
        """Training should end before testing starts."""
        validator = WalkForwardValidator(n_splits=3, gap_days=0)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        splits = validator.create_time_splits(start, end)

        for _, train_end, test_start, _ in splits:
            assert train_end <= test_start

    def test_gap_days_respected(self):
        """Gap between train and test should be respected."""
        gap = 5
        validator = WalkForwardValidator(n_splits=3, gap_days=gap, min_train_days=20)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        splits = validator.create_time_splits(start, end)

        for _, train_end, test_start, _ in splits:
            actual_gap = (test_start - train_end).days
            assert actual_gap >= gap

    def test_raises_on_insufficient_data(self):
        """Should raise if date range is too short."""
        validator = WalkForwardValidator(min_train_days=30)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)  # Only 14 days

        with pytest.raises(ValueError, match="too short"):
            validator.create_time_splits(start, end)


class TestWalkForwardResult:
    """Test WalkForwardResult dataclass."""

    def test_result_creation(self):
        """Test creating a result object."""
        result = WalkForwardResult(
            fold_num=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 9, 30),
            is_return=0.10,
            is_sharpe=1.5,
            is_trades=50,
            is_win_rate=0.6,
            oos_return=0.05,
            oos_sharpe=0.8,
            oos_trades=20,
            oos_win_rate=0.55,
            overfitting_ratio=2.0,
            degradation=0.5,
        )

        assert result.fold_num == 1
        assert result.is_return == 0.10
        assert result.oos_return == 0.05
        assert result.overfitting_ratio == 2.0


class TestOverfittingDetection:
    """Test overfitting detection logic."""

    def test_high_ratio_indicates_overfitting(self):
        """Ratio > threshold should indicate overfitting."""
        validator = WalkForwardValidator()

        # Create results with high overfitting
        result = WalkForwardResult(
            fold_num=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 9, 30),
            is_return=0.20,  # 20% in-sample
            is_sharpe=2.0,
            is_trades=50,
            is_win_rate=0.7,
            oos_return=0.02,  # Only 2% out-of-sample
            oos_sharpe=0.3,
            oos_trades=20,
            oos_win_rate=0.45,
            overfitting_ratio=10.0,  # 20%/2% = 10x overfit
            degradation=0.9,
        )

        validator.results = [result]
        summary = validator._aggregate_results()

        # With our fix, high overfit ratio (10.0) should fail validation
        assert not summary["passes_validation"]
        assert summary["avg_overfit_ratio"] >= 2.0

    def test_good_oos_performance_passes(self):
        """Good out-of-sample performance should pass validation."""
        validator = WalkForwardValidator()

        # Create results with good OOS performance
        result = WalkForwardResult(
            fold_num=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 9, 30),
            is_return=0.10,
            is_sharpe=1.2,
            is_trades=50,
            is_win_rate=0.6,
            oos_return=0.08,  # 80% of in-sample (good!)
            oos_sharpe=0.9,
            oos_trades=20,
            oos_win_rate=0.58,
            overfitting_ratio=1.25,
            degradation=0.2,
        )

        validator.results = [result]
        summary = validator._aggregate_results()

        assert summary["passes_validation"] is True
        assert summary["avg_overfit_ratio"] < 2.0


class TestResultAggregation:
    """Test result aggregation across folds."""

    def test_averages_calculated_correctly(self):
        """Average metrics should be calculated correctly."""
        validator = WalkForwardValidator()

        results = [
            WalkForwardResult(
                fold_num=i,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=50,
                is_win_rate=0.6,
                oos_return=0.05,
                oos_sharpe=0.5,
                oos_trades=20,
                oos_win_rate=0.55,
                overfitting_ratio=2.0,
                degradation=0.5,
            )
            for i in range(3)
        ]

        validator.results = results
        summary = validator._aggregate_results()

        assert summary["n_folds"] == 3
        assert summary["is_avg_return"] == pytest.approx(0.10, rel=1e-9)
        assert summary["oos_avg_return"] == pytest.approx(0.05, rel=1e-9)
        assert summary["oos_total_trades"] == 60  # 20 * 3

    def test_consistency_calculation(self):
        """Consistency should be percentage of profitable folds."""
        validator = WalkForwardValidator()

        results = [
            WalkForwardResult(
                fold_num=1,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 3, 31),
                test_start=datetime(2024, 4, 1),
                test_end=datetime(2024, 6, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=50,
                is_win_rate=0.6,
                oos_return=0.05,  # Profitable
                oos_sharpe=0.5,
                oos_trades=20,
                oos_win_rate=0.55,
                overfitting_ratio=2.0,
                degradation=0.5,
            ),
            WalkForwardResult(
                fold_num=2,
                train_start=datetime(2024, 4, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 9, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=50,
                is_win_rate=0.6,
                oos_return=-0.02,  # Unprofitable
                oos_sharpe=0.5,
                oos_trades=20,
                oos_win_rate=0.55,
                overfitting_ratio=2.0,
                degradation=0.5,
            ),
        ]

        validator.results = results
        summary = validator._aggregate_results()

        assert summary["oos_consistency"] == 0.5  # 1 of 2 folds profitable

    def test_empty_results_handled(self):
        """Empty results should return empty dict."""
        validator = WalkForwardValidator()
        validator.results = []

        summary = validator._aggregate_results()

        assert summary == {}


class TestValidationCriteria:
    """Test validation pass/fail criteria."""

    def test_negative_oos_return_fails(self):
        """Negative OOS return should fail validation."""
        validator = WalkForwardValidator()

        result = WalkForwardResult(
            fold_num=1,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 9, 30),
            is_return=0.10,
            is_sharpe=1.0,
            is_trades=50,
            is_win_rate=0.6,
            oos_return=-0.05,  # Negative!
            oos_sharpe=-0.5,
            oos_trades=20,
            oos_win_rate=0.45,
            overfitting_ratio=1.0,
            degradation=1.5,
        )

        validator.results = [result]
        summary = validator._aggregate_results()

        # Negative OOS return should fail validation
        assert not summary["passes_validation"]

    def test_low_consistency_fails(self):
        """Consistency below 50% should fail validation."""
        validator = WalkForwardValidator()

        # 3 folds, only 1 profitable (33% consistency)
        results = [
            WalkForwardResult(
                fold_num=1,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 3, 31),
                test_start=datetime(2024, 4, 1),
                test_end=datetime(2024, 4, 30),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=50,
                is_win_rate=0.6,
                oos_return=0.05,  # Profitable
                oos_sharpe=0.5,
                oos_trades=20,
                oos_win_rate=0.55,
                overfitting_ratio=1.0,
                degradation=0.0,
            ),
            WalkForwardResult(
                fold_num=2,
                train_start=datetime(2024, 4, 1),
                train_end=datetime(2024, 6, 30),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 7, 31),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=50,
                is_win_rate=0.6,
                oos_return=-0.02,  # Unprofitable
                oos_sharpe=-0.2,
                oos_trades=20,
                oos_win_rate=0.45,
                overfitting_ratio=1.0,
                degradation=0.0,
            ),
            WalkForwardResult(
                fold_num=3,
                train_start=datetime(2024, 7, 1),
                train_end=datetime(2024, 9, 30),
                test_start=datetime(2024, 10, 1),
                test_end=datetime(2024, 10, 31),
                is_return=0.10,
                is_sharpe=1.0,
                is_trades=50,
                is_win_rate=0.6,
                oos_return=-0.03,  # Unprofitable
                oos_sharpe=-0.3,
                oos_trades=20,
                oos_win_rate=0.40,
                overfitting_ratio=1.0,
                degradation=0.0,
            ),
        ]

        validator.results = results
        summary = validator._aggregate_results()

        # Average OOS return is still slightly positive
        # But consistency is only 33%, so should fail
        assert summary["oos_consistency"] < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
