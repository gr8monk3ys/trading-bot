from datetime import datetime, timedelta

from engine.walk_forward import (
    WalkForwardResult,
    WalkForwardValidator,
    check_degradation_significance,
)


def test_check_degradation_significance_insufficient_folds():
    result = check_degradation_significance([0.1, 0.2], [0.05, 0.1])

    assert result.degradation_significant is False
    assert "Insufficient folds" in result.interpretation


def test_check_degradation_significance_no_difference():
    result = check_degradation_significance([0.1] * 6, [0.1] * 6)

    assert result.degradation_significant is False
    assert "No difference" in result.interpretation


def test_create_time_splits_respects_gap_and_min_days():
    validator = WalkForwardValidator(train_ratio=0.5, n_splits=2, min_train_days=10, gap_days=3)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 1)

    splits = validator.create_time_splits(start, end)

    assert splits
    for train_start, train_end, test_start, test_end in splits:
        assert train_start == start
        assert (test_start - train_end).days >= 3
        assert test_end >= test_start


def test_aggregate_results_basic_passes_validation():
    validator = WalkForwardValidator()
    start = datetime(2024, 1, 1)

    validator.results = [
        WalkForwardResult(
            fold_num=1,
            train_start=start,
            train_end=start + timedelta(days=30),
            test_start=start + timedelta(days=35),
            test_end=start + timedelta(days=60),
            is_return=0.10,
            is_sharpe=1.2,
            is_trades=10,
            is_win_rate=0.6,
            oos_return=0.06,
            oos_sharpe=0.9,
            oos_trades=8,
            oos_win_rate=0.55,
            overfitting_ratio=1.2,
            degradation=0.4,
        ),
        WalkForwardResult(
            fold_num=2,
            train_start=start,
            train_end=start + timedelta(days=60),
            test_start=start + timedelta(days=65),
            test_end=start + timedelta(days=90),
            is_return=0.12,
            is_sharpe=1.1,
            is_trades=12,
            is_win_rate=0.62,
            oos_return=0.07,
            oos_sharpe=0.8,
            oos_trades=9,
            oos_win_rate=0.57,
            overfitting_ratio=1.3,
            degradation=0.42,
        ),
    ]

    agg = validator._aggregate_results()

    assert "passes_validation" in agg
    assert agg["n_folds"] == 2
    assert "degradation_test" in agg
