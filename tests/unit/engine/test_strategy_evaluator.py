"""Tests for engine.strategy_evaluator.StrategyEvaluator.

Exercises real computation (no mocking of the logic under test): score
composition/weighting/penalties/clamping in score_strategy, cross-period
consistency/trend analysis in evaluate_time_periods, best-parameter selection
and params-key parsing in get_optimal_parameters, and the small value/rating
helpers.
"""

import numpy as np
import pytest

from engine.strategy_evaluator import StrategyEvaluator


def good_metrics(**overrides):
    metrics = {
        "sharpe_ratio": 3.0,
        "annualized_return": 0.30,
        "max_drawdown": 0.0,
        "win_rate": 0.65,
        "profit_factor": 2.5,
        "calmar_ratio": 1.5,
        "trade_count": 50,
        "total_return": 0.30,
    }
    metrics.update(overrides)
    return metrics


# ---------------------------------------------------------------------------
# __init__ / weight normalization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_weights_sum_to_one_and_are_unchanged(self):
        evaluator = StrategyEvaluator()

        assert evaluator.weights == {
            "sharpe_ratio": 0.25,
            "returns": 0.20,
            "drawdown": 0.15,
            "consistency": 0.20,
            "win_rate": 0.10,
            "profit_factor": 0.10,
        }

    def test_weights_not_summing_to_one_are_normalized(self):
        evaluator = StrategyEvaluator(
            weight_sharpe=1.0,
            weight_returns=1.0,
            weight_drawdown=0.0,
            weight_consistency=0.0,
            weight_win_rate=0.0,
            weight_profit_factor=0.0,
        )

        total = sum(evaluator.weights.values())
        assert total == pytest.approx(1.0)
        assert evaluator.weights["sharpe_ratio"] == pytest.approx(0.5)
        assert evaluator.weights["returns"] == pytest.approx(0.5)

    def test_weights_within_floating_point_tolerance_are_not_renormalized(self):
        # 0.99 <= total <= 1.01 is treated as "close enough" and left as-is.
        evaluator = StrategyEvaluator(
            weight_sharpe=0.25,
            weight_returns=0.20,
            weight_drawdown=0.15,
            weight_consistency=0.20,
            weight_win_rate=0.10,
            weight_profit_factor=0.095,
        )

        assert evaluator.weights["profit_factor"] == pytest.approx(0.095)


# ---------------------------------------------------------------------------
# score_strategy
# ---------------------------------------------------------------------------


class TestScoreStrategy:
    def test_empty_metrics_scores_zero(self):
        evaluator = StrategyEvaluator()
        assert evaluator.score_strategy({}) == 0.0

    def test_best_case_metrics_score_near_one(self):
        evaluator = StrategyEvaluator()

        score = evaluator.score_strategy(good_metrics())

        assert score == pytest.approx(1.0)

    def test_missing_fields_default_to_worst_case_values(self):
        evaluator = StrategyEvaluator()

        # No fields at all except a couple: sharpe/returns/win_rate/profit_factor
        # default to 0, max_drawdown defaults to 1.0 (worst case) -> drawdown
        # component score is negative, clamped to 0 by max(0, ...).
        score = evaluator.score_strategy({"trade_count": 50, "total_return": 0.1})

        assert score == pytest.approx(0.0)

    def test_negative_sharpe_scores_zero_component(self):
        evaluator = StrategyEvaluator()

        score_negative = evaluator.score_strategy(good_metrics(sharpe_ratio=-1.0))
        score_zero = evaluator.score_strategy(good_metrics(sharpe_ratio=0.0))

        # A negative Sharpe contributes 0 to that component, same as Sharpe==0.
        assert score_negative == pytest.approx(score_zero)
        assert score_negative < 1.0

    def test_trade_count_below_ten_applies_proportional_penalty(self):
        evaluator = StrategyEvaluator()

        full = evaluator.score_strategy(good_metrics(trade_count=50))
        half = evaluator.score_strategy(good_metrics(trade_count=5))

        assert half == pytest.approx(full * 0.5)

    def test_zero_trade_count_scores_zero(self):
        evaluator = StrategyEvaluator()

        score = evaluator.score_strategy(good_metrics(trade_count=0))

        assert score == pytest.approx(0.0)

    def test_negative_total_return_applies_half_penalty(self):
        evaluator = StrategyEvaluator()

        positive = evaluator.score_strategy(good_metrics(total_return=0.1))
        negative = evaluator.score_strategy(good_metrics(total_return=-0.1))

        assert negative == pytest.approx(positive * 0.5)

    def test_score_is_clamped_between_zero_and_one(self):
        evaluator = StrategyEvaluator()

        # Absurdly good inputs would otherwise push components > 1.0 before
        # the final min/max clamp.
        score = evaluator.score_strategy(
            good_metrics(
                sharpe_ratio=100,
                annualized_return=10.0,
                profit_factor=100,
                calmar_ratio=100,
            )
        )

        assert 0.0 <= score <= 1.0

    def test_exception_during_scoring_is_caught_and_returns_zero(self):
        evaluator = StrategyEvaluator()

        class ExplodingMetrics(dict):
            def get(self, *args, **kwargs):
                raise RuntimeError("boom")

        score = evaluator.score_strategy(ExplodingMetrics(a=1))

        assert score == 0.0


# ---------------------------------------------------------------------------
# evaluate_time_periods
# ---------------------------------------------------------------------------


class TestEvaluateTimePeriods:
    def test_empty_input_returns_error(self):
        evaluator = StrategyEvaluator()

        result = evaluator.evaluate_time_periods({})

        assert result == {"error": "No data provided"}

    def test_single_period_is_fully_consistent_with_zero_trend(self):
        evaluator = StrategyEvaluator()

        result = evaluator.evaluate_time_periods({"recent": good_metrics()})

        assert result["scores"]["recent"] == pytest.approx(1.0)
        assert result["average_score"] == pytest.approx(1.0)
        assert result["consistency"] == pytest.approx(1.0)
        assert result["trend"] == 0
        assert result["improving"] is False
        assert result["overall_rating"] == "A+"

    def test_trend_is_recent_minus_long_when_both_present(self):
        evaluator = StrategyEvaluator()

        result = evaluator.evaluate_time_periods(
            {
                "recent": good_metrics(),
                "long": good_metrics(sharpe_ratio=0, trade_count=0),
            }
        )

        assert result["trend"] == pytest.approx(
            result["scores"]["recent"] - result["scores"]["long"]
        )
        assert result["improving"] is True

    def test_inconsistent_scores_lower_consistency(self):
        evaluator = StrategyEvaluator()

        wildly_varying = evaluator.evaluate_time_periods(
            {
                "recent": good_metrics(),
                "medium": good_metrics(trade_count=0),
                "long": good_metrics(total_return=-0.1),
            }
        )
        stable = evaluator.evaluate_time_periods(
            {
                "recent": good_metrics(),
                "medium": good_metrics(),
                "long": good_metrics(),
            }
        )

        assert wildly_varying["consistency"] < stable["consistency"]
        assert stable["consistency"] == pytest.approx(1.0)

    def test_matches_manual_numpy_average(self):
        evaluator = StrategyEvaluator()
        metrics_by_period = {
            "recent": good_metrics(),
            "long": good_metrics(trade_count=0),
        }

        result = evaluator.evaluate_time_periods(metrics_by_period)

        expected_avg = np.mean(list(result["scores"].values()))
        assert result["average_score"] == pytest.approx(expected_avg)


# ---------------------------------------------------------------------------
# get_optimal_parameters
# ---------------------------------------------------------------------------


class TestGetOptimalParameters:
    def test_empty_results_returns_error(self):
        evaluator = StrategyEvaluator()

        assert evaluator.get_optimal_parameters({}) == {"error": "No results provided"}

    def test_no_valid_results_returns_error(self):
        evaluator = StrategyEvaluator()

        result = evaluator.get_optimal_parameters({"window=10": {"no_metrics_key": True}})

        assert result == {"error": "No valid results found"}

    def test_picks_highest_scoring_parameter_set(self):
        evaluator = StrategyEvaluator()

        results = {
            "window=10,fast=True": {"metrics": good_metrics()},
            "window=20,fast=False": {"metrics": good_metrics(trade_count=0)},
        }

        result = evaluator.get_optimal_parameters(results)

        assert result["optimal_parameters"] == {"window": 10, "fast": True}
        assert result["score"] == pytest.approx(1.0)
        assert set(result["all_scores"]) == {"window=10,fast=True", "window=20,fast=False"}

    def test_malformed_params_key_falls_back_to_raw_key(self):
        evaluator = StrategyEvaluator()

        results = {"not-a-key-value-string": {"metrics": good_metrics()}}

        result = evaluator.get_optimal_parameters(results)

        assert result["optimal_parameters"] == "not-a-key-value-string"

    def test_numeric_and_boolean_param_values_are_converted(self):
        evaluator = StrategyEvaluator()

        results = {"int_p=5,float_p=1.5,bool_p=true,str_p=hello": {"metrics": good_metrics()}}

        result = evaluator.get_optimal_parameters(results)

        assert result["optimal_parameters"] == {
            "int_p": 5,
            "float_p": 1.5,
            "bool_p": True,
            "str_p": "hello",
        }


# ---------------------------------------------------------------------------
# _convert_value
# ---------------------------------------------------------------------------


class TestConvertValue:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("false", False),
            ("False", False),
            ("5", 5),
            ("5.5", 5.5),
            ("hello", "hello"),
        ],
    )
    def test_converts_expected_types(self, raw, expected):
        evaluator = StrategyEvaluator()
        assert evaluator._convert_value(raw) == expected

    def test_dotted_non_numeric_string_falls_back_to_original(self):
        evaluator = StrategyEvaluator()
        assert evaluator._convert_value("1.2.3") == "1.2.3"


# ---------------------------------------------------------------------------
# _get_rating
# ---------------------------------------------------------------------------


class TestGetRating:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.95, "A+"),
            (0.9, "A+"),
            (0.85, "A"),
            (0.8, "A"),
            (0.75, "B+"),
            (0.7, "B+"),
            (0.65, "B"),
            (0.6, "B"),
            (0.55, "C+"),
            (0.5, "C+"),
            (0.45, "C"),
            (0.4, "C"),
            (0.35, "D"),
            (0.3, "D"),
            (0.29, "F"),
            (0.0, "F"),
        ],
    )
    def test_boundaries(self, score, expected):
        evaluator = StrategyEvaluator()
        assert evaluator._get_rating(score) == expected
