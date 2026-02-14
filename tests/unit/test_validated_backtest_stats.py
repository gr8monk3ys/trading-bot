import pandas as pd

from engine.validated_backtest import ValidatedBacktestRunner


def test_calculate_significance_returns_false_for_small_sample():
    runner = ValidatedBacktestRunner(broker=None)
    sig, p_value = runner._calculate_significance(pd.Series([0.01] * 10))

    assert sig is False
    assert p_value is None


def test_calculate_permutation_tests_handles_none():
    runner = ValidatedBacktestRunner(broker=None)
    result = runner._calculate_permutation_tests(None)

    assert "error" in result


def test_calculate_permutation_tests_insufficient_returns():
    runner = ValidatedBacktestRunner(broker=None)
    result = runner._calculate_permutation_tests(pd.Series([0.01] * 5))

    assert "error" in result


def test_calculate_permutation_tests_fdr(monkeypatch):
    runner = ValidatedBacktestRunner(broker=None)

    def _fake_perm(_returns, statistic="mean", alpha=0.05):
        class _Res:
            def __init__(self):
                self.p_value = 0.01 if statistic == "mean" else 0.02
                self.n_permutations = 1000

        return _Res()

    monkeypatch.setattr("engine.validated_backtest.permutation_test_returns", _fake_perm)
    monkeypatch.setattr(
        "engine.validated_backtest.BACKTEST_PARAMS",
        {"MULTIPLE_TESTING_METHOD": "fdr", "PERMUTATION_P_THRESHOLD": 0.05},
    )

    result = runner._calculate_permutation_tests(pd.Series([0.01] * 20))

    assert "tests" in result
