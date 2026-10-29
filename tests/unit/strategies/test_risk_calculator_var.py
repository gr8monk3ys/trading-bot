"""
Tests for the VaR family in strategies/risk_manager/calculator.py.

``test_risk_manager.py`` covers historical VaR, ES, drawdown, position risk and
correlation. Nothing exercised the parametric, Monte-Carlo, Cornish-Fisher or
ensemble VaR methods, nor ``_compute_returns`` directly; these tests pin their
real behavior against hand-computed values.

The Cornish-Fisher expansion itself is deliberately not pinned: its skew term
has the wrong sign for the lower tail (negative skew makes the VaR *less*
severe), so only its fallback and invalid-input paths are asserted here.
"""

import numpy as np
import pytest

from strategies.risk_manager import RiskManager

SQRT_252 = np.sqrt(252)
Z_95 = 1.645
Z_99 = 2.326
INVALID_PRICE_VAR = -0.1


@pytest.fixture
def rm():
    return RiskManager()


@pytest.fixture
def gbm_prices():
    """300 days of a seeded random walk with 1% daily vol, starting at 100."""
    rng = np.random.default_rng(7)
    return (100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))).tolist()


@pytest.fixture
def crash_prices():
    """Mostly small gains with a few sharp drops: strongly negatively skewed returns."""
    rng = np.random.default_rng(11)
    daily = rng.normal(0.002, 0.003, 120)
    daily[[20, 55, 90]] = [-0.08, -0.10, -0.07]
    return (100 * np.cumprod(1 + daily)).tolist()


# =============================================================================
# _compute_returns
# =============================================================================
class TestComputeReturns:
    def test_simple_returns(self):
        returns = RiskManager._compute_returns([100.0, 110.0, 99.0])
        np.testing.assert_allclose(returns, [0.10, -0.10])

    @pytest.mark.parametrize("prices", [[], [100.0]])
    def test_fewer_than_two_prices_is_empty(self, prices):
        assert RiskManager._compute_returns(prices).size == 0

    def test_return_from_zero_price_is_dropped(self):
        # 0 -> 50 has no defined return; 50 -> 55 is +10%.
        returns = RiskManager._compute_returns([0.0, 50.0, 55.0])
        np.testing.assert_allclose(returns, [0.10])

    def test_non_finite_prices_drop_their_returns(self):
        returns = RiskManager._compute_returns([100.0, np.nan, 100.0, 105.0])
        np.testing.assert_allclose(returns, [0.05])


# =============================================================================
# Parametric VaR
# =============================================================================
class TestParametricVaR:
    def test_matches_gaussian_formula(self, rm, gbm_prices):
        returns = rm._compute_returns(gbm_prices)
        expected = (returns.mean() - Z_95 * returns.std()) * SQRT_252

        assert rm._calculate_parametric_var(gbm_prices) == pytest.approx(expected)

    def test_99_percent_uses_wider_quantile(self, rm, gbm_prices):
        returns = rm._compute_returns(gbm_prices)
        expected = (returns.mean() - Z_99 * returns.std()) * SQRT_252

        var_99 = rm._calculate_parametric_var(gbm_prices, confidence=0.99)
        assert var_99 == pytest.approx(expected)
        assert var_99 < rm._calculate_parametric_var(gbm_prices, confidence=0.95)

    def test_constant_prices_have_zero_var(self, rm):
        assert rm._calculate_parametric_var([50.0] * 10) == 0.0

    @pytest.mark.parametrize("prices", [[], [100.0]])
    def test_insufficient_history_returns_zero(self, rm, prices):
        assert rm._calculate_parametric_var(prices) == 0.0

    @pytest.mark.parametrize(
        "prices",
        [
            [100.0, 0.0, 101.0, 102.0],
            [100.0, np.nan, 101.0, 102.0],
            [100.0, np.inf, 101.0, 102.0],
        ],
    )
    def test_invalid_prices_signal_high_risk(self, rm, prices):
        assert rm._calculate_parametric_var(prices) == INVALID_PRICE_VAR


# =============================================================================
# Monte-Carlo VaR
# =============================================================================
class TestMonteCarloVaR:
    def test_short_history_falls_back_to_historical(self, rm):
        prices = [100.0, 98.0, 101.0, 97.0, 99.0, 102.0, 96.0]
        assert rm._calculate_monte_carlo_var(prices) == rm._calculate_var(prices)

    def test_zero_volatility_compounds_drift_for_one_year(self, rm):
        # A price rising exactly 1% a day has no dispersion, so every simulated
        # path compounds the mean return over 252 days.
        prices = (100 * 1.01 ** np.arange(40)).tolist()
        var_mc = rm._calculate_monte_carlo_var(prices, n_simulations=200)

        assert var_mc == pytest.approx(1.01**252 - 1, rel=1e-9)

    def test_flat_prices_have_zero_var(self, rm):
        assert rm._calculate_monte_carlo_var([25.0] * 30, n_simulations=100) == 0.0

    def test_seeded_result_is_a_one_year_loss_quantile(self, rm, gbm_prices):
        returns = rm._compute_returns(gbm_prices)
        np.random.seed(123)
        var_mc = rm._calculate_monte_carlo_var(gbm_prices, n_simulations=5000)

        np.random.seed(123)
        paths = np.random.normal(returns.mean(), returns.std(), (5000, 252))
        expected = np.percentile(np.cumprod(1 + paths, axis=1)[:, -1] - 1, 5)

        assert var_mc == pytest.approx(expected)

    def test_higher_confidence_is_more_conservative(self, rm, gbm_prices):
        np.random.seed(5)
        var_95 = rm._calculate_monte_carlo_var(gbm_prices, n_simulations=5000)
        np.random.seed(5)
        var_99 = rm._calculate_monte_carlo_var(gbm_prices, n_simulations=5000, confidence=0.99)

        assert var_99 < var_95

    def test_zero_price_in_long_history_signals_high_risk(self, rm):
        prices = [100.0 + i for i in range(25)]
        prices[10] = 0.0
        assert rm._calculate_monte_carlo_var(prices) == INVALID_PRICE_VAR


# =============================================================================
# Cornish-Fisher VaR
# =============================================================================
class TestCornishFisherVaR:
    def test_short_history_falls_back_to_parametric(self, rm):
        prices = [100.0, 98.0, 101.0, 97.0, 99.0, 102.0, 96.0, 100.0]
        assert rm._calculate_cornish_fisher_var(prices) == rm._calculate_parametric_var(prices)

    def test_nan_price_in_long_history_signals_high_risk(self, rm, gbm_prices):
        prices = list(gbm_prices[:40])
        prices[5] = np.nan
        assert rm._calculate_cornish_fisher_var(prices) == INVALID_PRICE_VAR


# =============================================================================
# Ensemble VaR
# =============================================================================
class TestVaREnsemble:
    def test_insufficient_data(self, rm):
        result = rm.calculate_var_ensemble([100.0])

        assert result == {
            "ensemble_var": 0.0,
            "method": "insufficient_data",
            "historical": 0.0,
            "parametric": 0.0,
            "monte_carlo": 0.0,
            "cornish_fisher": 0.0,
        }

    def test_picks_most_negative_estimate(self, rm, crash_prices):
        np.random.seed(42)
        result = rm.calculate_var_ensemble(crash_prices)
        estimates = {
            k: result[k] for k in ("historical", "parametric", "monte_carlo", "cornish_fisher")
        }

        assert result["ensemble_var"] == min(estimates.values())
        assert result["method"] == min(estimates, key=estimates.get)
        assert result["historical"] == rm._calculate_var(crash_prices)
        assert result["parametric"] == pytest.approx(rm._calculate_parametric_var(crash_prices))
        assert result["cornish_fisher"] == pytest.approx(
            rm._calculate_cornish_fisher_var(crash_prices)
        )

    def test_invalid_prices_propagate_high_risk(self, rm):
        prices = [100.0 + i for i in range(40)]
        prices[3] = 0.0
        result = rm.calculate_var_ensemble(prices)

        assert result["ensemble_var"] == INVALID_PRICE_VAR
        assert result["historical"] == INVALID_PRICE_VAR
        assert result["monte_carlo"] == INVALID_PRICE_VAR
        assert result["cornish_fisher"] == INVALID_PRICE_VAR
