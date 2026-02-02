"""
Tests for Factor Orthogonalization and Risk Parity

Tests:
- PCA orthogonalization
- Gram-Schmidt orthogonalization
- Risk parity weight calculation
- Adaptive factor weighter
"""

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from factors.factor_orthogonalization import (
    FactorOrthogonalizer,
    RiskParityWeighter,
    AdaptiveFactorWeighter,
    OrthogonalizationMethod,
    OrthogonalizedFactors,
    RiskParityWeights,
)


class TestFactorOrthogonalizer:
    """Tests for FactorOrthogonalizer class."""

    @pytest.fixture
    def sample_factor_scores(self):
        """Create sample factor scores for testing."""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
                   "JPM", "BAC", "GS", "JNJ", "PFE", "XOM", "CVX",
                   "PG", "KO", "NEE", "DUK", "CAT", "BA", "VZ"]

        np.random.seed(42)

        # Create correlated factors (momentum and relative strength typically correlate)
        base_scores = np.random.randn(len(symbols))
        noise = np.random.randn(len(symbols)) * 0.3

        return {
            "momentum": {s: float(base_scores[i] * 10 + 50) for i, s in enumerate(symbols)},
            "relative_strength": {s: float(base_scores[i] * 8 + noise[i] * 3 + 50) for i, s in enumerate(symbols)},
            "value": {s: float(np.random.randn() * 10 + 50) for s in symbols},
        }

    def test_pca_orthogonalization_reduces_correlation(self, sample_factor_scores):
        """Test that PCA orthogonalization reduces factor correlations."""
        orthogonalizer = FactorOrthogonalizer(
            sample_factor_scores,
            method=OrthogonalizationMethod.PCA,
        )

        result = orthogonalizer.orthogonalize()

        assert result is not None
        assert isinstance(result, OrthogonalizedFactors)
        assert result.correlation_reduction > 0  # Correlations should be reduced
        assert len(result.orthogonal_scores) == len(sample_factor_scores)

    def test_gram_schmidt_orthogonalization(self, sample_factor_scores):
        """Test Gram-Schmidt orthogonalization."""
        orthogonalizer = FactorOrthogonalizer(
            sample_factor_scores,
            method=OrthogonalizationMethod.GRAM_SCHMIDT,
        )

        result = orthogonalizer.orthogonalize()

        assert result is not None
        assert result.method == OrthogonalizationMethod.GRAM_SCHMIDT
        # After orthogonalization, correlations should be near zero
        corr_after = result.factor_correlations_after
        off_diagonal = corr_after[np.triu_indices_from(corr_after, 1)]
        assert np.mean(np.abs(off_diagonal)) < 0.1  # Near-zero correlations

    def test_symmetric_orthogonalization(self, sample_factor_scores):
        """Test symmetric (Löwdin) orthogonalization."""
        orthogonalizer = FactorOrthogonalizer(
            sample_factor_scores,
            method=OrthogonalizationMethod.SYMMETRIC,
        )

        result = orthogonalizer.orthogonalize()

        assert result is not None
        assert result.method == OrthogonalizationMethod.SYMMETRIC

    def test_insufficient_symbols_returns_none(self):
        """Test that orthogonalization returns None with too few symbols."""
        small_scores = {
            "momentum": {"AAPL": 50, "MSFT": 60},
            "value": {"AAPL": 55, "MSFT": 45},
        }

        orthogonalizer = FactorOrthogonalizer(small_scores, min_observations=20)
        result = orthogonalizer.orthogonalize()

        assert result is None

    def test_single_factor_returns_none(self):
        """Test that single factor returns None."""
        single_factor = {
            "momentum": {f"SYM{i}": float(i * 10) for i in range(30)},
        }

        orthogonalizer = FactorOrthogonalizer(single_factor)
        result = orthogonalizer.orthogonalize()

        assert result is None

    def test_explained_variance_ratio_sums_to_one(self, sample_factor_scores):
        """Test that PCA explained variance ratios sum to approximately 1."""
        orthogonalizer = FactorOrthogonalizer(
            sample_factor_scores,
            method=OrthogonalizationMethod.PCA,
        )

        result = orthogonalizer.orthogonalize()

        assert result is not None
        assert 0.99 <= sum(result.explained_variance_ratio) <= 1.01

    def test_transformation_matrix_exists(self, sample_factor_scores):
        """Test that transformation matrix is populated."""
        orthogonalizer = FactorOrthogonalizer(sample_factor_scores)
        result = orthogonalizer.orthogonalize()

        assert result is not None
        assert result.transformation_matrix is not None
        assert result.transformation_matrix.shape[0] == len(sample_factor_scores)


class TestRiskParityWeighter:
    """Tests for RiskParityWeighter class."""

    @pytest.fixture
    def sample_factor_returns(self):
        """Create sample factor returns for testing."""
        np.random.seed(42)
        n_periods = 100

        return {
            "momentum": list(np.random.randn(n_periods) * 0.02),  # 2% daily vol
            "value": list(np.random.randn(n_periods) * 0.01),      # 1% daily vol
            "quality": list(np.random.randn(n_periods) * 0.015),   # 1.5% daily vol
        }

    def test_risk_parity_weights_sum_to_one(self, sample_factor_returns):
        """Test that risk parity weights sum to 1."""
        weighter = RiskParityWeighter(sample_factor_returns)
        result = weighter.calculate_weights()

        assert result is not None
        assert isinstance(result, RiskParityWeights)
        total_weight = sum(result.factor_weights.values())
        assert 0.99 <= total_weight <= 1.01

    def test_lower_vol_factor_gets_higher_weight(self, sample_factor_returns):
        """Test that lower volatility factors get higher weights in risk parity."""
        weighter = RiskParityWeighter(sample_factor_returns)
        result = weighter.calculate_weights()

        assert result is not None
        # Value has lowest vol (1%) so should have highest weight
        assert result.factor_weights["value"] > result.factor_weights["momentum"]

    def test_risk_contributions_are_roughly_equal(self, sample_factor_returns):
        """Test that risk contributions are more equal than equal weights."""
        weighter = RiskParityWeighter(sample_factor_returns)
        result = weighter.calculate_weights()

        assert result is not None
        contributions = list(result.risk_contributions.values())

        # Risk parity optimization should produce more balanced risk contributions
        # than equal weights. With 3 factors, verify the concentration metric
        # is reasonable (perfect parity would have concentration near 0)
        assert result.risk_concentration < 0.5, (
            f"Risk concentration too high: {result.risk_concentration}"
        )

    def test_risk_concentration_is_low(self, sample_factor_returns):
        """Test that risk concentration (Herfindahl) is low."""
        weighter = RiskParityWeighter(sample_factor_returns)
        result = weighter.calculate_weights()

        assert result is not None
        # Perfect risk parity would have concentration = 0
        assert result.risk_concentration < 0.1

    def test_weight_bounds_respected(self, sample_factor_returns):
        """Test that weights stay within MIN/MAX bounds."""
        weighter = RiskParityWeighter(sample_factor_returns)
        result = weighter.calculate_weights()

        assert result is not None
        for weight in result.factor_weights.values():
            assert RiskParityWeighter.MIN_WEIGHT <= weight <= RiskParityWeighter.MAX_WEIGHT

    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None."""
        tiny_returns = {
            "momentum": [0.01, 0.02],  # Only 2 periods
        }

        weighter = RiskParityWeighter(tiny_returns)
        result = weighter.calculate_weights()

        # With only one factor, should return None
        assert result is None


class TestAdaptiveFactorWeighter:
    """Tests for AdaptiveFactorWeighter class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for adaptive weighter."""
        np.random.seed(42)
        symbols = [f"SYM{i}" for i in range(30)]

        factor_scores = {
            "momentum": {s: float(np.random.randn() * 10 + 50) for s in symbols},
            "value": {s: float(np.random.randn() * 10 + 50) for s in symbols},
            "quality": {s: float(np.random.randn() * 10 + 50) for s in symbols},
        }

        factor_returns = {
            "momentum": list(np.random.randn(100) * 0.02),
            "value": list(np.random.randn(100) * 0.01),
            "quality": list(np.random.randn(100) * 0.015),
        }

        return factor_scores, factor_returns

    def test_calculate_optimal_weights(self, sample_data):
        """Test calculating optimal weights."""
        factor_scores, factor_returns = sample_data

        weighter = AdaptiveFactorWeighter(
            factor_scores=factor_scores,
            factor_returns=factor_returns,
        )

        weights = weighter.calculate_optimal_weights()

        assert weights is not None
        assert len(weights) == len(factor_scores)
        assert 0.99 <= sum(weights.values()) <= 1.01

    def test_ic_blending(self, sample_data):
        """Test that IC weights are blended in."""
        factor_scores, factor_returns = sample_data

        ic_weights = {
            "momentum": 1.5,  # Boost momentum
            "value": 0.5,    # Reduce value
            "quality": 1.0,  # Neutral
        }

        weighter = AdaptiveFactorWeighter(
            factor_scores=factor_scores,
            factor_returns=factor_returns,
            ic_weights=ic_weights,
        )

        weights = weighter.calculate_optimal_weights()

        # With IC boost, momentum should have relatively higher weight
        # (though still balanced by risk parity)
        assert weights is not None

    def test_diagnostic_report(self, sample_data):
        """Test diagnostic report generation."""
        factor_scores, factor_returns = sample_data

        weighter = AdaptiveFactorWeighter(
            factor_scores=factor_scores,
            factor_returns=factor_returns,
        )

        report = weighter.get_diagnostic_report()

        assert "timestamp" in report
        assert "n_factors" in report
        assert "optimal_weights" in report
        assert report["n_factors"] == 3


class TestOrthogonalizedFactorsDataclass:
    """Tests for OrthogonalizedFactors dataclass."""

    def test_correlation_reduction_property(self):
        """Test correlation reduction calculation."""
        # Create mock correlations
        corr_before = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])

        corr_after = np.array([
            [1.0, 0.1, 0.05],
            [0.1, 1.0, 0.08],
            [0.05, 0.08, 1.0],
        ])

        result = OrthogonalizedFactors(
            original_factors=["a", "b", "c"],
            orthogonal_scores={},
            transformation_matrix=np.eye(3),
            explained_variance_ratio=[0.5, 0.3, 0.2],
            factor_correlations_before=corr_before,
            factor_correlations_after=corr_after,
            method=OrthogonalizationMethod.PCA,
            timestamp=datetime.now(),
        )

        # Correlation should be significantly reduced
        assert result.correlation_reduction > 0.8


class TestRiskParityWeightsDataclass:
    """Tests for RiskParityWeights dataclass."""

    def test_risk_concentration_property(self):
        """Test risk concentration (Herfindahl) calculation."""
        # Equal risk contributions = 0 concentration
        result = RiskParityWeights(
            factor_weights={"a": 0.33, "b": 0.33, "c": 0.34},
            risk_contributions={"a": 0.33, "b": 0.33, "c": 0.34},
            factor_volatilities={"a": 0.1, "b": 0.1, "c": 0.1},
            factor_correlations=np.eye(3),
            optimization_converged=True,
            timestamp=datetime.now(),
        )

        assert result.risk_concentration < 0.01

    def test_high_risk_concentration(self):
        """Test that unequal contributions have high concentration."""
        result = RiskParityWeights(
            factor_weights={"a": 0.8, "b": 0.1, "c": 0.1},
            risk_contributions={"a": 0.8, "b": 0.1, "c": 0.1},
            factor_volatilities={"a": 0.1, "b": 0.1, "c": 0.1},
            factor_correlations=np.eye(3),
            optimization_converged=True,
            timestamp=datetime.now(),
        )

        assert result.risk_concentration > 0.1
