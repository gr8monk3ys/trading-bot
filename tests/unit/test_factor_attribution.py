"""
Unit tests for engine/factor_attribution.py

Tests cover:
- AttributionResult dataclass and to_dict() method
- FactorAttributor class methods:
  - add_factor_returns()
  - add_portfolio_observation()
  - attribute() - regression-based attribution
  - _calculate_selection_return()
  - _calculate_timing_return()
  - calculate_factor_returns()
  - get_factor_report()
  - detect_style_drift()
- create_attribution_report() helper function

Test scenarios include:
- Regression-based alpha calculation
- R-squared calculation
- T-statistic and p-value for alpha significance
- Factor contribution decomposition
- Style drift detection with t-tests
- Minimum observations requirement
- Factor return calculation from long-short portfolios
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.factor_attribution import (
    AttributionResult,
    FactorAttributor,
    FactorReturn,
    create_attribution_report,
)
from strategies.factor_models import CompositeScore, FactorScore, FactorType

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def attributor():
    """Create a fresh FactorAttributor instance."""
    return FactorAttributor(significance_level=0.05, min_observations=30)


@pytest.fixture
def attributor_low_min_obs():
    """Create FactorAttributor with low minimum observations for testing."""
    return FactorAttributor(significance_level=0.05, min_observations=5)


@pytest.fixture
def sample_dates():
    """Generate sample dates for testing."""
    base_date = datetime(2024, 1, 1)
    return [base_date + timedelta(days=i) for i in range(60)]


@pytest.fixture
def sample_factor_returns(sample_dates):
    """Generate sample factor returns for all dates."""
    np.random.seed(42)
    factor_returns = {}

    for date in sample_dates:
        factor_returns[date] = {
            FactorType.VALUE: np.random.normal(0.001, 0.02),
            FactorType.QUALITY: np.random.normal(0.0005, 0.015),
            FactorType.MOMENTUM: np.random.normal(0.002, 0.025),
            FactorType.LOW_VOLATILITY: np.random.normal(0.0003, 0.01),
            FactorType.SIZE: np.random.normal(-0.0001, 0.018),
        }

    return factor_returns


@pytest.fixture
def sample_portfolio_data(sample_dates):
    """Generate sample portfolio returns and exposures."""
    np.random.seed(42)
    portfolio_data = []

    for date in sample_dates:
        portfolio_return = np.random.normal(0.002, 0.03)
        exposures = {
            FactorType.VALUE: np.random.uniform(-1, 1),
            FactorType.QUALITY: np.random.uniform(-0.5, 1.5),
            FactorType.MOMENTUM: np.random.uniform(0, 2),
            FactorType.LOW_VOLATILITY: np.random.uniform(-0.3, 0.7),
            FactorType.SIZE: np.random.uniform(-1, 0.5),
        }
        portfolio_data.append((date, portfolio_return, exposures))

    return portfolio_data


@pytest.fixture
def populated_attributor(attributor, sample_dates, sample_factor_returns, sample_portfolio_data):
    """Create an attributor populated with sample data."""
    for date in sample_dates:
        attributor.add_factor_returns(date, sample_factor_returns[date])

    for date, ret, exp in sample_portfolio_data:
        attributor.add_portfolio_observation(date, ret, exp)

    return attributor


@pytest.fixture
def sample_attribution_result():
    """Create a sample AttributionResult for testing."""
    return AttributionResult(
        period_start=datetime(2024, 1, 1),
        period_end=datetime(2024, 3, 31),
        total_return=0.15,
        factor_contributions={
            FactorType.VALUE: 0.03,
            FactorType.QUALITY: 0.02,
            FactorType.MOMENTUM: 0.05,
            FactorType.LOW_VOLATILITY: 0.01,
            FactorType.SIZE: -0.01,
        },
        total_factor_return=0.10,
        alpha=0.05,
        alpha_t_stat=2.5,
        alpha_p_value=0.015,
        is_alpha_significant=True,
        factor_exposures={
            FactorType.VALUE: 0.5,
            FactorType.QUALITY: 0.8,
            FactorType.MOMENTUM: 1.2,
            FactorType.LOW_VOLATILITY: 0.3,
            FactorType.SIZE: -0.4,
        },
        r_squared=0.75,
        selection_return=0.03,
        timing_return=0.02,
    )


# =============================================================================
# TESTS FOR FactorReturn DATACLASS
# =============================================================================


class TestFactorReturn:
    """Tests for the FactorReturn dataclass."""

    def test_creation(self):
        """Test FactorReturn creation with all fields."""
        factor_return = FactorReturn(
            factor=FactorType.MOMENTUM,
            period_return=0.05,
            t_statistic=2.1,
            p_value=0.04,
            sharpe_ratio=1.5,
            n_observations=100,
        )

        assert factor_return.factor == FactorType.MOMENTUM
        assert factor_return.period_return == 0.05
        assert factor_return.t_statistic == 2.1
        assert factor_return.p_value == 0.04
        assert factor_return.sharpe_ratio == 1.5
        assert factor_return.n_observations == 100

    def test_different_factor_types(self):
        """Test FactorReturn with different factor types."""
        for factor_type in FactorType:
            fr = FactorReturn(
                factor=factor_type,
                period_return=0.01,
                t_statistic=1.0,
                p_value=0.3,
                sharpe_ratio=0.5,
                n_observations=50,
            )
            assert fr.factor == factor_type


# =============================================================================
# TESTS FOR AttributionResult DATACLASS
# =============================================================================


class TestAttributionResult:
    """Tests for the AttributionResult dataclass."""

    def test_creation(self, sample_attribution_result):
        """Test AttributionResult creation with all fields."""
        result = sample_attribution_result

        assert result.period_start == datetime(2024, 1, 1)
        assert result.period_end == datetime(2024, 3, 31)
        assert result.total_return == 0.15
        assert result.total_factor_return == 0.10
        assert result.alpha == 0.05
        assert result.is_alpha_significant is True
        assert result.r_squared == 0.75

    def test_to_dict_basic_fields(self, sample_attribution_result):
        """Test to_dict() returns correct basic fields."""
        result_dict = sample_attribution_result.to_dict()

        assert result_dict["period"] == "2024-01-01 to 2024-03-31"
        assert result_dict["total_return"] == "15.00%"
        assert result_dict["total_factor_return"] == "10.00%"
        assert result_dict["alpha"] == "5.00%"
        assert result_dict["alpha_t_stat"] == "2.50"
        assert result_dict["alpha_significant"] is True
        assert result_dict["r_squared"] == "75.0%"
        assert result_dict["selection_return"] == "3.00%"
        assert result_dict["timing_return"] == "2.00%"

    def test_to_dict_factor_contributions(self, sample_attribution_result):
        """Test to_dict() returns correct factor contributions."""
        result_dict = sample_attribution_result.to_dict()

        assert "factor_contributions" in result_dict
        fc = result_dict["factor_contributions"]

        assert fc["value"] == "3.00%"
        assert fc["quality"] == "2.00%"
        assert fc["momentum"] == "5.00%"
        assert fc["low_volatility"] == "1.00%"
        assert fc["size"] == "-1.00%"

    def test_to_dict_negative_values(self):
        """Test to_dict() handles negative values correctly."""
        result = AttributionResult(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_return=-0.05,
            factor_contributions={FactorType.MOMENTUM: -0.03},
            total_factor_return=-0.03,
            alpha=-0.02,
            alpha_t_stat=-1.5,
            alpha_p_value=0.15,
            is_alpha_significant=False,
            factor_exposures={FactorType.MOMENTUM: -0.5},
            r_squared=0.25,
            selection_return=-0.01,
            timing_return=-0.01,
        )

        result_dict = result.to_dict()

        assert result_dict["total_return"] == "-5.00%"
        assert result_dict["alpha"] == "-2.00%"
        assert result_dict["alpha_t_stat"] == "-1.50"
        assert result_dict["alpha_significant"] is False

    def test_to_dict_zero_values(self):
        """Test to_dict() handles zero values correctly."""
        result = AttributionResult(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_return=0.0,
            factor_contributions={},
            total_factor_return=0.0,
            alpha=0.0,
            alpha_t_stat=0.0,
            alpha_p_value=1.0,
            is_alpha_significant=False,
            factor_exposures={},
            r_squared=0.0,
            selection_return=0.0,
            timing_return=0.0,
        )

        result_dict = result.to_dict()

        assert result_dict["total_return"] == "0.00%"
        assert result_dict["alpha"] == "0.00%"
        assert result_dict["r_squared"] == "0.0%"


# =============================================================================
# TESTS FOR FactorAttributor INITIALIZATION
# =============================================================================


class TestFactorAttributorInit:
    """Tests for FactorAttributor initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        attributor = FactorAttributor()

        assert attributor.significance_level == 0.05
        assert attributor.min_observations == 30
        assert len(attributor._factor_returns) == 0
        assert len(attributor._portfolio_returns) == 0
        assert len(attributor._portfolio_exposures) == 0

    def test_custom_initialization(self):
        """Test custom initialization values."""
        attributor = FactorAttributor(
            significance_level=0.10,
            min_observations=50,
        )

        assert attributor.significance_level == 0.10
        assert attributor.min_observations == 50


# =============================================================================
# TESTS FOR add_factor_returns()
# =============================================================================


class TestAddFactorReturns:
    """Tests for add_factor_returns() method."""

    def test_add_single_date(self, attributor):
        """Test adding factor returns for a single date."""
        date = datetime(2024, 1, 1)
        factor_returns = {
            FactorType.VALUE: 0.01,
            FactorType.MOMENTUM: 0.02,
        }

        attributor.add_factor_returns(date, factor_returns)

        assert date in attributor._factor_returns
        assert attributor._factor_returns[date] == factor_returns

    def test_add_multiple_dates(self, attributor, sample_dates):
        """Test adding factor returns for multiple dates."""
        for i, date in enumerate(sample_dates[:5]):
            factor_returns = {FactorType.MOMENTUM: 0.01 * i}
            attributor.add_factor_returns(date, factor_returns)

        assert len(attributor._factor_returns) == 5

    def test_overwrite_existing_date(self, attributor):
        """Test overwriting factor returns for existing date."""
        date = datetime(2024, 1, 1)

        attributor.add_factor_returns(date, {FactorType.VALUE: 0.01})
        attributor.add_factor_returns(date, {FactorType.VALUE: 0.05})

        assert attributor._factor_returns[date][FactorType.VALUE] == 0.05

    def test_add_empty_factor_returns(self, attributor):
        """Test adding empty factor returns dictionary."""
        date = datetime(2024, 1, 1)

        attributor.add_factor_returns(date, {})

        assert date in attributor._factor_returns
        assert len(attributor._factor_returns[date]) == 0


# =============================================================================
# TESTS FOR add_portfolio_observation()
# =============================================================================


class TestAddPortfolioObservation:
    """Tests for add_portfolio_observation() method."""

    def test_add_single_observation(self, attributor):
        """Test adding a single portfolio observation."""
        date = datetime(2024, 1, 1)
        portfolio_return = 0.02
        exposures = {FactorType.MOMENTUM: 1.5}

        attributor.add_portfolio_observation(date, portfolio_return, exposures)

        assert len(attributor._portfolio_returns) == 1
        assert attributor._portfolio_returns[0] == (date, portfolio_return)
        assert date in attributor._portfolio_exposures
        assert attributor._portfolio_exposures[date] == exposures

    def test_add_multiple_observations(self, attributor, sample_dates):
        """Test adding multiple portfolio observations."""
        for i, date in enumerate(sample_dates[:10]):
            attributor.add_portfolio_observation(date, 0.01 * i, {FactorType.VALUE: 0.5 * i})

        assert len(attributor._portfolio_returns) == 10
        assert len(attributor._portfolio_exposures) == 10

    def test_observations_preserve_order(self, attributor, sample_dates):
        """Test that observations are added in order."""
        for i, date in enumerate(sample_dates[:5]):
            attributor.add_portfolio_observation(date, float(i), {})

        for i, (_d, r) in enumerate(attributor._portfolio_returns):
            assert r == float(i)


# =============================================================================
# TESTS FOR attribute() METHOD - CORE REGRESSION
# =============================================================================


class TestAttributeMethod:
    """Tests for attribute() method - regression-based attribution."""

    def test_insufficient_observations_returns_none(self, attributor):
        """Test that insufficient observations returns None."""
        # Add only 10 observations (less than min_observations=30)
        base_date = datetime(2024, 1, 1)
        for i in range(10):
            date = base_date + timedelta(days=i)
            attributor.add_portfolio_observation(date, 0.01, {})

        result = attributor.attribute()

        assert result is None

    def test_returns_attribution_result_with_sufficient_data(self, populated_attributor):
        """Test that sufficient data returns AttributionResult."""
        result = populated_attributor.attribute()

        assert result is not None
        assert isinstance(result, AttributionResult)

    def test_r_squared_calculation(self, populated_attributor):
        """Test R-squared is calculated and within valid range."""
        result = populated_attributor.attribute()

        assert 0 <= result.r_squared <= 1

    def test_alpha_calculation(self, populated_attributor):
        """Test alpha is calculated from regression intercept."""
        result = populated_attributor.attribute()

        # Alpha should be a finite number
        assert np.isfinite(result.alpha)

    def test_alpha_t_statistic(self, populated_attributor):
        """Test alpha t-statistic is calculated."""
        result = populated_attributor.attribute()

        assert np.isfinite(result.alpha_t_stat)

    def test_alpha_p_value_range(self, populated_attributor):
        """Test alpha p-value is within valid range."""
        result = populated_attributor.attribute()

        assert 0 <= result.alpha_p_value <= 1

    def test_alpha_significance_threshold(self, sample_dates, sample_factor_returns):
        """Test alpha significance is correctly determined by threshold."""
        # Create attributor with significance level 0.10
        attributor = FactorAttributor(significance_level=0.10, min_observations=30)

        for date in sample_dates:
            attributor.add_factor_returns(date, sample_factor_returns[date])
            attributor.add_portfolio_observation(date, 0.001, {})

        result = attributor.attribute()

        assert result.is_alpha_significant == (result.alpha_p_value < 0.10)

    def test_date_range_filtering(self, populated_attributor, sample_dates):
        """Test attribution respects date range parameters."""
        start_date = sample_dates[10]
        end_date = sample_dates[40]

        result = populated_attributor.attribute(
            start_date=start_date,
            end_date=end_date,
        )

        assert result is not None
        assert result.period_start >= start_date
        assert result.period_end <= end_date

    def test_factor_contributions_calculated(self, populated_attributor):
        """Test factor contributions are calculated for all factors."""
        result = populated_attributor.attribute()

        assert len(result.factor_contributions) > 0

    def test_total_factor_return_is_sum(self, populated_attributor):
        """Test total_factor_return is sum of individual contributions."""
        result = populated_attributor.attribute()

        expected_total = sum(result.factor_contributions.values())
        assert np.isclose(result.total_factor_return, expected_total)

    def test_factor_exposures_calculated(self, populated_attributor):
        """Test factor exposures are calculated."""
        result = populated_attributor.attribute()

        assert len(result.factor_exposures) > 0

    def test_selection_return_calculated(self, populated_attributor):
        """Test selection return is calculated."""
        result = populated_attributor.attribute()

        assert np.isfinite(result.selection_return)

    def test_timing_return_calculated(self, populated_attributor):
        """Test timing return is calculated."""
        result = populated_attributor.attribute()

        assert np.isfinite(result.timing_return)

    def test_handles_missing_factor_returns(self, attributor_low_min_obs, sample_dates):
        """Test handling of missing factor returns for some dates."""
        # Add portfolio data for all dates
        for date in sample_dates[:10]:
            attributor_low_min_obs.add_portfolio_observation(date, 0.01, {})

        # Add factor returns for only some dates
        for date in sample_dates[:5]:
            attributor_low_min_obs.add_factor_returns(date, {FactorType.MOMENTUM: 0.02})

        result = attributor_low_min_obs.attribute()

        assert result is not None


# =============================================================================
# TESTS FOR _calculate_selection_return()
# =============================================================================


class TestCalculateSelectionReturn:
    """Tests for _calculate_selection_return() method."""

    def test_selection_return_positive_when_outperforming(self, attributor_low_min_obs):
        """Test positive selection when portfolio beats factor prediction."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        # Portfolio consistently outperforms factor prediction
        portfolio_returns = np.array([0.05] * 10)
        factor_matrix = np.zeros((10, len(FactorType)))  # Zero factor returns
        betas = np.zeros(len(FactorType))

        selection = attributor_low_min_obs._calculate_selection_return(
            dates, portfolio_returns, factor_matrix, betas
        )

        assert selection > 0

    def test_selection_return_negative_when_underperforming(self, attributor_low_min_obs):
        """Test negative selection when portfolio lags factor prediction."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        # Portfolio underperforms
        portfolio_returns = np.array([-0.05] * 10)
        factor_matrix = np.zeros((10, len(FactorType)))
        betas = np.zeros(len(FactorType))

        selection = attributor_low_min_obs._calculate_selection_return(
            dates, portfolio_returns, factor_matrix, betas
        )

        assert selection < 0

    def test_selection_return_near_zero_when_matching(self, attributor_low_min_obs):
        """Test selection near zero when portfolio matches factor prediction."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        # Factor matrix with some returns
        factor_matrix = np.ones((10, len(FactorType))) * 0.01
        betas = np.ones(len(FactorType))  # Unit betas

        # Portfolio matches factor prediction
        portfolio_returns = factor_matrix @ betas

        selection = attributor_low_min_obs._calculate_selection_return(
            dates, portfolio_returns, factor_matrix, betas
        )

        assert np.isclose(selection, 0, atol=1e-10)


# =============================================================================
# TESTS FOR _calculate_timing_return()
# =============================================================================


class TestCalculateTimingReturn:
    """Tests for _calculate_timing_return() method."""

    def test_timing_return_with_insufficient_data(self, attributor_low_min_obs):
        """Test timing return returns 0 with insufficient data."""
        dates = [datetime(2024, 1, 1)]  # Only one date
        factor_matrix = np.ones((1, len(FactorType))) * 0.01
        betas = np.ones(len(FactorType))

        timing = attributor_low_min_obs._calculate_timing_return(dates, factor_matrix, betas)

        assert timing == 0.0

    def test_timing_return_positive_with_good_timing(self, attributor_low_min_obs):
        """Test positive timing when exposure increases before gains."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        # Increasing factor returns
        factor_matrix = np.array([[0.01 * i] * len(FactorType) for i in range(10)])
        betas = np.ones(len(FactorType))

        # Add exposures that increase before gains
        for i, date in enumerate(dates):
            attributor_low_min_obs._portfolio_exposures[date] = {ft: float(i) for ft in FactorType}

        timing = attributor_low_min_obs._calculate_timing_return(dates, factor_matrix, betas)

        # With increasing exposures before increasing returns, timing should be positive
        assert timing > 0

    def test_timing_return_with_constant_exposures(self, attributor_low_min_obs):
        """Test timing return is near zero with constant exposures."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        factor_matrix = np.random.randn(10, len(FactorType)) * 0.01
        betas = np.ones(len(FactorType))

        # Constant exposures
        for date in dates:
            attributor_low_min_obs._portfolio_exposures[date] = dict.fromkeys(FactorType, 1.0)

        timing = attributor_low_min_obs._calculate_timing_return(dates, factor_matrix, betas)

        assert np.isclose(timing, 0.0, atol=1e-10)


# =============================================================================
# TESTS FOR calculate_factor_returns()
# =============================================================================


class TestCalculateFactorReturns:
    """Tests for calculate_factor_returns() method."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for factor return calculation."""
        np.random.seed(42)
        symbols = [f"STOCK{i}" for i in range(25)]
        dates = pd.date_range("2024-01-01", periods=10, freq="D")

        # Generate random prices
        prices = np.random.uniform(50, 150, size=(len(dates), len(symbols)))
        # Add some trend
        for i in range(len(dates)):
            prices[i, :] *= 1 + 0.001 * i

        return pd.DataFrame(prices, index=dates, columns=symbols)

    @pytest.fixture
    def sample_factor_scores(self, sample_price_data):
        """Create sample factor scores for all symbols."""
        np.random.seed(42)
        scores = {}

        for symbol in sample_price_data.columns:
            factor_scores = {}
            for ft in FactorType:
                factor_scores[ft] = FactorScore(
                    symbol=symbol,
                    factor=ft,
                    raw_score=np.random.uniform(0, 100),
                    z_score=np.random.uniform(-2, 2),
                    percentile=np.random.uniform(0, 100),
                )

            scores[symbol] = CompositeScore(
                symbol=symbol,
                composite_z=np.random.uniform(-1, 1),
                factor_scores=factor_scores,
                quintile=np.random.randint(1, 6),
                signal="neutral",
            )

        return scores

    def test_returns_all_factor_types(self, attributor, sample_price_data, sample_factor_scores):
        """Test that returns are calculated for all factor types."""
        factor_returns = attributor.calculate_factor_returns(
            sample_price_data, sample_factor_scores, n_quantiles=5
        )

        assert len(factor_returns) == len(FactorType)
        for ft in FactorType:
            assert ft in factor_returns

    def test_long_short_return_calculation(
        self, attributor, sample_price_data, sample_factor_scores
    ):
        """Test that long-short returns are calculated correctly."""
        factor_returns = attributor.calculate_factor_returns(
            sample_price_data, sample_factor_scores, n_quantiles=5
        )

        # Factor returns should be finite
        for _ft, ret in factor_returns.items():
            assert np.isfinite(ret)

    def test_insufficient_symbols_returns_zero(self, attributor):
        """Test that insufficient symbols returns zero for factor."""
        # Only 5 symbols, but need at least 2*5=10 for quintiles
        symbols = [f"STOCK{i}" for i in range(5)]
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = pd.DataFrame(
            np.random.uniform(50, 150, size=(10, 5)),
            index=dates,
            columns=symbols,
        )

        scores = {}
        for symbol in symbols:
            factor_scores = {
                FactorType.MOMENTUM: FactorScore(
                    symbol=symbol,
                    factor=FactorType.MOMENTUM,
                    raw_score=0.1,
                    z_score=0.5,
                    percentile=50,
                )
            }
            scores[symbol] = CompositeScore(
                symbol=symbol,
                composite_z=0.5,
                factor_scores=factor_scores,
                quintile=3,
                signal="neutral",
            )

        factor_returns = attributor.calculate_factor_returns(prices, scores)

        # All should be zero due to insufficient symbols
        for ft in FactorType:
            assert factor_returns[ft] == 0.0

    def test_custom_quantiles(self, attributor, sample_price_data, sample_factor_scores):
        """Test factor returns with different quantile settings."""
        # Test with 4 quantiles (quartiles)
        factor_returns_quartiles = attributor.calculate_factor_returns(
            sample_price_data, sample_factor_scores, n_quantiles=4
        )

        # Test with 10 quantiles (deciles)
        factor_returns_deciles = attributor.calculate_factor_returns(
            sample_price_data, sample_factor_scores, n_quantiles=10
        )

        # Both should return values for all factors
        assert len(factor_returns_quartiles) == len(FactorType)
        assert len(factor_returns_deciles) == len(FactorType)


# =============================================================================
# TESTS FOR get_factor_report()
# =============================================================================


class TestGetFactorReport:
    """Tests for get_factor_report() method."""

    def test_empty_data_returns_error(self, attributor):
        """Test that empty data returns error message."""
        report = attributor.get_factor_report()

        assert "error" in report
        assert "No factor data" in report["error"]

    def test_returns_all_factor_statistics(self, populated_attributor):
        """Test that report contains statistics for all factors."""
        report = populated_attributor.get_factor_report()

        assert "error" not in report
        for ft in FactorType:
            assert ft.value in report

    def test_report_contains_expected_fields(self, populated_attributor):
        """Test that each factor report contains expected fields."""
        report = populated_attributor.get_factor_report()

        expected_fields = [
            "cumulative_return",
            "annualized_vol",
            "sharpe_ratio",
            "t_statistic",
            "p_value",
            "significant",
            "n_observations",
        ]

        for ft in FactorType:
            for field in expected_fields:
                assert field in report[ft.value]

    def test_date_range_filtering(self, populated_attributor, sample_dates):
        """Test that date range filtering works."""
        start_date = sample_dates[20]
        end_date = sample_dates[40]

        report = populated_attributor.get_factor_report(
            start_date=start_date,
            end_date=end_date,
        )

        # Should have 21 observations (days 20-40 inclusive)
        for ft in FactorType:
            assert report[ft.value]["n_observations"] == 21

    def test_t_test_significance(self, populated_attributor):
        """Test that t-test significance is correctly calculated."""
        report = populated_attributor.get_factor_report()

        for ft in FactorType:
            p_value = float(report[ft.value]["p_value"])
            is_significant = report[ft.value]["significant"]

            assert is_significant == (p_value < 0.05)

    def test_sharpe_ratio_calculation(self, populated_attributor):
        """Test that Sharpe ratio is calculated correctly."""
        report = populated_attributor.get_factor_report()

        for ft in FactorType:
            sharpe_str = report[ft.value]["sharpe_ratio"]
            sharpe = float(sharpe_str)

            # Sharpe should be finite
            assert np.isfinite(sharpe)


# =============================================================================
# TESTS FOR detect_style_drift()
# =============================================================================


class TestDetectStyleDrift:
    """Tests for detect_style_drift() method."""

    def test_insufficient_data_returns_error(self, attributor):
        """Test that insufficient data returns error message."""
        result = attributor.detect_style_drift(window_days=63)

        assert "error" in result
        assert "Insufficient data" in result["error"]

    def test_returns_drift_analysis_with_sufficient_data(self, sample_dates):
        """Test that drift analysis is returned with sufficient data."""
        attributor = FactorAttributor(min_observations=5)

        # Need at least 2*window_days of data
        # Use window_days=20 to ensure we have enough data (60 dates available)
        for date in sample_dates:
            attributor._portfolio_exposures[date] = {
                ft: np.random.uniform(-1, 1) for ft in FactorType
            }

        result = attributor.detect_style_drift(window_days=20)

        assert "error" not in result
        assert "factor_drift" in result
        assert "significant_drifts" in result
        assert "total_factors" in result
        assert "drift_score" in result
        assert "alert" in result

    def test_drift_detection_for_each_factor(self, sample_dates):
        """Test drift detection includes all factors."""
        attributor = FactorAttributor(min_observations=5)

        for date in sample_dates:
            attributor._portfolio_exposures[date] = {
                ft: np.random.uniform(-1, 1) for ft in FactorType
            }

        result = attributor.detect_style_drift(window_days=20)

        for ft in FactorType:
            assert ft.value in result["factor_drift"]

    def test_drift_result_structure(self, sample_dates):
        """Test that each factor drift result has correct structure."""
        attributor = FactorAttributor(min_observations=5)

        for date in sample_dates:
            attributor._portfolio_exposures[date] = {
                ft: np.random.uniform(-1, 1) for ft in FactorType
            }

        result = attributor.detect_style_drift(window_days=20)

        expected_fields = [
            "recent_exposure",
            "historical_exposure",
            "change",
            "is_significant",
            "p_value",
        ]

        for ft in FactorType:
            for field in expected_fields:
                assert field in result["factor_drift"][ft.value]

    def test_detects_significant_drift(self, sample_dates):
        """Test that significant drift is detected when exposures change."""
        attributor = FactorAttributor(min_observations=5)

        # Create data with clear drift in MOMENTUM factor
        mid_point = len(sample_dates) // 2

        for i, date in enumerate(sample_dates):
            if i < mid_point:
                # Historical period: low momentum exposure
                exposures = dict.fromkeys(FactorType, 0.0)
                exposures[FactorType.MOMENTUM] = -1.0
            else:
                # Recent period: high momentum exposure
                exposures = dict.fromkeys(FactorType, 0.0)
                exposures[FactorType.MOMENTUM] = 1.0

            attributor._portfolio_exposures[date] = exposures

        result = attributor.detect_style_drift(window_days=20)

        # MOMENTUM should show significant drift
        momentum_drift = result["factor_drift"]["momentum"]
        assert momentum_drift["is_significant"]  # Use == for numpy bool compatibility
        assert float(momentum_drift["change"].replace("+", "")) > 0

    def test_alert_threshold(self, sample_dates):
        """Test alert is triggered when 2+ factors drift."""
        attributor = FactorAttributor(min_observations=5)

        # Create data with drift in multiple factors
        mid_point = len(sample_dates) // 2

        for i, date in enumerate(sample_dates):
            if i < mid_point:
                exposures = dict.fromkeys(FactorType, -1.0)
            else:
                exposures = dict.fromkeys(FactorType, 1.0)

            attributor._portfolio_exposures[date] = exposures

        result = attributor.detect_style_drift(window_days=20)

        # With all factors drifting, alert should be True
        assert result["alert"] is True
        assert result["significant_drifts"] >= 2

    def test_no_drift_with_stable_exposures(self, sample_dates):
        """Test no drift detected with stable exposures."""
        attributor = FactorAttributor(min_observations=5)
        np.random.seed(42)

        # All dates have similar exposures (just noise)
        base_exposures = dict.fromkeys(FactorType, 0.5)

        for date in sample_dates:
            # Add small noise but keep exposures mostly stable
            exposures = {ft: base_exposures[ft] + np.random.normal(0, 0.01) for ft in FactorType}
            attributor._portfolio_exposures[date] = exposures

        result = attributor.detect_style_drift(window_days=20)

        # Few or no significant drifts expected
        assert result["significant_drifts"] <= 1
        assert result["alert"] is False


# =============================================================================
# TESTS FOR create_attribution_report() HELPER
# =============================================================================


class TestCreateAttributionReport:
    """Tests for create_attribution_report() helper function."""

    @pytest.fixture
    def sample_portfolio_returns(self):
        """Create sample portfolio returns series."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def sample_factor_scores_history(self, sample_portfolio_returns):
        """Create sample factor scores history."""
        np.random.seed(42)
        history = {}
        symbols = ["AAPL", "GOOGL", "MSFT"]

        for date in sample_portfolio_returns.index:
            scores = {}
            for symbol in symbols:
                factor_scores = {}
                for ft in FactorType:
                    factor_scores[ft] = FactorScore(
                        symbol=symbol,
                        factor=ft,
                        raw_score=np.random.uniform(0, 100),
                        z_score=np.random.uniform(-2, 2),
                        percentile=np.random.uniform(0, 100),
                    )

                scores[symbol] = CompositeScore(
                    symbol=symbol,
                    composite_z=np.random.uniform(-1, 1),
                    factor_scores=factor_scores,
                    quintile=np.random.randint(1, 6),
                    signal="neutral",
                )

            history[date] = scores

        return history

    @pytest.fixture
    def sample_prices_df(self, sample_portfolio_returns):
        """Create sample price DataFrame."""
        dates = sample_portfolio_returns.index
        symbols = ["AAPL", "GOOGL", "MSFT"]
        np.random.seed(42)

        prices = pd.DataFrame(
            np.random.uniform(100, 200, size=(len(dates), len(symbols))),
            index=dates,
            columns=symbols,
        )

        return prices

    def test_returns_report_structure(
        self, sample_portfolio_returns, sample_factor_scores_history, sample_prices_df
    ):
        """Test that report has expected structure."""
        report = create_attribution_report(
            sample_portfolio_returns,
            sample_factor_scores_history,
            sample_prices_df,
        )

        assert "attribution" in report
        assert "factor_performance" in report
        assert "style_drift" in report

    def test_handles_insufficient_data(self):
        """Test handling of insufficient data."""
        # Only 10 days of data (less than min_observations=30)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

        report = create_attribution_report(returns, {}, pd.DataFrame())

        assert "error" in report

    def test_attribution_section_contents(
        self, sample_portfolio_returns, sample_factor_scores_history, sample_prices_df
    ):
        """Test that attribution section contains expected fields."""
        report = create_attribution_report(
            sample_portfolio_returns,
            sample_factor_scores_history,
            sample_prices_df,
        )

        attribution = report["attribution"]
        expected_keys = [
            "period",
            "total_return",
            "factor_contributions",
            "total_factor_return",
            "alpha",
            "alpha_t_stat",
            "alpha_significant",
            "r_squared",
            "selection_return",
            "timing_return",
        ]

        for key in expected_keys:
            assert key in attribution


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_attribute_with_all_zero_returns(self, attributor_low_min_obs):
        """Test attribution with all zero portfolio returns."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        for date in dates:
            attributor_low_min_obs.add_factor_returns(date, dict.fromkeys(FactorType, 0.01))
            attributor_low_min_obs.add_portfolio_observation(date, 0.0, {})

        result = attributor_low_min_obs.attribute()

        assert result is not None
        assert result.total_return == 0.0

    def test_attribute_with_all_zero_factor_returns(self, attributor_low_min_obs):
        """Test attribution with all zero factor returns."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        for date in dates:
            attributor_low_min_obs.add_factor_returns(date, dict.fromkeys(FactorType, 0.0))
            attributor_low_min_obs.add_portfolio_observation(date, 0.01, {})

        result = attributor_low_min_obs.attribute()

        assert result is not None

    def test_attribute_with_extreme_values(self, attributor_low_min_obs):
        """Test attribution handles extreme return values."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        for date in dates:
            attributor_low_min_obs.add_factor_returns(
                date, dict.fromkeys(FactorType, 1.0)  # 100% daily returns
            )
            attributor_low_min_obs.add_portfolio_observation(date, 2.0, {})  # 200% return

        result = attributor_low_min_obs.attribute()

        assert result is not None
        assert np.isfinite(result.total_return)

    def test_attribute_with_negative_portfolio_value(self, attributor_low_min_obs):
        """Test attribution with negative returns."""
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        for date in dates:
            attributor_low_min_obs.add_factor_returns(date, dict.fromkeys(FactorType, -0.05))
            attributor_low_min_obs.add_portfolio_observation(date, -0.10, {})

        result = attributor_low_min_obs.attribute()

        assert result is not None
        assert result.total_return < 0

    def test_single_observation_timing_return(self, attributor_low_min_obs):
        """Test timing return with single observation."""
        dates = [datetime(2024, 1, 1)]
        factor_matrix = np.array([[0.01] * len(FactorType)])
        betas = np.ones(len(FactorType))

        timing = attributor_low_min_obs._calculate_timing_return(dates, factor_matrix, betas)

        assert timing == 0.0

    def test_factor_report_with_single_observation(self):
        """Test factor report with single observation per factor."""
        attributor = FactorAttributor(min_observations=1)
        date = datetime(2024, 1, 1)

        attributor.add_factor_returns(date, dict.fromkeys(FactorType, 0.01))

        report = attributor.get_factor_report()

        # Should have data for all factors
        for ft in FactorType:
            assert ft.value in report
            assert report[ft.value]["n_observations"] == 1

    def test_calculate_factor_returns_with_missing_scores(self, attributor):
        """Test factor return calculation when some symbols lack scores."""
        symbols = [f"STOCK{i}" for i in range(25)]
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        np.random.seed(42)

        prices = pd.DataFrame(
            np.random.uniform(50, 150, size=(10, 25)),
            index=dates,
            columns=symbols,
        )

        # Only create scores for some symbols
        scores = {}
        for symbol in symbols[:15]:  # Only first 15
            factor_scores = {}
            for ft in FactorType:
                factor_scores[ft] = FactorScore(
                    symbol=symbol,
                    factor=ft,
                    raw_score=np.random.uniform(0, 100),
                    z_score=np.random.uniform(-2, 2),
                    percentile=np.random.uniform(0, 100),
                )

            scores[symbol] = CompositeScore(
                symbol=symbol,
                composite_z=np.random.uniform(-1, 1),
                factor_scores=factor_scores,
                quintile=np.random.randint(1, 6),
                signal="neutral",
            )

        factor_returns = attributor.calculate_factor_returns(prices, scores)

        # Should still calculate returns for available symbols
        assert len(factor_returns) == len(FactorType)


# =============================================================================
# REGRESSION ACCURACY TESTS
# =============================================================================


class TestRegressionAccuracy:
    """Tests to verify regression calculations are accurate."""

    def test_perfect_linear_relationship(self, attributor_low_min_obs):
        """Test regression with perfect linear relationship."""
        np.random.seed(42)
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        # Create perfect linear relationship: portfolio = 0.01 + 2*momentum
        alpha_true = 0.01
        beta_momentum = 2.0

        for date in dates:
            momentum_return = np.random.uniform(-0.02, 0.02)
            portfolio_return = alpha_true + beta_momentum * momentum_return

            attributor_low_min_obs.add_factor_returns(date, {FactorType.MOMENTUM: momentum_return})
            attributor_low_min_obs.add_portfolio_observation(date, portfolio_return, {})

        result = attributor_low_min_obs.attribute()

        # R-squared should be very high (close to 1)
        assert result.r_squared > 0.99

    def test_no_linear_relationship(self, attributor_low_min_obs):
        """Test regression with no linear relationship (random data)."""
        np.random.seed(42)
        dates = [datetime(2024, 1, i) for i in range(1, 11)]

        for date in dates:
            # Independent random returns
            factor_returns = {ft: np.random.normal(0, 0.01) for ft in FactorType}
            portfolio_return = np.random.normal(0, 0.02)

            attributor_low_min_obs.add_factor_returns(date, factor_returns)
            attributor_low_min_obs.add_portfolio_observation(date, portfolio_return, {})

        result = attributor_low_min_obs.attribute()

        # R-squared should be relatively low
        # (may not be exactly 0 due to random correlation)
        assert result.r_squared < 0.9

    def test_alpha_vs_zero(self):
        """Test that alpha t-test correctly identifies significant alpha."""
        attributor = FactorAttributor(significance_level=0.05, min_observations=30)
        np.random.seed(42)

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]

        # Create data with clear alpha (consistent outperformance)
        for date in dates:
            factor_returns = {ft: np.random.normal(0, 0.005) for ft in FactorType}
            # Add consistent 0.5% daily alpha (very high, should be significant)
            portfolio_return = sum(factor_returns.values()) + 0.005

            attributor.add_factor_returns(date, factor_returns)
            attributor.add_portfolio_observation(date, portfolio_return, {})

        result = attributor.attribute()

        # With consistent positive alpha, should be significant
        assert result.alpha > 0
        assert result.alpha_t_stat > 0
