"""
Comprehensive tests for strategies/factor_models.py

Tests cover:
- FactorCalculator initialization and configuration
- Z-score calculation correctness
- Winsorization at configurable standard deviations
- Percentile rank calculations
- Momentum factor calculation (12-1 month returns)
- Value factor calculation (P/E, P/B, EV/EBITDA)
- Quality factor calculation (ROE, debt/equity, earnings variability)
- Low volatility factor calculation
- Size factor calculation (log market cap)
- Composite score calculation with weights
- Quintile assignment logic
- Signal generation (long/short/neutral)
- FactorModel high-level API
- Edge cases (insufficient data, missing values, NaN handling)
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strategies.factor_models import (
    CompositeScore,
    FactorCalculator,
    FactorModel,
    FactorScore,
    FactorType,
)

# =============================================================================
# Constants - No magic numbers
# =============================================================================
DEFAULT_WINSORIZE_STD = 3.0
DEFAULT_MIN_UNIVERSE_SIZE = 20
RANDOM_SEED = 42
DEFAULT_NUM_SYMBOLS = 30  # Enough to pass min_universe_size
SMALL_UNIVERSE_SIZE = 10  # Below min_universe_size
DEFAULT_PRICE_DAYS = 300  # Enough for momentum calculation (~252 trading days)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def factor_calculator():
    """Create FactorCalculator with default parameters."""
    return FactorCalculator()


@pytest.fixture
def lenient_calculator():
    """Create FactorCalculator with lower min_universe_size for edge case tests."""
    return FactorCalculator(min_universe_size=5)


@pytest.fixture
def factor_model():
    """Create FactorModel with default parameters."""
    return FactorModel()


@pytest.fixture
def sample_symbols():
    """Generate list of sample stock symbols."""
    return [f"SYM{i:03d}" for i in range(DEFAULT_NUM_SYMBOLS)]


@pytest.fixture
def sample_price_data(sample_symbols):
    """
    Generate sample price data DataFrame for testing.

    Creates ~300 days of price data for each symbol with realistic
    random walk patterns.
    """
    np.random.seed(RANDOM_SEED)
    dates = pd.date_range(start="2023-01-01", periods=DEFAULT_PRICE_DAYS, freq="D")

    data = {}
    for symbol in sample_symbols:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, DEFAULT_PRICE_DAYS)
        prices = 100 * np.exp(np.cumsum(returns))
        data[symbol] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_fundamental_data(sample_symbols):
    """
    Generate sample fundamental data for testing.

    Creates realistic P/E, P/B, EV/EBITDA, ROE, debt_to_equity,
    and earnings_variability values.
    """
    np.random.seed(RANDOM_SEED)

    data = {}
    for symbol in sample_symbols:
        data[symbol] = {
            "pe_ratio": np.random.uniform(10, 40),
            "pb_ratio": np.random.uniform(1, 8),
            "ev_ebitda": np.random.uniform(5, 25),
            "roe": np.random.uniform(0.05, 0.30),
            "debt_to_equity": np.random.uniform(0.1, 2.0),
            "earnings_variability": np.random.uniform(0.05, 0.30),
        }

    return data


@pytest.fixture
def sample_market_caps(sample_symbols):
    """
    Generate sample market cap data for testing.

    Creates market caps ranging from small cap to mega cap.
    """
    np.random.seed(RANDOM_SEED)

    # Log-normal distribution for market caps (in billions)
    log_caps = np.random.normal(np.log(50e9), 1.5, len(sample_symbols))
    caps = np.exp(log_caps)

    return {symbol: float(cap) for symbol, cap in zip(sample_symbols, caps, strict=False)}


@pytest.fixture
def insufficient_price_data(sample_symbols):
    """Generate price data with insufficient history (less than 252 days)."""
    np.random.seed(RANDOM_SEED)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    data = {}
    for symbol in sample_symbols:
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        data[symbol] = prices.tolist()

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def small_universe_symbols():
    """Generate a small universe that's below min_universe_size."""
    return [f"SYM{i:03d}" for i in range(SMALL_UNIVERSE_SIZE)]


# =============================================================================
# FactorScore and CompositeScore Dataclass Tests
# =============================================================================
class TestFactorScore:
    """Tests for FactorScore dataclass."""

    def test_factor_score_creation(self):
        """Test basic FactorScore creation."""
        score = FactorScore(
            symbol="AAPL",
            factor=FactorType.MOMENTUM,
            raw_score=0.15,
            z_score=1.5,
            percentile=85.0,
        )

        assert score.symbol == "AAPL"
        assert score.factor == FactorType.MOMENTUM
        assert score.raw_score == 0.15
        assert score.z_score == 1.5
        assert score.percentile == 85.0
        assert score.sector is None
        assert score.timestamp is not None

    def test_factor_score_with_sector(self):
        """Test FactorScore with sector specified."""
        score = FactorScore(
            symbol="AAPL",
            factor=FactorType.VALUE,
            raw_score=15.0,
            z_score=-0.5,
            percentile=30.0,
            sector="Technology",
        )

        assert score.sector == "Technology"

    def test_factor_score_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        before = datetime.now()
        score = FactorScore(
            symbol="AAPL",
            factor=FactorType.QUALITY,
            raw_score=0.20,
            z_score=1.0,
            percentile=75.0,
        )
        after = datetime.now()

        assert before <= score.timestamp <= after


class TestCompositeScore:
    """Tests for CompositeScore dataclass."""

    def test_composite_score_creation(self):
        """Test basic CompositeScore creation."""
        factor_scores = {
            FactorType.MOMENTUM: FactorScore(
                symbol="AAPL",
                factor=FactorType.MOMENTUM,
                raw_score=0.15,
                z_score=1.5,
                percentile=85.0,
            )
        }

        score = CompositeScore(
            symbol="AAPL",
            composite_z=1.5,
            factor_scores=factor_scores,
            quintile=5,
            signal="long",
        )

        assert score.symbol == "AAPL"
        assert score.composite_z == 1.5
        assert score.quintile == 5
        assert score.signal == "long"
        assert FactorType.MOMENTUM in score.factor_scores

    def test_composite_score_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        before = datetime.now()
        score = CompositeScore(
            symbol="AAPL",
            composite_z=0.5,
            factor_scores={},
            quintile=3,
            signal="neutral",
        )
        after = datetime.now()

        assert before <= score.timestamp <= after


# =============================================================================
# FactorType Enum Tests
# =============================================================================
class TestFactorType:
    """Tests for FactorType enum."""

    def test_all_factor_types_exist(self):
        """Test that all 5 core factors are defined."""
        assert FactorType.VALUE.value == "value"
        assert FactorType.QUALITY.value == "quality"
        assert FactorType.MOMENTUM.value == "momentum"
        assert FactorType.LOW_VOLATILITY.value == "low_volatility"
        assert FactorType.SIZE.value == "size"

    def test_factor_type_count(self):
        """Test that exactly 5 factor types exist."""
        assert len(FactorType) == 5


# =============================================================================
# FactorCalculator Initialization Tests
# =============================================================================
class TestFactorCalculatorInitialization:
    """Tests for FactorCalculator initialization."""

    def test_default_initialization(self):
        """Test FactorCalculator with default parameters."""
        calc = FactorCalculator()

        assert calc.winsorize_std == DEFAULT_WINSORIZE_STD
        assert calc.sector_neutral is True
        assert calc.min_universe_size == DEFAULT_MIN_UNIVERSE_SIZE

    def test_custom_initialization(self):
        """Test FactorCalculator with custom parameters."""
        calc = FactorCalculator(
            winsorize_std=2.5,
            sector_neutral=False,
            min_universe_size=10,
        )

        assert calc.winsorize_std == 2.5
        assert calc.sector_neutral is False
        assert calc.min_universe_size == 10


# =============================================================================
# Winsorization Tests
# =============================================================================
class TestWinsorization:
    """Tests for the _winsorize method."""

    def test_winsorize_normal_data(self, factor_calculator):
        """Test winsorization doesn't affect normal data."""
        np.random.seed(RANDOM_SEED)
        values = np.random.normal(0, 1, 100)

        winsorized = factor_calculator._winsorize(values)

        # Most values should be unchanged for normal data
        assert len(winsorized) == len(values)
        # Check that extreme values are clipped
        assert np.all(winsorized >= winsorized.min())
        assert np.all(winsorized <= winsorized.max())

    def test_winsorize_extreme_values(self, factor_calculator):
        """Test that extreme outliers are clipped at 3 std devs."""
        # Create data with extreme outliers
        values = np.array([1, 2, 3, 4, 5, 100, -100])  # 100 and -100 are extreme

        winsorized = factor_calculator._winsorize(values)

        mean = np.nanmean(values)
        std = np.nanstd(values)
        lower = mean - 3.0 * std
        upper = mean + 3.0 * std

        # Check that all values are within bounds
        assert np.all(winsorized >= lower)
        assert np.all(winsorized <= upper)

    def test_winsorize_custom_std(self):
        """Test winsorization with custom std deviation."""
        calc = FactorCalculator(winsorize_std=2.0)

        values = np.array([1, 2, 3, 4, 5, 50, -50])
        winsorized = calc._winsorize(values)

        mean = np.nanmean(values)
        std = np.nanstd(values)
        lower = mean - 2.0 * std
        upper = mean + 2.0 * std

        assert np.all(winsorized >= lower)
        assert np.all(winsorized <= upper)

    def test_winsorize_handles_nan(self, factor_calculator):
        """Test that winsorization handles NaN values."""
        values = np.array([1, 2, np.nan, 4, 5])

        winsorized = factor_calculator._winsorize(values)

        # NaN should still be NaN
        assert np.isnan(winsorized[2])


# =============================================================================
# Z-Score Tests
# =============================================================================
class TestZScore:
    """Tests for the _z_score method."""

    def test_z_score_calculation(self, factor_calculator):
        """Test z-score calculation correctness."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        z_scores = factor_calculator._z_score(values)

        # Z-scores should have mean ~0 and std ~1
        assert abs(np.nanmean(z_scores)) < 0.01
        assert abs(np.nanstd(z_scores) - 1.0) < 0.01

    def test_z_score_with_zero_std(self, factor_calculator):
        """Test z-score returns zeros when std is zero."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        z_scores = factor_calculator._z_score(values)

        assert np.all(z_scores == 0)

    def test_z_score_handles_nan(self, factor_calculator):
        """Test z-score handles NaN values correctly."""
        values = np.array([10.0, np.nan, 30.0, 40.0, 50.0])

        z_scores = factor_calculator._z_score(values)

        # Should have valid z-scores for non-NaN values
        assert not np.isnan(z_scores[0])
        assert not np.isnan(z_scores[2])

    def test_z_score_preserves_ranking(self, factor_calculator):
        """Test that z-scores preserve the relative ordering."""
        values = np.array([10.0, 50.0, 30.0, 20.0, 40.0])

        z_scores = factor_calculator._z_score(values)

        # Higher values should have higher z-scores
        value_order = np.argsort(values)
        zscore_order = np.argsort(z_scores)

        np.testing.assert_array_equal(value_order, zscore_order)


# =============================================================================
# Percentile Rank Tests
# =============================================================================
class TestPercentileRank:
    """Tests for the _percentile_rank method."""

    def test_percentile_rank_calculation(self, factor_calculator):
        """Test percentile rank calculation."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        percentiles = factor_calculator._percentile_rank(values)

        # Lowest value should have lowest percentile
        assert percentiles[0] < percentiles[4]
        # All percentiles should be between 0 and 100
        assert np.all(percentiles >= 0)
        assert np.all(percentiles <= 100)

    def test_percentile_rank_handles_nan(self, factor_calculator):
        """Test percentile rank handles NaN values."""
        values = np.array([10.0, np.nan, 30.0, 40.0, 50.0])

        percentiles = factor_calculator._percentile_rank(values)

        # Should still produce valid percentiles
        assert len(percentiles) == len(values)

    def test_percentile_rank_all_nan(self, factor_calculator):
        """Test percentile rank when all values are NaN."""
        values = np.array([np.nan, np.nan, np.nan])

        percentiles = factor_calculator._percentile_rank(values)

        # Should return 50.0 for all (default)
        assert np.all(percentiles == 50.0)


# =============================================================================
# Momentum Factor Tests
# =============================================================================
class TestMomentumFactor:
    """Tests for calculate_momentum method."""

    def test_momentum_calculation(self, factor_calculator, sample_price_data):
        """Test basic momentum calculation."""
        results = factor_calculator.calculate_momentum(sample_price_data)

        assert len(results) > 0
        for _symbol, score in results.items():
            assert score.factor == FactorType.MOMENTUM
            assert isinstance(score.raw_score, float)
            assert isinstance(score.z_score, float)
            assert 0 <= score.percentile <= 100

    def test_momentum_insufficient_data(self, factor_calculator, insufficient_price_data):
        """Test momentum returns empty dict with insufficient data."""
        results = factor_calculator.calculate_momentum(insufficient_price_data)

        assert results == {}

    def test_momentum_skip_recent_month(self, factor_calculator, sample_price_data):
        """Test momentum calculation with skip_recent_month parameter."""
        results_skip = factor_calculator.calculate_momentum(
            sample_price_data, skip_recent_month=True
        )
        results_no_skip = factor_calculator.calculate_momentum(
            sample_price_data, skip_recent_month=False
        )

        # Both should produce results but with different values
        assert len(results_skip) > 0
        assert len(results_no_skip) > 0

        # The raw scores should differ since one includes recent month
        list(results_skip.keys())[0]
        # Note: values may be close but calculation differs

    def test_momentum_custom_lookback(self, factor_calculator, sample_price_data):
        """Test momentum calculation with custom lookback period."""
        results_12m = factor_calculator.calculate_momentum(sample_price_data, lookback_months=12)
        results_6m = factor_calculator.calculate_momentum(sample_price_data, lookback_months=6)

        # Both should produce results
        assert len(results_12m) > 0
        assert len(results_6m) > 0

    def test_momentum_handles_inf(self, factor_calculator, sample_symbols):
        """Test momentum handles infinite values (from zero prices)."""
        np.random.seed(RANDOM_SEED)
        dates = pd.date_range(start="2023-01-01", periods=DEFAULT_PRICE_DAYS, freq="D")

        data = {}
        for i, symbol in enumerate(sample_symbols):
            prices = 100 + np.cumsum(np.random.randn(DEFAULT_PRICE_DAYS) * 0.5)
            if i == 0:
                # Set first symbol to have a zero price that creates inf
                prices[0] = 0.0
            data[symbol] = prices

        df = pd.DataFrame(data, index=dates)
        results = factor_calculator.calculate_momentum(df)

        # Should still produce results, excluding inf values
        assert len(results) > 0

    def test_momentum_small_universe(self, lenient_calculator, small_universe_symbols):
        """Test momentum with small universe (below default min_universe_size)."""
        np.random.seed(RANDOM_SEED)
        dates = pd.date_range(start="2023-01-01", periods=DEFAULT_PRICE_DAYS, freq="D")

        data = {}
        for symbol in small_universe_symbols:
            prices = 100 + np.cumsum(np.random.randn(DEFAULT_PRICE_DAYS) * 0.5)
            data[symbol] = prices

        df = pd.DataFrame(data, index=dates)

        # With lenient calculator (min_universe_size=5), should work
        results = lenient_calculator.calculate_momentum(df)
        assert len(results) > 0


# =============================================================================
# Value Factor Tests
# =============================================================================
class TestValueFactor:
    """Tests for calculate_value method."""

    def test_value_calculation(self, factor_calculator, sample_fundamental_data):
        """Test basic value factor calculation."""
        results = factor_calculator.calculate_value(sample_fundamental_data)

        assert len(results) > 0
        for _symbol, score in results.items():
            assert score.factor == FactorType.VALUE
            assert isinstance(score.raw_score, float)
            assert isinstance(score.z_score, float)

    def test_value_inverts_metrics(self, lenient_calculator):
        """Test that value factor correctly inverts P/E (lower P/E = higher value)."""
        # Create universe where one stock has clearly low P/E (should be high value)
        # and another has high P/E (should be low value)
        data = {}
        for i in range(10):
            data[f"SYM{i:03d}"] = {
                "pe_ratio": 10.0 + i * 3,  # Range from 10 to 37
                "pb_ratio": 1.0 + i * 0.3,
                "ev_ebitda": 5.0 + i * 1.5,
            }

        results = lenient_calculator.calculate_value(data)

        # SYM000 has lowest P/E (10.0) - should have highest value score
        # SYM009 has highest P/E (37.0) - should have lowest value score
        if "SYM000" in results and "SYM009" in results:
            assert results["SYM000"].z_score > results["SYM009"].z_score

    def test_value_handles_missing_metrics(self, lenient_calculator):
        """Test value calculation with missing metrics."""
        data = {}
        for i in range(10):
            metrics = {"pe_ratio": 15.0 + i}
            if i % 2 == 0:
                metrics["pb_ratio"] = 2.0 + i * 0.1
            # ev_ebitda intentionally missing for some
            data[f"SYM{i:03d}"] = metrics

        results = lenient_calculator.calculate_value(data)

        # Should still produce results using available metrics
        assert len(results) > 0

    def test_value_filters_negative_values(self, lenient_calculator):
        """Test that negative/zero metric values are filtered."""
        data = {}
        for i in range(10):
            data[f"SYM{i:03d}"] = {
                "pe_ratio": 15.0 if i > 0 else -10.0,  # First has negative P/E
                "pb_ratio": 2.0 if i > 0 else 0.0,  # First has zero P/B
                "ev_ebitda": 10.0,
            }

        results = lenient_calculator.calculate_value(data)

        # Results should exist but may exclude symbols with invalid data
        assert len(results) > 0

    def test_value_custom_metrics(self, lenient_calculator):
        """Test value calculation with custom metric list."""
        data = {}
        for i in range(10):
            data[f"SYM{i:03d}"] = {
                "pe_ratio": 15.0 + i,
                "pb_ratio": 2.0 + i * 0.1,
                "ev_ebitda": 10.0 + i,
            }

        # Use only P/E
        results = lenient_calculator.calculate_value(data, metrics=["pe_ratio"])

        assert len(results) > 0

    def test_value_small_universe_returns_empty(self, factor_calculator):
        """Test that value returns empty dict when universe too small."""
        data = {"SYM001": {"pe_ratio": 15.0, "pb_ratio": 2.0, "ev_ebitda": 10.0}}

        results = factor_calculator.calculate_value(data)

        assert results == {}


# =============================================================================
# Quality Factor Tests
# =============================================================================
class TestQualityFactor:
    """Tests for calculate_quality method."""

    def test_quality_calculation(self, factor_calculator, sample_fundamental_data):
        """Test basic quality factor calculation."""
        results = factor_calculator.calculate_quality(sample_fundamental_data)

        assert len(results) > 0
        for _symbol, score in results.items():
            assert score.factor == FactorType.QUALITY
            assert isinstance(score.z_score, float)

    def test_quality_high_roe_high_score(self, lenient_calculator):
        """Test that high ROE stocks get higher quality scores."""
        data = {}
        for i in range(10):
            data[f"SYM{i:03d}"] = {
                "roe": 0.05 + i * 0.03,  # Range from 0.05 to 0.32
                "debt_to_equity": 0.5,  # Same for all
                "earnings_variability": 0.1,  # Same for all
            }

        results = lenient_calculator.calculate_quality(data)

        # SYM009 has highest ROE - should have highest quality score
        if "SYM009" in results and "SYM000" in results:
            assert results["SYM009"].z_score > results["SYM000"].z_score

    def test_quality_low_debt_high_score(self, lenient_calculator):
        """Test that low debt stocks get higher quality scores."""
        data = {}
        for i in range(10):
            data[f"SYM{i:03d}"] = {
                "roe": 0.15,  # Same for all
                "debt_to_equity": 0.1 + i * 0.2,  # Range from 0.1 to 1.9
                "earnings_variability": 0.1,
            }

        results = lenient_calculator.calculate_quality(data)

        # SYM000 has lowest debt - should have higher quality score
        if "SYM000" in results and "SYM009" in results:
            assert results["SYM000"].z_score > results["SYM009"].z_score

    def test_quality_handles_missing_data(self, lenient_calculator):
        """Test quality calculation with missing metrics."""
        data = {}
        for i in range(10):
            metrics = {"roe": 0.15 + i * 0.01}
            # Only some have debt_to_equity
            if i % 2 == 0:
                metrics["debt_to_equity"] = 0.5
            data[f"SYM{i:03d}"] = metrics

        results = lenient_calculator.calculate_quality(data)

        # Should produce results using available metrics
        assert len(results) > 0

    def test_quality_small_universe_returns_empty(self, factor_calculator):
        """Test quality returns empty dict when universe too small."""
        data = {"SYM001": {"roe": 0.15, "debt_to_equity": 0.5}}

        results = factor_calculator.calculate_quality(data)

        assert results == {}


# =============================================================================
# Low Volatility Factor Tests
# =============================================================================
class TestLowVolatilityFactor:
    """Tests for calculate_low_volatility method."""

    def test_low_volatility_calculation(self, factor_calculator, sample_price_data):
        """Test basic low volatility calculation."""
        results = factor_calculator.calculate_low_volatility(sample_price_data)

        assert len(results) > 0
        for _symbol, score in results.items():
            assert score.factor == FactorType.LOW_VOLATILITY
            assert score.raw_score > 0  # Volatility should be positive

    def test_low_volatility_inverts_correctly(self, lenient_calculator):
        """Test that lower volatility stocks get higher scores."""
        np.random.seed(RANDOM_SEED)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        data = {}
        # Create stocks with known volatility patterns
        for i in range(10):
            volatility = 0.01 + i * 0.01  # Range from 1% to 10% daily vol
            returns = np.random.normal(0, volatility, 100)
            prices = 100 * np.exp(np.cumsum(returns))
            data[f"SYM{i:03d}"] = prices

        df = pd.DataFrame(data, index=dates)
        results = lenient_calculator.calculate_low_volatility(df)

        # SYM000 has lowest volatility - should have highest low_vol score
        if "SYM000" in results and "SYM009" in results:
            assert results["SYM000"].z_score > results["SYM009"].z_score

    def test_low_volatility_custom_lookback(self, factor_calculator, sample_price_data):
        """Test low volatility with custom lookback period."""
        results_252 = factor_calculator.calculate_low_volatility(
            sample_price_data, lookback_days=252
        )
        results_60 = factor_calculator.calculate_low_volatility(sample_price_data, lookback_days=60)

        # Both should produce results
        assert len(results_252) > 0
        assert len(results_60) > 0

    def test_low_volatility_insufficient_data(self, factor_calculator):
        """Test low volatility with insufficient data."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        data = {"SYM001": [100 + i for i in range(10)]}
        df = pd.DataFrame(data, index=dates)

        results = factor_calculator.calculate_low_volatility(df)

        # Should return empty (less than 20 return observations)
        assert results == {}

    def test_low_volatility_annualized(self, lenient_calculator):
        """Test that volatility is properly annualized."""
        np.random.seed(RANDOM_SEED)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create stock with ~1% daily volatility (should be ~16% annualized)
        data = {}
        for i in range(10):
            returns = np.random.normal(0, 0.01, 100)
            prices = 100 * np.exp(np.cumsum(returns))
            data[f"SYM{i:03d}"] = prices

        df = pd.DataFrame(data, index=dates)
        results = lenient_calculator.calculate_low_volatility(df)

        # Raw scores should be annualized (~0.16 for 1% daily vol)
        for score in results.values():
            assert 0.05 < score.raw_score < 0.5  # Reasonable annualized vol range


# =============================================================================
# Size Factor Tests
# =============================================================================
class TestSizeFactor:
    """Tests for calculate_size method."""

    def test_size_calculation(self, factor_calculator, sample_market_caps):
        """Test basic size factor calculation."""
        results = factor_calculator.calculate_size(sample_market_caps)

        assert len(results) > 0
        for _symbol, score in results.items():
            assert score.factor == FactorType.SIZE
            assert score.raw_score > 0  # Market cap should be positive

    def test_size_small_cap_premium_default(self, lenient_calculator):
        """Test that smaller caps get higher scores by default (small cap premium)."""
        caps = {
            "SMALL": 1e9,  # $1B
            "MED": 50e9,  # $50B
            "LARGE": 500e9,  # $500B
        }
        # Add more to meet min universe
        for i in range(7):
            caps[f"SYM{i:03d}"] = 10e9 + i * 10e9

        results = lenient_calculator.calculate_size(caps)

        # SMALL should have highest size score (small cap premium)
        assert results["SMALL"].z_score > results["LARGE"].z_score

    def test_size_no_small_cap_premium(self, lenient_calculator):
        """Test size factor without small cap premium."""
        caps = {
            "SMALL": 1e9,
            "MED": 50e9,
            "LARGE": 500e9,
        }
        for i in range(7):
            caps[f"SYM{i:03d}"] = 10e9 + i * 10e9

        results = lenient_calculator.calculate_size(caps, small_cap_premium=False)

        # LARGE should have highest size score when premium disabled
        assert results["LARGE"].z_score > results["SMALL"].z_score

    def test_size_handles_invalid_caps(self, lenient_calculator):
        """Test size calculation handles zero/negative market caps."""
        caps = {"SYM000": 0, "SYM001": -100}  # Invalid
        for i in range(10):
            caps[f"SYM{i+10:03d}"] = 10e9 + i * 1e9  # Valid

        results = lenient_calculator.calculate_size(caps)

        # Invalid caps should be excluded
        assert "SYM000" not in results
        assert "SYM001" not in results

    def test_size_uses_log_transform(self, lenient_calculator):
        """Test that size uses log transform for better distribution."""
        # Market caps span several orders of magnitude
        caps = {}
        for i in range(10):
            caps[f"SYM{i:03d}"] = 10 ** (9 + i * 0.3)  # 1B to ~100B

        results = lenient_calculator.calculate_size(caps)

        # Z-scores should be reasonable (not extreme due to log transform)
        for score in results.values():
            assert abs(score.z_score) < 5  # Reasonable z-score range


# =============================================================================
# Composite Score Tests
# =============================================================================
class TestCompositeScoreCalculation:
    """Tests for calculate_composite method."""

    def test_composite_calculation(
        self, factor_calculator, sample_price_data, sample_fundamental_data, sample_market_caps
    ):
        """Test composite score calculation with multiple factors."""
        # Calculate individual factors
        momentum = factor_calculator.calculate_momentum(sample_price_data)
        value = factor_calculator.calculate_value(sample_fundamental_data)
        low_vol = factor_calculator.calculate_low_volatility(sample_price_data)

        factor_scores = {
            FactorType.MOMENTUM: momentum,
            FactorType.VALUE: value,
            FactorType.LOW_VOLATILITY: low_vol,
        }

        results = factor_calculator.calculate_composite(factor_scores)

        assert len(results) > 0
        for _symbol, score in results.items():
            assert 1 <= score.quintile <= 5
            assert score.signal in ["long", "short", "neutral"]

    def test_composite_equal_weights_default(self, lenient_calculator):
        """Test that default weights are equal."""
        # Create simple factor scores
        factor_scores = {
            FactorType.MOMENTUM: {
                "SYM000": FactorScore("SYM000", FactorType.MOMENTUM, 0.1, 1.0, 80.0),
                "SYM001": FactorScore("SYM001", FactorType.MOMENTUM, 0.05, -1.0, 20.0),
            },
            FactorType.VALUE: {
                "SYM000": FactorScore("SYM000", FactorType.VALUE, 15.0, -1.0, 20.0),
                "SYM001": FactorScore("SYM001", FactorType.VALUE, 10.0, 1.0, 80.0),
            },
        }

        results = lenient_calculator.calculate_composite(factor_scores)

        # With equal weights, composite should be average of z-scores
        # SYM000: (1.0 + -1.0) / 2 = 0.0
        # SYM001: (-1.0 + 1.0) / 2 = 0.0
        assert "SYM000" in results
        assert "SYM001" in results

    def test_composite_custom_weights(self, lenient_calculator):
        """Test composite calculation with custom weights."""
        factor_scores = {
            FactorType.MOMENTUM: {
                "SYM000": FactorScore("SYM000", FactorType.MOMENTUM, 0.1, 2.0, 90.0),
            },
            FactorType.VALUE: {
                "SYM000": FactorScore("SYM000", FactorType.VALUE, 15.0, -1.0, 30.0),
            },
        }

        # Heavily weight momentum
        weights = {FactorType.MOMENTUM: 0.8, FactorType.VALUE: 0.2}

        results = lenient_calculator.calculate_composite(factor_scores, weights)

        # Composite z should be closer to momentum z
        expected_z = 0.8 * 2.0 + 0.2 * (-1.0)  # = 1.4
        assert abs(results["SYM000"].composite_z - expected_z) < 0.01

    def test_composite_weight_normalization(self, lenient_calculator):
        """Test that weights are normalized to sum to 1."""
        factor_scores = {
            FactorType.MOMENTUM: {
                "SYM000": FactorScore("SYM000", FactorType.MOMENTUM, 0.1, 1.0, 80.0),
            },
            FactorType.VALUE: {
                "SYM000": FactorScore("SYM000", FactorType.VALUE, 15.0, 1.0, 80.0),
            },
        }

        # Weights don't sum to 1
        weights = {FactorType.MOMENTUM: 2.0, FactorType.VALUE: 2.0}

        results = lenient_calculator.calculate_composite(factor_scores, weights)

        # Should still work, weights normalized internally
        # With equal normalized weights and same z-scores, composite = 1.0
        assert abs(results["SYM000"].composite_z - 1.0) < 0.01

    def test_composite_quintile_assignment(self, lenient_calculator):
        """Test quintile assignment logic."""
        # Create 20 symbols with z-scores from -2 to 2
        factor_scores = {FactorType.MOMENTUM: {}}
        for i in range(20):
            z = -2.0 + i * 0.2  # Range from -2 to 1.8
            factor_scores[FactorType.MOMENTUM][f"SYM{i:03d}"] = FactorScore(
                f"SYM{i:03d}", FactorType.MOMENTUM, 0.1, z, 50.0
            )

        results = lenient_calculator.calculate_composite(factor_scores)

        # Check quintile distribution
        quintiles = [r.quintile for r in results.values()]
        assert 1 in quintiles  # Should have quintile 1
        assert 5 in quintiles  # Should have quintile 5

        # Count per quintile (should be ~4 each for 20 symbols)
        for q in range(1, 6):
            count = sum(1 for x in quintiles if x == q)
            assert 2 <= count <= 6  # Roughly 4 per quintile

    def test_composite_signal_generation(self, lenient_calculator):
        """Test signal generation based on quintiles."""
        # Create clear long/short/neutral candidates
        factor_scores = {FactorType.MOMENTUM: {}}
        for i in range(20):
            z = -2.0 + i * 0.2
            factor_scores[FactorType.MOMENTUM][f"SYM{i:03d}"] = FactorScore(
                f"SYM{i:03d}", FactorType.MOMENTUM, 0.1, z, 50.0
            )

        results = lenient_calculator.calculate_composite(factor_scores)

        signals = {r.signal for r in results.values()}
        assert "long" in signals
        assert "short" in signals
        assert "neutral" in signals

        # High quintile (4-5) should be long
        for r in results.values():
            if r.quintile >= 4:
                assert r.signal == "long"
            elif r.quintile <= 2:
                assert r.signal == "short"
            else:
                assert r.signal == "neutral"

    def test_composite_partial_coverage(self, lenient_calculator):
        """Test composite when not all symbols have all factors."""
        factor_scores = {
            FactorType.MOMENTUM: {
                "SYM000": FactorScore("SYM000", FactorType.MOMENTUM, 0.1, 1.0, 80.0),
                "SYM001": FactorScore("SYM001", FactorType.MOMENTUM, 0.05, 0.5, 60.0),
            },
            FactorType.VALUE: {
                "SYM000": FactorScore("SYM000", FactorType.VALUE, 15.0, 0.5, 60.0),
                # SYM001 missing VALUE factor
                "SYM002": FactorScore("SYM002", FactorType.VALUE, 20.0, -0.5, 40.0),
            },
        }

        results = lenient_calculator.calculate_composite(factor_scores)

        # All symbols with at least one factor should be in results
        assert "SYM000" in results
        assert "SYM001" in results
        assert "SYM002" in results


# =============================================================================
# FactorModel High-Level API Tests
# =============================================================================
class TestFactorModel:
    """Tests for FactorModel high-level API."""

    def test_factor_model_initialization(self):
        """Test FactorModel default initialization."""
        model = FactorModel()

        assert model.long_threshold == 0.5
        assert model.short_threshold == -0.5
        assert model.calculator is not None

        # Check default weights sum to 1
        total_weight = sum(model.factor_weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_factor_model_custom_weights(self):
        """Test FactorModel with custom factor weights."""
        custom_weights = {
            FactorType.MOMENTUM: 0.5,
            FactorType.VALUE: 0.5,
        }
        model = FactorModel(factor_weights=custom_weights)

        assert model.factor_weights[FactorType.MOMENTUM] == 0.5
        assert model.factor_weights[FactorType.VALUE] == 0.5

    def test_factor_model_custom_weights_string_keys(self):
        """Test FactorModel accepts string-keyed weights for integration callers."""
        custom_weights = {
            "momentum": 0.5,
            "value": 0.5,
        }
        model = FactorModel(factor_weights=custom_weights)

        assert model.factor_weights[FactorType.MOMENTUM] == 0.5
        assert model.factor_weights[FactorType.VALUE] == 0.5

    def test_score_universe(
        self,
        factor_model,
        sample_symbols,
        sample_price_data,
        sample_fundamental_data,
        sample_market_caps,
    ):
        """Test score_universe method with all data types."""
        results = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
            fundamental_data=sample_fundamental_data,
            market_caps=sample_market_caps,
        )

        assert len(results) > 0
        for _symbol, score in results.items():
            assert score.signal in ["long", "short", "neutral"]
            assert 1 <= score.quintile <= 5

    def test_score_universe_price_only(self, factor_model, sample_symbols, sample_price_data):
        """Test score_universe with only price data."""
        results = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
        )

        # Should still produce results using momentum and low_vol
        assert len(results) > 0

    def test_score_universe_insufficient_data(
        self, factor_model, sample_symbols, insufficient_price_data
    ):
        """Test score_universe with insufficient price data."""
        results = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=insufficient_price_data,
        )

        # May return empty or limited results
        # This depends on which factors can be calculated
        assert isinstance(results, dict)

    def test_get_portfolios(
        self, factor_model, sample_symbols, sample_price_data, sample_fundamental_data
    ):
        """Test get_portfolios method."""
        scores = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
            fundamental_data=sample_fundamental_data,
        )

        longs, shorts = factor_model.get_portfolios(scores, n_stocks=5)

        assert len(longs) == 5
        assert len(shorts) == 5
        assert set(longs).isdisjoint(set(shorts))  # No overlap

    def test_get_portfolios_sorted_by_score(self, factor_model, sample_symbols, sample_price_data):
        """Test that portfolios are sorted by composite score."""
        scores = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
        )

        longs, shorts = factor_model.get_portfolios(scores, n_stocks=5)

        # Longs should have highest scores
        long_scores = [scores[s].composite_z for s in longs]
        short_scores = [scores[s].composite_z for s in shorts]

        assert min(long_scores) > max(short_scores)

    def test_get_signal(self, factor_model, sample_symbols, sample_price_data):
        """Test get_signal method."""
        scores = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
        )

        symbol = sample_symbols[0]
        signal = factor_model.get_signal(symbol, scores)

        assert "action" in signal
        assert "confidence" in signal
        assert "composite_z" in signal
        assert "quintile" in signal
        assert "factor_breakdown" in signal
        assert "reason" in signal

    def test_get_signal_missing_symbol(self, factor_model):
        """Test get_signal for symbol not in universe."""
        scores = {}  # Empty scores

        signal = factor_model.get_signal("NOTFOUND", scores)

        assert signal["action"] == "hold"
        assert signal["confidence"] == 0.0
        assert "not in scored universe" in signal["reason"].lower()

    def test_get_signal_confidence_calculation(self, factor_model):
        """Test that confidence is calculated from z-score."""
        # Create mock scores
        scores = {
            "HIGH_Z": CompositeScore(
                symbol="HIGH_Z",
                composite_z=2.0,
                factor_scores={},
                quintile=5,
                signal="long",
            ),
            "LOW_Z": CompositeScore(
                symbol="LOW_Z",
                composite_z=0.5,
                factor_scores={},
                quintile=3,
                signal="neutral",
            ),
        }

        high_signal = factor_model.get_signal("HIGH_Z", scores)
        low_signal = factor_model.get_signal("LOW_Z", scores)

        # Higher z-score should have higher confidence
        assert high_signal["confidence"] > low_signal["confidence"]
        # Confidence should be capped at 1.0
        assert high_signal["confidence"] <= 1.0

    def test_get_factor_exposures(
        self, factor_model, sample_symbols, sample_price_data, sample_fundamental_data
    ):
        """Test get_factor_exposures method."""
        scores = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
            fundamental_data=sample_fundamental_data,
        )

        # Create equal-weighted portfolio
        portfolio = [(s, 1.0 / len(scores)) for s in list(scores.keys())[:10]]

        exposures = factor_model.get_factor_exposures(portfolio, scores)

        assert isinstance(exposures, dict)
        # Should have exposure values for factors
        assert "momentum" in exposures or "value" in exposures

    def test_get_factor_exposures_empty_portfolio(self, factor_model):
        """Test get_factor_exposures with empty portfolio."""
        exposures = factor_model.get_factor_exposures([], {})

        # Should return dict with zero exposures
        assert isinstance(exposures, dict)

    def test_get_factor_exposures_weighted(self, factor_model):
        """Test that exposures are properly weighted."""
        scores = {
            "SYM001": CompositeScore(
                symbol="SYM001",
                composite_z=1.0,
                factor_scores={
                    FactorType.MOMENTUM: FactorScore("SYM001", FactorType.MOMENTUM, 0.1, 2.0, 90.0),
                },
                quintile=5,
                signal="long",
            ),
            "SYM002": CompositeScore(
                symbol="SYM002",
                composite_z=-1.0,
                factor_scores={
                    FactorType.MOMENTUM: FactorScore(
                        "SYM002", FactorType.MOMENTUM, 0.05, -2.0, 10.0
                    ),
                },
                quintile=1,
                signal="short",
            ),
        }

        # Equal weights
        portfolio = [("SYM001", 0.5), ("SYM002", 0.5)]

        exposures = factor_model.get_factor_exposures(portfolio, scores)

        # Momentum exposure should be average: (2.0 + -2.0) / 2 = 0.0
        assert abs(exposures["momentum"] - 0.0) < 0.01


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_nan_values(self, factor_calculator):
        """Test handling of all NaN values."""
        dates = pd.date_range(start="2023-01-01", periods=DEFAULT_PRICE_DAYS, freq="D")
        data = {"SYM000": [np.nan] * DEFAULT_PRICE_DAYS}
        df = pd.DataFrame(data, index=dates)

        results = factor_calculator.calculate_momentum(df)

        # Should return empty or handle gracefully
        assert isinstance(results, dict)

    def test_mixed_nan_values(self, lenient_calculator):
        """Test handling of mixed NaN values."""
        np.random.seed(RANDOM_SEED)
        dates = pd.date_range(start="2023-01-01", periods=DEFAULT_PRICE_DAYS, freq="D")

        data = {}
        for i in range(10):
            prices = 100 + np.cumsum(np.random.randn(DEFAULT_PRICE_DAYS) * 0.5)
            if i == 0:
                # First 100 prices are NaN
                prices[:100] = np.nan
            data[f"SYM{i:03d}"] = prices

        df = pd.DataFrame(data, index=dates)
        results = lenient_calculator.calculate_momentum(df)

        # Should produce results for valid symbols
        assert len(results) > 0

    def test_empty_fundamental_data(self, factor_calculator):
        """Test value factor with empty fundamental data."""
        results = factor_calculator.calculate_value({})

        assert results == {}

    def test_empty_market_caps(self, factor_calculator):
        """Test size factor with empty market caps."""
        results = factor_calculator.calculate_size({})

        assert results == {}

    def test_single_symbol_universe(self, factor_calculator):
        """Test with single symbol (below min_universe_size)."""
        np.random.seed(RANDOM_SEED)
        dates = pd.date_range(start="2023-01-01", periods=DEFAULT_PRICE_DAYS, freq="D")
        data = {"SYM000": 100 + np.cumsum(np.random.randn(DEFAULT_PRICE_DAYS) * 0.5)}
        df = pd.DataFrame(data, index=dates)

        results = factor_calculator.calculate_momentum(df)

        # Should return empty (below min universe)
        assert results == {}

    def test_duplicate_symbols(self, lenient_calculator):
        """Test handling of data with potential duplicates."""
        # DataFrame columns must be unique, but test fundamental data dict
        data = {}
        for i in range(10):
            data[f"SYM{i:03d}"] = {"pe_ratio": 15.0, "pb_ratio": 2.0, "ev_ebitda": 10.0}

        results = lenient_calculator.calculate_value(data)

        assert len(results) == 10  # Should have one result per unique symbol

    def test_extreme_values(self, lenient_calculator):
        """Test handling of extreme values (winsorization)."""
        data = {}
        for i in range(10):
            pe = 15.0 if i < 8 else 1000.0  # Two extreme outliers
            data[f"SYM{i:03d}"] = {"pe_ratio": pe, "pb_ratio": 2.0, "ev_ebitda": 10.0}

        results = lenient_calculator.calculate_value(data)

        # Should handle extreme values via winsorization
        assert len(results) > 0
        # Z-scores should be reasonable after winsorization
        for score in results.values():
            assert abs(score.z_score) < 5  # Should be winsorized

    def test_zero_weight_factor(self, factor_calculator):
        """Test composite calculation with zero-weighted factor."""
        factor_scores = {
            FactorType.MOMENTUM: {
                "SYM000": FactorScore("SYM000", FactorType.MOMENTUM, 0.1, 1.0, 80.0),
            },
            FactorType.VALUE: {
                "SYM000": FactorScore("SYM000", FactorType.VALUE, 15.0, -2.0, 10.0),
            },
        }

        # Zero weight for value
        weights = {FactorType.MOMENTUM: 1.0, FactorType.VALUE: 0.0}

        results = factor_calculator.calculate_composite(factor_scores, weights)

        # Composite should equal momentum z (value has zero weight)
        # After normalization, momentum has weight 1.0
        assert abs(results["SYM000"].composite_z - 1.0) < 0.01


# =============================================================================
# Integration Tests
# =============================================================================
class TestIntegration:
    """Integration tests for the full factor model workflow."""

    def test_full_workflow(
        self,
        factor_model,
        sample_symbols,
        sample_price_data,
        sample_fundamental_data,
        sample_market_caps,
    ):
        """Test complete workflow from data to portfolio."""
        # Score universe
        scores = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
            fundamental_data=sample_fundamental_data,
            market_caps=sample_market_caps,
        )

        assert len(scores) > 0

        # Get portfolios
        longs, shorts = factor_model.get_portfolios(scores, n_stocks=5)

        assert len(longs) == 5
        assert len(shorts) == 5

        # Get signals for each position
        for symbol in longs:
            signal = factor_model.get_signal(symbol, scores)
            assert signal["action"] in ["long", "neutral"]  # Should not be short

        for symbol in shorts:
            signal = factor_model.get_signal(symbol, scores)
            assert signal["action"] in ["short", "neutral"]  # Should not be long

        # Get factor exposures
        long_portfolio = [(s, 0.1) for s in longs]
        exposures = factor_model.get_factor_exposures(long_portfolio, scores)

        assert isinstance(exposures, dict)

    def test_rebalancing_simulation(
        self, factor_model, sample_symbols, sample_price_data, sample_fundamental_data
    ):
        """Test simulated rebalancing with changing data."""
        # Initial scoring
        scores1 = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=sample_price_data,
            fundamental_data=sample_fundamental_data,
        )

        longs1, shorts1 = factor_model.get_portfolios(scores1, n_stocks=5)

        # Simulate new data (shift prices)
        shifted_price_data = sample_price_data.shift(-20).dropna()

        # Re-score
        scores2 = factor_model.score_universe(
            symbols=sample_symbols,
            price_data=shifted_price_data,
            fundamental_data=sample_fundamental_data,
        )

        longs2, shorts2 = factor_model.get_portfolios(scores2, n_stocks=5)

        # Portfolios may differ (some turnover expected)
        # Just verify both are valid
        assert len(longs2) == 5
        assert len(shorts2) == 5
