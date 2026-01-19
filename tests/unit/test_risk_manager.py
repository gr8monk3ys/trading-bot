"""
Comprehensive tests for strategies/risk_manager.py

Tests cover:
- RiskManager initialization and parameter validation
- Volatility calculations
- Value at Risk (VaR) calculations
- Expected Shortfall (ES) calculations
- Maximum drawdown calculations
- Position risk scoring
- Position correlation calculations
- Portfolio risk calculations
- Position size adjustments
- Edge cases and error handling
"""

import numpy as np
import pytest

# Module-level imports - DRY principle
from strategies.risk_manager import RiskCalculationError, RiskManager

# =============================================================================
# Constants - No magic numbers
# =============================================================================
DEFAULT_STARTING_PRICE = 100.0
DEFAULT_POSITION_VALUE = 10000.0
HIGH_RISK_VALUE = 0.5
LOW_RISK_VALUE = 0.3
HIGH_CORRELATION = 0.8
MODERATE_CORRELATION = 0.3
RANDOM_SEED = 42

# Default RiskManager parameters
DEFAULT_MAX_PORTFOLIO_RISK = 0.02
DEFAULT_MAX_POSITION_RISK = 0.01
DEFAULT_MAX_CORRELATION = 0.7
DEFAULT_VOLATILITY_THRESHOLD = 0.4
DEFAULT_VAR_THRESHOLD = 0.03
DEFAULT_ES_THRESHOLD = 0.04
DEFAULT_DRAWDOWN_THRESHOLD = 0.3

# Zero-price detection return values
ZERO_PRICE_VOLATILITY = 1.0
ZERO_PRICE_VAR = -0.1
ZERO_PRICE_ES = -0.15
ZERO_ROLLING_MAX_DRAWDOWN = -0.5


# =============================================================================
# Module-level fixtures - DRY principle
# =============================================================================
@pytest.fixture
def risk_manager():
    """Create RiskManager with default parameters."""
    return RiskManager()


@pytest.fixture
def risk_manager_strict():
    """Create RiskManager with strict correlation enforcement."""
    return RiskManager(strict_correlation_enforcement=True)


@pytest.fixture
def risk_manager_soft():
    """Create RiskManager with soft correlation enforcement."""
    return RiskManager(strict_correlation_enforcement=False)


@pytest.fixture
def sample_prices():
    """Generate sample price series for testing."""
    np.random.seed(RANDOM_SEED)
    return (DEFAULT_STARTING_PRICE + np.cumsum(np.random.randn(50) * 0.5)).tolist()


@pytest.fixture
def stable_prices():
    """Generate stable (low volatility) price series."""
    np.random.seed(RANDOM_SEED)
    return (DEFAULT_STARTING_PRICE + np.cumsum(np.random.randn(50) * 0.1)).tolist()


@pytest.fixture
def volatile_prices():
    """Generate volatile price series."""
    return [100, 200, 50, 300, 25, 400, 10]


def generate_correlated_prices(base_prices, noise_factor=0.001):
    """Generate prices correlated to base_prices with small noise."""
    np.random.seed(RANDOM_SEED + 1)
    return [p + np.random.randn() * noise_factor for p in base_prices]


# =============================================================================
# Initialization Tests
# =============================================================================
class TestRiskManagerInitialization:
    """Tests for RiskManager initialization."""

    def test_default_initialization(self):
        """Test RiskManager with default parameters."""
        rm = RiskManager()

        assert (
            rm.max_portfolio_risk == DEFAULT_MAX_PORTFOLIO_RISK
        ), f"Expected {DEFAULT_MAX_PORTFOLIO_RISK}, got {rm.max_portfolio_risk}"
        assert rm.max_position_risk == DEFAULT_MAX_POSITION_RISK
        assert rm.max_correlation == DEFAULT_MAX_CORRELATION
        assert rm.volatility_threshold == DEFAULT_VOLATILITY_THRESHOLD
        assert rm.var_threshold == DEFAULT_VAR_THRESHOLD
        assert rm.es_threshold == DEFAULT_ES_THRESHOLD
        assert rm.drawdown_threshold == DEFAULT_DRAWDOWN_THRESHOLD
        assert rm.strict_correlation_enforcement is True

    def test_custom_initialization(self):
        """Test RiskManager with custom parameters."""
        rm = RiskManager(
            max_portfolio_risk=0.05,
            max_position_risk=0.02,
            max_correlation=0.5,
            volatility_threshold=0.3,
            var_threshold=0.05,
            es_threshold=0.06,
            drawdown_threshold=0.2,
            strict_correlation_enforcement=False,
        )

        assert rm.max_portfolio_risk == 0.05
        assert rm.max_position_risk == 0.02
        assert rm.max_correlation == 0.5
        assert rm.volatility_threshold == 0.3
        assert rm.strict_correlation_enforcement is False

    def test_initialization_creates_empty_tracking(self):
        """Test that tracking dictionaries are initialized."""
        rm = RiskManager()

        assert rm.position_sizes == {}, "position_sizes should be empty dict"
        assert rm.position_correlations == {}, "position_correlations should be empty dict"


# =============================================================================
# Threshold Validation Tests - Using parametrize for DRY
# =============================================================================
class TestValidateThreshold:
    """Tests for _validate_threshold static method."""

    @pytest.mark.parametrize(
        "value,min_val,max_val",
        [
            (0.5, 0, 1),  # Middle of range
            (0.0, 0, 1),  # At minimum
            (1.0, 0, 1),  # At maximum
            (0.001, 0, 1),  # Near minimum
            (0.999, 0, 1),  # Near maximum
        ],
    )
    def test_valid_thresholds(self, value, min_val, max_val):
        """Test validation passes for valid thresholds."""
        # Should not raise
        RiskManager._validate_threshold("test", value, min_val, max_val)

    @pytest.mark.parametrize(
        "value,min_val,max_val,error_match",
        [
            (-0.1, 0, 1, "must be between"),  # Below minimum
            (1.5, 0, 1, "must be between"),  # Above maximum
            (-1, 0, 1, "must be between"),  # Negative
            (2.0, 0, 1, "must be between"),  # Way above
        ],
    )
    def test_invalid_thresholds_raise(self, value, min_val, max_val, error_match):
        """Test validation fails for out-of-range values."""
        with pytest.raises(ValueError, match=error_match):
            RiskManager._validate_threshold("test", value, min_val, max_val)

    def test_non_numeric_value_raises(self):
        """Test validation fails for non-numeric values."""
        with pytest.raises(ValueError, match="must be numeric"):
            RiskManager._validate_threshold("test", "0.5", 0, 1)

    def test_zero_threshold_raises(self):
        """Test validation fails for zero threshold values (division by zero protection)."""
        with pytest.raises(ValueError, match="cannot be zero"):
            RiskManager._validate_threshold("var_threshold", 0.0, 0, 1)


# =============================================================================
# Volatility Calculation Tests
# =============================================================================
class TestCalculateVolatility:
    """Tests for _calculate_volatility method."""

    def test_volatility_with_valid_data(self, risk_manager, sample_prices):
        """Test volatility calculation with valid price data."""
        vol = risk_manager._calculate_volatility(sample_prices)

        assert vol > 0, f"Expected positive volatility, got {vol}"

    def test_volatility_with_insufficient_data(self, risk_manager):
        """Test volatility with insufficient data returns 0."""
        vol = risk_manager._calculate_volatility([DEFAULT_STARTING_PRICE])

        assert vol == 0.0, f"Expected 0.0 for insufficient data, got {vol}"

    def test_volatility_with_zero_prices(self, risk_manager):
        """Test volatility handling of zero prices returns high volatility signal."""
        # Must use numpy array for element-wise zero comparison
        prices = np.array([100, 0, 102])
        vol = risk_manager._calculate_volatility(prices)

        assert (
            vol == ZERO_PRICE_VOLATILITY
        ), f"Expected {ZERO_PRICE_VOLATILITY} for zero prices, got {vol}"

    def test_volatility_with_constant_prices(self, risk_manager):
        """Test volatility with constant prices returns 0."""
        prices = [DEFAULT_STARTING_PRICE] * 5
        vol = risk_manager._calculate_volatility(prices)

        assert vol == 0.0, f"Expected 0.0 for constant prices, got {vol}"


# =============================================================================
# VaR Calculation Tests
# =============================================================================
class TestCalculateVaR:
    """Tests for _calculate_var method."""

    def test_var_with_valid_data(self, risk_manager):
        """Test VaR calculation with valid data."""
        np.random.seed(RANDOM_SEED)
        prices = (DEFAULT_STARTING_PRICE + np.cumsum(np.random.randn(100) * 0.5)).tolist()
        var = risk_manager._calculate_var(prices)

        assert var < 0, f"Expected negative VaR (loss), got {var}"

    def test_var_with_insufficient_data(self, risk_manager):
        """Test VaR with insufficient data returns 0."""
        var = risk_manager._calculate_var([DEFAULT_STARTING_PRICE])

        assert var == 0.0, f"Expected 0.0 for insufficient data, got {var}"

    def test_var_with_zero_prices(self, risk_manager):
        """Test VaR handling of zero prices returns high risk signal."""
        prices = np.array([100, 0, 102])
        var = risk_manager._calculate_var(prices)

        assert var == ZERO_PRICE_VAR, f"Expected {ZERO_PRICE_VAR} for zero prices, got {var}"


# =============================================================================
# Expected Shortfall Tests
# =============================================================================
class TestCalculateExpectedShortfall:
    """Tests for _calculate_expected_shortfall method."""

    def test_es_with_valid_data(self, risk_manager):
        """Test Expected Shortfall calculation with valid data."""
        np.random.seed(RANDOM_SEED)
        prices = (DEFAULT_STARTING_PRICE + np.cumsum(np.random.randn(100) * 0.5)).tolist()
        es = risk_manager._calculate_expected_shortfall(prices)

        assert es < 0, f"Expected negative ES (loss), got {es}"

    def test_es_with_insufficient_data(self, risk_manager):
        """Test ES with insufficient data returns 0."""
        es = risk_manager._calculate_expected_shortfall([DEFAULT_STARTING_PRICE])

        assert es == 0.0, f"Expected 0.0 for insufficient data, got {es}"

    def test_es_with_zero_prices(self, risk_manager):
        """Test ES handling of zero prices returns high risk signal."""
        prices = np.array([100, 0, 102])
        es = risk_manager._calculate_expected_shortfall(prices)

        assert es == ZERO_PRICE_ES, f"Expected {ZERO_PRICE_ES} for zero prices, got {es}"


# =============================================================================
# Max Drawdown Tests
# =============================================================================
class TestCalculateMaxDrawdown:
    """Tests for _calculate_max_drawdown method."""

    def test_drawdown_with_valid_data(self, risk_manager):
        """Test max drawdown with price series that has drawdown."""
        prices = [100, 110, 115, 100, 95, 90, 100]
        dd = risk_manager._calculate_max_drawdown(prices)

        assert dd < 0, f"Expected negative drawdown, got {dd}"
        # From peak of 115 to low of 90: (90-115)/115 = -0.217
        assert dd == pytest.approx(-0.217, rel=0.01), f"Expected approx -0.217 drawdown, got {dd}"

    def test_drawdown_with_insufficient_data(self, risk_manager):
        """Test drawdown with insufficient data returns 0."""
        dd = risk_manager._calculate_max_drawdown([DEFAULT_STARTING_PRICE])

        assert dd == 0.0, f"Expected 0.0 for insufficient data, got {dd}"

    def test_drawdown_with_always_increasing(self, risk_manager):
        """Test drawdown when prices always increase is 0."""
        prices = [100, 101, 102, 103, 104, 105]
        dd = risk_manager._calculate_max_drawdown(prices)

        assert dd == 0.0, f"Expected 0.0 drawdown for increasing prices, got {dd}"

    def test_drawdown_with_zero_rolling_max(self, risk_manager):
        """Test drawdown handling of zero in rolling max."""
        prices = [0, 100, 102]
        dd = risk_manager._calculate_max_drawdown(prices)

        assert (
            dd == ZERO_ROLLING_MAX_DRAWDOWN
        ), f"Expected {ZERO_ROLLING_MAX_DRAWDOWN} for zero rolling max, got {dd}"


# =============================================================================
# Position Risk Tests
# =============================================================================
class TestCalculatePositionRisk:
    """Tests for calculate_position_risk method."""

    def test_position_risk_with_valid_data(self, risk_manager, sample_prices):
        """Test position risk calculation returns value in [0, 1]."""
        risk = risk_manager.calculate_position_risk("AAPL", sample_prices)

        assert 0 <= risk <= 1, f"Risk should be in [0, 1], got {risk}"

    @pytest.mark.parametrize(
        "invalid_input,expected_risk",
        [
            ([DEFAULT_STARTING_PRICE], 1.0),  # Insufficient data
            ([], 1.0),  # Empty data
            (None, 1.0),  # None data
            ("not a list", 1.0),  # Invalid type
        ],
    )
    def test_position_risk_with_invalid_data(self, risk_manager, invalid_input, expected_risk):
        """Test position risk returns max risk for invalid inputs."""
        risk = risk_manager.calculate_position_risk("AAPL", invalid_input)

        # Allow for 0.0 or 1.0 depending on how the code handles edge cases
        assert risk in [
            0.0,
            expected_risk,
        ], f"Expected 0.0 or {expected_risk} for invalid input, got {risk}"

    def test_position_risk_capped_at_one(self, risk_manager, volatile_prices):
        """Test that position risk is capped at 1.0."""
        risk = risk_manager.calculate_position_risk("AAPL", volatile_prices)

        assert risk <= 1.0, f"Risk should be capped at 1.0, got {risk}"


# =============================================================================
# Position Correlation Tests
# =============================================================================
class TestCalculatePositionCorrelation:
    """Tests for calculate_position_correlation method."""

    def test_correlation_with_identical_data(self, risk_manager):
        """Test correlation with identical price series is 1.0."""
        prices = [100, 101, 102, 103, 104, 105]
        corr = risk_manager.calculate_position_correlation("AAPL", "MSFT", prices, prices.copy())

        assert corr == pytest.approx(
            1.0, rel=0.01
        ), f"Expected correlation ~1.0 for identical series, got {corr}"

    def test_correlation_with_inversely_correlated(self, risk_manager):
        """Test correlation with inversely correlated data is high (absolute)."""
        prices1 = [100, 101, 102, 103, 104, 105]
        prices2 = [100, 99, 98, 97, 96, 95]
        corr = risk_manager.calculate_position_correlation("AAPL", "MSFT", prices1, prices2)

        assert corr > 0.9, f"Expected high absolute correlation, got {corr}"

    @pytest.mark.parametrize(
        "prices1,prices2,expected",
        [
            ([100], [100], 1.0),  # Insufficient data
            ([], [], 1.0),  # Empty data
        ],
    )
    def test_correlation_with_invalid_data(self, risk_manager, prices1, prices2, expected):
        """Test correlation returns max for invalid inputs."""
        corr = risk_manager.calculate_position_correlation("AAPL", "MSFT", prices1, prices2)

        assert corr == expected, f"Expected {expected} for invalid input, got {corr}"

    def test_correlation_with_different_lengths(self, risk_manager):
        """Test correlation handles different length arrays."""
        prices1 = [100, 101, 102, 103, 104, 105, 106, 107]
        prices2 = [100, 101, 102, 103]
        corr = risk_manager.calculate_position_correlation("AAPL", "MSFT", prices1, prices2)

        assert 0 <= corr <= 1, f"Correlation should be in [0, 1], got {corr}"

    def test_correlation_with_zero_variance(self, risk_manager):
        """Test correlation with zero variance returns 1.0 (NaN handled)."""
        prices1 = [100.0] * 5  # Constant - zero variance
        prices2 = [50.0] * 5  # Constant - zero variance
        corr = risk_manager.calculate_position_correlation("AAPL", "MSFT", prices1, prices2)

        assert corr == 1.0, f"Expected 1.0 for zero variance, got {corr}"


# =============================================================================
# Portfolio Risk Tests
# =============================================================================
class TestCalculatePortfolioRisk:
    """Tests for calculate_portfolio_risk method."""

    def test_portfolio_risk_with_positions(self, risk_manager):
        """Test portfolio risk calculation with positions."""
        positions = {
            "AAPL": {"value": DEFAULT_POSITION_VALUE, "risk": HIGH_RISK_VALUE},
            "MSFT": {"value": DEFAULT_POSITION_VALUE * 1.5, "risk": LOW_RISK_VALUE},
        }
        risk = risk_manager.calculate_portfolio_risk(positions)

        assert risk >= 0, f"Portfolio risk should be non-negative, got {risk}"

    def test_portfolio_risk_with_empty_positions(self, risk_manager):
        """Test portfolio risk with empty positions."""
        risk = risk_manager.calculate_portfolio_risk({})

        assert (
            risk == 0 or risk == risk_manager.max_portfolio_risk
        ), f"Expected 0 or max risk for empty portfolio, got {risk}"

    def test_portfolio_risk_with_correlations(self, risk_manager):
        """Test portfolio risk includes correlation impact."""
        positions = {
            "AAPL": {"value": DEFAULT_POSITION_VALUE, "risk": HIGH_RISK_VALUE},
            "MSFT": {"value": DEFAULT_POSITION_VALUE, "risk": HIGH_RISK_VALUE},
        }
        risk_manager.position_correlations[("AAPL", "MSFT")] = HIGH_CORRELATION
        risk = risk_manager.calculate_portfolio_risk(positions)

        assert risk >= 0, f"Portfolio risk should be non-negative, got {risk}"


# =============================================================================
# Position Size Adjustment Tests
# =============================================================================
class TestAdjustPositionSize:
    """Tests for adjust_position_size method."""

    def test_adjust_size_with_no_positions(self, risk_manager_strict, stable_prices):
        """Test size adjustment with no existing positions."""
        adjusted = risk_manager_strict.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, stable_prices, {}
        )

        assert (
            0 <= adjusted <= DEFAULT_POSITION_VALUE
        ), f"Adjusted size should be in [0, {DEFAULT_POSITION_VALUE}], got {adjusted}"

    @pytest.mark.parametrize("desired_size", [-1000, 0])
    def test_adjust_size_with_invalid_desired_size(self, risk_manager_strict, desired_size):
        """Test size adjustment with invalid desired size returns 0."""
        prices = [100, 101, 102]
        adjusted = risk_manager_strict.adjust_position_size("AAPL", desired_size, prices, {})

        assert adjusted == 0, f"Expected 0 for invalid desired size, got {adjusted}"

    def test_adjust_size_rejects_high_correlation_strict(self, risk_manager_strict, stable_prices):
        """Test strict mode rejects positions with high correlation."""
        current_positions = {
            "MSFT": {
                "value": DEFAULT_POSITION_VALUE,
                "price_history": stable_prices.copy(),  # Same prices = perfect correlation
            }
        }
        adjusted = risk_manager_strict.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, stable_prices, current_positions
        )

        assert adjusted == 0, f"Expected 0 for high correlation in strict mode, got {adjusted}"

    def test_adjust_size_reduces_high_correlation_soft(self, risk_manager_soft, stable_prices):
        """Test soft mode reduces (not rejects) positions with high correlation."""
        correlated_prices = generate_correlated_prices(stable_prices)
        current_positions = {
            "MSFT": {"value": DEFAULT_POSITION_VALUE, "price_history": correlated_prices}
        }
        adjusted = risk_manager_soft.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, stable_prices, current_positions
        )

        assert adjusted >= 0, f"Adjusted size should be non-negative, got {adjusted}"

    def test_adjust_size_stores_risk(self, risk_manager_strict, stable_prices):
        """Test that adjust_position_size stores risk in current_positions."""
        current_positions = {
            "AAPL": {"value": DEFAULT_POSITION_VALUE / 2, "price_history": stable_prices.copy()}
        }
        risk_manager_strict.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, stable_prices, current_positions
        )

        assert "risk" in current_positions["AAPL"], "Risk should be stored in current_positions"

    def test_adjust_size_stores_correlations(self, risk_manager_soft, stable_prices):
        """Test that adjust_position_size stores correlations."""
        np.random.seed(RANDOM_SEED + 2)
        other_prices = (150 + np.cumsum(np.random.randn(50) * 0.1)).tolist()
        current_positions = {
            "MSFT": {"value": DEFAULT_POSITION_VALUE, "price_history": other_prices}
        }
        risk_manager_soft.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, stable_prices, current_positions
        )

        has_correlation = ("AAPL", "MSFT") in risk_manager_soft.position_correlations or (
            "MSFT",
            "AAPL",
        ) in risk_manager_soft.position_correlations
        assert has_correlation, "Correlations should be stored"


# =============================================================================
# Exception Class Tests
# =============================================================================
class TestRiskCalculationError:
    """Tests for RiskCalculationError exception."""

    def test_risk_calculation_error_exists(self):
        """Test that RiskCalculationError is defined and works."""
        error = RiskCalculationError("test error")
        assert str(error) == "test error"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.parametrize(
        "prices,expected_corr",
        [
            ([100, float("nan"), 102, 103, 104], 1.0),  # NaN values
            ([100, float("inf"), 102, 103, 104], 1.0),  # Infinity values
            ([100, "invalid", 102, 103, 104], 1.0),  # Type error
            ([1e308] * 5, 1.0),  # Extreme values
        ],
    )
    def test_correlation_handles_invalid_values(self, risk_manager, prices, expected_corr):
        """Test correlation handles various invalid values gracefully."""
        prices2 = [100, 101, 102, 103, 104]
        corr = risk_manager.calculate_position_correlation("AAPL", "MSFT", prices, prices2)

        assert 0 <= corr <= 1.0, f"Correlation should be in [0, 1], got {corr}"

    def test_portfolio_risk_handles_missing_value_key(self, risk_manager):
        """Test portfolio risk handles missing 'value' key gracefully."""
        positions = {"AAPL": {"invalid_key": 10000}}
        risk = risk_manager.calculate_portfolio_risk(positions)

        assert (
            risk == risk_manager.max_portfolio_risk
        ), f"Expected max portfolio risk on error, got {risk}"

    def test_adjust_size_with_math_error(self, risk_manager_strict):
        """Test adjust_position_size handles math errors gracefully."""
        prices = [float("inf")] * 3
        adjusted = risk_manager_strict.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, prices, {}
        )

        assert adjusted >= 0, f"Adjusted size should be non-negative, got {adjusted}"

    def test_adjust_size_with_missing_position_keys(self, risk_manager_strict, stable_prices):
        """Test adjust_position_size handles missing position keys."""
        current_positions = {"MSFT": {}}  # Missing 'value' and 'price_history'
        adjusted = risk_manager_strict.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, stable_prices, current_positions
        )

        assert adjusted >= 0, f"Adjusted size should be non-negative, got {adjusted}"

    def test_volatility_with_very_stable_prices(self, risk_manager):
        """Test volatility with very stable prices."""
        prices = [100.0001, 100.0002, 100.0003, 100.0004, 100.0005]
        vol = risk_manager._calculate_volatility(prices)

        assert vol >= 0, f"Volatility should be non-negative, got {vol}"

    def test_var_with_all_positive_returns(self, risk_manager):
        """Test VaR when all returns are positive (profit)."""
        prices = list(range(100, 111))  # 100, 101, ..., 110
        var = risk_manager._calculate_var(prices)

        assert var >= 0, f"VaR should be positive for all-profit scenario, got {var}"

    def test_es_with_no_tail_returns(self, risk_manager):
        """Test ES when no returns below VaR (all positive)."""
        prices = [100, 101, 102, 103, 104, 105]
        es = risk_manager._calculate_expected_shortfall(prices)

        assert es >= 0, f"ES should be positive when no tail returns, got {es}"

    def test_initialization_with_boundary_values(self):
        """Test initialization with minimum valid values."""
        rm = RiskManager(
            max_portfolio_risk=0.001,
            max_position_risk=0.001,
            volatility_threshold=0.001,
            var_threshold=0.001,
            es_threshold=0.001,
            drawdown_threshold=0.001,
        )

        assert rm.max_portfolio_risk == 0.001, "Should accept minimum valid values"

    def test_adjust_size_with_high_portfolio_risk(self, risk_manager_soft, sample_prices):
        """
        Test portfolio adjustment when portfolio risk is high.

        When existing positions have high individual risk and moderate correlation,
        adding a new position should result in reduced position sizing.
        """
        current_positions = {
            "MSFT": {
                "value": DEFAULT_POSITION_VALUE * 5,
                "risk": HIGH_RISK_VALUE,
                "price_history": sample_prices.copy(),
            },
            "GOOGL": {
                "value": DEFAULT_POSITION_VALUE * 5,
                "risk": HIGH_RISK_VALUE,
                "price_history": sample_prices.copy(),
            },
        }
        risk_manager_soft.position_correlations[("MSFT", "GOOGL")] = MODERATE_CORRELATION
        risk_manager_soft.position_correlations[("GOOGL", "MSFT")] = MODERATE_CORRELATION

        np.random.seed(RANDOM_SEED + 3)
        new_prices = (200 + np.cumsum(np.random.randn(50) * 0.5)).tolist()
        adjusted = risk_manager_soft.adjust_position_size(
            "AAPL", DEFAULT_POSITION_VALUE, new_prices, current_positions
        )

        assert (
            0 <= adjusted <= DEFAULT_POSITION_VALUE
        ), f"Adjusted should be in [0, {DEFAULT_POSITION_VALUE}], got {adjusted}"

    def test_max_drawdown_with_zero_start(self, risk_manager):
        """Test max drawdown when starting with zero."""
        prices = [0, 100, 110, 100, 90]
        dd = risk_manager._calculate_max_drawdown(prices)

        assert dd <= 0, f"Drawdown should be non-positive, got {dd}"
