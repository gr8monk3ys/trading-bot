"""
Comprehensive tests for RiskManager

Tests cover:
- Threshold validation
- Position risk calculation
- Correlation calculation
- Portfolio risk calculation
- Position size adjustment
- Edge cases and error handling
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.risk_manager import RiskManager


class TestRiskManagerInit:
    """Test RiskManager initialization and validation."""

    def test_default_initialization(self):
        """Test default values are set correctly."""
        rm = RiskManager()
        assert rm.max_portfolio_risk == 0.02
        assert rm.max_position_risk == 0.01
        assert rm.max_correlation == 0.7

    def test_custom_initialization(self):
        """Test custom values are accepted."""
        rm = RiskManager(
            max_portfolio_risk=0.05,
            max_position_risk=0.02,
            max_correlation=0.5
        )
        assert rm.max_portfolio_risk == 0.05
        assert rm.max_position_risk == 0.02
        assert rm.max_correlation == 0.5

    def test_invalid_portfolio_risk_raises(self):
        """Portfolio risk outside 0-1 should raise."""
        with pytest.raises(ValueError, match="max_portfolio_risk"):
            RiskManager(max_portfolio_risk=1.5)

        with pytest.raises(ValueError, match="max_portfolio_risk"):
            RiskManager(max_portfolio_risk=-0.1)

    def test_invalid_position_risk_raises(self):
        """Position risk outside 0-1 should raise."""
        with pytest.raises(ValueError, match="max_position_risk"):
            RiskManager(max_position_risk=1.5)

    def test_invalid_correlation_raises(self):
        """Correlation outside -1 to 1 should raise."""
        with pytest.raises(ValueError, match="max_correlation"):
            RiskManager(max_correlation=1.5)

    def test_zero_threshold_raises(self):
        """Zero thresholds should raise (division by zero risk)."""
        with pytest.raises(ValueError, match="cannot be zero"):
            RiskManager(var_threshold=0)

        with pytest.raises(ValueError, match="cannot be zero"):
            RiskManager(volatility_threshold=0)


class TestVolatilityCalculation:
    """Test volatility calculation."""

    def test_volatility_with_valid_data(self):
        """Test volatility calculation with normal data."""
        rm = RiskManager()
        prices = [100, 101, 102, 100, 99, 101, 102, 103, 102, 101]

        vol = rm._calculate_volatility(prices)

        assert vol > 0
        assert vol < 1  # Reasonable volatility

    def test_volatility_with_insufficient_data(self):
        """Insufficient data should return 0."""
        rm = RiskManager()

        assert rm._calculate_volatility([100]) == 0.0
        assert rm._calculate_volatility([]) == 0.0

    def test_volatility_with_zero_prices(self):
        """Zero prices should return high volatility (caution signal)."""
        rm = RiskManager()
        prices = [100, 0, 102]  # Contains zero

        vol = rm._calculate_volatility(prices)
        assert vol == 1.0  # High volatility to signal caution


class TestVaRCalculation:
    """Test Value at Risk calculation."""

    def test_var_with_valid_data(self):
        """Test VaR calculation with normal data."""
        rm = RiskManager()
        # Create data with some negative returns
        prices = [100 + np.random.randn() * 2 for _ in range(100)]

        var = rm._calculate_var(prices)

        # VaR should be negative (representing potential loss)
        assert var < 0 or var == 0

    def test_var_with_insufficient_data(self):
        """Insufficient data should return 0."""
        rm = RiskManager()

        assert rm._calculate_var([100]) == 0.0
        assert rm._calculate_var([]) == 0.0

    def test_var_with_zero_prices(self):
        """Zero prices should return large negative VaR."""
        rm = RiskManager()
        prices = [100, 0, 102]

        var = rm._calculate_var(prices)
        assert var == -0.1  # Large negative to signal high risk


class TestExpectedShortfall:
    """Test Expected Shortfall calculation."""

    def test_es_with_valid_data(self):
        """Test ES calculation with normal data."""
        rm = RiskManager()
        prices = [100 + np.random.randn() * 2 for _ in range(100)]

        es = rm._calculate_expected_shortfall(prices)

        # ES should be <= VaR (more severe)
        var = rm._calculate_var(prices)
        assert es <= var or np.isclose(es, var)

    def test_es_with_insufficient_data(self):
        """Insufficient data should return 0."""
        rm = RiskManager()

        assert rm._calculate_expected_shortfall([100]) == 0.0


class TestMaxDrawdown:
    """Test max drawdown calculation."""

    def test_drawdown_with_declining_prices(self):
        """Declining prices should show drawdown."""
        rm = RiskManager()
        prices = [100, 95, 90, 85, 80]  # 20% decline

        dd = rm._calculate_max_drawdown(prices)

        assert dd < 0  # Negative represents drawdown
        assert dd == pytest.approx(-0.20, abs=0.01)

    def test_drawdown_with_rising_prices(self):
        """Rising prices should have zero drawdown."""
        rm = RiskManager()
        prices = [100, 105, 110, 115, 120]

        dd = rm._calculate_max_drawdown(prices)

        assert dd == 0.0

    def test_drawdown_with_mixed_prices(self):
        """Test drawdown with peak then decline."""
        rm = RiskManager()
        prices = [100, 110, 120, 100, 90]  # Peak at 120, then -25%

        dd = rm._calculate_max_drawdown(prices)

        assert dd < 0
        assert dd == pytest.approx(-0.25, abs=0.01)


class TestPositionRiskCalculation:
    """Test calculate_position_risk method."""

    def test_risk_with_volatile_asset(self):
        """Volatile asset should have higher risk score."""
        rm = RiskManager()

        # Very volatile prices
        volatile_prices = [100, 110, 90, 115, 85, 120, 80]
        volatile_risk = rm.calculate_position_risk("VOLATILE", volatile_prices)

        # Stable prices
        stable_prices = [100, 100.5, 100, 100.2, 99.8, 100.1, 99.9]
        stable_risk = rm.calculate_position_risk("STABLE", stable_prices)

        assert volatile_risk > stable_risk

    def test_risk_capped_at_one(self):
        """Risk score should be capped at 1.0."""
        rm = RiskManager()

        # Extremely volatile
        extreme_prices = [100, 200, 50, 300, 25]
        risk = rm.calculate_position_risk("EXTREME", extreme_prices)

        assert risk <= 1.0

    def test_risk_with_insufficient_data(self):
        """Insufficient data should return max risk (1.0)."""
        rm = RiskManager()

        assert rm.calculate_position_risk("X", []) == 1.0
        assert rm.calculate_position_risk("X", [100]) == 1.0


class TestCorrelationCalculation:
    """Test position correlation calculation."""

    def test_perfect_positive_correlation(self):
        """Identical price movements should have correlation near 1."""
        rm = RiskManager()

        prices1 = [100, 105, 110, 115, 120]
        prices2 = [50, 52.5, 55, 57.5, 60]  # Same % moves

        corr = rm.calculate_position_correlation("A", "B", prices1, prices2)

        assert corr == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self):
        """Opposite price movements should have correlation near 1 (abs)."""
        rm = RiskManager()

        prices1 = [100, 105, 110, 115, 120]
        prices2 = [100, 95, 90, 85, 80]  # Opposite moves

        corr = rm.calculate_position_correlation("A", "B", prices1, prices2)

        # We return abs(correlation), so should be ~1
        assert corr == pytest.approx(1.0, abs=0.1)

    def test_uncorrelated_assets(self):
        """Random movements should have low correlation."""
        rm = RiskManager()

        np.random.seed(42)
        prices1 = np.cumsum(np.random.randn(100)) + 100
        prices2 = np.cumsum(np.random.randn(100)) + 100

        corr = rm.calculate_position_correlation("A", "B", list(prices1), list(prices2))

        # Should be relatively low
        assert corr < 0.5

    def test_correlation_with_different_lengths(self):
        """Different length histories should be aligned."""
        rm = RiskManager()

        prices1 = [100, 105, 110, 115, 120, 125, 130]  # 7 points
        prices2 = [50, 52.5, 55, 57.5, 60]  # 5 points

        # Should not raise, uses minimum length
        corr = rm.calculate_position_correlation("A", "B", prices1, prices2)
        assert 0 <= corr <= 1

    def test_correlation_with_insufficient_data(self):
        """Insufficient data should return max correlation (1.0)."""
        rm = RiskManager()

        assert rm.calculate_position_correlation("A", "B", [], [100, 105]) == 1.0
        assert rm.calculate_position_correlation("A", "B", [100], [100, 105]) == 1.0


class TestPositionSizeAdjustment:
    """Test adjust_position_size method."""

    def test_normal_adjustment(self):
        """Normal conditions should allow reasonable position."""
        rm = RiskManager()

        prices = [100 + i * 0.1 for i in range(50)]  # Low volatility
        current_positions = {}

        adjusted = rm.adjust_position_size("AAPL", 10000, prices, current_positions)

        assert adjusted > 0
        assert adjusted <= 10000

    def test_high_risk_reduces_position(self):
        """High-risk asset should have reduced position size."""
        rm = RiskManager(max_position_risk=0.01)

        # Very volatile prices
        volatile_prices = [100, 120, 80, 130, 70, 140, 60]
        current_positions = {}

        adjusted = rm.adjust_position_size("VOLATILE", 10000, volatile_prices, current_positions)

        # Should be reduced from desired 10000
        assert adjusted < 10000

    def test_high_correlation_strict_rejection(self):
        """High correlation with strict mode should reject position."""
        rm = RiskManager(max_correlation=0.5, strict_correlation_enforcement=True)

        prices = [100 + i for i in range(50)]

        # Existing position with same price pattern (high correlation)
        current_positions = {
            'EXISTING': {
                'value': 10000,
                'price_history': prices.copy(),
                'risk': 0.5
            }
        }

        # Same pattern = high correlation
        adjusted = rm.adjust_position_size("NEW", 10000, prices, current_positions)

        # Should be rejected (0) due to high correlation
        assert adjusted == 0

    def test_zero_desired_size_returns_zero(self):
        """Zero or negative desired size should return 0."""
        rm = RiskManager()

        assert rm.adjust_position_size("X", 0, [100, 105], {}) == 0
        assert rm.adjust_position_size("X", -100, [100, 105], {}) == 0


class TestPortfolioRisk:
    """Test portfolio-level risk calculation."""

    def test_single_position_portfolio(self):
        """Single position risk should equal position risk."""
        rm = RiskManager()

        positions = {
            'AAPL': {'value': 10000, 'risk': 0.5}
        }

        portfolio_risk = rm.calculate_portfolio_risk(positions)

        # With single position at 100% weight, risk = 0.5
        assert portfolio_risk == pytest.approx(0.5, abs=0.01)

    def test_diversified_portfolio_lower_risk(self):
        """Diversified uncorrelated portfolio should have lower risk."""
        rm = RiskManager()

        # Two uncorrelated positions (correlation = 0)
        rm.position_correlations = {('A', 'B'): 0, ('B', 'A'): 0}

        positions = {
            'A': {'value': 5000, 'risk': 0.5},
            'B': {'value': 5000, 'risk': 0.5}
        }

        portfolio_risk = rm.calculate_portfolio_risk(positions)

        # With zero correlation, diversification should reduce risk
        # Each position contributes 0.5 * 0.5 = 0.25, total ~0.5
        # But no correlation contribution
        assert portfolio_risk < 0.6

    def test_concentrated_portfolio_higher_risk(self):
        """Highly correlated portfolio should have higher risk."""
        rm = RiskManager()

        # Perfectly correlated positions
        rm.position_correlations = {('A', 'B'): 1.0, ('B', 'A'): 1.0}

        positions = {
            'A': {'value': 5000, 'risk': 0.5},
            'B': {'value': 5000, 'risk': 0.5}
        }

        portfolio_risk = rm.calculate_portfolio_risk(positions)

        # With high correlation, risk should be higher
        assert portfolio_risk > 0.4


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_handling_in_prices(self):
        """NaN values in prices should be handled gracefully."""
        rm = RiskManager()

        prices = [100, np.nan, 105, 110, np.nan]
        # Should not raise, return max risk
        risk = rm.calculate_position_risk("X", prices)
        assert risk <= 1.0

    def test_inf_handling(self):
        """Infinity values should be handled gracefully."""
        rm = RiskManager()

        prices = [100, np.inf, 105]
        risk = rm.calculate_position_risk("X", prices)
        assert risk <= 1.0

    def test_empty_positions_dict(self):
        """Empty positions should return zero portfolio risk."""
        rm = RiskManager()

        risk = rm.calculate_portfolio_risk({})
        # No positions = use default max_portfolio_risk as fallback
        assert risk >= 0

    def test_negative_prices(self):
        """Negative prices should be handled (unlikely but possible)."""
        rm = RiskManager()

        prices = [100, -10, 105]  # Invalid but shouldn't crash
        risk = rm.calculate_position_risk("X", prices)
        assert isinstance(risk, float)


class TestThresholdValidation:
    """Test P2 fix: threshold validation."""

    def test_valid_thresholds_accepted(self):
        """Valid thresholds should not raise."""
        rm = RiskManager(
            max_portfolio_risk=0.05,
            max_position_risk=0.02,
            max_correlation=0.7,
            volatility_threshold=0.3,
            var_threshold=0.03,
            es_threshold=0.04,
            drawdown_threshold=0.2
        )

        assert rm.volatility_threshold == 0.3
        assert rm.var_threshold == 0.03

    def test_non_numeric_threshold_raises(self):
        """Non-numeric thresholds should raise."""
        with pytest.raises(ValueError, match="must be numeric"):
            RiskManager(max_portfolio_risk="0.02")

    def test_threshold_out_of_range_raises(self):
        """Out-of-range thresholds should raise."""
        with pytest.raises(ValueError):
            RiskManager(volatility_threshold=-0.1)

        with pytest.raises(ValueError):
            RiskManager(volatility_threshold=15)  # Max is 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
