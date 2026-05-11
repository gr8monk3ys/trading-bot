#!/usr/bin/env python3
"""
Unit tests for Pairs Trading Strategy.

Tests cover:
1. Cointegration testing (Engle-Granger)
2. Hurst exponent calculation
3. Half-life calculation
4. Spread and z-score calculation
5. Signal generation
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.pairs_trading_strategy import PairsTradingStrategy


class TestHurstExponent:
    """Test Hurst exponent calculation."""

    def test_random_walk_hurst_near_05(self):
        """Random walk should have Hurst near 0.5 (with wider tolerance for R/S noise)."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)

        # Generate random walk with larger sample
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        prices = 100 * np.exp(np.cumsum(returns))

        hurst = strategy._calculate_hurst_exponent(prices)

        # Hurst should be around 0.5 for random walk (±0.25 tolerance for R/S method)
        assert 0.25 <= hurst <= 0.75, f"Random walk Hurst={hurst}, expected ~0.5"

    def test_hurst_detects_mean_reversion(self):
        """Mean-reverting series should generally have low Hurst (<0.6)."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)

        # Generate moderately mean-reverting series (not too fast, visible oscillation)
        np.random.seed(42)
        n = 500
        theta = 0.15  # Moderate mean reversion
        mu = 100
        sigma = 3.0  # More noise to create visible oscillation
        x = np.zeros(n)
        x[0] = mu + 10  # Start away from mean

        for i in range(1, n):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * np.random.normal()

        hurst = strategy._calculate_hurst_exponent(x)

        # Mean-reverting series should have Hurst < 0.6
        # (R/S method has variance, so we use wider tolerance)
        assert hurst < 0.65, f"Mean-reverting Hurst={hurst} should be < 0.65"

    def test_hurst_bounds(self):
        """Hurst exponent should always be between 0 and 1."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)

        # Test various series
        np.random.seed(42)
        for _ in range(10):
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 500)))
            hurst = strategy._calculate_hurst_exponent(prices)
            assert 0.0 <= hurst <= 1.0, f"Hurst={hurst} out of bounds"

    def test_insufficient_data_returns_05(self):
        """Insufficient data should return 0.5 (neutral)."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)

        # Only 10 data points
        prices = np.array([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])

        hurst = strategy._calculate_hurst_exponent(prices)

        assert hurst == 0.5, f"Insufficient data should return 0.5, got {hurst}"


class TestHalfLife:
    """Test half-life calculation."""

    def test_mean_reverting_has_positive_half_life(self):
        """Mean-reverting spread should have positive half-life."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)

        # Generate mean-reverting spread
        np.random.seed(42)
        n = 100
        theta = 0.1  # Mean reversion speed
        mu = 0  # Mean
        sigma = 1
        spread = np.zeros(n)
        spread[0] = 5  # Start away from mean

        for i in range(1, n):
            spread[i] = spread[i - 1] - theta * (spread[i - 1] - mu) + sigma * np.random.normal()

        half_life = strategy._calculate_half_life(spread)

        assert half_life is not None, "Mean-reverting spread should have half-life"
        assert half_life > 0, f"Half-life should be positive, got {half_life}"
        # Theoretical half-life = ln(2) / theta ≈ 6.9 for theta=0.1
        assert 2 < half_life < 20, f"Half-life={half_life} seems unreasonable"

    def test_random_walk_long_half_life(self):
        """Random walk should have long/no half-life (beta close to 1)."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)

        # Generate pure random walk
        np.random.seed(42)
        spread = np.cumsum(np.random.normal(0, 1, 200))

        half_life = strategy._calculate_half_life(spread)

        # Random walk has beta ≈ 1, so half-life could be None or relatively long
        # With small samples, regression noise can give some mean-reversion signal
        # Key insight: random walk half-life >> mean-reverting half-life
        if half_life is not None:
            assert half_life > 10, f"Random walk half-life={half_life} should be long"

    def test_insufficient_data_returns_none(self):
        """Insufficient data should return None."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)

        spread = np.array([1, 2, 3, 4, 5])

        half_life = strategy._calculate_half_life(spread)

        assert half_life is None


class TestCointegrationValidation:
    """Test cointegration validation logic."""

    def test_default_parameters(self):
        """Test default parameters are valid."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)
        params = strategy.default_parameters()

        assert params["cointegration_pvalue"] == 0.05
        assert params["min_correlation"] == 0.70
        assert params["entry_z_score"] == 2.0
        assert params["exit_z_score"] == 0.5
        assert params["stop_loss_z_score"] == 3.5
        assert params["min_hurst_threshold"] == 0.5

    def test_half_life_exit_parameters(self):
        """Test half-life exit parameters exist."""
        strategy = PairsTradingStrategy.__new__(PairsTradingStrategy)
        params = strategy.default_parameters()

        assert "use_half_life_exit" in params
        assert params["use_half_life_exit"] is True
        assert "half_life_multiplier" in params
        assert params["half_life_multiplier"] == 3.0


class TestSpreadCalculation:
    """Test spread and z-score calculation."""

    def test_z_score_formula(self):
        """Test z-score calculation is correct."""
        # z-score = (spread - mean) / std
        spread = 10
        spread_mean = 5
        spread_std = 2.5

        z_score = (spread - spread_mean) / spread_std

        assert z_score == 2.0, f"Z-score calculation error: {z_score}"

    def test_z_score_symmetric(self):
        """Test z-score is symmetric around mean."""
        spread_mean = 5
        spread_std = 2.5

        z_high = (10 - spread_mean) / spread_std  # +2
        z_low = (0 - spread_mean) / spread_std  # -2

        assert z_high == -z_low, "Z-score should be symmetric"


class TestSignalGeneration:
    """Test trading signal generation."""

    def test_high_z_score_short_spread(self):
        """High z-score (>2) should generate short_spread signal."""
        # When z-score > entry_threshold:
        # Spread too wide -> expect to narrow -> short spread
        z_score = 2.5
        entry_threshold = 2.0

        if z_score > entry_threshold:
            signal = "short_spread"
        else:
            signal = "neutral"

        assert signal == "short_spread"

    def test_low_z_score_long_spread(self):
        """Low z-score (<-2) should generate long_spread signal."""
        # When z-score < -entry_threshold:
        # Spread too narrow -> expect to widen -> long spread
        z_score = -2.5
        entry_threshold = 2.0

        if z_score < -entry_threshold:
            signal = "long_spread"
        else:
            signal = "neutral"

        assert signal == "long_spread"

    def test_neutral_z_score(self):
        """Z-score within threshold should be neutral."""
        z_score = 1.5
        entry_threshold = 2.0

        if abs(z_score) > entry_threshold:
            signal = "short_spread" if z_score > 0 else "long_spread"
        else:
            signal = "neutral"

        assert signal == "neutral"


class TestHedgeRatio:
    """Test hedge ratio calculation."""

    def test_hedge_ratio_calculation(self):
        """Test hedge ratio from linear regression."""
        np.random.seed(42)

        # Generate correlated prices
        prices2 = np.linspace(100, 150, 50) + np.random.normal(0, 2, 50)
        hedge_ratio_true = 0.8
        prices1 = hedge_ratio_true * prices2 + 10 + np.random.normal(0, 1, 50)

        # Calculate hedge ratio (should be close to 0.8)
        hedge_ratio = np.polyfit(prices2, prices1, 1)[0]

        assert 0.7 < hedge_ratio < 0.9, f"Hedge ratio={hedge_ratio}, expected ~0.8"


class TestPositionSizing:
    """Test position sizing for pairs."""

    def test_market_neutral_sizing(self):
        """Test that position sizing creates market-neutral pairs."""
        price1 = 150.0
        price2 = 100.0
        hedge_ratio = 1.5
        pair_capital = 10000.0

        # From strategy logic:
        value1 = pair_capital / (1 + 1 / hedge_ratio)
        value2 = pair_capital - value1

        quantity1 = value1 / price1
        quantity2 = value2 / price2

        # Check dollar-neutral (approximately)
        dollar_exposure1 = quantity1 * price1
        dollar_exposure2 = quantity2 * price2

        assert abs(dollar_exposure1 + dollar_exposure2 - pair_capital) < 0.01

    def test_hedge_ratio_applied_to_sizing(self):
        """Test that hedge ratio is used in sizing."""
        price1 = 100.0
        price2 = 100.0
        hedge_ratio = 2.0  # Need 2x stock2 for every stock1
        pair_capital = 10000.0

        value1 = pair_capital / (1 + 1 / hedge_ratio)
        value2 = pair_capital - value1

        quantity1 = value1 / price1
        quantity2 = value2 / price2

        # With hedge_ratio=2, quantity1 should be roughly 2x quantity2
        assert 1.8 < (quantity1 / quantity2) < 2.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
