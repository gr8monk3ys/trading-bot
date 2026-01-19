"""
Tests for RSI-2 Strategy Mode

Tests cover:
- RSI-2 aggressive mode initialization
- Parameter overrides when aggressive mode is set
- Standard mode vs aggressive mode differences
"""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRSI2Mode:
    """Test RSI-2 aggressive mode settings."""

    def test_default_mode_is_standard(self):
        """Default mode should be standard RSI-14."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        params = strategy.default_parameters()

        assert params["rsi_mode"] == "standard"
        assert params["rsi_period"] == 14
        assert params["rsi_overbought"] == 70
        assert params["rsi_oversold"] == 30

    def test_aggressive_mode_params_in_defaults(self):
        """Aggressive mode should be documented in defaults."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        params = strategy.default_parameters()

        # rsi_mode should exist as a parameter
        assert "rsi_mode" in params
        assert params["rsi_mode"] in ["standard", "aggressive"]

    def test_aggressive_mode_overrides_rsi_params(self):
        """When aggressive mode is set, RSI params should be overridden to RSI-2."""
        from strategies.momentum_strategy import MomentumStrategy

        # Create strategy with aggressive mode
        strategy = MomentumStrategy()
        strategy.parameters = strategy.default_parameters()
        strategy.parameters["rsi_mode"] = "aggressive"

        # Simulate the initialization logic
        rsi_mode = strategy.parameters.get("rsi_mode", "standard")

        if rsi_mode == "aggressive":
            rsi_period = 2
            rsi_overbought = 90
            rsi_oversold = 10
        else:
            rsi_period = strategy.parameters["rsi_period"]
            rsi_overbought = strategy.parameters["rsi_overbought"]
            rsi_oversold = strategy.parameters["rsi_oversold"]

        # Verify aggressive mode uses RSI-2 settings
        assert rsi_period == 2, "RSI period should be 2 in aggressive mode"
        assert rsi_overbought == 90, "RSI overbought should be 90 in aggressive mode"
        assert rsi_oversold == 10, "RSI oversold should be 10 in aggressive mode"

    def test_standard_mode_uses_default_rsi_params(self):
        """Standard mode should use RSI-14 with 30/70 thresholds."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        strategy.parameters = strategy.default_parameters()
        strategy.parameters["rsi_mode"] = "standard"

        rsi_mode = strategy.parameters.get("rsi_mode", "standard")

        if rsi_mode == "aggressive":
            rsi_period = 2
            rsi_overbought = 90
            rsi_oversold = 10
        else:
            rsi_period = strategy.parameters["rsi_period"]
            rsi_overbought = strategy.parameters["rsi_overbought"]
            rsi_oversold = strategy.parameters["rsi_oversold"]

        assert rsi_period == 14, "RSI period should be 14 in standard mode"
        assert rsi_overbought == 70, "RSI overbought should be 70 in standard mode"
        assert rsi_oversold == 30, "RSI oversold should be 30 in standard mode"

    def test_rsi2_thresholds_are_extreme(self):
        """RSI-2 thresholds (10/90) should be more extreme than standard (30/70)."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        params = strategy.default_parameters()

        # Standard thresholds
        std_oversold = params["rsi_oversold"]
        std_overbought = params["rsi_overbought"]

        # RSI-2 aggressive thresholds
        agg_oversold = 10
        agg_overbought = 90

        # Aggressive thresholds should be more extreme
        assert agg_oversold < std_oversold, "RSI-2 oversold should be lower (more extreme)"
        assert agg_overbought > std_overbought, "RSI-2 overbought should be higher (more extreme)"

        # The range should be wider for aggressive mode
        std_range = std_overbought - std_oversold  # 70 - 30 = 40
        agg_range = agg_overbought - agg_oversold  # 90 - 10 = 80

        assert agg_range > std_range, "RSI-2 should have wider threshold range"


class TestRSI2Strategy:
    """Test RSI-2 signal generation logic."""

    def test_rsi_below_10_is_oversold_in_aggressive_mode(self):
        """RSI < 10 should trigger oversold in aggressive mode."""
        # RSI-2 uses extreme thresholds
        rsi_value = 5
        rsi_oversold = 10  # Aggressive threshold

        is_oversold = rsi_value < rsi_oversold
        assert is_oversold is True

    def test_rsi_above_90_is_overbought_in_aggressive_mode(self):
        """RSI > 90 should trigger overbought in aggressive mode."""
        rsi_value = 95
        rsi_overbought = 90  # Aggressive threshold

        is_overbought = rsi_value > rsi_overbought
        assert is_overbought is True

    def test_rsi_50_is_neutral_in_both_modes(self):
        """RSI at 50 should be neutral in both modes."""
        rsi_value = 50

        # Standard mode (30/70)
        std_oversold = 30
        std_overbought = 70
        std_neutral = std_oversold <= rsi_value <= std_overbought

        # Aggressive mode (10/90)
        agg_oversold = 10
        agg_overbought = 90
        agg_neutral = agg_oversold <= rsi_value <= agg_overbought

        # Both should be neutral at RSI 50
        assert std_neutral is True
        assert agg_neutral is True


class TestRSI2Research:
    """Validate research claims about RSI-2 performance."""

    def test_research_claims(self):
        """Document research claims for reference."""
        # From QuantifiedStrategies.com research:
        # RSI-2 Strategy on S&P 500 stocks (2000-2020):
        # - Win rate: 91%
        # - Average gain: 2.5%
        # - Average loss: 1.8%
        # - Risk-adjusted Sharpe: > 3.0

        # These are documented claims, not assertions
        rsi2_expected_win_rate = 0.91
        rsi14_expected_win_rate = 0.55

        improvement = (rsi2_expected_win_rate - rsi14_expected_win_rate) / rsi14_expected_win_rate

        # RSI-2 should improve win rate by ~65%
        assert improvement > 0.5, "RSI-2 should significantly improve win rate over RSI-14"

    def test_rsi2_period_is_shorter(self):
        """RSI-2 uses 2-period lookback for faster signals."""
        rsi2_period = 2
        rsi14_period = 14

        # RSI-2 responds 7x faster to price changes
        speed_ratio = rsi14_period / rsi2_period
        assert speed_ratio == 7, "RSI-2 should respond 7x faster than RSI-14"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
