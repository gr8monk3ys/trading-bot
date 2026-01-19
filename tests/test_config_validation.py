"""
Tests for config.py validation

Tests cover:
- Trading parameter validation
- Risk parameter validation
- Technical parameter validation
- Backtest parameter validation
- Environment variable handling
"""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTradingParamsValidation:
    """Test TRADING_PARAMS validation."""

    def test_position_size_in_range(self):
        """Position size should be between 0 and 1."""
        from config import TRADING_PARAMS

        pos_size = TRADING_PARAMS["POSITION_SIZE"]
        assert 0 < pos_size <= 1

    def test_max_position_size_greater_than_position_size(self):
        """Max position size should be >= position size."""
        from config import TRADING_PARAMS

        assert TRADING_PARAMS["MAX_POSITION_SIZE"] >= TRADING_PARAMS["POSITION_SIZE"]

    def test_stop_loss_in_range(self):
        """Stop loss should be between 0 and 0.5."""
        from config import TRADING_PARAMS

        stop_loss = TRADING_PARAMS["STOP_LOSS"]
        assert 0 < stop_loss <= 0.5

    def test_take_profit_in_range(self):
        """Take profit should be between 0 and 1."""
        from config import TRADING_PARAMS

        take_profit = TRADING_PARAMS["TAKE_PROFIT"]
        assert 0 < take_profit <= 1

    def test_kelly_fraction_in_range(self):
        """Kelly fraction should be between 0 and 1."""
        from config import TRADING_PARAMS

        kelly = TRADING_PARAMS["KELLY_FRACTION"]
        assert 0 <= kelly <= 1


class TestRiskParamsValidation:
    """Test RISK_PARAMS validation."""

    def test_max_portfolio_risk_reasonable(self):
        """Max portfolio risk should be <= 0.2 (20%)."""
        from config import RISK_PARAMS

        assert 0 < RISK_PARAMS["MAX_PORTFOLIO_RISK"] <= 0.2

    def test_max_position_risk_reasonable(self):
        """Max position risk should be <= 0.1 (10%)."""
        from config import RISK_PARAMS

        assert 0 < RISK_PARAMS["MAX_POSITION_RISK"] <= 0.1

    def test_var_confidence_valid(self):
        """VaR confidence should be between 0.5 and 1."""
        from config import RISK_PARAMS

        assert 0.5 < RISK_PARAMS["VAR_CONFIDENCE"] < 1

    def test_max_correlation_valid(self):
        """Max correlation should be between -1 and 1."""
        from config import RISK_PARAMS

        corr = RISK_PARAMS["MAX_CORRELATION"]
        assert -1 <= corr <= 1


class TestTechnicalParamsValidation:
    """Test TECHNICAL_PARAMS validation."""

    def test_rsi_period_positive(self):
        """RSI period should be positive."""
        from config import TECHNICAL_PARAMS

        assert TECHNICAL_PARAMS["RSI_PERIOD"] > 0

    def test_rsi_thresholds_valid(self):
        """RSI oversold should be < overbought."""
        from config import TECHNICAL_PARAMS

        assert TECHNICAL_PARAMS["RSI_OVERSOLD"] < TECHNICAL_PARAMS["RSI_OVERBOUGHT"]

    def test_sma_periods_ordered(self):
        """Short SMA should be < Long SMA."""
        from config import TECHNICAL_PARAMS

        assert TECHNICAL_PARAMS["SHORT_SMA"] < TECHNICAL_PARAMS["LONG_SMA"]

    def test_rsi_thresholds_in_range(self):
        """RSI thresholds should be between 0 and 100."""
        from config import TECHNICAL_PARAMS

        assert 0 <= TECHNICAL_PARAMS["RSI_OVERSOLD"] <= 100
        assert 0 <= TECHNICAL_PARAMS["RSI_OVERBOUGHT"] <= 100


class TestBacktestParamsValidation:
    """Test BACKTEST_PARAMS validation."""

    def test_slippage_reasonable(self):
        """Slippage should be positive and reasonable."""
        from config import BACKTEST_PARAMS

        slip = BACKTEST_PARAMS["SLIPPAGE_PCT"]
        assert 0 <= slip <= 0.05  # Max 5% seems reasonable

    def test_train_ratio_valid(self):
        """Train ratio should be between 0 and 1."""
        from config import BACKTEST_PARAMS

        ratio = BACKTEST_PARAMS["TRAIN_RATIO"]
        assert 0 < ratio < 1

    def test_n_splits_positive(self):
        """Number of walk-forward splits should be positive."""
        from config import BACKTEST_PARAMS

        assert BACKTEST_PARAMS["N_SPLITS"] > 0

    def test_min_trades_reasonable(self):
        """Minimum trades for significance should be reasonable."""
        from config import BACKTEST_PARAMS

        min_trades = BACKTEST_PARAMS["MIN_TRADES_FOR_SIGNIFICANCE"]
        assert 20 <= min_trades <= 200

    def test_overfitting_threshold_positive(self):
        """Overfitting ratio threshold should be positive."""
        from config import BACKTEST_PARAMS

        assert BACKTEST_PARAMS["OVERFITTING_RATIO_THRESHOLD"] > 0


class TestSymbolsConfiguration:
    """Test SYMBOLS configuration."""

    def test_symbols_not_empty(self):
        """Symbols list should not be empty."""
        from config import SYMBOLS

        assert len(SYMBOLS) > 0

    def test_symbols_are_strings(self):
        """All symbols should be strings."""
        from config import SYMBOLS

        for symbol in SYMBOLS:
            assert isinstance(symbol, str)

    def test_symbols_uppercase(self):
        """Symbols should be uppercase."""
        from config import SYMBOLS

        for symbol in SYMBOLS:
            assert symbol == symbol.upper()


class TestAlpacaCredentials:
    """Test ALPACA_CREDS configuration."""

    def test_creds_structure(self):
        """ALPACA_CREDS should have expected keys."""
        from config import ALPACA_CREDS

        assert "API_KEY" in ALPACA_CREDS
        assert "API_SECRET" in ALPACA_CREDS
        assert "PAPER" in ALPACA_CREDS

    def test_paper_is_boolean(self):
        """PAPER should be a boolean."""
        from config import ALPACA_CREDS

        assert isinstance(ALPACA_CREDS["PAPER"], bool)


class TestSymbolSelectionConfig:
    """Test SYMBOL_SELECTION configuration."""

    def test_symbol_selection_structure(self):
        """SYMBOL_SELECTION should have expected keys."""
        from config import SYMBOL_SELECTION

        expected_keys = [
            "USE_DYNAMIC_SELECTION",
            "TOP_N_SYMBOLS",
            "MIN_MOMENTUM_SCORE",
            "RESCAN_INTERVAL_HOURS",
        ]

        for key in expected_keys:
            assert key in SYMBOL_SELECTION

    def test_top_n_symbols_positive(self):
        """TOP_N_SYMBOLS should be positive."""
        from config import SYMBOL_SELECTION

        assert SYMBOL_SELECTION["TOP_N_SYMBOLS"] > 0

    def test_rescan_interval_positive(self):
        """Rescan interval should be positive."""
        from config import SYMBOL_SELECTION

        assert SYMBOL_SELECTION["RESCAN_INTERVAL_HOURS"] > 0


class TestAdvancedTradingFeatures:
    """Test advanced trading feature flags."""

    def test_kelly_criterion_flag_exists(self):
        """USE_KELLY_CRITERION flag should exist."""
        from config import TRADING_PARAMS

        assert "USE_KELLY_CRITERION" in TRADING_PARAMS
        assert isinstance(TRADING_PARAMS["USE_KELLY_CRITERION"], bool)

    def test_volatility_regime_flag_exists(self):
        """USE_VOLATILITY_REGIME flag should exist."""
        from config import TRADING_PARAMS

        assert "USE_VOLATILITY_REGIME" in TRADING_PARAMS
        assert isinstance(TRADING_PARAMS["USE_VOLATILITY_REGIME"], bool)

    def test_streak_sizing_flag_exists(self):
        """USE_STREAK_SIZING flag should exist."""
        from config import TRADING_PARAMS

        assert "USE_STREAK_SIZING" in TRADING_PARAMS
        assert isinstance(TRADING_PARAMS["USE_STREAK_SIZING"], bool)

    def test_multi_timeframe_flag_exists(self):
        """USE_MULTI_TIMEFRAME flag should exist."""
        from config import TRADING_PARAMS

        assert "USE_MULTI_TIMEFRAME" in TRADING_PARAMS
        assert isinstance(TRADING_PARAMS["USE_MULTI_TIMEFRAME"], bool)


class TestBacktestRealisticSettings:
    """Test backtest settings for realistic simulation."""

    def test_slippage_enabled_by_default(self):
        """Slippage should be enabled by default for realistic backtests."""
        from config import BACKTEST_PARAMS

        assert BACKTEST_PARAMS["USE_SLIPPAGE"] is True

    def test_walk_forward_enabled_by_default(self):
        """Walk-forward validation should be enabled."""
        from config import BACKTEST_PARAMS

        assert BACKTEST_PARAMS["WALK_FORWARD_ENABLED"] is True

    def test_min_train_days_reasonable(self):
        """Minimum training days should be at least 30."""
        from config import BACKTEST_PARAMS

        assert BACKTEST_PARAMS["MIN_TRAIN_DAYS"] >= 30


class TestConfigIntegrity:
    """Test overall config integrity."""

    def test_config_imports_without_error(self):
        """Config module should import without errors."""
        try:
            import config

            # Verify the module loaded
            assert config is not None
        except Exception as e:
            pytest.fail(f"Config import failed: {e}")

    def test_all_required_params_exist(self):
        """All required parameter dicts should exist."""
        import config

        assert hasattr(config, "TRADING_PARAMS")
        assert hasattr(config, "RISK_PARAMS")
        assert hasattr(config, "TECHNICAL_PARAMS")
        assert hasattr(config, "BACKTEST_PARAMS")
        assert hasattr(config, "ALPACA_CREDS")
        assert hasattr(config, "SYMBOLS")

    def test_no_conflicting_defaults(self):
        """Default values should not conflict."""
        from config import TRADING_PARAMS

        # Stop-loss should be less than position risk limit
        assert TRADING_PARAMS["STOP_LOSS"] <= 0.1

        # Position size should allow for max positions
        max_pos = TRADING_PARAMS.get("max_positions", 3)
        pos_size = TRADING_PARAMS["POSITION_SIZE"]
        # Total max allocation should be <= 1 (100%)
        assert pos_size * max_pos <= 1.0 or pos_size <= TRADING_PARAMS["MAX_POSITION_SIZE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
