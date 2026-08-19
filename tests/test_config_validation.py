"""
Tests for config.py validation

Tests cover:
- Risk parameter validation
- Backtest parameter validation (research/ harness input)
- Symbols and credential structure

TRADING_PARAMS / TECHNICAL_PARAMS / SYMBOL_SELECTION were deleted in the
2026-08 slop cleanup: nothing in the production path ever read them
(strategies carry their own default_parameters()).
"""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRiskParamsValidation:
    """Test RISK_PARAMS validation."""

    def test_max_portfolio_risk_reasonable(self):
        """Max portfolio risk should be a valid fraction."""
        from config import RISK_PARAMS

        assert 0 < RISK_PARAMS["MAX_PORTFOLIO_RISK"] <= 1

    def test_max_position_risk_reasonable(self):
        """Max position risk should be a valid fraction."""
        from config import RISK_PARAMS

        assert 0 < RISK_PARAMS["MAX_POSITION_RISK"] <= 1

    def test_var_confidence_valid(self):
        """VaR confidence should be between 0.5 and 1."""
        from config import RISK_PARAMS

        assert 0.5 < RISK_PARAMS["VAR_CONFIDENCE"] < 1

    def test_max_correlation_valid(self):
        """Max correlation should be between -1 and 1."""
        from config import RISK_PARAMS

        corr = RISK_PARAMS["MAX_CORRELATION"]
        assert -1 <= corr <= 1


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
        """All surviving parameter dicts should exist."""
        import config

        assert hasattr(config, "RISK_PARAMS")
        assert hasattr(config, "BACKTEST_PARAMS")
        assert hasattr(config, "ALPACA_CREDS")
        assert hasattr(config, "SYMBOLS")

    def test_deleted_param_blocks_stay_deleted(self):
        """The dead config blocks must not quietly return."""
        import config

        assert not hasattr(config, "TRADING_PARAMS")
        assert not hasattr(config, "TECHNICAL_PARAMS")
        assert not hasattr(config, "SYMBOL_SELECTION")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
