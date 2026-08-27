"""
Tests for config.py validation

Tests cover:
- Risk parameter validation
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
        assert hasattr(config, "ALPACA_CREDS")
        assert hasattr(config, "SYMBOLS")

    def test_deleted_param_blocks_stay_deleted(self):
        """The dead config blocks must not quietly return."""
        import config

        assert not hasattr(config, "TRADING_PARAMS")
        assert not hasattr(config, "TECHNICAL_PARAMS")
        assert not hasattr(config, "SYMBOL_SELECTION")
        assert not hasattr(config, "BACKTEST_PARAMS")  # only research/ read it


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
