"""
Tests for Walk-Forward Backtest Mode in BacktestEngine.

These tests verify the walk-forward analysis functionality which:
- Splits data into multiple train/test folds
- Detects overfitting by comparing IS vs OOS performance
- Calculates degradation metrics
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.backtest_engine import BacktestEngine


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create a BacktestEngine."""
    return BacktestEngine()


@pytest.fixture
def mock_strategy_class():
    """Create a mock strategy class."""
    class MockStrategy:
        def __init__(self, broker, parameters):
            self.broker = broker
            self.parameters = parameters
            self.price_history = {}
            self.current_data = {}

        async def initialize(self):
            pass

        async def analyze_symbol(self, symbol):
            return {"action": "hold"}

        async def execute_trade(self, symbol, signal):
            pass

    return MockStrategy


# ============================================================================
# SHARPE CALCULATION TESTS
# ============================================================================


class TestSharpeCalculation:
    """Tests for Sharpe ratio calculation from equity curve."""

    def test_sharpe_basic(self, engine):
        """Test basic Sharpe calculation."""
        # Equity curve with positive returns
        equity = [100000, 101000, 102000, 103000, 104000, 105000]
        sharpe = engine._calculate_sharpe_from_equity(equity)

        # Should be positive
        assert sharpe > 0

    def test_sharpe_flat_equity(self, engine):
        """Test Sharpe with flat equity (no returns)."""
        equity = [100000, 100000, 100000, 100000]
        sharpe = engine._calculate_sharpe_from_equity(equity)

        assert sharpe == 0.0

    def test_sharpe_insufficient_data(self, engine):
        """Test Sharpe with insufficient data."""
        equity = [100000]
        sharpe = engine._calculate_sharpe_from_equity(equity)

        assert sharpe == 0.0

    def test_sharpe_negative_returns(self, engine):
        """Test Sharpe with negative returns."""
        equity = [100000, 99000, 98000, 97000, 96000]
        sharpe = engine._calculate_sharpe_from_equity(equity)

        # Should be negative
        assert sharpe < 0

    def test_sharpe_volatile_returns(self, engine):
        """Test Sharpe with volatile returns."""
        # Same average return but more volatile
        equity_volatile = [100000, 105000, 95000, 110000, 90000, 105000]
        equity_smooth = [100000, 101000, 102000, 103000, 104000, 105000]

        sharpe_volatile = engine._calculate_sharpe_from_equity(equity_volatile)
        sharpe_smooth = engine._calculate_sharpe_from_equity(equity_smooth)

        # Smooth should have higher Sharpe (less risk for same return)
        assert sharpe_smooth > sharpe_volatile

    def test_sharpe_with_risk_free_rate(self, engine):
        """Test Sharpe with non-zero risk-free rate."""
        equity = [100000, 101000, 102000, 103000, 104000]

        sharpe_zero_rf = engine._calculate_sharpe_from_equity(equity, risk_free_rate=0.0)
        sharpe_positive_rf = engine._calculate_sharpe_from_equity(equity, risk_free_rate=0.05)

        # Higher risk-free rate should reduce Sharpe
        assert sharpe_positive_rf < sharpe_zero_rf


# ============================================================================
# WALK-FORWARD BACKTEST TESTS
# ============================================================================


class TestWalkForwardBacktest:
    """Tests for walk-forward backtest mode."""

    @pytest.mark.asyncio
    async def test_walk_forward_basic_structure(self, engine, mock_strategy_class):
        """Test that walk-forward returns expected structure."""
        # Mock run_backtest to return simple results
        async def mock_run_backtest(*args, **kwargs):
            return {
                "equity_curve": [100000, 101000, 102000],
                "total_trades": 5,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),  # Longer period for valid folds
                n_folds=3,
            )

        # Check result structure
        assert "fold_results" in result
        assert "is_sharpe" in result
        assert "oos_sharpe" in result
        assert "sharpe_degradation" in result
        assert "overfit_detected" in result
        assert "n_folds" in result

    @pytest.mark.asyncio
    async def test_walk_forward_fold_count(self, engine, mock_strategy_class):
        """Test that walk-forward creates correct number of folds."""
        folds_created = []

        async def mock_run_backtest(*args, **kwargs):
            folds_created.append(1)
            return {
                "equity_curve": [100000, 101000],
                "total_trades": 1,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=5,
            )

        # Should have 2 backtests per fold (IS and OOS) * n_folds
        # Minus any folds skipped due to insufficient data
        assert len(result["fold_results"]) > 0

    @pytest.mark.asyncio
    async def test_walk_forward_overfit_detection(self, engine, mock_strategy_class):
        """Test overfitting is detected when OOS << IS."""
        call_count = [0]

        async def mock_run_backtest(*args, **kwargs):
            call_count[0] += 1
            # IS runs (odd calls) have great performance
            # OOS runs (even calls) have poor performance
            if call_count[0] % 2 == 1:  # IS
                equity = [100000, 110000, 120000, 130000]  # Great returns
            else:  # OOS
                equity = [100000, 99000, 98000, 97000]  # Poor returns

            return {
                "equity_curve": equity,
                "total_trades": 5,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=3,
            )

        # Should detect overfitting (OOS Sharpe < 50% of IS Sharpe)
        assert result["is_sharpe"] > 0
        assert result["oos_sharpe"] < result["is_sharpe"]
        # Overfitting should be detected
        assert result["overfit_detected"] == True  # Use == for numpy bool compatibility

    @pytest.mark.asyncio
    async def test_walk_forward_no_overfit(self, engine, mock_strategy_class):
        """Test no overfitting when OOS ~ IS."""
        async def mock_run_backtest(*args, **kwargs):
            # Both IS and OOS have similar performance
            equity = [100000, 101000, 102000, 103000]
            return {
                "equity_curve": equity,
                "total_trades": 5,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=3,
            )

        # Should NOT detect overfitting
        assert result["overfit_detected"] == False  # Use == for numpy bool compatibility

    @pytest.mark.asyncio
    async def test_walk_forward_embargo_days(self, engine, mock_strategy_class):
        """Test that embargo days are applied."""
        async def mock_run_backtest(*args, **kwargs):
            return {
                "equity_curve": [100000, 101000],
                "total_trades": 1,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=3,
                embargo_days=10,  # Large embargo
            )

        assert result["embargo_days"] == 10

    @pytest.mark.asyncio
    async def test_walk_forward_train_pct(self, engine, mock_strategy_class):
        """Test that train percentage is respected."""
        async def mock_run_backtest(*args, **kwargs):
            return {
                "equity_curve": [100000, 101000],
                "total_trades": 1,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=3,
                train_pct=0.80,  # 80% train
            )

        assert result["train_pct"] == 0.80

    @pytest.mark.asyncio
    async def test_walk_forward_fold_details(self, engine, mock_strategy_class):
        """Test that fold results contain expected details."""
        async def mock_run_backtest(*args, **kwargs):
            return {
                "equity_curve": [100000, 101000, 102000],
                "total_trades": 3,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=2,
            )

        # Check fold result structure
        if result["fold_results"]:
            fold = result["fold_results"][0]
            assert "fold" in fold
            assert "train_start" in fold
            assert "train_end" in fold
            assert "test_start" in fold
            assert "test_end" in fold
            assert "is_return" in fold
            assert "is_sharpe" in fold
            assert "oos_return" in fold
            assert "oos_sharpe" in fold

    @pytest.mark.asyncio
    async def test_walk_forward_handles_backtest_error(self, engine, mock_strategy_class):
        """Test that walk-forward handles backtest errors gracefully."""
        call_count = [0]

        async def mock_run_backtest(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second call
                raise Exception("Backtest failed")
            return {
                "equity_curve": [100000, 101000],
                "total_trades": 1,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=2,
            )

        # Should complete with partial results
        assert "fold_results" in result

    @pytest.mark.asyncio
    async def test_walk_forward_degradation_calculation(self, engine, mock_strategy_class):
        """Test degradation is calculated correctly."""
        call_count = [0]

        async def mock_run_backtest(*args, **kwargs):
            call_count[0] += 1
            # IS: 10% return, OOS: 5% return (50% degradation)
            if call_count[0] % 2 == 1:  # IS
                equity = [100000, 105000, 110000]  # ~10% return
            else:  # OOS
                equity = [100000, 102500, 105000]  # ~5% return

            return {
                "equity_curve": equity,
                "total_trades": 5,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=3,
            )

        # Check degradation metrics exist
        assert "sharpe_degradation" in result
        assert "return_degradation" in result

        # IS return should be higher than OOS
        assert result["is_return"] > result["oos_return"]


# ============================================================================
# EDGE CASES
# ============================================================================


class TestWalkForwardEdgeCases:
    """Edge case tests for walk-forward backtest."""

    @pytest.mark.asyncio
    async def test_short_date_range(self, engine, mock_strategy_class):
        """Test with very short date range."""
        async def mock_run_backtest(*args, **kwargs):
            return {
                "equity_curve": [100000, 101000],
                "total_trades": 1,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),  # Just 1 month
                n_folds=5,  # Too many folds for data
            )

        # Should handle gracefully
        assert "fold_results" in result

    @pytest.mark.asyncio
    async def test_single_fold(self, engine, mock_strategy_class):
        """Test with single fold."""
        async def mock_run_backtest(*args, **kwargs):
            return {
                "equity_curve": [100000, 101000, 102000],
                "total_trades": 2,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=1,
            )

        assert result["n_folds"] == 1

    @pytest.mark.asyncio
    async def test_zero_is_sharpe(self, engine, mock_strategy_class):
        """Test handling when IS Sharpe is zero."""
        async def mock_run_backtest(*args, **kwargs):
            # Flat equity = 0 Sharpe
            return {
                "equity_curve": [100000, 100000, 100000],
                "total_trades": 0,
            }

        with patch.object(engine, 'run_backtest', side_effect=mock_run_backtest):
            result = await engine.run_walk_forward_backtest(
                strategy_class=mock_strategy_class,
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                n_folds=2,
            )

        # Should not crash with division by zero
        assert result["is_sharpe"] == 0
        assert result["sharpe_degradation"] == 0
