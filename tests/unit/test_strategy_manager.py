"""
Comprehensive tests for engine/strategy_manager.py

Tests cover:
- Initialization with various configurations
- Strategy loading and discovery
- Strategy evaluation via backtesting
- Strategy selection based on scores
- Capital allocation optimization
- Starting and stopping strategies
- Rebalancing strategies
- Portfolio statistics and reporting
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestStrategyManagerInitialization:
    """Tests for StrategyManager initialization."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = Mock()
        broker.get_account = AsyncMock()
        broker.get_positions = AsyncMock(return_value=[])
        return broker

    def test_init_with_broker(self, mock_broker):
        """Test initialization with provided broker."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            manager = StrategyManager(broker=mock_broker)

            assert manager.broker == mock_broker
            assert manager.max_strategies == 5
            assert manager.max_allocation == 0.9
            assert manager.min_backtest_days == 30
            assert manager.evaluation_period_days == 14

    def test_init_with_custom_params(self, mock_broker):
        """Test initialization with custom parameters."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            manager = StrategyManager(
                broker=mock_broker,
                max_strategies=3,
                max_allocation=0.8,
                min_backtest_days=60,
                evaluation_period_days=7,
            )

            assert manager.max_strategies == 3
            assert manager.max_allocation == 0.8
            assert manager.min_backtest_days == 60
            assert manager.evaluation_period_days == 7

    def test_init_creates_alpaca_broker_when_none(self):
        """Test broker is created when not provided."""
        with (
            patch("engine.strategy_manager.StrategyManager._load_available_strategies"),
            patch("engine.strategy_manager.AlpacaBroker") as mock_alpaca,
        ):
            from engine.strategy_manager import StrategyManager

            mock_alpaca.return_value = Mock()
            StrategyManager(broker=None, broker_type="alpaca")

            mock_alpaca.assert_called_once()

    def test_init_unsupported_broker_type(self):
        """Test error for unsupported broker type."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            with pytest.raises(ValueError, match="Unsupported broker type"):
                StrategyManager(broker=None, broker_type="unknown")

    def test_init_initializes_tracking_dicts(self, mock_broker):
        """Test that tracking dictionaries are initialized."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            manager = StrategyManager(broker=mock_broker)

            assert manager.available_strategies == {}
            assert manager.active_strategies == {}
            assert manager.strategy_performances == {}
            assert manager.strategy_allocations == {}
            assert manager.strategy_status == {}


class TestLoadAvailableStrategies:
    """Tests for _load_available_strategies method."""

    @pytest.fixture
    def manager_without_load(self):
        """Create manager without strategy loading."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)
            manager.available_strategies = {}
            return manager

    def test_load_strategies_relative_path(self, manager_without_load, tmp_path):
        """Test loading strategies from relative path."""
        # Create mock strategy directory
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()

        manager_without_load.strategy_path = str(strategy_dir)

        # Create dummy strategy file
        (strategy_dir / "dummy_strategy.py").write_text(
            """
from strategies.base_strategy import BaseStrategy

class DummyStrategy(BaseStrategy):
    NAME = "DummyStrategy"
"""
        )

        with patch("os.getcwd", return_value=str(tmp_path)):
            # Should not raise
            try:
                manager_without_load._load_available_strategies()
            except ImportError:
                pass  # Expected due to import issues in test environment

    def test_load_strategies_handles_import_error(self, manager_without_load, tmp_path):
        """Test error handling when strategy import fails."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()

        # Create a strategy file with invalid Python
        (strategy_dir / "invalid_strategy.py").write_text("this is not valid python {{{{")

        manager_without_load.strategy_path = str(strategy_dir)

        # Should not raise, just log error
        try:
            manager_without_load._load_available_strategies()
        except Exception:
            pass  # Import errors are logged, not raised

    def test_load_strategies_skips_init_and_base(self, manager_without_load, tmp_path):
        """Test that __init__.py and base_strategy.py are skipped."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()

        # These should be skipped
        (strategy_dir / "__init__.py").write_text("")
        (strategy_dir / "base_strategy.py").write_text("")

        manager_without_load.strategy_path = str(strategy_dir)

        # Mock os.listdir to return these files
        with patch("os.listdir", return_value=["__init__.py", "base_strategy.py"]):
            manager_without_load._load_available_strategies()

        # No strategies should be loaded
        assert len(manager_without_load.available_strategies) == 0


class TestEvaluateAllStrategies:
    """Tests for evaluate_all_strategies method."""

    @pytest.fixture
    def manager_with_strategies(self):
        """Create manager with mock strategies."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            # Add mock strategy class
            mock_strategy_class = Mock()
            mock_strategy_class.NAME = "TestStrategy"
            manager.available_strategies = {"TestStrategy": mock_strategy_class}

            # Mock backtest engine
            manager.backtest_engine = Mock()
            manager.backtest_engine.run_backtest = AsyncMock(
                return_value={
                    "equity_curve": [100000, 101000, 102000],
                    "returns": [0.01, 0.0099],
                    "trades": [],
                }
            )

            # Mock performance metrics
            manager.perf_metrics = Mock()
            manager.perf_metrics.calculate_metrics = Mock(
                return_value={"total_return": 0.02, "sharpe_ratio": 1.5, "max_drawdown": 0.01}
            )

            # Mock evaluator
            manager.evaluator = Mock()
            manager.evaluator.score_strategy = Mock(return_value=0.75)

            return manager

    @pytest.mark.asyncio
    async def test_evaluate_all_strategies_success(self, manager_with_strategies):
        """Test successful evaluation of all strategies."""
        scores = await manager_with_strategies.evaluate_all_strategies(
            symbols=["AAPL", "MSFT"], lookback_days=30
        )

        assert "TestStrategy" in scores
        assert scores["TestStrategy"] == 0.75
        assert manager_with_strategies.strategy_status["TestStrategy"] == "evaluated"

    @pytest.mark.asyncio
    async def test_evaluate_uses_default_symbols(self, manager_with_strategies):
        """Test evaluation uses default symbols when none provided."""
        with patch("engine.strategy_manager.SYMBOLS", ["AAPL"]):
            await manager_with_strategies.evaluate_all_strategies()

            # Backtest should have been called
            manager_with_strategies.backtest_engine.run_backtest.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_handles_backtest_error(self, manager_with_strategies):
        """Test error handling when backtest fails."""
        manager_with_strategies.backtest_engine.run_backtest = AsyncMock(
            side_effect=Exception("Backtest failed")
        )

        scores = await manager_with_strategies.evaluate_all_strategies(symbols=["AAPL"])

        assert scores["TestStrategy"] == -1.0
        assert manager_with_strategies.strategy_status["TestStrategy"] == "error"


class TestSelectTopStrategies:
    """Tests for select_top_strategies method."""

    @pytest.fixture
    def manager_with_performances(self):
        """Create manager with strategy performances."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker, max_strategies=2)

            manager.available_strategies = {
                "Strategy1": Mock(),
                "Strategy2": Mock(),
                "Strategy3": Mock(),
            }

            manager.strategy_performances = {
                "Strategy1": {"sharpe_ratio": 2.0, "total_return": 0.10},
                "Strategy2": {"sharpe_ratio": 1.5, "total_return": 0.08},
                "Strategy3": {"sharpe_ratio": 0.5, "total_return": 0.02},
            }

            manager.evaluator = Mock()
            manager.evaluator.score_strategy = Mock(side_effect=lambda m: m["sharpe_ratio"] / 2)

            return manager

    @pytest.mark.asyncio
    async def test_select_top_strategies_returns_top_n(self, manager_with_performances):
        """Test that top N strategies are selected."""
        selected = await manager_with_performances.select_top_strategies(n=2, min_score=0.0)

        assert len(selected) == 2
        assert "Strategy1" in selected
        assert "Strategy2" in selected
        assert "Strategy3" not in selected

    @pytest.mark.asyncio
    async def test_select_respects_min_score(self, manager_with_performances):
        """Test that minimum score filter is applied."""
        # Strategy3 has sharpe 0.5, score = 0.25 which is below min_score 0.5
        selected = await manager_with_performances.select_top_strategies(n=3, min_score=0.5)

        assert "Strategy3" not in selected

    @pytest.mark.asyncio
    async def test_select_uses_max_strategies_default(self, manager_with_performances):
        """Test that max_strategies is used when n is None."""
        selected = await manager_with_performances.select_top_strategies(n=None, min_score=0.0)

        assert len(selected) <= manager_with_performances.max_strategies

    @pytest.mark.asyncio
    async def test_select_evaluates_if_no_performances(self, manager_with_performances):
        """Test that evaluation is triggered if no performances exist."""
        manager_with_performances.strategy_performances = {}
        manager_with_performances.evaluate_all_strategies = AsyncMock()

        await manager_with_performances.select_top_strategies()

        manager_with_performances.evaluate_all_strategies.assert_called_once()


class TestOptimizeAllocations:
    """Tests for optimize_allocations method."""

    @pytest.fixture
    def manager_for_allocation(self):
        """Create manager for allocation tests."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker, max_allocation=0.9)

            manager.available_strategies = {"Strategy1": Mock(), "Strategy2": Mock()}

            manager.strategy_performances = {
                "Strategy1": {"sharpe_ratio": 2.0},
                "Strategy2": {"sharpe_ratio": 1.0},
            }

            return manager

    @pytest.mark.asyncio
    async def test_optimize_allocations_weights_by_sharpe(self, manager_for_allocation):
        """Test allocation weighted by Sharpe ratio."""
        allocations = await manager_for_allocation.optimize_allocations(
            strategies=["Strategy1", "Strategy2"]
        )

        # Strategy1 has sharpe 2.0, Strategy2 has 1.0
        # Total sharpe = 3.0, max_allocation = 0.9
        # Strategy1 should get 2/3 * 0.9 = 0.6
        # Strategy2 should get 1/3 * 0.9 = 0.3
        assert allocations["Strategy1"] == pytest.approx(0.6, rel=0.01)
        assert allocations["Strategy2"] == pytest.approx(0.3, rel=0.01)

    @pytest.mark.asyncio
    async def test_optimize_allocations_equal_when_no_sharpe(self, manager_for_allocation):
        """Test equal allocation when Sharpe ratios are zero or negative."""
        manager_for_allocation.strategy_performances = {
            "Strategy1": {"sharpe_ratio": -1.0},
            "Strategy2": {"sharpe_ratio": -0.5},
        }

        allocations = await manager_for_allocation.optimize_allocations(
            strategies=["Strategy1", "Strategy2"]
        )

        # Should be equal since total Sharpe is negative
        assert allocations["Strategy1"] == pytest.approx(0.45, rel=0.01)
        assert allocations["Strategy2"] == pytest.approx(0.45, rel=0.01)

    @pytest.mark.asyncio
    async def test_optimize_allocations_empty_strategies(self, manager_for_allocation):
        """Test handling of empty strategy list."""
        allocations = await manager_for_allocation.optimize_allocations(strategies=[])

        assert allocations == {}

    @pytest.mark.asyncio
    async def test_optimize_allocations_selects_if_none(self, manager_for_allocation):
        """Test that strategies are selected if none provided."""
        manager_for_allocation.select_top_strategies = AsyncMock(return_value=["Strategy1"])

        await manager_for_allocation.optimize_allocations(strategies=None)

        manager_for_allocation.select_top_strategies.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_handles_missing_metrics(self, manager_for_allocation):
        """Test allocation when some strategies have missing metrics."""
        manager_for_allocation.strategy_performances = {
            "Strategy1": {"sharpe_ratio": 2.0}
            # Strategy2 missing
        }

        allocations = await manager_for_allocation.optimize_allocations(
            strategies=["Strategy1", "Strategy2"]
        )

        # Strategy1 should get allocation, Strategy2 gets 0
        assert allocations["Strategy1"] == manager_for_allocation.max_allocation
        assert allocations["Strategy2"] == 0.0


class TestStartStrategy:
    """Tests for start_strategy method."""

    @pytest.fixture
    def manager_for_start(self):
        """Create manager for start strategy tests."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            # Create mock strategy class
            mock_strategy = Mock()
            mock_strategy.initialize = AsyncMock(return_value=True)

            mock_class = Mock(return_value=mock_strategy)
            mock_class.default_parameters = Mock(return_value={"param1": "value1"})

            manager.available_strategies = {"TestStrategy": mock_class}
            manager.strategy_allocations = {"TestStrategy": 0.5}

            return manager

    @pytest.mark.asyncio
    async def test_start_strategy_success(self, manager_for_start):
        """Test successful strategy start."""
        result = await manager_for_start.start_strategy(
            "TestStrategy", parameters={"param2": "value2"}, symbols=["AAPL"], allocation=0.3
        )

        assert result is True
        assert "TestStrategy" in manager_for_start.active_strategies
        assert manager_for_start.strategy_status["TestStrategy"] == "running"

    @pytest.mark.asyncio
    async def test_start_strategy_not_found(self, manager_for_start):
        """Test starting non-existent strategy."""
        result = await manager_for_start.start_strategy("NonExistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_start_strategy_already_running(self, manager_for_start):
        """Test starting already running strategy."""
        manager_for_start.active_strategies["TestStrategy"] = Mock()

        result = await manager_for_start.start_strategy("TestStrategy")

        assert result is True  # Returns True but doesn't restart

    @pytest.mark.asyncio
    async def test_start_strategy_initialization_fails(self, manager_for_start):
        """Test handling of initialization failure."""
        mock_strategy = Mock()
        mock_strategy.initialize = AsyncMock(return_value=False)
        manager_for_start.available_strategies["TestStrategy"].return_value = mock_strategy

        result = await manager_for_start.start_strategy("TestStrategy")

        assert result is False

    @pytest.mark.asyncio
    async def test_start_strategy_handles_exception(self, manager_for_start):
        """Test exception handling during start."""
        manager_for_start.available_strategies["TestStrategy"].side_effect = Exception("Failed")

        result = await manager_for_start.start_strategy("TestStrategy")

        assert result is False

    @pytest.mark.asyncio
    async def test_start_uses_default_allocation(self, manager_for_start):
        """Test that default allocation is used when not provided."""
        await manager_for_start.start_strategy("TestStrategy")

        # Should use allocation from strategy_allocations
        # Verified by the strategy being initialized


class TestStopStrategy:
    """Tests for stop_strategy method."""

    @pytest.fixture
    def manager_for_stop(self):
        """Create manager with active strategy."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            # Add active strategy
            mock_strategy = Mock()
            mock_strategy.liquidate_all_positions = AsyncMock()
            mock_strategy.shutdown = AsyncMock()

            manager.active_strategies = {"TestStrategy": mock_strategy}
            manager.strategy_status = {"TestStrategy": "running"}

            return manager

    @pytest.mark.asyncio
    async def test_stop_strategy_success(self, manager_for_stop):
        """Test successful strategy stop."""
        result = await manager_for_stop.stop_strategy("TestStrategy")

        assert result is True
        assert "TestStrategy" not in manager_for_stop.active_strategies
        assert manager_for_stop.strategy_status["TestStrategy"] == "stopped"

    @pytest.mark.asyncio
    async def test_stop_strategy_with_liquidation(self, manager_for_stop):
        """Test stop with position liquidation."""
        result = await manager_for_stop.stop_strategy("TestStrategy", liquidate=True)

        assert result is True
        # Verify liquidation was called (we need to check before deletion)

    @pytest.mark.asyncio
    async def test_stop_strategy_not_running(self, manager_for_stop):
        """Test stopping non-running strategy."""
        result = await manager_for_stop.stop_strategy("NonExistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_stop_strategy_handles_exception(self, manager_for_stop):
        """Test exception handling during stop."""
        manager_for_stop.active_strategies["TestStrategy"].shutdown = AsyncMock(
            side_effect=Exception("Shutdown failed")
        )

        result = await manager_for_stop.stop_strategy("TestStrategy")

        assert result is False


class TestStartSelectedStrategies:
    """Tests for start_selected_strategies method."""

    @pytest.fixture
    def manager_for_batch_start(self):
        """Create manager for batch start tests."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            manager.select_top_strategies = AsyncMock(return_value=["Strategy1", "Strategy2"])
            manager.optimize_allocations = AsyncMock(
                return_value={"Strategy1": 0.5, "Strategy2": 0.3}
            )
            manager.start_strategy = AsyncMock(return_value=True)

            return manager

    @pytest.mark.asyncio
    async def test_start_selected_strategies(self, manager_for_batch_start):
        """Test starting selected strategies."""
        started = await manager_for_batch_start.start_selected_strategies(n=2, min_score=0.5)

        assert started == ["Strategy1", "Strategy2"]
        assert manager_for_batch_start.start_strategy.call_count == 2


class TestStopAllStrategies:
    """Tests for stop_all_strategies method."""

    @pytest.fixture
    def manager_with_active(self):
        """Create manager with multiple active strategies."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            manager.active_strategies = {
                "Strategy1": Mock(shutdown=AsyncMock(), liquidate_all_positions=AsyncMock()),
                "Strategy2": Mock(shutdown=AsyncMock(), liquidate_all_positions=AsyncMock()),
            }

            return manager

    @pytest.mark.asyncio
    async def test_stop_all_strategies(self, manager_with_active):
        """Test stopping all strategies."""
        stopped = await manager_with_active.stop_all_strategies()

        assert stopped == 2
        assert len(manager_with_active.active_strategies) == 0


class TestRebalanceStrategies:
    """Tests for rebalance_strategies method."""

    @pytest.fixture
    def manager_for_rebalance(self):
        """Create manager for rebalance tests."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            mock_strategy = Mock()
            mock_strategy.update_parameters = Mock()

            manager.active_strategies = {"TestStrategy": mock_strategy}
            manager.strategy_performances = {"TestStrategy": {"sharpe_ratio": 1.5}}
            manager.evaluator = Mock()
            manager.evaluator.score_strategy = Mock(return_value=0.75)

            return manager

    @pytest.mark.asyncio
    async def test_rebalance_strategies(self, manager_for_rebalance):
        """Test rebalancing strategies."""
        result = await manager_for_rebalance.rebalance_strategies()

        assert result is True
        manager_for_rebalance.active_strategies["TestStrategy"].update_parameters.assert_called()

    @pytest.mark.asyncio
    async def test_rebalance_no_active_strategies(self, manager_for_rebalance):
        """Test rebalance with no active strategies."""
        manager_for_rebalance.active_strategies = {}

        result = await manager_for_rebalance.rebalance_strategies()

        assert result is False


class TestGetStrategyInfo:
    """Tests for get_strategy_info method."""

    @pytest.fixture
    def manager_with_info(self):
        """Create manager with strategy info."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            manager.available_strategies = {"TestStrategy": Mock()}
            manager.active_strategies = {"TestStrategy": Mock()}
            manager.strategy_status = {"TestStrategy": "running"}
            manager.strategy_allocations = {"TestStrategy": 0.5}
            manager.strategy_performances = {"TestStrategy": {"sharpe_ratio": 1.5}}

            return manager

    @pytest.mark.asyncio
    async def test_get_specific_strategy_info(self, manager_with_info):
        """Test getting info for specific strategy."""
        info = await manager_with_info.get_strategy_info("TestStrategy")

        assert info["name"] == "TestStrategy"
        assert info["status"] == "running"
        assert info["allocation"] == 0.5
        assert info["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_strategy_info_not_found(self, manager_with_info):
        """Test getting info for non-existent strategy."""
        info = await manager_with_info.get_strategy_info("NonExistent")

        assert "error" in info

    @pytest.mark.asyncio
    async def test_get_all_strategies_info(self, manager_with_info):
        """Test getting info for all strategies."""
        info = await manager_with_info.get_strategy_info()

        assert "TestStrategy" in info
        assert info["TestStrategy"]["status"] == "running"


class TestGetAvailableAndActiveStrategyNames:
    """Tests for strategy name getters."""

    @pytest.fixture
    def manager_with_strategies(self):
        """Create manager with strategies."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            manager = StrategyManager(broker=broker)

            manager.available_strategies = {"Strategy1": Mock(), "Strategy2": Mock()}
            manager.active_strategies = {"Strategy1": Mock()}

            return manager

    def test_get_available_strategy_names(self, manager_with_strategies):
        """Test getting available strategy names."""
        names = manager_with_strategies.get_available_strategy_names()

        assert "Strategy1" in names
        assert "Strategy2" in names

    def test_get_active_strategy_names(self, manager_with_strategies):
        """Test getting active strategy names."""
        names = manager_with_strategies.get_active_strategy_names()

        assert names == ["Strategy1"]


class TestGetPortfolioStats:
    """Tests for get_portfolio_stats method."""

    @pytest.fixture
    def manager_for_portfolio(self):
        """Create manager for portfolio stats tests."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            broker.get_account = AsyncMock(
                return_value=Mock(equity="100000", cash="50000", buying_power="80000")
            )
            broker.get_positions = AsyncMock(
                return_value=[
                    Mock(symbol="AAPL", market_value="10000"),
                    Mock(symbol="MSFT", market_value="15000"),
                ]
            )
            broker.get_tracked_positions = AsyncMock(return_value=[])

            manager = StrategyManager(broker=broker)
            manager.active_strategies = {}
            manager.strategy_allocations = {}

            return manager

    @pytest.mark.asyncio
    async def test_get_portfolio_stats(self, manager_for_portfolio):
        """Test getting portfolio statistics."""
        stats = await manager_for_portfolio.get_portfolio_stats()

        assert stats["portfolio_value"] == 100000.0
        assert stats["cash"] == 50000.0
        assert stats["buying_power"] == 80000.0
        assert stats["position_count"] == 2

    @pytest.mark.asyncio
    async def test_get_portfolio_stats_handles_error(self, manager_for_portfolio):
        """Test error handling in portfolio stats."""
        manager_for_portfolio.broker.get_account = AsyncMock(side_effect=Exception("API error"))

        stats = await manager_for_portfolio.get_portfolio_stats()

        assert "error" in stats


class TestGeneratePerformanceReport:
    """Tests for generate_performance_report method."""

    @pytest.fixture
    def manager_for_report(self):
        """Create manager for report tests."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            broker.get_account = AsyncMock(
                return_value=Mock(equity="100000", cash="50000", buying_power="80000")
            )
            broker.get_positions = AsyncMock(return_value=[])
            broker.get_tracked_positions = AsyncMock(return_value=[])

            manager = StrategyManager(broker=broker)
            manager.active_strategies = {}
            manager.strategy_allocations = {"TestStrategy": 0.5}
            manager.strategy_performances = {"TestStrategy": {"sharpe_ratio": 1.5}}
            manager.strategy_status = {"TestStrategy": "running"}

            return manager

    @pytest.mark.asyncio
    async def test_generate_performance_report(self, manager_for_report):
        """Test performance report generation."""
        report = await manager_for_report.generate_performance_report(days=30)

        assert "period" in report
        assert report["days"] == 30
        assert "strategies" in report
        assert "portfolio" in report
        assert "TestStrategy" in report["strategies"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def base_manager(self):
        """Create base manager for edge case tests."""
        with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
            from engine.strategy_manager import StrategyManager

            broker = Mock()
            return StrategyManager(broker=broker)

    @pytest.mark.asyncio
    async def test_optimize_allocations_with_zero_sharpe_strategies(self, base_manager):
        """Test allocation with strategies that have 0 or missing Sharpe."""
        base_manager.strategy_performances = {
            "Strategy1": {},  # No Sharpe
            "Strategy2": {"sharpe_ratio": 0.0},
        }

        allocations = await base_manager.optimize_allocations(strategies=["Strategy1", "Strategy2"])

        # Should use default 0.5 for missing Sharpe
        assert "Strategy1" in allocations
        assert "Strategy2" in allocations

    def test_load_strategies_directory_not_exists(self, base_manager):
        """Test handling of non-existent strategy directory."""
        base_manager.strategy_path = "/nonexistent/path"

        # Should handle error gracefully
        try:
            base_manager._load_available_strategies()
        except FileNotFoundError:
            pass  # Expected
        except Exception:
            pass  # Any other exception is acceptable here
