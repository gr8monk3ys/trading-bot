"""
Tests for Quant Feature Integration

Tests the integration of:
- SignalAggregator with AdaptiveStrategy
- FactorPortfolio with symbol selection
- PortfolioOptimizer with position sizing
- ML predictions with signal enrichment
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestSignalAggregatorIntegration:
    """Test SignalAggregator integration with AdaptiveStrategy."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = AsyncMock()
        broker.get_account = AsyncMock(return_value=MagicMock(equity=100000, buying_power=200000))
        broker.get_bars = AsyncMock(return_value=[])
        broker.get_positions = AsyncMock(return_value=[])
        return broker

    @pytest.fixture
    def adaptive_strategy(self, mock_broker):
        """Create AdaptiveStrategy with quant features enabled."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy(
            broker=mock_broker,
            symbols=["AAPL", "MSFT"],
            enable_signal_aggregator=True,
            enable_portfolio_optimizer=True,
        )
        return strategy

    def test_adaptive_strategy_has_quant_flags(self, adaptive_strategy):
        """Test that AdaptiveStrategy has quant feature flags."""
        assert hasattr(adaptive_strategy, "enable_signal_aggregator")
        assert hasattr(adaptive_strategy, "enable_portfolio_optimizer")
        assert adaptive_strategy.enable_signal_aggregator is True
        assert adaptive_strategy.enable_portfolio_optimizer is True

    def test_adaptive_strategy_has_quant_attributes(self, adaptive_strategy):
        """Test that AdaptiveStrategy has quant component placeholders."""
        assert hasattr(adaptive_strategy, "signal_aggregator")
        assert hasattr(adaptive_strategy, "portfolio_optimizer")
        assert hasattr(adaptive_strategy, "cached_portfolio_weights")

    def test_enrich_signal_method_exists(self, adaptive_strategy):
        """Test that enrich_signal method exists and is async."""
        assert hasattr(adaptive_strategy, "enrich_signal")
        assert asyncio.iscoroutinefunction(adaptive_strategy.enrich_signal)

    def test_update_portfolio_weights_method_exists(self, adaptive_strategy):
        """Test that update_portfolio_weights method exists and is async."""
        assert hasattr(adaptive_strategy, "update_portfolio_weights")
        assert asyncio.iscoroutinefunction(adaptive_strategy.update_portfolio_weights)

    async def test_enrich_signal_passthrough_without_aggregator(self, adaptive_strategy):
        """Test enrich_signal passes through when aggregator not initialized."""
        # Without initialization, signal_aggregator is None
        technical_signal = {"action": "buy", "confidence": 0.7, "reason": "Test"}

        result = await adaptive_strategy.enrich_signal("AAPL", technical_signal)

        # Should pass through unchanged
        assert result["action"] == "buy"
        assert result["confidence"] == 0.7


class TestSignalAggregator:
    """Test SignalAggregator standalone functionality."""

    @pytest.fixture
    def signal_aggregator(self):
        """Create SignalAggregator."""
        from utils.signal_aggregator import SignalAggregator

        return SignalAggregator(
            broker=None,
            enable_sentiment=False,  # Disable to avoid API calls
            enable_ml=False,
            min_agreement=0.5,
        )

    def test_signal_aggregator_initialization(self, signal_aggregator):
        """Test SignalAggregator initializes correctly."""
        assert signal_aggregator.min_agreement == 0.5
        assert signal_aggregator.enable_sentiment is False
        assert signal_aggregator.enable_ml is False

    def test_signal_aggregator_has_default_weights(self, signal_aggregator):
        """Test SignalAggregator has default source weights."""
        from utils.signal_aggregator import SignalSource

        assert SignalSource.TECHNICAL in signal_aggregator.source_weights
        assert SignalSource.REGIME in signal_aggregator.source_weights

    def test_convert_technical_signal(self, signal_aggregator):
        """Test converting technical signal to SourceSignal."""
        from utils.signal_aggregator import SignalDirection

        tech_signal = {"action": "buy", "confidence": 0.8, "reason": "RSI oversold"}

        result = signal_aggregator._convert_technical_signal(tech_signal)

        assert result is not None
        assert result.direction == SignalDirection.BUY
        assert result.confidence == 0.8

    def test_convert_technical_signal_sell(self, signal_aggregator):
        """Test converting sell signal."""
        from utils.signal_aggregator import SignalDirection

        tech_signal = {"action": "sell", "confidence": 0.6}

        result = signal_aggregator._convert_technical_signal(tech_signal)

        assert result.direction == SignalDirection.SELL

    def test_convert_technical_signal_neutral(self, signal_aggregator):
        """Test converting neutral signal."""
        from utils.signal_aggregator import SignalDirection

        tech_signal = {"action": "neutral", "confidence": 0.5}

        result = signal_aggregator._convert_technical_signal(tech_signal)

        assert result.direction == SignalDirection.NEUTRAL


class TestFactorPortfolio:
    """Test FactorPortfolio for factor-based ranking."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker with price data."""
        broker = AsyncMock()

        # Create mock bars with proper attributes
        def create_bar(close, volume=1000000):
            bar = MagicMock()
            bar.close = close
            bar.volume = volume
            bar.open = close * 0.99
            bar.high = close * 1.01
            bar.low = close * 0.98
            return bar

        # Return 300 days of data for momentum calculation
        bars = [create_bar(100 + i * 0.1) for i in range(300)]
        broker.get_bars = AsyncMock(return_value=bars)

        return broker

    @pytest.fixture
    def factor_portfolio(self, mock_broker):
        """Create FactorPortfolio."""
        from factors.factor_portfolio import FactorPortfolio

        return FactorPortfolio(broker=mock_broker)

    def test_factor_portfolio_initialization(self, factor_portfolio):
        """Test FactorPortfolio initializes correctly."""
        assert factor_portfolio.broker is not None
        assert hasattr(factor_portfolio, "factors")

    def test_factor_portfolio_has_default_weights(self, factor_portfolio):
        """Test FactorPortfolio has default factor weights."""
        assert hasattr(factor_portfolio, "DEFAULT_WEIGHTS")
        assert len(factor_portfolio.DEFAULT_WEIGHTS) > 0

    async def test_get_composite_rankings(self, factor_portfolio, mock_broker):
        """Test getting composite rankings for symbols."""
        symbols = ["AAPL", "MSFT"]

        # This should run without errors
        rankings = await factor_portfolio.get_composite_rankings(symbols)

        # Should return dict
        assert isinstance(rankings, dict)


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer for position sizing."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = AsyncMock()

        def create_bar(close):
            bar = MagicMock()
            bar.close = close
            return bar

        # Return price data
        broker.get_bars = AsyncMock(return_value=[create_bar(100 + i * 0.5) for i in range(100)])
        return broker

    @pytest.fixture
    def portfolio_optimizer(self, mock_broker):
        """Create PortfolioOptimizer."""
        from utils.portfolio_optimizer import PortfolioOptimizer

        return PortfolioOptimizer(
            broker=mock_broker,
            lookback_days=60,
            risk_free_rate=0.05,
        )

    def test_portfolio_optimizer_initialization(self, portfolio_optimizer):
        """Test PortfolioOptimizer initializes correctly."""
        assert portfolio_optimizer.lookback_days == 60
        assert portfolio_optimizer.risk_free_rate == 0.05

    async def test_optimize_risk_parity(self, portfolio_optimizer):
        """Test risk parity optimization."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        result = await portfolio_optimizer.optimize_risk_parity(symbols=symbols, max_weight=0.5)

        # Should return OptimizationResult with weights
        assert hasattr(result, "weights")
        assert hasattr(result, "method")
        assert len(result.weights) > 0

    async def test_optimize_mean_variance(self, portfolio_optimizer):
        """Test mean-variance optimization."""
        symbols = ["AAPL", "MSFT"]

        result = await portfolio_optimizer.optimize_mean_variance(symbols=symbols, max_weight=0.5)

        assert hasattr(result, "weights")
        assert hasattr(result, "sharpe_ratio")


class TestStrategyPerformanceTracker:
    """Test StrategyPerformanceTracker for adaptive weights."""

    @pytest.fixture
    def performance_tracker(self):
        """Create StrategyPerformanceTracker."""
        from utils.strategy_performance_tracker import StrategyPerformanceTracker

        return StrategyPerformanceTracker(
            lookback_trades=100,
            min_trades_for_adjustment=20,
        )

    def test_tracker_initialization(self, performance_tracker):
        """Test tracker initializes correctly."""
        assert performance_tracker.lookback_trades == 100
        assert performance_tracker.min_trades == 20

    def test_record_signal_outcome(self, performance_tracker):
        """Test recording signal outcomes."""
        performance_tracker.record_signal_outcome(
            strategy_name="MomentumStrategy",
            symbol="AAPL",
            predicted="buy",
            actual="up",
            pnl=100.0,
            confidence=0.7,
            regime="bull",
        )

        # Should have recorded the outcome
        assert "MomentumStrategy" in performance_tracker._outcomes
        assert len(performance_tracker._outcomes["MomentumStrategy"]) == 1

    def test_get_adaptive_weights(self, performance_tracker):
        """Test getting adaptive weights."""
        # Record enough outcomes for adjustment
        for i in range(25):
            performance_tracker.record_signal_outcome(
                strategy_name="MomentumStrategy",
                symbol="AAPL",
                predicted="buy",
                actual="up" if i % 2 == 0 else "down",
                pnl=10.0 if i % 2 == 0 else -5.0,
                confidence=0.7,
            )

        weights = performance_tracker.get_adaptive_weights(["MomentumStrategy"])

        assert "MomentumStrategy" in weights
        assert 0 < weights["MomentumStrategy"] <= 1.0


class TestEnsembleVotingStrategy:
    """Test EnsembleVotingStrategy."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = AsyncMock()
        broker.get_account = AsyncMock(return_value=MagicMock(equity=100000))
        broker.get_bars = AsyncMock(return_value=[])
        return broker

    def test_ensemble_voting_initialization(self, mock_broker):
        """Test EnsembleVotingStrategy initializes correctly."""
        from strategies.ensemble_voting_strategy import EnsembleVotingStrategy

        strategy = EnsembleVotingStrategy(
            broker=mock_broker,
            sub_strategies=[],
            min_agreement=0.6,
        )

        assert strategy.min_agreement == 0.6
        assert strategy.use_adaptive_weights is True

    def test_ensemble_has_performance_tracker(self, mock_broker):
        """Test EnsembleVotingStrategy has performance tracker."""
        from strategies.ensemble_voting_strategy import EnsembleVotingStrategy

        strategy = EnsembleVotingStrategy(
            broker=mock_broker,
            sub_strategies=[],
        )

        assert hasattr(strategy, "performance_tracker")
        assert strategy.performance_tracker is not None


class TestCompositeSignal:
    """Test CompositeSignal dataclass."""

    def test_composite_signal_is_actionable(self):
        """Test CompositeSignal actionability check."""
        from utils.signal_aggregator import CompositeSignal, SignalDirection

        # Actionable signal
        signal = CompositeSignal(
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.7,
            agreement_pct=0.6,
            contributing_signals=[],
        )

        assert signal.is_actionable(min_confidence=0.5, min_agreement=0.5) is True

    def test_composite_signal_not_actionable_low_confidence(self):
        """Test CompositeSignal with low confidence."""
        from utils.signal_aggregator import CompositeSignal, SignalDirection

        signal = CompositeSignal(
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.3,  # Too low
            agreement_pct=0.6,
            contributing_signals=[],
        )

        assert signal.is_actionable(min_confidence=0.5, min_agreement=0.5) is False

    def test_composite_signal_not_actionable_blocked(self):
        """Test CompositeSignal when blocked."""
        from utils.signal_aggregator import CompositeSignal, SignalDirection

        signal = CompositeSignal(
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.7,
            agreement_pct=0.6,
            contributing_signals=[],
            blocked_by=["FOMC event"],
        )

        assert signal.is_actionable() is False


class TestMLPipeline:
    """Test MLPipeline for walk-forward validation."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = AsyncMock()
        broker.get_bars = AsyncMock(return_value=[])
        return broker

    def test_ml_pipeline_initialization(self, mock_broker):
        """Test MLPipeline initializes correctly."""
        from ml.ml_pipeline import MLPipeline

        pipeline = MLPipeline(
            broker=mock_broker,
            max_overfit_ratio=1.5,
            min_sharpe=0.3,
        )

        assert pipeline.max_overfit_ratio == 1.5
        assert pipeline.min_sharpe == 0.3

    def test_ml_pipeline_has_feature_engineer(self, mock_broker):
        """Test MLPipeline has feature engineer."""
        from ml.ml_pipeline import MLPipeline

        pipeline = MLPipeline(broker=mock_broker)

        assert hasattr(pipeline, "feature_engineer")


class TestAlphaMonitor:
    """Test AlphaMonitor for decay detection."""

    def test_alpha_monitor_initialization(self):
        """Test AlphaMonitor initializes correctly."""
        from utils.portfolio_optimizer import AlphaMonitor

        monitor = AlphaMonitor(
            min_sharpe=0.5,
            lookback_trades=100,
        )

        assert monitor.min_sharpe == 0.5
        assert monitor.lookback_trades == 100

    def test_alpha_monitor_record_trade(self):
        """Test recording trade for performance tracking."""
        from utils.portfolio_optimizer import AlphaMonitor

        monitor = AlphaMonitor()

        monitor.record_trade(
            strategy_name="Momentum",
            symbol="AAPL",
            pnl=100.0,
            pnl_pct=0.01,
            signal_strength=0.8,
        )

        # Trade history is a list of dicts
        assert len(monitor._trade_history) == 1
        assert monitor._trade_history[0]["strategy"] == "Momentum"

    def test_alpha_monitor_check_decay(self):
        """Test checking for alpha decay."""
        from utils.portfolio_optimizer import AlphaMonitor

        monitor = AlphaMonitor()

        # Record enough trades for rolling window analysis
        for i in range(120):  # Need enough for rolling window
            monitor.record_trade(
                strategy_name="Momentum",
                symbol="AAPL",
                pnl=10.0 if i < 80 else -5.0,  # Declining performance
                pnl_pct=0.01 if i < 80 else -0.005,
                signal_strength=0.8,
            )

        result = monitor.check_alpha_decay("Momentum")
        assert isinstance(result, dict)
        # Check for expected keys in result
        assert "has_sufficient_data" in result or "alert" in result
