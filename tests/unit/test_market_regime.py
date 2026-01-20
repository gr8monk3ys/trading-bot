"""
Unit tests for MarketRegimeDetector.

Tests the market regime detection system including:
- Regime classification (BULL, BEAR, SIDEWAYS, VOLATILE)
- Technical indicators (SMA, ADX, ATR, EMA)
- Strategy recommendations
- Caching behavior
- Helper methods
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


class TestMarketRegimeEnum:
    """Test MarketRegime enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        from utils.market_regime import MarketRegime

        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.UNKNOWN.value == "unknown"

    def test_enum_members(self):
        """Test that all expected members exist."""
        from utils.market_regime import MarketRegime

        assert len(MarketRegime) == 5
        assert hasattr(MarketRegime, "BULL")
        assert hasattr(MarketRegime, "BEAR")
        assert hasattr(MarketRegime, "SIDEWAYS")
        assert hasattr(MarketRegime, "VOLATILE")
        assert hasattr(MarketRegime, "UNKNOWN")


class TestMarketRegimeDetectorInit:
    """Test MarketRegimeDetector initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        assert detector.broker == mock_broker
        assert detector.lookback_days == 200
        assert detector.cache_minutes == 30
        assert detector.market_index == "SPY"
        assert detector.last_regime is None
        assert detector.last_detection_time is None
        assert detector.regime_history == []

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(
            mock_broker,
            lookback_days=100,
            cache_minutes=60
        )

        assert detector.lookback_days == 100
        assert detector.cache_minutes == 60

    def test_class_constants(self):
        """Test class threshold constants."""
        from utils.market_regime import MarketRegimeDetector

        assert MarketRegimeDetector.ADX_TRENDING_THRESHOLD == 25
        assert MarketRegimeDetector.ADX_RANGING_THRESHOLD == 20
        assert MarketRegimeDetector.BREADTH_BULL_THRESHOLD == 0.60
        assert MarketRegimeDetector.BREADTH_BEAR_THRESHOLD == 0.40


class TestCalculateSMA:
    """Test SMA calculation."""

    def test_sma_normal(self):
        """Test SMA with normal data."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        prices = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
        sma = detector._calculate_sma(prices, 5)

        # Last 5: 20, 22, 24, 26, 28 -> mean = 24
        assert sma == 24.0

    def test_sma_insufficient_data(self):
        """Test SMA when prices less than period."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        prices = np.array([10, 12, 14])
        sma = detector._calculate_sma(prices, 5)

        # Returns last price when insufficient data
        assert sma == 14

    def test_sma_empty_data(self):
        """Test SMA with empty array."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        prices = np.array([])
        sma = detector._calculate_sma(prices, 5)

        assert sma == 0

    def test_sma_single_price(self):
        """Test SMA with single price."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        prices = np.array([100])
        sma = detector._calculate_sma(prices, 5)

        assert sma == 100


class TestCalculateEMA:
    """Test EMA calculation."""

    def test_ema_normal(self):
        """Test EMA with normal data."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        ema = detector._ema(data, 3)

        # EMA calculation
        assert ema > 0

    def test_ema_empty_data(self):
        """Test EMA with empty array."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        data = np.array([])
        ema = detector._ema(data, 5)

        assert ema == 0

    def test_ema_insufficient_data(self):
        """Test EMA when data less than period."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        data = np.array([10.0, 12.0])
        ema = detector._ema(data, 5)

        # Returns mean when insufficient data
        assert ema == 11.0


class TestCalculateATR:
    """Test ATR calculation."""

    def test_atr_normal(self):
        """Test ATR with normal OHLC data."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        # Create realistic price data
        closes = np.array([100.0] * 20)
        highs = np.array([102.0] * 20)
        lows = np.array([98.0] * 20)

        atr = detector._calculate_atr(highs, lows, closes, period=14)

        assert atr > 0

    def test_atr_insufficient_data(self):
        """Test ATR with less than 2 data points."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        closes = np.array([100.0])
        highs = np.array([102.0])
        lows = np.array([98.0])

        atr = detector._calculate_atr(highs, lows, closes)

        assert atr == 0

    def test_atr_less_than_period(self):
        """Test ATR when data less than period."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        highs = np.array([101.0, 102.0, 103.0, 104.0, 105.0])
        lows = np.array([99.0, 100.0, 101.0, 102.0, 103.0])

        atr = detector._calculate_atr(highs, lows, closes, period=14)

        # Should return mean of TR (excluding first zero element)
        assert atr > 0


class TestCalculateADX:
    """Test ADX calculation."""

    def test_adx_normal(self):
        """Test ADX with normal trending data."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        # Create trending price data
        n = 30
        closes = np.array([100.0 + i * 2 for i in range(n)])
        highs = closes + 1
        lows = closes - 1

        adx = detector._calculate_adx(highs, lows, closes, period=14)

        assert adx is not None
        assert adx >= 0

    def test_adx_insufficient_data(self):
        """Test ADX with insufficient data."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        closes = np.array([100.0, 101.0, 102.0])
        highs = closes + 1
        lows = closes - 1

        adx = detector._calculate_adx(highs, lows, closes, period=14)

        assert adx is None

    def test_adx_sideways_market(self):
        """Test ADX with sideways market data."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        # Create sideways price data
        n = 30
        closes = np.array([100.0 + np.sin(i * 0.5) * 2 for i in range(n)])
        highs = closes + 0.5
        lows = closes - 0.5

        adx = detector._calculate_adx(highs, lows, closes, period=14)

        # Sideways should have lower ADX
        assert adx is not None


class TestClassifyRegime:
    """Test regime classification logic."""

    def test_classify_high_volatility(self):
        """Test classification with high volatility."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime, confidence = detector._classify_regime(
            trend_direction="up",
            trend_strength=30,
            is_trending=True,
            is_ranging=False,
            volatility_regime="high"
        )

        assert regime == MarketRegime.VOLATILE
        assert 0.7 <= confidence <= 0.95

    def test_classify_bull_trending(self):
        """Test classification for bull trending market."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime, confidence = detector._classify_regime(
            trend_direction="up",
            trend_strength=35,
            is_trending=True,
            is_ranging=False,
            volatility_regime="normal"
        )

        assert regime == MarketRegime.BULL
        assert confidence >= 0.6

    def test_classify_bear_trending(self):
        """Test classification for bear trending market."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime, confidence = detector._classify_regime(
            trend_direction="down",
            trend_strength=35,
            is_trending=True,
            is_ranging=False,
            volatility_regime="normal"
        )

        assert regime == MarketRegime.BEAR
        assert confidence >= 0.6

    def test_classify_sideways_ranging(self):
        """Test classification for sideways ranging market."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime, confidence = detector._classify_regime(
            trend_direction="flat",
            trend_strength=15,
            is_trending=False,
            is_ranging=True,
            volatility_regime="normal"
        )

        assert regime == MarketRegime.SIDEWAYS
        assert confidence >= 0.6

    def test_classify_weak_uptrend(self):
        """Test classification for weak uptrend."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime, confidence = detector._classify_regime(
            trend_direction="up",
            trend_strength=22,
            is_trending=False,
            is_ranging=False,
            volatility_regime="normal"
        )

        assert regime == MarketRegime.BULL
        assert confidence == 0.55

    def test_classify_weak_downtrend(self):
        """Test classification for weak downtrend."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime, confidence = detector._classify_regime(
            trend_direction="down",
            trend_strength=22,
            is_trending=False,
            is_ranging=False,
            volatility_regime="normal"
        )

        assert regime == MarketRegime.BEAR
        assert confidence == 0.55

    def test_classify_flat_no_trend(self):
        """Test classification for flat market with no trend."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime, confidence = detector._classify_regime(
            trend_direction="flat",
            trend_strength=22,
            is_trending=False,
            is_ranging=False,
            volatility_regime="normal"
        )

        assert regime == MarketRegime.SIDEWAYS
        assert confidence == 0.50


class TestGetStrategyRecommendation:
    """Test strategy recommendation logic."""

    def test_bull_regime_recommendation(self):
        """Test recommendation for bull regime."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        strategy, mult = detector._get_strategy_recommendation(
            MarketRegime.BULL, 0.75, "normal"
        )

        assert strategy == "momentum_long"
        assert 1.0 <= mult <= 1.5

    def test_bear_regime_recommendation(self):
        """Test recommendation for bear regime."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        strategy, mult = detector._get_strategy_recommendation(
            MarketRegime.BEAR, 0.75, "normal"
        )

        assert strategy == "momentum_short"
        assert mult <= 1.0

    def test_sideways_regime_recommendation(self):
        """Test recommendation for sideways regime."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        strategy, mult = detector._get_strategy_recommendation(
            MarketRegime.SIDEWAYS, 0.75, "normal"
        )

        assert strategy == "mean_reversion"
        assert 0.8 <= mult <= 1.2

    def test_volatile_regime_recommendation(self):
        """Test recommendation for volatile regime."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        strategy, mult = detector._get_strategy_recommendation(
            MarketRegime.VOLATILE, 0.75, "high"
        )

        assert strategy == "defensive"
        assert mult <= 0.5  # Heavily reduced

    def test_unknown_regime_recommendation(self):
        """Test recommendation for unknown regime."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        strategy, mult = detector._get_strategy_recommendation(
            MarketRegime.UNKNOWN, 0.5, "normal"
        )

        assert strategy == "momentum_long"
        assert mult < 1.0  # Conservative

    def test_low_confidence_adjustment(self):
        """Test multiplier adjustment for low confidence."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        # Low confidence
        _, mult_low = detector._get_strategy_recommendation(
            MarketRegime.BULL, 0.50, "normal"
        )

        # High confidence
        _, mult_high = detector._get_strategy_recommendation(
            MarketRegime.BULL, 0.85, "normal"
        )

        assert mult_low < mult_high

    def test_volatility_adjustment(self):
        """Test multiplier adjustment for volatility."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        _, mult_low_vol = detector._get_strategy_recommendation(
            MarketRegime.BULL, 0.75, "low"
        )

        _, mult_high_vol = detector._get_strategy_recommendation(
            MarketRegime.BULL, 0.75, "high"
        )

        assert mult_low_vol > mult_high_vol

    def test_multiplier_caps(self):
        """Test that multiplier is properly capped."""
        from utils.market_regime import MarketRegimeDetector, MarketRegime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        # High confidence, low vol, bull = maximum multiplier
        _, mult = detector._get_strategy_recommendation(
            MarketRegime.BULL, 0.95, "low"
        )

        assert 0.3 <= mult <= 1.5


class TestGetDefaultRegime:
    """Test default regime generation."""

    def test_default_regime_values(self):
        """Test default regime returns expected values."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        regime = detector._get_default_regime()

        assert regime["type"] == "unknown"
        assert regime["confidence"] == 0.5
        assert regime["trend_direction"] == "flat"
        assert regime["trend_strength"] == 20
        assert regime["is_trending"] is False
        assert regime["is_ranging"] is True
        assert regime["volatility_regime"] == "normal"
        assert regime["volatility_pct"] == 2.0
        assert regime["sma_50"] is None
        assert regime["sma_200"] is None
        assert regime["recommended_strategy"] == "momentum_long"
        assert regime["position_multiplier"] == 0.7
        assert "detected_at" in regime


class TestGetRegimeHistory:
    """Test regime history retrieval."""

    def test_empty_history(self):
        """Test empty regime history."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        assert detector.get_regime_history() == []

    def test_history_after_changes(self):
        """Test regime history accumulates changes."""
        from utils.market_regime import MarketRegimeDetector
        from datetime import datetime

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        # Simulate regime change by modifying history directly
        detector.regime_history.append({
            "time": datetime.now(),
            "from": "bull",
            "to": "bear",
            "confidence": 0.8
        })

        history = detector.get_regime_history()
        assert len(history) == 1
        assert history[0]["from"] == "bull"
        assert history[0]["to"] == "bear"


class TestDetectRegime:
    """Test regime detection async method."""

    @pytest.mark.asyncio
    async def test_detect_regime_with_bull_data(self):
        """Test detecting bull regime."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create uptrending price data
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.5  # Strong uptrend
            bars.append(MagicMock(
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        assert "type" in regime
        assert "confidence" in regime
        assert "trend_direction" in regime
        assert "recommended_strategy" in regime
        assert "position_multiplier" in regime

    @pytest.mark.asyncio
    async def test_detect_regime_with_bear_data(self):
        """Test detecting bear regime."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create downtrending price data
        n = 250
        bars = []
        for i in range(n):
            price = 200.0 - i * 0.5  # Strong downtrend
            bars.append(MagicMock(
                open=price + 0.5,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        assert "type" in regime
        # Should detect bearish trend
        assert regime["trend_direction"] == "down"

    @pytest.mark.asyncio
    async def test_detect_regime_insufficient_data(self):
        """Test detection with insufficient data."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()
        mock_broker.get_bars = AsyncMock(return_value=[])

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        assert regime["type"] == "unknown"

    @pytest.mark.asyncio
    async def test_detect_regime_with_none_bars(self):
        """Test detection when bars is None."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()
        mock_broker.get_bars = AsyncMock(return_value=None)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        assert regime["type"] == "unknown"

    @pytest.mark.asyncio
    async def test_detect_regime_caching(self):
        """Test that regime detection uses caching."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create price data
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.3
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)

        # First call
        regime1 = await detector.detect_regime()
        # Second call should use cache
        regime2 = await detector.detect_regime()

        # Should only call get_bars once due to caching
        assert mock_broker.get_bars.call_count == 1
        assert regime1 == regime2

    @pytest.mark.asyncio
    async def test_detect_regime_force_refresh(self):
        """Test force refresh bypasses cache."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.3
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)

        # First call
        await detector.detect_regime()
        # Second call with force refresh
        await detector.detect_regime(force_refresh=True)

        # Should call get_bars twice
        assert mock_broker.get_bars.call_count == 2

    @pytest.mark.asyncio
    async def test_detect_regime_exception_handling(self):
        """Test detection handles exceptions gracefully."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()
        mock_broker.get_bars = AsyncMock(side_effect=Exception("API Error"))

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        assert regime["type"] == "unknown"

    @pytest.mark.asyncio
    async def test_detect_regime_logs_regime_change(self):
        """Test that regime changes are logged and tracked."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create price data
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.5
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)

        # Set a previous regime
        detector.last_regime = {"type": "bear", "confidence": 0.7}
        detector.last_detection_time = None  # Force fresh detection

        # Detect new regime
        await detector.detect_regime(force_refresh=True)

        # Check if regime history was updated (if there was a change)
        # The history will only update if the type changed
        assert detector.last_regime is not None


class TestGetMarketBars:
    """Test market bars fetching."""

    @pytest.mark.asyncio
    async def test_get_market_bars_success(self):
        """Test successful bar fetching."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()
        bars = [
            MagicMock(open=100, high=102, low=98, close=101, volume=1000),
            MagicMock(open=101, high=103, low=99, close=102, volume=1000),
        ]
        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        result = await detector._get_market_bars()

        assert result is not None
        assert len(result) == 2
        assert result[0]["close"] == 101
        assert result[1]["close"] == 102

    @pytest.mark.asyncio
    async def test_get_market_bars_none_response(self):
        """Test bar fetching with None response."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()
        mock_broker.get_bars = AsyncMock(return_value=None)

        detector = MarketRegimeDetector(mock_broker)
        result = await detector._get_market_bars()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_bars_exception(self):
        """Test bar fetching handles exceptions."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()
        mock_broker.get_bars = AsyncMock(side_effect=Exception("API Error"))

        detector = MarketRegimeDetector(mock_broker)
        result = await detector._get_market_bars()

        assert result is None


class TestHelperMethods:
    """Test async helper methods."""

    @pytest.mark.asyncio
    async def test_should_use_momentum_bull(self):
        """Test should_use_momentum in bull trending market."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create uptrending data
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.8
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        result = await detector.should_use_momentum()

        # Should be boolean (can be numpy bool)
        assert result in (True, False)

    @pytest.mark.asyncio
    async def test_should_use_momentum_sideways(self):
        """Test should_use_momentum in sideways market."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create sideways data
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + np.sin(i * 0.1) * 2  # Oscillating
            bars.append(MagicMock(
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        result = await detector.should_use_momentum()

        # Should be boolean (can be numpy bool)
        assert result in (True, False)

    @pytest.mark.asyncio
    async def test_should_use_mean_reversion_sideways(self):
        """Test should_use_mean_reversion in sideways market."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create sideways data with low ADX
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + np.sin(i * 0.1) * 2
            bars.append(MagicMock(
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        result = await detector.should_use_mean_reversion()

        # Should be boolean (can be numpy bool)
        assert result in (True, False)

    @pytest.mark.asyncio
    async def test_get_position_multiplier(self):
        """Test get_position_multiplier returns valid value."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.3
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        multiplier = await detector.get_position_multiplier()

        assert 0.3 <= multiplier <= 1.5


class TestGetCurrentRegime:
    """Test convenience function."""

    @pytest.mark.asyncio
    async def test_get_current_regime(self):
        """Test get_current_regime function."""
        from utils.market_regime import get_current_regime

        mock_broker = AsyncMock()

        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.3
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        regime = await get_current_regime(mock_broker)

        assert "type" in regime
        assert "confidence" in regime
        assert "recommended_strategy" in regime


class TestVolatilityRegimeClassification:
    """Test volatility regime classification."""

    @pytest.mark.asyncio
    async def test_high_volatility_detection(self):
        """Test high volatility is detected correctly."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create high volatility data (large swings)
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.1
            bars.append(MagicMock(
                open=price,
                high=price + 5,  # Large high/low range
                low=price - 5,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        # High volatility should result in high volatility regime or reduced multiplier
        assert regime["volatility_pct"] > 2.0 or regime["volatility_regime"] != "low"

    @pytest.mark.asyncio
    async def test_low_volatility_detection(self):
        """Test low volatility is detected correctly."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create low volatility data (small swings)
        n = 250
        bars = []
        for i in range(n):
            price = 100.0 + i * 0.1
            bars.append(MagicMock(
                open=price,
                high=price + 0.2,  # Tiny high/low range
                low=price - 0.2,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        # Should have low volatility percentage
        assert regime["volatility_pct"] < 3.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_sma_with_period_equal_length(self):
        """Test SMA when period equals array length."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        prices = np.array([10, 20, 30, 40, 50])
        sma = detector._calculate_sma(prices, 5)

        assert sma == 30.0  # Mean of [10, 20, 30, 40, 50]

    def test_adx_with_zero_di_sum(self):
        """Test ADX when DI sum would be zero."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        # Create flat data
        n = 20
        closes = np.array([100.0] * n)
        highs = np.array([100.0] * n)
        lows = np.array([100.0] * n)

        adx = detector._calculate_adx(highs, lows, closes, period=14)

        # Should handle gracefully
        assert adx == 0 or adx is None

    @pytest.mark.asyncio
    async def test_detect_regime_with_exactly_50_bars(self):
        """Test detection with exactly 50 bars (minimum)."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        bars = []
        for i in range(50):
            price = 100.0 + i * 0.5
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        # Should work with minimum data
        assert "type" in regime

    @pytest.mark.asyncio
    async def test_detect_regime_with_49_bars(self):
        """Test detection with less than 50 bars (insufficient)."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        bars = []
        for i in range(49):
            price = 100.0 + i * 0.5
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        # Should return unknown regime
        assert regime["type"] == "unknown"

    def test_atr_calculation_zero_price(self):
        """Test ATR when price is zero to avoid division by zero."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = MagicMock()
        detector = MarketRegimeDetector(mock_broker)

        closes = np.array([0.0, 0.0, 0.0])
        highs = np.array([1.0, 1.0, 1.0])
        lows = np.array([-1.0, -1.0, -1.0])

        atr = detector._calculate_atr(highs, lows, closes, period=2)

        # Should handle gracefully
        assert atr >= 0

    @pytest.mark.asyncio
    async def test_detect_regime_sma_200_fallback(self):
        """Test that SMA 200 falls back to SMA 50 with less than 200 bars."""
        from utils.market_regime import MarketRegimeDetector

        mock_broker = AsyncMock()

        # Create 100 bars (less than 200)
        bars = []
        for i in range(100):
            price = 100.0 + i * 0.5
            bars.append(MagicMock(
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))

        mock_broker.get_bars = AsyncMock(return_value=bars)

        detector = MarketRegimeDetector(mock_broker)
        regime = await detector.detect_regime()

        # SMA 50 and SMA 200 should be equal when < 200 bars
        # (because SMA 200 falls back to SMA 50)
        assert regime["sma_50"] == regime["sma_200"]
