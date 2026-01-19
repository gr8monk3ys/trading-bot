"""
Comprehensive tests for utils/indicators.py

Tests cover:
- TechnicalIndicators class initialization
- Trend indicators: SMA, EMA, MACD, ADX, Parabolic SAR
- Momentum indicators: RSI, Stochastic, CCI, Williams %R, ROC
- Volatility indicators: Bollinger Bands, ATR, Keltner Channels
- Volume indicators: VWAP, OBV, Volume SMA, MFI
- Support/Resistance: Pivot Points, Fibonacci Retracements
- Composite indicators
- Quick analysis functions
"""

from datetime import datetime, timedelta

import numpy as np
import pytest


class TestTechnicalIndicatorsInitialization:
    """Tests for TechnicalIndicators class initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.2
        volume = np.random.randint(100000, 1000000, n).astype(np.float64)
        timestamps = [datetime.now() - timedelta(minutes=100 - i) for i in range(n)]

        return {
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
            "close": close.astype(np.float64),
            "open_": open_.astype(np.float64),
            "volume": volume,
            "timestamps": timestamps,
        }

    def test_initialization_with_all_data(self, sample_data):
        """Test initialization with all price data."""
        from utils.indicators import TechnicalIndicators

        ind = TechnicalIndicators(
            high=sample_data["high"],
            low=sample_data["low"],
            close=sample_data["close"],
            open_=sample_data["open_"],
            volume=sample_data["volume"],
            timestamps=sample_data["timestamps"],
        )

        assert ind.high is not None
        assert ind.low is not None
        assert ind.close is not None
        assert ind.open is not None
        assert ind.volume is not None
        assert ind.timestamps is not None
        assert len(ind.close) == 100

    def test_initialization_with_minimal_data(self, sample_data):
        """Test initialization with only close prices."""
        from utils.indicators import TechnicalIndicators

        ind = TechnicalIndicators(close=sample_data["close"])

        assert ind.close is not None
        assert ind.high is None
        assert ind.volume is None

    def test_initialization_with_none(self):
        """Test initialization with no data."""
        from utils.indicators import TechnicalIndicators

        ind = TechnicalIndicators()

        assert ind.close is None
        assert ind.high is None


class TestTrendIndicators:
    """Tests for trend indicators."""

    @pytest.fixture
    def indicators(self):
        """Create indicators instance with sample data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 250  # Need enough data for SMA 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return TechnicalIndicators(
            high=high.astype(np.float64), low=low.astype(np.float64), close=close.astype(np.float64)
        )

    def test_sma_default_period(self, indicators):
        """Test SMA with default period."""
        sma = indicators.sma()

        assert len(sma) == len(indicators.close)
        assert not np.isnan(sma[-1])
        # First 19 values should be NaN (period=20)
        assert np.isnan(sma[0])

    def test_sma_custom_period(self, indicators):
        """Test SMA with custom period."""
        sma_10 = indicators.sma(period=10)
        sma_50 = indicators.sma(period=50)

        # SMA_10 should have more valid values than SMA_50
        assert np.sum(~np.isnan(sma_10)) > np.sum(~np.isnan(sma_50))

    def test_sma_different_prices(self, indicators):
        """Test SMA with different price types."""
        sma_close = indicators.sma(price="close")
        sma_high = indicators.sma(price="high")

        # High prices should produce higher SMA
        valid_idx = ~np.isnan(sma_close) & ~np.isnan(sma_high)
        assert np.mean(sma_high[valid_idx]) > np.mean(sma_close[valid_idx])

    def test_ema_default_period(self, indicators):
        """Test EMA with default period."""
        ema = indicators.ema()

        assert len(ema) == len(indicators.close)
        assert not np.isnan(ema[-1])

    def test_ema_custom_period(self, indicators):
        """Test EMA with custom period."""
        ema_10 = indicators.ema(period=10)
        ema_50 = indicators.ema(period=50)

        # Both should have same length
        assert len(ema_10) == len(ema_50)

    def test_macd(self, indicators):
        """Test MACD calculation."""
        macd, signal, histogram = indicators.macd()

        assert len(macd) == len(indicators.close)
        assert len(signal) == len(indicators.close)
        assert len(histogram) == len(indicators.close)

        # Histogram should be MACD - Signal
        valid_idx = ~np.isnan(macd) & ~np.isnan(signal)
        np.testing.assert_array_almost_equal(
            histogram[valid_idx], macd[valid_idx] - signal[valid_idx]
        )

    def test_macd_custom_periods(self, indicators):
        """Test MACD with custom periods."""
        macd, signal, histogram = indicators.macd(fast_period=8, slow_period=17, signal_period=9)

        assert len(macd) == len(indicators.close)

    def test_adx(self, indicators):
        """Test ADX calculation."""
        adx = indicators.adx()

        assert len(adx) == len(indicators.close)
        # ADX should be between 0 and 100
        valid_adx = adx[~np.isnan(adx)]
        assert np.all(valid_adx >= 0)
        assert np.all(valid_adx <= 100)

    def test_adx_di(self, indicators):
        """Test ADX with directional indicators."""
        adx, plus_di, minus_di = indicators.adx_di()

        assert len(adx) == len(indicators.close)
        assert len(plus_di) == len(indicators.close)
        assert len(minus_di) == len(indicators.close)

    def test_parabolic_sar(self, indicators):
        """Test Parabolic SAR calculation."""
        sar = indicators.parabolic_sar()

        assert len(sar) == len(indicators.close)
        # SAR should be reasonable values (not extremely different from price)
        valid_sar = sar[~np.isnan(sar)]
        assert len(valid_sar) > 0

    def test_parabolic_sar_custom_params(self, indicators):
        """Test Parabolic SAR with custom parameters."""
        sar = indicators.parabolic_sar(acceleration=0.01, maximum=0.1)

        assert len(sar) == len(indicators.close)


class TestMomentumIndicators:
    """Tests for momentum indicators."""

    @pytest.fixture
    def indicators(self):
        """Create indicators instance with sample data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return TechnicalIndicators(
            high=high.astype(np.float64), low=low.astype(np.float64), close=close.astype(np.float64)
        )

    def test_rsi_default_period(self, indicators):
        """Test RSI with default period."""
        rsi = indicators.rsi()

        assert len(rsi) == len(indicators.close)
        # RSI should be between 0 and 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_rsi_custom_period(self, indicators):
        """Test RSI with custom period."""
        rsi = indicators.rsi(period=7)

        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_stochastic(self, indicators):
        """Test Stochastic oscillator."""
        slowk, slowd = indicators.stochastic()

        assert len(slowk) == len(indicators.close)
        assert len(slowd) == len(indicators.close)

        # Stochastic should be between 0 and 100
        valid_k = slowk[~np.isnan(slowk)]
        assert np.all(valid_k >= 0)
        assert np.all(valid_k <= 100)

    def test_stochastic_rsi(self, indicators):
        """Test Stochastic RSI."""
        fastk, fastd = indicators.stochastic_rsi()

        assert len(fastk) == len(indicators.close)
        assert len(fastd) == len(indicators.close)

    def test_cci(self, indicators):
        """Test CCI calculation."""
        cci = indicators.cci()

        assert len(cci) == len(indicators.close)
        # CCI can be any value, but typically -200 to +200

    def test_williams_r(self, indicators):
        """Test Williams %R."""
        willr = indicators.williams_r()

        assert len(willr) == len(indicators.close)
        # Williams %R should be between -100 and 0
        valid_willr = willr[~np.isnan(willr)]
        assert np.all(valid_willr >= -100)
        assert np.all(valid_willr <= 0)

    def test_roc(self, indicators):
        """Test Rate of Change."""
        roc = indicators.roc()

        assert len(roc) == len(indicators.close)
        # ROC is a percentage

    def test_roc_custom_period(self, indicators):
        """Test ROC with custom period."""
        roc = indicators.roc(period=6)

        assert len(roc) == len(indicators.close)


class TestVolatilityIndicators:
    """Tests for volatility indicators."""

    @pytest.fixture
    def indicators(self):
        """Create indicators instance with sample data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return TechnicalIndicators(
            high=high.astype(np.float64), low=low.astype(np.float64), close=close.astype(np.float64)
        )

    def test_bollinger_bands(self, indicators):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = indicators.bollinger_bands()

        assert len(upper) == len(indicators.close)
        assert len(middle) == len(indicators.close)
        assert len(lower) == len(indicators.close)

        # Upper should be > middle > lower
        valid_idx = ~np.isnan(upper) & ~np.isnan(middle) & ~np.isnan(lower)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_bollinger_bands_custom_params(self, indicators):
        """Test Bollinger Bands with custom parameters."""
        upper, middle, lower = indicators.bollinger_bands(period=10, std=1.5)

        valid_idx = ~np.isnan(upper) & ~np.isnan(lower)
        assert np.any(valid_idx)

    def test_atr(self, indicators):
        """Test ATR calculation."""
        atr = indicators.atr()

        assert len(atr) == len(indicators.close)
        # ATR should be positive
        valid_atr = atr[~np.isnan(atr)]
        assert np.all(valid_atr >= 0)

    def test_keltner_channels(self, indicators):
        """Test Keltner Channels."""
        upper, middle, lower = indicators.keltner_channels()

        assert len(upper) == len(indicators.close)
        assert len(middle) == len(indicators.close)
        assert len(lower) == len(indicators.close)

        # Upper should be >= middle >= lower
        valid_idx = ~np.isnan(upper) & ~np.isnan(middle) & ~np.isnan(lower)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_stddev(self, indicators):
        """Test Standard Deviation calculation."""
        std = indicators.stddev()

        assert len(std) == len(indicators.close)
        # Std dev should be non-negative
        valid_std = std[~np.isnan(std)]
        assert np.all(valid_std >= 0)


class TestVolumeIndicators:
    """Tests for volume indicators."""

    @pytest.fixture
    def indicators_with_volume(self):
        """Create indicators instance with volume data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(100000, 1000000, n).astype(np.float64)

        # Create timestamps spanning 2 days
        base_date = datetime(2024, 1, 1, 9, 30)
        timestamps = []
        for i in range(n):
            if i < 50:
                ts = base_date + timedelta(minutes=i)
            else:
                ts = base_date + timedelta(days=1, minutes=i - 50)
            timestamps.append(ts)

        return TechnicalIndicators(
            high=high.astype(np.float64),
            low=low.astype(np.float64),
            close=close.astype(np.float64),
            volume=volume,
            timestamps=timestamps,
        )

    @pytest.fixture
    def indicators_no_volume(self):
        """Create indicators instance without volume data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return TechnicalIndicators(
            high=high.astype(np.float64), low=low.astype(np.float64), close=close.astype(np.float64)
        )

    def test_vwap_with_timestamps(self, indicators_with_volume):
        """Test VWAP calculation with timestamps."""
        vwap = indicators_with_volume.vwap()

        assert len(vwap) == len(indicators_with_volume.close)
        # VWAP should be positive
        assert np.all(vwap > 0)

    def test_vwap_without_timestamps(self):
        """Test VWAP calculation without timestamps (cumulative)."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(100000, 1000000, n).astype(np.float64)

        ind = TechnicalIndicators(
            high=high.astype(np.float64),
            low=low.astype(np.float64),
            close=close.astype(np.float64),
            volume=volume,
        )

        vwap = ind.vwap()
        assert len(vwap) == n
        assert np.all(vwap > 0)

    def test_vwap_requires_volume(self, indicators_no_volume):
        """Test VWAP raises error without volume."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.vwap()

    def test_obv(self, indicators_with_volume):
        """Test OBV calculation."""
        obv = indicators_with_volume.obv()

        assert len(obv) == len(indicators_with_volume.close)

    def test_obv_requires_volume(self, indicators_no_volume):
        """Test OBV raises error without volume."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.obv()

    def test_volume_sma(self, indicators_with_volume):
        """Test Volume SMA calculation."""
        vol_sma = indicators_with_volume.volume_sma()

        assert len(vol_sma) == len(indicators_with_volume.close)

    def test_volume_sma_requires_volume(self, indicators_no_volume):
        """Test Volume SMA raises error without volume."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.volume_sma()

    def test_mfi(self, indicators_with_volume):
        """Test MFI calculation."""
        mfi = indicators_with_volume.mfi()

        assert len(mfi) == len(indicators_with_volume.close)
        # MFI should be between 0 and 100
        valid_mfi = mfi[~np.isnan(mfi)]
        assert np.all(valid_mfi >= 0)
        assert np.all(valid_mfi <= 100)

    def test_mfi_requires_volume(self, indicators_no_volume):
        """Test MFI raises error without volume."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.mfi()


class TestSupportResistance:
    """Tests for support/resistance indicators."""

    @pytest.fixture
    def indicators(self):
        """Create indicators instance with sample data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return TechnicalIndicators(
            high=high.astype(np.float64), low=low.astype(np.float64), close=close.astype(np.float64)
        )

    def test_pivot_points(self, indicators):
        """Test pivot point calculation."""
        pivots = indicators.pivot_points()

        assert "PP" in pivots
        assert "R1" in pivots
        assert "R2" in pivots
        assert "R3" in pivots
        assert "S1" in pivots
        assert "S2" in pivots
        assert "S3" in pivots

        # R levels should be above PP, S levels should be below
        assert pivots["R1"] >= pivots["PP"]
        assert pivots["S1"] <= pivots["PP"]
        assert pivots["R2"] >= pivots["R1"]
        assert pivots["S2"] <= pivots["S1"]

    def test_fibonacci_retracement(self, indicators):
        """Test Fibonacci retracement levels."""
        fib = indicators.fibonacci_retracement()

        assert "0.0%" in fib
        assert "23.6%" in fib
        assert "38.2%" in fib
        assert "50.0%" in fib
        assert "61.8%" in fib
        assert "78.6%" in fib
        assert "100.0%" in fib

        # Levels should be in descending order
        assert fib["0.0%"] >= fib["23.6%"]
        assert fib["23.6%"] >= fib["38.2%"]
        assert fib["38.2%"] >= fib["50.0%"]
        assert fib["50.0%"] >= fib["61.8%"]
        assert fib["61.8%"] >= fib["100.0%"]

    def test_fibonacci_custom_swings(self, indicators):
        """Test Fibonacci with custom swing high/low."""
        fib = indicators.fibonacci_retracement(swing_high=110.0, swing_low=90.0)

        assert fib["0.0%"] == 110.0
        assert fib["100.0%"] == 90.0
        assert fib["50.0%"] == pytest.approx(100.0, rel=0.01)


class TestHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def indicators(self):
        """Create indicators instance with sample data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 50
        close = np.array([100 + i for i in range(n)], dtype=np.float64)
        high = close + 1
        low = close - 1
        open_ = close + 0.5

        return TechnicalIndicators(high=high, low=low, close=close, open_=open_)

    def test_get_price_array_close(self, indicators):
        """Test getting close price array."""
        prices = indicators._get_price_array("close")
        np.testing.assert_array_equal(prices, indicators.close)

    def test_get_price_array_high(self, indicators):
        """Test getting high price array."""
        prices = indicators._get_price_array("high")
        np.testing.assert_array_equal(prices, indicators.high)

    def test_get_price_array_low(self, indicators):
        """Test getting low price array."""
        prices = indicators._get_price_array("low")
        np.testing.assert_array_equal(prices, indicators.low)

    def test_get_price_array_open(self, indicators):
        """Test getting open price array."""
        prices = indicators._get_price_array("open")
        np.testing.assert_array_equal(prices, indicators.open)

    def test_get_price_array_invalid(self, indicators):
        """Test invalid price type raises error."""
        with pytest.raises(ValueError, match="Invalid price type"):
            indicators._get_price_array("invalid")


class TestCompositeIndicators:
    """Tests for composite indicator methods."""

    @pytest.fixture
    def indicators(self):
        """Create indicators instance with sample data."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 250
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return TechnicalIndicators(
            high=high.astype(np.float64), low=low.astype(np.float64), close=close.astype(np.float64)
        )

    def test_all_momentum_indicators(self, indicators):
        """Test all momentum indicators at once."""
        momentum = indicators.all_momentum_indicators()

        assert "rsi" in momentum
        assert "stochastic" in momentum
        assert "cci" in momentum
        assert "williams_r" in momentum
        assert "roc" in momentum
        assert "macd" in momentum

    def test_all_trend_indicators(self, indicators):
        """Test all trend indicators at once."""
        trend = indicators.all_trend_indicators()

        assert "sma_20" in trend
        assert "sma_50" in trend
        assert "sma_200" in trend
        assert "ema_20" in trend
        assert "ema_50" in trend
        assert "macd" in trend
        assert "adx" in trend
        assert "plus_di" in trend
        assert "minus_di" in trend
        assert "parabolic_sar" in trend

    def test_all_volatility_indicators(self, indicators):
        """Test all volatility indicators at once."""
        volatility = indicators.all_volatility_indicators()

        assert "bollinger_bands" in volatility
        assert "atr" in volatility
        assert "keltner_channels" in volatility
        assert "stddev" in volatility


class TestAnalyzeTrend:
    """Tests for analyze_trend function."""

    @pytest.fixture
    def price_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 250
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return {
            "close": close.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
        }

    def test_analyze_trend_returns_expected_keys(self, price_data):
        """Test analyze_trend returns all expected keys."""
        from utils.indicators import analyze_trend

        result = analyze_trend(price_data["close"], price_data["high"], price_data["low"])

        assert "direction" in result
        assert "strength" in result
        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result
        assert "sma_50" in result
        assert "sma_200" in result
        assert "price_vs_sma_50" in result
        assert "price_vs_sma_200" in result

    def test_analyze_trend_direction_values(self, price_data):
        """Test analyze_trend returns valid direction."""
        from utils.indicators import analyze_trend

        result = analyze_trend(price_data["close"], price_data["high"], price_data["low"])

        assert result["direction"] in ["bullish", "bearish", "neutral"]

    def test_analyze_trend_strength_values(self, price_data):
        """Test analyze_trend returns valid strength."""
        from utils.indicators import analyze_trend

        result = analyze_trend(price_data["close"], price_data["high"], price_data["low"])

        assert result["strength"] in ["very_strong", "strong", "weak", "no_trend"]


class TestAnalyzeMomentum:
    """Tests for analyze_momentum function."""

    @pytest.fixture
    def price_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return {
            "close": close.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
        }

    def test_analyze_momentum_returns_expected_keys(self, price_data):
        """Test analyze_momentum returns all expected keys."""
        from utils.indicators import analyze_momentum

        result = analyze_momentum(price_data["close"], price_data["high"], price_data["low"])

        assert "condition" in result
        assert "rsi" in result
        assert "stochastic_k" in result
        assert "stochastic_d" in result

    def test_analyze_momentum_condition_values(self, price_data):
        """Test analyze_momentum returns valid condition."""
        from utils.indicators import analyze_momentum

        result = analyze_momentum(price_data["close"], price_data["high"], price_data["low"])

        assert result["condition"] in ["overbought", "oversold", "neutral"]


class TestAnalyzeVolatility:
    """Tests for analyze_volatility function."""

    @pytest.fixture
    def price_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        return {
            "close": close.astype(np.float64),
            "high": high.astype(np.float64),
            "low": low.astype(np.float64),
        }

    def test_analyze_volatility_returns_expected_keys(self, price_data):
        """Test analyze_volatility returns all expected keys."""
        from utils.indicators import analyze_volatility

        result = analyze_volatility(price_data["close"], price_data["high"], price_data["low"])

        assert "state" in result
        assert "atr" in result
        assert "bb_upper" in result
        assert "bb_middle" in result
        assert "bb_lower" in result
        assert "bb_width_pct" in result
        assert "price_position" in result

    def test_analyze_volatility_state_values(self, price_data):
        """Test analyze_volatility returns valid state."""
        from utils.indicators import analyze_volatility

        result = analyze_volatility(price_data["close"], price_data["high"], price_data["low"])

        assert result["state"] in ["squeeze", "expansion", "normal"]

    def test_analyze_volatility_price_position_range(self, price_data):
        """Test price_position is in valid range."""
        from utils.indicators import analyze_volatility

        result = analyze_volatility(price_data["close"], price_data["high"], price_data["low"])

        # Price position should be between 0 and 1 (or close)
        assert 0 <= result["price_position"] <= 1.5


class TestEdgeCases:
    """Tests for edge cases."""

    def test_insufficient_data_for_sma(self):
        """Test handling of insufficient data for SMA."""
        from utils.indicators import TechnicalIndicators

        close = np.array([100.0, 101.0, 102.0])  # Only 3 data points
        ind = TechnicalIndicators(close=close)

        sma = ind.sma(period=20)  # Needs 20 periods

        # All values should be NaN
        assert np.all(np.isnan(sma))

    def test_vwap_handles_zero_volume(self):
        """Test VWAP handles zero volume gracefully."""
        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 0.5
        low = close - 0.5
        volume = np.zeros(n)  # All zero volume
        volume[0] = 100000  # First one non-zero to avoid all zeros

        ind = TechnicalIndicators(
            high=high.astype(np.float64),
            low=low.astype(np.float64),
            close=close.astype(np.float64),
            volume=volume.astype(np.float64),
        )

        # Should not raise
        vwap = ind.vwap()
        assert len(vwap) == n

    def test_vwap_with_date_objects(self):
        """Test VWAP with date objects instead of datetime."""
        from datetime import date

        from utils.indicators import TechnicalIndicators

        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 0.5
        low = close - 0.5
        volume = np.random.randint(100000, 1000000, n).astype(np.float64)

        # Use date objects instead of datetime
        timestamps = [date(2024, 1, 1) for _ in range(n)]

        ind = TechnicalIndicators(
            high=high.astype(np.float64),
            low=low.astype(np.float64),
            close=close.astype(np.float64),
            volume=volume,
            timestamps=timestamps,
        )

        # Should not raise
        vwap = ind.vwap()
        assert len(vwap) == n
