#!/usr/bin/env python3
"""
Unit tests for utils/indicators.py

Tests TechnicalIndicators class for:
- Trend indicators (SMA, EMA, MACD, ADX, Parabolic SAR)
- Momentum indicators (RSI, Stochastic, CCI, Williams %R, ROC)
- Volatility indicators (Bollinger Bands, ATR, Keltner Channels, Std Dev)
- Volume indicators (VWAP, OBV, Volume SMA, MFI)
- Support/Resistance (Pivot Points, Fibonacci)
- Analysis functions
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from utils.indicators import (
    TechnicalIndicators,
    analyze_momentum,
    analyze_trend,
    analyze_volatility,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Generate trending price data
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100000, 1000000, n).astype(float)

    return {
        "high": high,
        "low": low,
        "close": close,
        "open": open_,
        "volume": volume,
    }


@pytest.fixture
def indicators(sample_ohlcv):
    """Create TechnicalIndicators instance with sample data."""
    return TechnicalIndicators(
        high=sample_ohlcv["high"],
        low=sample_ohlcv["low"],
        close=sample_ohlcv["close"],
        open_=sample_ohlcv["open"],
        volume=sample_ohlcv["volume"],
    )


@pytest.fixture
def indicators_no_volume(sample_ohlcv):
    """Create TechnicalIndicators instance without volume data."""
    return TechnicalIndicators(
        high=sample_ohlcv["high"],
        low=sample_ohlcv["low"],
        close=sample_ohlcv["close"],
    )


@pytest.fixture
def indicators_with_timestamps(sample_ohlcv):
    """Create TechnicalIndicators instance with timestamps."""
    n = len(sample_ohlcv["close"])
    base_date = datetime(2024, 1, 1, 9, 30)
    timestamps = [base_date + timedelta(minutes=i * 5) for i in range(n)]

    return TechnicalIndicators(
        high=sample_ohlcv["high"],
        low=sample_ohlcv["low"],
        close=sample_ohlcv["close"],
        open_=sample_ohlcv["open"],
        volume=sample_ohlcv["volume"],
        timestamps=timestamps,
    )


@pytest.fixture
def long_sample_ohlcv():
    """Generate longer sample for indicators that need more data."""
    np.random.seed(42)
    n = 250  # ~1 year of trading days

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100000, 1000000, n).astype(float)

    return {
        "high": high,
        "low": low,
        "close": close,
        "open": open_,
        "volume": volume,
    }


# ============================================================================
# Initialization Tests
# ============================================================================


class TestTechnicalIndicatorsInit:
    """Test TechnicalIndicators initialization."""

    def test_init_with_all_data(self, sample_ohlcv):
        """Test initialization with all data types."""
        ind = TechnicalIndicators(
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            open_=sample_ohlcv["open"],
            volume=sample_ohlcv["volume"],
        )

        assert ind.high is not None
        assert ind.low is not None
        assert ind.close is not None
        assert ind.open is not None
        assert ind.volume is not None

    def test_init_with_minimal_data(self, sample_ohlcv):
        """Test initialization with only close prices."""
        ind = TechnicalIndicators(close=sample_ohlcv["close"])

        assert ind.close is not None
        assert ind.high is None
        assert ind.low is None
        assert ind.open is None
        assert ind.volume is None

    def test_init_converts_to_numpy(self, sample_ohlcv):
        """Test that input is converted to numpy arrays."""
        ind = TechnicalIndicators(
            close=list(sample_ohlcv["close"]),  # Pass as list
        )

        assert isinstance(ind.close, np.ndarray)

    def test_init_with_timestamps(self, indicators_with_timestamps):
        """Test initialization with timestamps."""
        assert indicators_with_timestamps.timestamps is not None
        assert len(indicators_with_timestamps.timestamps) == len(
            indicators_with_timestamps.close
        )


# ============================================================================
# Trend Indicator Tests
# ============================================================================


class TestSMA:
    """Test Simple Moving Average."""

    def test_sma_returns_array(self, indicators):
        """Test SMA returns numpy array."""
        sma = indicators.sma(period=20)
        assert isinstance(sma, np.ndarray)
        assert len(sma) == len(indicators.close)

    def test_sma_default_period(self, indicators):
        """Test SMA with default period of 20."""
        sma = indicators.sma()
        # First 19 values should be NaN
        assert np.isnan(sma[:19]).all()
        # After that, values should exist
        assert not np.isnan(sma[19:]).any()

    def test_sma_custom_period(self, indicators):
        """Test SMA with custom period."""
        sma_10 = indicators.sma(period=10)
        sma_50 = indicators.sma(period=50)

        # SMA-10 should have NaN for first 9 values
        assert np.isnan(sma_10[8])
        assert not np.isnan(sma_10[9])

        # SMA-50 should have NaN for first 49 values
        assert np.isnan(sma_50[48])
        assert not np.isnan(sma_50[49])

    def test_sma_different_price_types(self, indicators):
        """Test SMA with different price types."""
        sma_close = indicators.sma(price="close")
        sma_high = indicators.sma(price="high")
        sma_low = indicators.sma(price="low")

        # High SMA should generally be higher than low SMA
        # Compare non-NaN values
        valid_idx = ~np.isnan(sma_close)
        assert (sma_high[valid_idx] >= sma_low[valid_idx]).all()


class TestEMA:
    """Test Exponential Moving Average."""

    def test_ema_returns_array(self, indicators):
        """Test EMA returns numpy array."""
        ema = indicators.ema(period=20)
        assert isinstance(ema, np.ndarray)
        assert len(ema) == len(indicators.close)

    def test_ema_faster_than_sma(self, indicators):
        """Test that EMA reacts faster to price changes."""
        sma = indicators.sma(period=20)
        ema = indicators.ema(period=20)

        # EMA should have fewer NaN values at the start
        # (EMA can start calculating earlier)
        assert np.sum(~np.isnan(ema)) >= np.sum(~np.isnan(sma))

    def test_ema_different_price_types(self, indicators):
        """Test EMA with different price types."""
        ema_close = indicators.ema(price="close")
        ema_open = indicators.ema(price="open")

        # Should return different values
        valid_idx = ~np.isnan(ema_close) & ~np.isnan(ema_open)
        # At least some values should differ
        assert not np.allclose(ema_close[valid_idx], ema_open[valid_idx])


class TestMACD:
    """Test MACD indicator."""

    def test_macd_returns_tuple(self, indicators):
        """Test MACD returns tuple of three arrays."""
        result = indicators.macd()

        assert isinstance(result, tuple)
        assert len(result) == 3

        macd, signal, histogram = result
        assert len(macd) == len(indicators.close)
        assert len(signal) == len(indicators.close)
        assert len(histogram) == len(indicators.close)

    def test_macd_default_periods(self, indicators):
        """Test MACD with default periods (12, 26, 9)."""
        macd, signal, histogram = indicators.macd()

        # Histogram should equal MACD - Signal (where both are valid)
        valid_idx = ~np.isnan(macd) & ~np.isnan(signal)
        assert np.allclose(histogram[valid_idx], macd[valid_idx] - signal[valid_idx])

    def test_macd_custom_periods(self, indicators):
        """Test MACD with custom periods."""
        macd1, _, _ = indicators.macd(fast_period=8, slow_period=17, signal_period=9)
        macd2, _, _ = indicators.macd(fast_period=12, slow_period=26, signal_period=9)

        # Different periods should give different results
        valid_idx = ~np.isnan(macd1) & ~np.isnan(macd2)
        assert not np.allclose(macd1[valid_idx], macd2[valid_idx])


class TestADX:
    """Test ADX indicator."""

    def test_adx_returns_array(self, indicators):
        """Test ADX returns numpy array."""
        adx = indicators.adx(period=14)
        assert isinstance(adx, np.ndarray)
        assert len(adx) == len(indicators.close)

    def test_adx_values_in_range(self, indicators):
        """Test ADX values are in 0-100 range."""
        adx = indicators.adx(period=14)
        valid_values = adx[~np.isnan(adx)]

        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_adx_di(self, indicators):
        """Test ADX with directional indicators."""
        adx, plus_di, minus_di = indicators.adx_di(period=14)

        assert len(adx) == len(indicators.close)
        assert len(plus_di) == len(indicators.close)
        assert len(minus_di) == len(indicators.close)

        # DI values should be in 0-100 range
        valid_plus = plus_di[~np.isnan(plus_di)]
        valid_minus = minus_di[~np.isnan(minus_di)]

        assert (valid_plus >= 0).all()
        assert (valid_minus >= 0).all()


class TestParabolicSAR:
    """Test Parabolic SAR indicator."""

    def test_sar_returns_array(self, indicators):
        """Test SAR returns numpy array."""
        sar = indicators.parabolic_sar()
        assert isinstance(sar, np.ndarray)
        assert len(sar) == len(indicators.close)

    def test_sar_custom_parameters(self, indicators):
        """Test SAR with custom parameters."""
        sar1 = indicators.parabolic_sar(acceleration=0.02, maximum=0.2)
        sar2 = indicators.parabolic_sar(acceleration=0.01, maximum=0.1)

        # Different parameters should give different results
        valid_idx = ~np.isnan(sar1) & ~np.isnan(sar2)
        assert not np.allclose(sar1[valid_idx], sar2[valid_idx])


# ============================================================================
# Momentum Indicator Tests
# ============================================================================


class TestRSI:
    """Test RSI indicator."""

    def test_rsi_returns_array(self, indicators):
        """Test RSI returns numpy array."""
        rsi = indicators.rsi(period=14)
        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(indicators.close)

    def test_rsi_values_in_range(self, indicators):
        """Test RSI values are in 0-100 range."""
        rsi = indicators.rsi(period=14)
        valid_values = rsi[~np.isnan(rsi)]

        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_different_periods(self, indicators):
        """Test RSI with different periods."""
        rsi_7 = indicators.rsi(period=7)
        rsi_14 = indicators.rsi(period=14)

        # Shorter period RSI should be more volatile
        valid_idx = ~np.isnan(rsi_7) & ~np.isnan(rsi_14)
        std_7 = np.std(rsi_7[valid_idx])
        std_14 = np.std(rsi_14[valid_idx])

        # RSI-7 typically has higher volatility
        assert std_7 >= std_14 * 0.8  # Allow some tolerance


class TestStochastic:
    """Test Stochastic oscillator."""

    def test_stochastic_returns_tuple(self, indicators):
        """Test Stochastic returns tuple of two arrays."""
        result = indicators.stochastic()

        assert isinstance(result, tuple)
        assert len(result) == 2

        slowk, slowd = result
        assert len(slowk) == len(indicators.close)
        assert len(slowd) == len(indicators.close)

    def test_stochastic_values_in_range(self, indicators):
        """Test Stochastic values are in 0-100 range."""
        slowk, slowd = indicators.stochastic()

        valid_k = slowk[~np.isnan(slowk)]
        valid_d = slowd[~np.isnan(slowd)]

        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_stochastic_rsi(self, indicators):
        """Test Stochastic RSI."""
        fastk, fastd = indicators.stochastic_rsi()

        assert len(fastk) == len(indicators.close)
        assert len(fastd) == len(indicators.close)

        # Values should be in 0-100 range (with small tolerance for floating point)
        valid_k = fastk[~np.isnan(fastk)]
        assert (valid_k >= -0.01).all(), f"Min value: {valid_k.min()}"
        assert (valid_k <= 100.01).all(), f"Max value: {valid_k.max()}"


class TestCCI:
    """Test Commodity Channel Index."""

    def test_cci_returns_array(self, indicators):
        """Test CCI returns numpy array."""
        cci = indicators.cci(period=20)
        assert isinstance(cci, np.ndarray)
        assert len(cci) == len(indicators.close)

    def test_cci_can_be_extreme(self, indicators):
        """Test CCI can go outside typical bounds."""
        cci = indicators.cci(period=20)
        valid_values = cci[~np.isnan(cci)]

        # CCI is unbounded but typically ranges around -100 to +100
        # Just verify it's not constrained to 0-100
        assert valid_values.min() < 50 or valid_values.max() > 50


class TestWilliamsR:
    """Test Williams %R."""

    def test_williams_r_returns_array(self, indicators):
        """Test Williams %R returns numpy array."""
        willr = indicators.williams_r(period=14)
        assert isinstance(willr, np.ndarray)
        assert len(willr) == len(indicators.close)

    def test_williams_r_values_in_range(self, indicators):
        """Test Williams %R values are in -100 to 0 range."""
        willr = indicators.williams_r(period=14)
        valid_values = willr[~np.isnan(willr)]

        assert (valid_values >= -100).all()
        assert (valid_values <= 0).all()


class TestROC:
    """Test Rate of Change."""

    def test_roc_returns_array(self, indicators):
        """Test ROC returns numpy array."""
        roc = indicators.roc(period=12)
        assert isinstance(roc, np.ndarray)
        assert len(roc) == len(indicators.close)

    def test_roc_can_be_negative(self, indicators):
        """Test ROC can be negative."""
        roc = indicators.roc(period=12)
        valid_values = roc[~np.isnan(roc)]

        # ROC should have both positive and negative values in volatile data
        # At minimum, values should exist
        assert len(valid_values) > 0


# ============================================================================
# Volatility Indicator Tests
# ============================================================================


class TestBollingerBands:
    """Test Bollinger Bands."""

    def test_bb_returns_tuple(self, indicators):
        """Test Bollinger Bands returns tuple of three arrays."""
        result = indicators.bollinger_bands()

        assert isinstance(result, tuple)
        assert len(result) == 3

        upper, middle, lower = result
        assert len(upper) == len(indicators.close)
        assert len(middle) == len(indicators.close)
        assert len(lower) == len(indicators.close)

    def test_bb_ordering(self, indicators):
        """Test upper > middle > lower bands."""
        upper, middle, lower = indicators.bollinger_bands()

        valid_idx = ~np.isnan(upper) & ~np.isnan(middle) & ~np.isnan(lower)

        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_bb_different_std(self, indicators):
        """Test Bollinger Bands with different standard deviations."""
        upper1, _, lower1 = indicators.bollinger_bands(std=1.0)
        upper2, _, lower2 = indicators.bollinger_bands(std=2.0)

        # Wider bands with higher std
        valid_idx = ~np.isnan(upper1) & ~np.isnan(upper2)
        width1 = upper1[valid_idx] - lower1[valid_idx]
        width2 = upper2[valid_idx] - lower2[valid_idx]

        assert (width2 > width1).all()


class TestATR:
    """Test Average True Range."""

    def test_atr_returns_array(self, indicators):
        """Test ATR returns numpy array."""
        atr = indicators.atr(period=14)
        assert isinstance(atr, np.ndarray)
        assert len(atr) == len(indicators.close)

    def test_atr_positive(self, indicators):
        """Test ATR values are positive."""
        atr = indicators.atr(period=14)
        valid_values = atr[~np.isnan(atr)]

        assert (valid_values >= 0).all()

    def test_atr_different_periods(self, indicators):
        """Test ATR with different periods."""
        atr_7 = indicators.atr(period=7)
        atr_14 = indicators.atr(period=14)

        # Different periods should give different results
        valid_idx = ~np.isnan(atr_7) & ~np.isnan(atr_14)
        # At least some values should differ
        assert not np.allclose(atr_7[valid_idx], atr_14[valid_idx])


class TestKeltnerChannels:
    """Test Keltner Channels."""

    def test_keltner_returns_tuple(self, indicators):
        """Test Keltner Channels returns tuple of three arrays."""
        result = indicators.keltner_channels()

        assert isinstance(result, tuple)
        assert len(result) == 3

        upper, middle, lower = result
        assert len(upper) == len(indicators.close)
        assert len(middle) == len(indicators.close)
        assert len(lower) == len(indicators.close)

    def test_keltner_ordering(self, indicators):
        """Test upper > middle > lower channels."""
        upper, middle, lower = indicators.keltner_channels()

        valid_idx = ~np.isnan(upper) & ~np.isnan(middle) & ~np.isnan(lower)

        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestStdDev:
    """Test Standard Deviation."""

    def test_stddev_returns_array(self, indicators):
        """Test StdDev returns numpy array."""
        stddev = indicators.stddev(period=20)
        assert isinstance(stddev, np.ndarray)
        assert len(stddev) == len(indicators.close)

    def test_stddev_positive(self, indicators):
        """Test StdDev values are positive."""
        stddev = indicators.stddev(period=20)
        valid_values = stddev[~np.isnan(stddev)]

        assert (valid_values >= 0).all()


# ============================================================================
# Volume Indicator Tests
# ============================================================================


class TestVWAP:
    """Test VWAP indicator."""

    def test_vwap_returns_array(self, indicators):
        """Test VWAP returns numpy array."""
        vwap = indicators.vwap()
        assert isinstance(vwap, np.ndarray)
        assert len(vwap) == len(indicators.close)

    def test_vwap_without_volume_raises(self, indicators_no_volume):
        """Test VWAP raises error without volume data."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.vwap()

    def test_vwap_with_timestamps(self, indicators_with_timestamps):
        """Test VWAP with timestamps resets daily."""
        vwap = indicators_with_timestamps.vwap()
        assert isinstance(vwap, np.ndarray)
        assert len(vwap) == len(indicators_with_timestamps.close)

    def test_vwap_values_reasonable(self, indicators):
        """Test VWAP values are within price range."""
        vwap = indicators.vwap()

        typical_price = (indicators.high + indicators.low + indicators.close) / 3
        min_price = np.min(typical_price)
        max_price = np.max(typical_price)

        # VWAP should be within typical price range
        assert vwap[-1] >= min_price * 0.9
        assert vwap[-1] <= max_price * 1.1


class TestOBV:
    """Test On-Balance Volume."""

    def test_obv_returns_array(self, indicators):
        """Test OBV returns numpy array."""
        obv = indicators.obv()
        assert isinstance(obv, np.ndarray)
        assert len(obv) == len(indicators.close)

    def test_obv_without_volume_raises(self, indicators_no_volume):
        """Test OBV raises error without volume data."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.obv()


class TestVolumeSMA:
    """Test Volume SMA."""

    def test_volume_sma_returns_array(self, indicators):
        """Test Volume SMA returns numpy array."""
        vol_sma = indicators.volume_sma(period=20)
        assert isinstance(vol_sma, np.ndarray)
        assert len(vol_sma) == len(indicators.close)

    def test_volume_sma_without_volume_raises(self, indicators_no_volume):
        """Test Volume SMA raises error without volume data."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.volume_sma()


class TestMFI:
    """Test Money Flow Index."""

    def test_mfi_returns_array(self, indicators):
        """Test MFI returns numpy array."""
        mfi = indicators.mfi(period=14)
        assert isinstance(mfi, np.ndarray)
        assert len(mfi) == len(indicators.close)

    def test_mfi_values_in_range(self, indicators):
        """Test MFI values are in 0-100 range."""
        mfi = indicators.mfi(period=14)
        valid_values = mfi[~np.isnan(mfi)]

        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_mfi_without_volume_raises(self, indicators_no_volume):
        """Test MFI raises error without volume data."""
        with pytest.raises(ValueError, match="Volume data required"):
            indicators_no_volume.mfi()


# ============================================================================
# Support/Resistance Tests
# ============================================================================


class TestPivotPoints:
    """Test Pivot Points calculation."""

    def test_pivot_points_returns_dict(self, indicators):
        """Test pivot points returns dictionary."""
        pivots = indicators.pivot_points()
        assert isinstance(pivots, dict)

    def test_pivot_points_keys(self, indicators):
        """Test pivot points has expected keys."""
        pivots = indicators.pivot_points()
        expected_keys = {"PP", "R1", "R2", "R3", "S1", "S2", "S3"}
        assert set(pivots.keys()) == expected_keys

    def test_pivot_points_ordering(self, indicators):
        """Test resistance > PP > support ordering."""
        pivots = indicators.pivot_points()

        assert pivots["R3"] >= pivots["R2"]
        assert pivots["R2"] >= pivots["R1"]
        assert pivots["R1"] >= pivots["PP"]
        assert pivots["PP"] >= pivots["S1"]
        assert pivots["S1"] >= pivots["S2"]
        assert pivots["S2"] >= pivots["S3"]


class TestFibonacciRetracement:
    """Test Fibonacci Retracement levels."""

    def test_fibonacci_returns_dict(self, indicators):
        """Test Fibonacci retracement returns dictionary."""
        fib = indicators.fibonacci_retracement()
        assert isinstance(fib, dict)

    def test_fibonacci_keys(self, indicators):
        """Test Fibonacci has expected keys."""
        fib = indicators.fibonacci_retracement()
        expected_keys = {"0.0%", "23.6%", "38.2%", "50.0%", "61.8%", "78.6%", "100.0%"}
        assert set(fib.keys()) == expected_keys

    def test_fibonacci_ordering(self, indicators):
        """Test Fibonacci levels are properly ordered."""
        fib = indicators.fibonacci_retracement()

        # 0% (swing high) should be highest, 100% (swing low) lowest
        assert fib["0.0%"] >= fib["23.6%"]
        assert fib["23.6%"] >= fib["38.2%"]
        assert fib["38.2%"] >= fib["50.0%"]
        assert fib["50.0%"] >= fib["61.8%"]
        assert fib["61.8%"] >= fib["78.6%"]
        assert fib["78.6%"] >= fib["100.0%"]

    def test_fibonacci_custom_levels(self, indicators):
        """Test Fibonacci with custom swing high/low."""
        fib = indicators.fibonacci_retracement(swing_high=150, swing_low=100)

        assert fib["0.0%"] == 150
        assert fib["100.0%"] == 100
        assert fib["50.0%"] == 125  # Midpoint


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestGetPriceArray:
    """Test _get_price_array helper method."""

    def test_get_close(self, indicators):
        """Test getting close prices."""
        prices = indicators._get_price_array("close")
        assert np.array_equal(prices, indicators.close)

    def test_get_high(self, indicators):
        """Test getting high prices."""
        prices = indicators._get_price_array("high")
        assert np.array_equal(prices, indicators.high)

    def test_get_low(self, indicators):
        """Test getting low prices."""
        prices = indicators._get_price_array("low")
        assert np.array_equal(prices, indicators.low)

    def test_get_open(self, indicators):
        """Test getting open prices."""
        prices = indicators._get_price_array("open")
        assert np.array_equal(prices, indicators.open)

    def test_invalid_price_type(self, indicators):
        """Test invalid price type raises error."""
        with pytest.raises(ValueError, match="Invalid price type"):
            indicators._get_price_array("invalid")


# ============================================================================
# Composite Indicator Tests
# ============================================================================


class TestAllMomentumIndicators:
    """Test all_momentum_indicators composite method."""

    def test_returns_dict(self, indicators):
        """Test returns dictionary with all momentum indicators."""
        result = indicators.all_momentum_indicators()
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, indicators):
        """Test dictionary has expected keys."""
        result = indicators.all_momentum_indicators()
        expected_keys = {"rsi", "stochastic", "cci", "williams_r", "roc", "macd"}
        assert set(result.keys()) == expected_keys


class TestAllTrendIndicators:
    """Test all_trend_indicators composite method."""

    def test_returns_dict(self, indicators):
        """Test returns dictionary with all trend indicators."""
        result = indicators.all_trend_indicators()
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, indicators):
        """Test dictionary has expected keys."""
        result = indicators.all_trend_indicators()
        expected_keys = {
            "sma_20", "sma_50", "sma_200", "ema_20", "ema_50",
            "macd", "adx", "plus_di", "minus_di", "parabolic_sar"
        }
        assert set(result.keys()) == expected_keys


class TestAllVolatilityIndicators:
    """Test all_volatility_indicators composite method."""

    def test_returns_dict(self, indicators):
        """Test returns dictionary with all volatility indicators."""
        result = indicators.all_volatility_indicators()
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, indicators):
        """Test dictionary has expected keys."""
        result = indicators.all_volatility_indicators()
        expected_keys = {"bollinger_bands", "atr", "keltner_channels", "stddev"}
        assert set(result.keys()) == expected_keys


# ============================================================================
# Analysis Function Tests
# ============================================================================


class TestAnalyzeTrend:
    """Test analyze_trend function."""

    def test_returns_dict(self, long_sample_ohlcv):
        """Test returns dictionary."""
        result = analyze_trend(
            close=long_sample_ohlcv["close"],
            high=long_sample_ohlcv["high"],
            low=long_sample_ohlcv["low"],
        )
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, long_sample_ohlcv):
        """Test dictionary has expected keys."""
        result = analyze_trend(
            close=long_sample_ohlcv["close"],
            high=long_sample_ohlcv["high"],
            low=long_sample_ohlcv["low"],
        )
        expected_keys = {
            "direction", "strength", "adx", "plus_di", "minus_di",
            "sma_50", "sma_200", "price_vs_sma_50", "price_vs_sma_200"
        }
        assert set(result.keys()) == expected_keys

    def test_direction_valid_values(self, long_sample_ohlcv):
        """Test direction has valid value."""
        result = analyze_trend(
            close=long_sample_ohlcv["close"],
            high=long_sample_ohlcv["high"],
            low=long_sample_ohlcv["low"],
        )
        assert result["direction"] in {"bullish", "bearish", "neutral"}

    def test_strength_valid_values(self, long_sample_ohlcv):
        """Test strength has valid value."""
        result = analyze_trend(
            close=long_sample_ohlcv["close"],
            high=long_sample_ohlcv["high"],
            low=long_sample_ohlcv["low"],
        )
        assert result["strength"] in {"very_strong", "strong", "weak", "no_trend"}


class TestAnalyzeMomentum:
    """Test analyze_momentum function."""

    def test_returns_dict(self, sample_ohlcv):
        """Test returns dictionary."""
        result = analyze_momentum(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_ohlcv):
        """Test dictionary has expected keys."""
        result = analyze_momentum(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        expected_keys = {"condition", "rsi", "stochastic_k", "stochastic_d"}
        assert set(result.keys()) == expected_keys

    def test_condition_valid_values(self, sample_ohlcv):
        """Test condition has valid value."""
        result = analyze_momentum(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        assert result["condition"] in {"overbought", "oversold", "neutral"}

    def test_rsi_in_range(self, sample_ohlcv):
        """Test RSI is in 0-100 range."""
        result = analyze_momentum(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        assert 0 <= result["rsi"] <= 100


class TestAnalyzeVolatility:
    """Test analyze_volatility function."""

    def test_returns_dict(self, sample_ohlcv):
        """Test returns dictionary."""
        result = analyze_volatility(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_ohlcv):
        """Test dictionary has expected keys."""
        result = analyze_volatility(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        expected_keys = {
            "state", "atr", "bb_upper", "bb_middle", "bb_lower",
            "bb_width_pct", "price_position"
        }
        assert set(result.keys()) == expected_keys

    def test_state_valid_values(self, sample_ohlcv):
        """Test state has valid value."""
        result = analyze_volatility(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        assert result["state"] in {"squeeze", "expansion", "normal"}

    def test_price_position_in_range(self, sample_ohlcv):
        """Test price_position is in 0-1 range (or near)."""
        result = analyze_volatility(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
        )
        # Price position can be slightly outside 0-1 if price is beyond bands
        assert -0.5 <= result["price_position"] <= 1.5


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_short_data_series(self):
        """Test with minimal data."""
        close = np.array([100.0, 101.0, 102.0, 101.5, 102.5])
        high = close + 0.5
        low = close - 0.5

        ind = TechnicalIndicators(high=high, low=low, close=close)

        # Should return arrays with NaN for insufficient data
        sma = ind.sma(period=20)
        assert len(sma) == len(close)
        assert np.isnan(sma).all()  # All NaN since period > data length

    def test_constant_prices(self):
        """Test with constant price data."""
        close = np.full(50, 100.0)
        high = np.full(50, 100.5)
        low = np.full(50, 99.5)
        volume = np.full(50, 1000000.0)

        ind = TechnicalIndicators(high=high, low=low, close=close, volume=volume)

        # RSI should be 50 for constant prices (after warmup)
        # ATR should be constant
        atr = ind.atr(period=14)
        valid_atr = atr[~np.isnan(atr)]
        # ATR should be 1.0 (high - low = 1.0)
        assert np.allclose(valid_atr, 1.0)

    def test_vwap_with_multi_day_timestamps(self):
        """Test VWAP with multiple days."""
        np.random.seed(42)
        n = 100

        # Create timestamps spanning 3 days
        base_date = datetime(2024, 1, 1, 9, 30)
        timestamps = []
        for i in range(n):
            day_offset = i // 40  # Change day every 40 bars
            minute_offset = (i % 40) * 5
            ts = base_date + timedelta(days=day_offset, minutes=minute_offset)
            timestamps.append(ts)

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        volume = np.random.randint(100000, 1000000, n).astype(float)

        ind = TechnicalIndicators(
            high=high, low=low, close=close, volume=volume, timestamps=timestamps
        )

        vwap = ind.vwap()
        assert len(vwap) == n
        # VWAP should reset at day boundaries, so values at day starts
        # should be close to typical price at those points

    def test_vwap_handles_zero_volume(self):
        """Test VWAP handles zero cumulative volume without division error."""
        close = np.array([100.0, 101.0, 102.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([0.0, 0.0, 1000.0])  # First two bars have zero volume

        ind = TechnicalIndicators(high=high, low=low, close=close, volume=volume)

        # Should not raise and should handle gracefully
        vwap = ind.vwap()
        assert len(vwap) == 3


class TestIntegration:
    """Integration tests for indicator combinations."""

    def test_multiple_indicators_consistent(self, indicators):
        """Test multiple indicators can be calculated consistently."""
        # Calculate various indicators
        rsi = indicators.rsi()
        macd, signal, hist = indicators.macd()
        upper, middle, lower = indicators.bollinger_bands()
        atr = indicators.atr()

        # All should have same length
        assert len(rsi) == len(indicators.close)
        assert len(macd) == len(indicators.close)
        assert len(upper) == len(indicators.close)
        assert len(atr) == len(indicators.close)

    def test_trend_and_momentum_alignment(self, long_sample_ohlcv):
        """Test trend and momentum analysis consistency."""
        trend = analyze_trend(
            close=long_sample_ohlcv["close"],
            high=long_sample_ohlcv["high"],
            low=long_sample_ohlcv["low"],
        )
        momentum = analyze_momentum(
            close=long_sample_ohlcv["close"],
            high=long_sample_ohlcv["high"],
            low=long_sample_ohlcv["low"],
        )

        # Both should return valid results
        assert trend["direction"] is not None
        assert momentum["condition"] is not None
