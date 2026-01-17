"""
Advanced Technical Indicators Library

Comprehensive collection of technical analysis indicators for algorithmic trading.
All indicators use TA-Lib for consistency and performance.

Categories:
- Trend Indicators: SMA, EMA, MACD, ADX, Parabolic SAR
- Momentum Indicators: RSI, Stochastic, CCI, Williams %R, ROC
- Volatility Indicators: Bollinger Bands, ATR, Keltner Channels, Standard Deviation
- Volume Indicators: VWAP, OBV, Volume SMA, Money Flow Index
- Support/Resistance: Pivot Points, Fibonacci Retracements

Usage:
    from utils.indicators import TechnicalIndicators

    # Initialize with price data
    indicators = TechnicalIndicators(
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volumes
    )

    # Calculate indicators
    rsi = indicators.rsi(period=14)
    vwap = indicators.vwap()
    bb_upper, bb_middle, bb_lower = indicators.bollinger_bands(period=20, std=2.0)
"""

import numpy as np
import talib
from typing import Tuple, Dict, List
from datetime import datetime


class TechnicalIndicators:
    """
    Technical indicators calculator using TA-Lib.

    All methods return numpy arrays aligned with the input data.
    NaN values indicate insufficient data for calculation.
    """

    def __init__(self,
                 high: np.ndarray = None,
                 low: np.ndarray = None,
                 close: np.ndarray = None,
                 open_: np.ndarray = None,
                 volume: np.ndarray = None,
                 timestamps: List[datetime] = None):
        """
        Initialize with price/volume data.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open_: Open prices (optional)
            volume: Trading volume (optional)
            timestamps: Timestamps for each bar (optional, needed for VWAP)
        """
        self.high = np.array(high) if high is not None else None
        self.low = np.array(low) if low is not None else None
        self.close = np.array(close) if close is not None else None
        self.open = np.array(open_) if open_ is not None else None
        self.volume = np.array(volume) if volume is not None else None
        self.timestamps = timestamps

    # ==================== TREND INDICATORS ====================

    def sma(self, period: int = 20, price: str = 'close') -> np.ndarray:
        """
        Simple Moving Average.

        Args:
            period: Number of periods (default: 20)
            price: Price type ('close', 'high', 'low', 'open')

        Returns:
            SMA values
        """
        prices = self._get_price_array(price)
        return talib.SMA(prices, timeperiod=period)

    def ema(self, period: int = 20, price: str = 'close') -> np.ndarray:
        """
        Exponential Moving Average.

        Args:
            period: Number of periods (default: 20)
            price: Price type ('close', 'high', 'low', 'open')

        Returns:
            EMA values
        """
        prices = self._get_price_array(price)
        return talib.EMA(prices, timeperiod=period)

    def macd(self,
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence Divergence.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)

        Returns:
            Tuple of (macd, signal, histogram)
        """
        return talib.MACD(
            self.close,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )

    def adx(self, period: int = 14) -> np.ndarray:
        """
        Average Directional Index (trend strength).

        Values:
            0-25: Weak or absent trend
            25-50: Strong trend
            50-75: Very strong trend
            75-100: Extremely strong trend

        Args:
            period: Number of periods (default: 14)

        Returns:
            ADX values
        """
        return talib.ADX(self.high, self.low, self.close, timeperiod=period)

    def adx_di(self, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ADX with Directional Indicators.

        Returns ADX, +DI (bullish), and -DI (bearish).
        When +DI > -DI: Uptrend
        When -DI > +DI: Downtrend

        Args:
            period: Number of periods (default: 14)

        Returns:
            Tuple of (adx, plus_di, minus_di)
        """
        adx = talib.ADX(self.high, self.low, self.close, timeperiod=period)
        plus_di = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=period)
        minus_di = talib.MINUS_DI(self.high, self.low, self.close, timeperiod=period)
        return adx, plus_di, minus_di

    def parabolic_sar(self,
                      acceleration: float = 0.02,
                      maximum: float = 0.2) -> np.ndarray:
        """
        Parabolic SAR (Stop and Reverse).

        Trailing stop and trend indicator.
        When price > SAR: Uptrend
        When price < SAR: Downtrend

        Args:
            acceleration: Acceleration factor (default: 0.02)
            maximum: Maximum acceleration (default: 0.2)

        Returns:
            SAR values
        """
        return talib.SAR(self.high, self.low, acceleration=acceleration, maximum=maximum)

    # ==================== MOMENTUM INDICATORS ====================

    def rsi(self, period: int = 14, price: str = 'close') -> np.ndarray:
        """
        Relative Strength Index.

        Values:
            > 70: Overbought
            < 30: Oversold

        Args:
            period: Number of periods (default: 14)
            price: Price type ('close', 'high', 'low', 'open')

        Returns:
            RSI values (0-100)
        """
        prices = self._get_price_array(price)
        return talib.RSI(prices, timeperiod=period)

    def stochastic(self,
                   fastk_period: int = 14,
                   slowk_period: int = 3,
                   slowd_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator.

        Values:
            > 80: Overbought
            < 20: Oversold

        Signals:
            %K crosses above %D: Bullish
            %K crosses below %D: Bearish

        Args:
            fastk_period: Fast %K period (default: 14)
            slowk_period: Slow %K period (default: 3)
            slowd_period: Slow %D period (default: 3)

        Returns:
            Tuple of (slowk, slowd)
        """
        slowk, slowd = talib.STOCH(
            self.high,
            self.low,
            self.close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )
        return slowk, slowd

    def stochastic_rsi(self,
                       rsi_period: int = 14,
                       stoch_period: int = 14,
                       k_period: int = 3,
                       d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic RSI (more sensitive than regular Stochastic).

        Combines RSI with Stochastic for faster signals.

        Args:
            rsi_period: RSI calculation period (default: 14)
            stoch_period: Stochastic calculation period (default: 14)
            k_period: %K smoothing (default: 3)
            d_period: %D smoothing (default: 3)

        Returns:
            Tuple of (fastk, fastd)
        """
        fastk, fastd = talib.STOCHRSI(
            self.close,
            timeperiod=rsi_period,
            fastk_period=k_period,
            fastd_period=d_period,
            fastd_matype=0
        )
        return fastk, fastd

    def cci(self, period: int = 20) -> np.ndarray:
        """
        Commodity Channel Index.

        Values:
            > 100: Overbought
            < -100: Oversold

        Args:
            period: Number of periods (default: 20)

        Returns:
            CCI values
        """
        return talib.CCI(self.high, self.low, self.close, timeperiod=period)

    def williams_r(self, period: int = 14) -> np.ndarray:
        """
        Williams %R (momentum indicator).

        Values:
            > -20: Overbought
            < -80: Oversold

        Args:
            period: Number of periods (default: 14)

        Returns:
            Williams %R values (-100 to 0)
        """
        return talib.WILLR(self.high, self.low, self.close, timeperiod=period)

    def roc(self, period: int = 12, price: str = 'close') -> np.ndarray:
        """
        Rate of Change (price momentum).

        Measures percentage change over period.
        Positive: Upward momentum
        Negative: Downward momentum

        Args:
            period: Number of periods (default: 12)
            price: Price type ('close', 'high', 'low', 'open')

        Returns:
            ROC values (percentage)
        """
        prices = self._get_price_array(price)
        return talib.ROC(prices, timeperiod=period)

    # ==================== VOLATILITY INDICATORS ====================

    def bollinger_bands(self,
                        period: int = 20,
                        std: float = 2.0,
                        price: str = 'close') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands (volatility and overbought/oversold).

        Interpretation:
            Price near upper band: Overbought
            Price near lower band: Oversold
            Bands squeeze: Low volatility (breakout coming)
            Bands expand: High volatility

        Args:
            period: Number of periods (default: 20)
            std: Standard deviations (default: 2.0)
            price: Price type ('close', 'high', 'low', 'open')

        Returns:
            Tuple of (upper, middle, lower)
        """
        prices = self._get_price_array(price)
        upper, middle, lower = talib.BBANDS(
            prices,
            timeperiod=period,
            nbdevup=std,
            nbdevdn=std,
            matype=0
        )
        return upper, middle, lower

    def atr(self, period: int = 14) -> np.ndarray:
        """
        Average True Range (volatility).

        Measures market volatility.
        Higher ATR = Higher volatility
        Used for stop-loss placement and position sizing.

        Args:
            period: Number of periods (default: 14)

        Returns:
            ATR values
        """
        return talib.ATR(self.high, self.low, self.close, timeperiod=period)

    def keltner_channels(self,
                         ema_period: int = 20,
                         atr_period: int = 10,
                         atr_multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Keltner Channels (volatility-based bands).

        Similar to Bollinger Bands but uses ATR instead of standard deviation.

        Args:
            ema_period: EMA period for middle line (default: 20)
            atr_period: ATR period (default: 10)
            atr_multiplier: ATR multiplier (default: 2.0)

        Returns:
            Tuple of (upper, middle, lower)
        """
        middle = talib.EMA(self.close, timeperiod=ema_period)
        atr = talib.ATR(self.high, self.low, self.close, timeperiod=atr_period)

        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)

        return upper, middle, lower

    def stddev(self, period: int = 20, price: str = 'close') -> np.ndarray:
        """
        Standard Deviation (volatility measure).

        Args:
            period: Number of periods (default: 20)
            price: Price type ('close', 'high', 'low', 'open')

        Returns:
            Standard deviation values
        """
        prices = self._get_price_array(price)
        return talib.STDDEV(prices, timeperiod=period, nbdev=1)

    # ==================== VOLUME INDICATORS ====================

    def vwap(self) -> np.ndarray:
        """
        Volume Weighted Average Price.

        Institutional benchmark price.
        Price above VWAP: Bullish
        Price below VWAP: Bearish

        Requires volume and timestamps to be set.
        Resets at start of each trading day.

        Returns:
            VWAP values
        """
        if self.volume is None:
            raise ValueError("Volume data required for VWAP calculation")

        # Typical price: (H + L + C) / 3
        typical_price = (self.high + self.low + self.close) / 3

        # If timestamps provided, reset VWAP daily
        if self.timestamps:
            vwap_values = np.zeros(len(typical_price))

            # Group by date
            dates = [ts.date() if hasattr(ts, 'date') else ts for ts in self.timestamps]
            unique_dates = sorted(set(dates))

            for date in unique_dates:
                # Find indices for this date
                date_mask = np.array([d == date for d in dates])
                date_indices = np.where(date_mask)[0]

                if len(date_indices) == 0:
                    continue

                # Calculate VWAP for this day
                day_typical = typical_price[date_indices]
                day_volume = self.volume[date_indices]

                # Cumulative (typical_price * volume) / cumulative volume
                pv = day_typical * day_volume
                cumsum_pv = np.cumsum(pv)
                cumsum_volume = np.cumsum(day_volume)

                # Avoid division by zero
                cumsum_volume[cumsum_volume == 0] = 1

                vwap_values[date_indices] = cumsum_pv / cumsum_volume

            return vwap_values
        else:
            # Simple cumulative VWAP (no daily reset)
            pv = typical_price * self.volume
            cumsum_pv = np.cumsum(pv)
            cumsum_volume = np.cumsum(self.volume)

            # Avoid division by zero
            cumsum_volume[cumsum_volume == 0] = 1

            return cumsum_pv / cumsum_volume

    def obv(self) -> np.ndarray:
        """
        On-Balance Volume (volume momentum).

        Cumulative volume indicator:
            Price up: Add volume
            Price down: Subtract volume

        Rising OBV: Buying pressure
        Falling OBV: Selling pressure

        Returns:
            OBV values
        """
        if self.volume is None:
            raise ValueError("Volume data required for OBV calculation")

        return talib.OBV(self.close, self.volume)

    def volume_sma(self, period: int = 20) -> np.ndarray:
        """
        Volume Simple Moving Average.

        Average volume over period.
        Current volume > SMA: High volume (confirmation)
        Current volume < SMA: Low volume (weak signal)

        Args:
            period: Number of periods (default: 20)

        Returns:
            Volume SMA values
        """
        if self.volume is None:
            raise ValueError("Volume data required for Volume SMA")

        return talib.SMA(self.volume, timeperiod=period)

    def mfi(self, period: int = 14) -> np.ndarray:
        """
        Money Flow Index (volume-weighted RSI).

        Values:
            > 80: Overbought
            < 20: Oversold

        Args:
            period: Number of periods (default: 14)

        Returns:
            MFI values (0-100)
        """
        if self.volume is None:
            raise ValueError("Volume data required for MFI")

        return talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=period)

    # ==================== SUPPORT/RESISTANCE ====================

    def pivot_points(self) -> Dict[str, float]:
        """
        Classic Pivot Points (support/resistance levels).

        Uses last complete bar's high, low, close.

        Returns:
            Dict with keys: PP, R1, R2, R3, S1, S2, S3
        """
        # Use last bar
        high = self.high[-1]
        low = self.low[-1]
        close = self.close[-1]

        # Pivot Point
        pp = (high + low + close) / 3

        # Support and Resistance levels
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)

        s1 = 2 * pp - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)

        return {
            'PP': pp,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }

    def fibonacci_retracement(self,
                              swing_high: float = None,
                              swing_low: float = None) -> Dict[str, float]:
        """
        Fibonacci Retracement Levels.

        Args:
            swing_high: Swing high price (default: max of high array)
            swing_low: Swing low price (default: min of low array)

        Returns:
            Dict with retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%, 100%)
        """
        if swing_high is None:
            swing_high = np.max(self.high)

        if swing_low is None:
            swing_low = np.min(self.low)

        diff = swing_high - swing_low

        return {
            '0.0%': swing_high,
            '23.6%': swing_high - 0.236 * diff,
            '38.2%': swing_high - 0.382 * diff,
            '50.0%': swing_high - 0.500 * diff,
            '61.8%': swing_high - 0.618 * diff,
            '78.6%': swing_high - 0.786 * diff,
            '100.0%': swing_low
        }

    # ==================== HELPER METHODS ====================

    def _get_price_array(self, price: str) -> np.ndarray:
        """Get price array by name."""
        if price == 'close':
            return self.close
        elif price == 'high':
            return self.high
        elif price == 'low':
            return self.low
        elif price == 'open':
            return self.open
        else:
            raise ValueError(f"Invalid price type: {price}")

    # ==================== COMPOSITE INDICATORS ====================

    def all_momentum_indicators(self) -> Dict[str, any]:
        """
        Calculate all momentum indicators at once.

        Returns:
            Dict with all momentum indicator values
        """
        return {
            'rsi': self.rsi(),
            'stochastic': self.stochastic(),
            'cci': self.cci(),
            'williams_r': self.williams_r(),
            'roc': self.roc(),
            'macd': self.macd()
        }

    def all_trend_indicators(self) -> Dict[str, any]:
        """
        Calculate all trend indicators at once.

        Returns:
            Dict with all trend indicator values
        """
        adx, plus_di, minus_di = self.adx_di()

        return {
            'sma_20': self.sma(20),
            'sma_50': self.sma(50),
            'sma_200': self.sma(200),
            'ema_20': self.ema(20),
            'ema_50': self.ema(50),
            'macd': self.macd(),
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'parabolic_sar': self.parabolic_sar()
        }

    def all_volatility_indicators(self) -> Dict[str, any]:
        """
        Calculate all volatility indicators at once.

        Returns:
            Dict with all volatility indicator values
        """
        return {
            'bollinger_bands': self.bollinger_bands(),
            'atr': self.atr(),
            'keltner_channels': self.keltner_channels(),
            'stddev': self.stddev()
        }


# ==================== QUICK ANALYSIS FUNCTIONS ====================

def analyze_trend(close: np.ndarray,
                  high: np.ndarray,
                  low: np.ndarray) -> Dict[str, any]:
    """
    Quick trend analysis.

    Returns trend strength, direction, and key levels.
    """
    ind = TechnicalIndicators(high=high, low=low, close=close)

    adx, plus_di, minus_di = ind.adx_di(period=14)
    sma_50 = ind.sma(period=50)
    sma_200 = ind.sma(period=200)

    # Current values
    current_price = close[-1]
    current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
    current_plus_di = plus_di[-1] if not np.isnan(plus_di[-1]) else 0
    current_minus_di = minus_di[-1] if not np.isnan(minus_di[-1]) else 0

    # Trend direction
    if current_plus_di > current_minus_di:
        direction = 'bullish'
    elif current_minus_di > current_plus_di:
        direction = 'bearish'
    else:
        direction = 'neutral'

    # Trend strength
    if current_adx > 50:
        strength = 'very_strong'
    elif current_adx > 25:
        strength = 'strong'
    elif current_adx > 15:
        strength = 'weak'
    else:
        strength = 'no_trend'

    return {
        'direction': direction,
        'strength': strength,
        'adx': current_adx,
        'plus_di': current_plus_di,
        'minus_di': current_minus_di,
        'sma_50': sma_50[-1] if not np.isnan(sma_50[-1]) else None,
        'sma_200': sma_200[-1] if not np.isnan(sma_200[-1]) else None,
        'price_vs_sma_50': 'above' if current_price > sma_50[-1] else 'below',
        'price_vs_sma_200': 'above' if current_price > sma_200[-1] else 'below'
    }


def analyze_momentum(close: np.ndarray,
                     high: np.ndarray,
                     low: np.ndarray) -> Dict[str, any]:
    """
    Quick momentum analysis.

    Returns overbought/oversold conditions and momentum strength.
    """
    ind = TechnicalIndicators(high=high, low=low, close=close)

    rsi = ind.rsi(period=14)
    slowk, slowd = ind.stochastic()

    current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
    current_stoch_k = slowk[-1] if not np.isnan(slowk[-1]) else 50

    # Conditions
    if current_rsi > 70 and current_stoch_k > 80:
        condition = 'overbought'
    elif current_rsi < 30 and current_stoch_k < 20:
        condition = 'oversold'
    else:
        condition = 'neutral'

    return {
        'condition': condition,
        'rsi': current_rsi,
        'stochastic_k': current_stoch_k,
        'stochastic_d': slowd[-1] if not np.isnan(slowd[-1]) else 50
    }


def analyze_volatility(close: np.ndarray,
                       high: np.ndarray,
                       low: np.ndarray) -> Dict[str, any]:
    """
    Quick volatility analysis.

    Returns volatility level and key bands.
    """
    ind = TechnicalIndicators(high=high, low=low, close=close)

    atr = ind.atr(period=14)
    upper, middle, lower = ind.bollinger_bands(period=20, std=2.0)

    current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
    current_price = close[-1]

    # BB squeeze detection
    bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] > 0 else 0

    if bb_width < 0.04:  # Less than 4% width
        volatility_state = 'squeeze'  # Breakout coming
    elif bb_width > 0.12:  # More than 12% width
        volatility_state = 'expansion'  # High volatility
    else:
        volatility_state = 'normal'

    return {
        'state': volatility_state,
        'atr': current_atr,
        'bb_upper': upper[-1],
        'bb_middle': middle[-1],
        'bb_lower': lower[-1],
        'bb_width_pct': bb_width * 100,
        'price_position': (current_price - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
    }
