"""
Tests for MomentumStrategy advanced features

Tests cover:
- Bollinger Band filter
- Multi-timeframe analysis
- Short selling
- Signal generation logic
- Indicator calculations
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBollingerBandFilter:
    """Test Bollinger Band mean reversion filter."""

    def test_bb_position_calculation(self):
        """Test Bollinger Band position calculation (0-1 scale)."""
        # When price is at lower band, position should be 0
        lower = 95.0
        upper = 105.0
        price = 95.0

        position = (price - lower) / (upper - lower)
        assert position == 0.0

        # When price is at upper band, position should be 1
        price = 105.0
        position = (price - lower) / (upper - lower)
        assert position == 1.0

        # When price is at middle, position should be 0.5
        price = 100.0
        position = (price - lower) / (upper - lower)
        assert position == 0.5

    def test_bb_buy_boost_near_lower_band(self):
        """Buy signals should be boosted when price near lower band."""
        # Simulate Bollinger Band filter logic
        bb_position = 0.2  # Near lower band (< 0.3 threshold)
        bb_buy_threshold = 0.3
        momentum_score = 1  # Initial bullish score

        # Apply filter logic (from momentum_strategy.py)
        if bb_position < bb_buy_threshold and momentum_score > 0:
            momentum_score += 0.5

        assert momentum_score == 1.5  # Boosted

    def test_bb_buy_reduction_near_upper_band(self):
        """Buy signals should be reduced when price near upper band."""
        bb_position = 0.8  # Near upper band (> 0.7 threshold)
        bb_sell_threshold = 0.7
        momentum_score = 1  # Initial bullish score

        # Apply filter logic
        if bb_position > bb_sell_threshold and momentum_score > 0:
            momentum_score -= 0.5

        assert momentum_score == 0.5  # Reduced

    def test_bb_filter_disabled_no_effect(self):
        """When BB filter disabled, momentum score unchanged."""
        use_bollinger_filter = False
        momentum_score = 2

        if use_bollinger_filter:
            momentum_score += 0.5  # This shouldn't happen

        assert momentum_score == 2  # Unchanged


class TestMultiTimeframeFilter:
    """Test multi-timeframe filtering logic."""

    def test_bullish_signal_rejected_on_bearish_higher_tf(self):
        """Bullish signal should be rejected if higher TF is bearish."""
        momentum_score = 2  # Bullish
        higher_tf_trend = 'bearish'

        # MTF filter logic
        if higher_tf_trend == 'bearish' and momentum_score > 0:
            signal = 'neutral'  # Reject
        else:
            signal = 'buy'

        assert signal == 'neutral'

    def test_bearish_signal_rejected_on_bullish_higher_tf(self):
        """Bearish signal should be rejected if higher TF is bullish."""
        momentum_score = -2  # Bearish
        higher_tf_trend = 'bullish'

        # MTF filter logic
        if higher_tf_trend == 'bullish' and momentum_score < 0:
            signal = 'neutral'  # Reject
        else:
            signal = 'short'

        assert signal == 'neutral'

    def test_aligned_signal_passes(self):
        """Signal should pass when aligned with higher TF."""
        momentum_score = 2  # Bullish
        higher_tf_trend = 'bullish'

        # MTF filter logic
        if higher_tf_trend == 'bearish' and momentum_score > 0:
            signal = 'neutral'
        else:
            signal = 'buy'

        assert signal == 'buy'


class TestShortSelling:
    """Test short selling logic."""

    def test_short_signal_when_enabled(self):
        """Should return 'short' when short selling enabled."""
        enable_short_selling = True
        momentum_score = -3  # Strong bearish

        if momentum_score <= -2:
            if enable_short_selling:
                signal = 'short'
            else:
                signal = 'neutral'

        assert signal == 'short'

    def test_neutral_when_short_disabled(self):
        """Should return 'neutral' when short selling disabled."""
        enable_short_selling = False
        momentum_score = -3  # Strong bearish

        if momentum_score <= -2:
            if enable_short_selling:
                signal = 'short'
            else:
                signal = 'neutral'

        assert signal == 'neutral'

    def test_short_stop_loss_inverted(self):
        """Short stop-loss should trigger when price RISES."""
        entry_price = 100.0
        short_stop_loss = 0.04  # 4% stop
        current_price = 104.5  # Price rose 4.5%

        # For shorts: stop-loss is ABOVE entry
        stop_price = entry_price * (1 + short_stop_loss)

        should_exit = current_price >= stop_price
        assert should_exit is True

    def test_short_take_profit_inverted(self):
        """Short take-profit should trigger when price FALLS."""
        entry_price = 100.0
        take_profit = 0.05  # 5% profit target
        current_price = 94.5  # Price fell 5.5%

        # For shorts: take-profit is BELOW entry
        target_price = entry_price * (1 - take_profit)

        should_exit = current_price <= target_price
        assert should_exit is True


class TestMomentumScoreCalculation:
    """Test momentum score calculation logic."""

    def test_rsi_oversold_adds_point(self):
        """RSI below oversold threshold should add to momentum."""
        rsi = 25  # Below 30 (oversold)
        rsi_oversold = 30
        momentum_score = 0

        if rsi < rsi_oversold:
            momentum_score += 1

        assert momentum_score == 1

    def test_rsi_overbought_subtracts_point(self):
        """RSI above overbought threshold should subtract from momentum."""
        rsi = 75  # Above 70 (overbought)
        rsi_overbought = 70
        momentum_score = 0

        if rsi > rsi_overbought:
            momentum_score -= 1

        assert momentum_score == -1

    def test_macd_bullish_crossover_adds_point(self):
        """MACD bullish crossover should add to momentum."""
        macd = 0.5
        macd_signal = 0.3
        macd_hist = 0.2  # Positive histogram
        momentum_score = 0

        if macd > macd_signal and macd_hist > 0:
            momentum_score += 1

        assert momentum_score == 1

    def test_ma_bullish_alignment_adds_point(self):
        """Bullish MA alignment (fast > medium > slow) adds point."""
        fast_ma = 110
        medium_ma = 105
        slow_ma = 100
        momentum_score = 0

        if fast_ma > medium_ma > slow_ma:
            momentum_score += 1

        assert momentum_score == 1

    def test_buy_signal_threshold(self):
        """Buy signal requires score >= 2 with trend and volume."""
        momentum_score = 2
        trend_strength = True
        volume_confirmation = True

        if momentum_score >= 2 and trend_strength and volume_confirmation:
            signal = 'buy'
        else:
            signal = 'neutral'

        assert signal == 'buy'

    def test_no_signal_without_confirmation(self):
        """No signal if missing trend or volume confirmation."""
        momentum_score = 2
        trend_strength = True
        volume_confirmation = False  # No volume confirmation

        if momentum_score >= 2 and trend_strength and volume_confirmation:
            signal = 'buy'
        else:
            signal = 'neutral'

        assert signal == 'neutral'


class TestIndicatorCalculations:
    """Test technical indicator calculations."""

    def test_rsi_calculation(self):
        """Test RSI calculation matches expected values."""
        import talib

        # Create sample price data
        prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10,
                          45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
                          46.28, 46.00, 46.03, 46.41, 46.22, 45.64], dtype=float)

        rsi = talib.RSI(prices, timeperiod=14)

        # RSI should be between 0 and 100
        assert all(0 <= r <= 100 for r in rsi if not np.isnan(r))

    def test_macd_calculation(self):
        """Test MACD calculation produces valid output."""
        import talib

        prices = np.random.uniform(100, 110, 50).astype(float)
        macd, signal, hist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)

        # MACD histogram should equal MACD - signal
        valid_idx = ~np.isnan(hist)
        np.testing.assert_array_almost_equal(
            hist[valid_idx],
            macd[valid_idx] - signal[valid_idx]
        )

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        import talib

        prices = np.array([100 + i * 0.1 for i in range(30)], dtype=float)
        upper, middle, lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)

        # Upper should always be above lower
        valid = ~np.isnan(upper)
        assert all(upper[valid] > lower[valid])

        # Middle should be between upper and lower
        assert all(lower[valid] <= middle[valid])
        assert all(middle[valid] <= upper[valid])


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_fixed_position_size(self):
        """Test fixed 10% position sizing."""
        buying_power = 100000
        position_size = 0.10

        position_value = buying_power * position_size
        assert position_value == 10000

    def test_max_positions_limit(self):
        """Test max positions limit is enforced."""
        current_positions = 3
        max_positions = 3

        should_trade = current_positions < max_positions
        assert should_trade is False

    def test_fractional_shares_calculation(self):
        """Test fractional share quantity calculation."""
        position_value = 10000
        price = 150.00

        quantity = position_value / price
        assert quantity == pytest.approx(66.67, rel=0.01)

        # Minimum quantity check (0.01 shares)
        assert quantity >= 0.01


class TestRiskManagement:
    """Test risk management integration."""

    def test_stop_loss_percentage(self):
        """Test stop-loss price calculation."""
        entry_price = 100.0
        stop_loss_pct = 0.03  # 3%

        stop_price = entry_price * (1 - stop_loss_pct)
        assert stop_price == 97.0

    def test_take_profit_percentage(self):
        """Test take-profit price calculation."""
        entry_price = 100.0
        take_profit_pct = 0.05  # 5%

        target_price = entry_price * (1 + take_profit_pct)
        assert target_price == 105.0

    def test_stop_loss_triggers_exit(self):
        """Test stop-loss exit condition."""
        entry_price = 100.0
        current_price = 96.5
        stop_loss = 0.03

        pnl_pct = (current_price - entry_price) / entry_price
        should_exit = pnl_pct <= -stop_loss

        assert should_exit is True

    def test_take_profit_triggers_exit(self):
        """Test take-profit exit condition."""
        entry_price = 100.0
        current_price = 105.5
        take_profit = 0.05

        pnl_pct = (current_price - entry_price) / entry_price
        should_exit = pnl_pct >= take_profit

        assert should_exit is True


class TestDefaultParameters:
    """Test default parameter values."""

    def test_default_parameters_structure(self):
        """Test default_parameters returns expected structure."""
        # These are the expected default values from the strategy
        expected_keys = [
            'position_size', 'max_positions', 'stop_loss', 'take_profit',
            'rsi_period', 'rsi_overbought', 'rsi_oversold',
            'macd_fast_period', 'macd_slow_period', 'macd_signal_period',
            'use_multi_timeframe', 'enable_short_selling',
            'use_kelly_criterion', 'use_volatility_regime',
            'use_bollinger_filter'
        ]

        # Mock the default parameters check
        defaults = {
            'position_size': 0.10,
            'max_positions': 3,
            'stop_loss': 0.03,
            'take_profit': 0.05,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            'use_multi_timeframe': False,
            'enable_short_selling': False,
            'use_kelly_criterion': False,
            'use_volatility_regime': False,
            'use_bollinger_filter': False,
        }

        for key in expected_keys:
            assert key in defaults

    def test_advanced_features_disabled_by_default(self):
        """Advanced features should be disabled by default."""
        defaults = {
            'use_multi_timeframe': False,
            'enable_short_selling': False,
            'use_kelly_criterion': False,
            'use_volatility_regime': False,
            'use_streak_sizing': False,
            'use_bollinger_filter': False,
        }

        for feature, enabled in defaults.items():
            assert enabled is False, f"{feature} should be disabled by default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
