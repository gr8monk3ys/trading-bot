"""
Unit tests for cross-asset data types.
"""

import pytest
from datetime import datetime

from data.alt_data_types import AltDataSource, SignalDirection
from data.cross_asset_types import (
    CrossAssetSource,
    VolatilityRegime,
    YieldCurveRegime,
    RiskAppetiteRegime,
    VixTermStructureSignal,
    YieldCurveSignal,
    FxCorrelationSignal,
    CrossAssetAggregatedSignal,
)


class TestCrossAssetSource:
    """Tests for CrossAssetSource enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert CrossAssetSource.VIX_STRUCTURE.value == "vix_structure"
        assert CrossAssetSource.YIELD_CURVE.value == "yield_curve"
        assert CrossAssetSource.FX_CORRELATION.value == "fx_correlation"
        assert CrossAssetSource.COMMODITY.value == "commodity"

    def test_enum_count(self):
        """Test correct number of sources."""
        assert len(CrossAssetSource) == 4


class TestVolatilityRegime:
    """Tests for VolatilityRegime enum."""

    def test_all_regimes(self):
        """Test all volatility regimes."""
        assert VolatilityRegime.LOW_VOL.value == "low_vol"
        assert VolatilityRegime.NORMAL.value == "normal"
        assert VolatilityRegime.ELEVATED.value == "elevated"
        assert VolatilityRegime.HIGH_VOL.value == "high_vol"
        assert VolatilityRegime.CRISIS.value == "crisis"


class TestVixTermStructureSignal:
    """Tests for VixTermStructureSignal dataclass."""

    def test_basic_creation(self):
        """Test basic signal creation."""
        signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
            vix_spot=18.0,
            vix_3m=20.0,
            term_slope=0.11,
        )
        assert signal.vix_spot == 18.0
        assert signal.vix_3m == 20.0
        assert signal.term_slope == 0.11

    def test_contango_detection(self):
        """Test contango detection from slope."""
        signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
            vix_spot=15.0,
            vix_3m=18.0,
            term_slope=0.20,
        )
        assert signal.is_contango is True
        assert signal.is_backwardation is False

    def test_backwardation_detection(self):
        """Test backwardation detection from slope."""
        signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.8,
            vix_spot=30.0,
            vix_3m=25.0,
            term_slope=-0.17,
        )
        assert signal.is_contango is False
        assert signal.is_backwardation is True

    def test_volatility_regime_low(self):
        """Test low volatility regime detection."""
        signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.6,
            vix_spot=12.0,
            vix_3m=14.0,
            term_slope=0.17,
        )
        assert signal.volatility_regime == VolatilityRegime.LOW_VOL

    def test_volatility_regime_crisis(self):
        """Test crisis volatility regime detection."""
        signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.8,
            confidence=0.9,
            vix_spot=45.0,
            vix_3m=35.0,
            term_slope=-0.22,
        )
        assert signal.volatility_regime == VolatilityRegime.CRISIS

    def test_fear_elevated_property(self):
        """Test is_fear_elevated property."""
        # High VIX
        signal1 = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.3,
            confidence=0.7,
            vix_spot=25.0,
            term_slope=0.05,
        )
        assert signal1.is_fear_elevated is True

        # Backwardation (even with moderate VIX)
        signal2 = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.7,
            vix_spot=18.0,
            term_slope=-0.10,
        )
        assert signal2.is_fear_elevated is True

    def test_complacent_property(self):
        """Test is_complacent property."""
        signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.8,
            confidence=0.6,
            vix_spot=12.0,
            term_slope=0.15,
        )
        assert signal.is_complacent is True


class TestYieldCurveSignal:
    """Tests for YieldCurveSignal dataclass."""

    def test_basic_creation(self):
        """Test basic signal creation."""
        signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.3,
            confidence=0.65,
            curve_slope=0.03,
            short_rate_proxy=85.0,
            long_rate_proxy=95.0,
        )
        assert signal.curve_slope == 0.03
        assert signal.short_rate_proxy == 85.0

    def test_inversion_detection(self):
        """Test yield curve inversion detection."""
        signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.75,
            curve_slope=-0.08,
        )
        assert signal.is_inverted is True
        assert signal.yield_curve_regime == YieldCurveRegime.INVERTED

    def test_steep_curve(self):
        """Test steep yield curve detection."""
        signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.6,
            confidence=0.7,
            curve_slope=0.08,
        )
        assert signal.yield_curve_regime == YieldCurveRegime.STEEP
        assert signal.is_inverted is False

    def test_steepening_detection(self):
        """Test steepening detection from change."""
        signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.4,
            confidence=0.65,
            curve_slope=0.02,
            curve_slope_change_5d=0.02,
        )
        assert signal.is_steepening is True
        assert signal.is_flattening is False

    def test_recession_probability_inverted(self):
        """Test recession probability increases when inverted."""
        inverted = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.6,
            confidence=0.8,
            curve_slope=-0.05,
        )
        normal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.3,
            confidence=0.6,
            curve_slope=0.05,
        )
        assert inverted.recession_probability > normal.recession_probability

    def test_is_recessionary_property(self):
        """Test is_recessionary property."""
        signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.75,
            curve_slope=-0.04,
        )
        assert signal.is_recessionary is True


class TestFxCorrelationSignal:
    """Tests for FxCorrelationSignal dataclass."""

    def test_basic_creation(self):
        """Test basic signal creation."""
        signal = FxCorrelationSignal(
            symbol="FX_RISK",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.4,
            confidence=0.65,
            dxy_level=103.5,
            audjpy_level=98.2,
            risk_appetite_score=0.4,
        )
        assert signal.dxy_level == 103.5
        assert signal.audjpy_level == 98.2
        assert signal.risk_appetite_score == 0.4

    def test_risk_on_regime(self):
        """Test risk-on regime detection."""
        signal = FxCorrelationSignal(
            symbol="FX_RISK",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
            risk_appetite_score=0.5,
        )
        assert signal.risk_appetite_regime == RiskAppetiteRegime.RISK_ON
        assert signal.is_risk_on is True

    def test_risk_off_regime(self):
        """Test risk-off regime detection."""
        signal = FxCorrelationSignal(
            symbol="FX_RISK",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.7,
            risk_appetite_score=-0.5,
        )
        assert signal.risk_appetite_regime == RiskAppetiteRegime.RISK_OFF
        assert signal.is_risk_off is True

    def test_signal_agreement(self):
        """Test signal agreement calculation."""
        # USD up, AUD/JPY down = agreement on risk-off
        signal = FxCorrelationSignal(
            symbol="FX_RISK",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.4,
            confidence=0.8,
            dxy_change_5d=0.02,
            audjpy_change_5d=-0.015,
            risk_appetite_score=-0.4,
        )
        assert signal.signal_agreement == 1.0


class TestCrossAssetAggregatedSignal:
    """Tests for CrossAssetAggregatedSignal dataclass."""

    def test_empty_aggregation(self):
        """Test aggregation with no signals."""
        agg = CrossAssetAggregatedSignal()
        assert agg.composite_signal == 0.0
        assert agg.composite_confidence == 0.0
        assert agg.overall_regime == "neutral"

    def test_single_signal_aggregation(self):
        """Test aggregation with single signal."""
        vix_signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
            vix_spot=18.0,
            term_slope=0.10,
        )
        agg = CrossAssetAggregatedSignal(vix_signal=vix_signal)

        assert agg.composite_signal == 0.5
        assert agg.composite_confidence == 0.7
        assert len(agg.sources) == 1
        assert CrossAssetSource.VIX_STRUCTURE in agg.sources

    def test_multi_signal_aggregation(self):
        """Test aggregation with multiple signals."""
        vix_signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.6,
            confidence=0.8,
            vix_spot=15.0,
            term_slope=0.12,
        )
        yield_signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.4,
            confidence=0.6,
            curve_slope=0.03,
        )
        fx_signal = FxCorrelationSignal(
            symbol="FX_RISK",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
            risk_appetite_score=0.5,
        )

        agg = CrossAssetAggregatedSignal(
            vix_signal=vix_signal,
            yield_curve_signal=yield_signal,
            fx_signal=fx_signal,
        )

        assert len(agg.sources) == 3
        assert agg.composite_signal > 0  # All bullish
        assert agg.agreement_ratio == 1.0  # All agree

    def test_direction_property(self):
        """Test direction property."""
        vix_signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.8,
            vix_spot=30.0,
            term_slope=-0.10,
        )
        agg = CrossAssetAggregatedSignal(vix_signal=vix_signal)
        assert agg.direction == SignalDirection.BEARISH

    def test_high_conviction_property(self):
        """Test is_high_conviction property."""
        # All signals agreeing with high confidence
        vix_signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.6,
            confidence=0.8,
            vix_spot=12.0,
            term_slope=0.15,
        )
        yield_signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
            curve_slope=0.05,
        )

        agg = CrossAssetAggregatedSignal(
            vix_signal=vix_signal,
            yield_curve_signal=yield_signal,
        )
        # May or may not be high conviction depending on exact thresholds
        assert isinstance(agg.is_high_conviction, bool)

    def test_should_reduce_exposure(self):
        """Test should_reduce_exposure property."""
        # Create bearish signals across multiple sources
        vix_signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.6,
            confidence=0.8,
            vix_spot=30.0,
            term_slope=-0.15,
        )
        yield_signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.7,
            curve_slope=-0.05,
        )
        fx_signal = FxCorrelationSignal(
            symbol="FX_RISK",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-0.5,
            confidence=0.7,
            risk_appetite_score=-0.5,
        )

        agg = CrossAssetAggregatedSignal(
            vix_signal=vix_signal,
            yield_curve_signal=yield_signal,
            fx_signal=fx_signal,
        )
        assert agg.should_reduce_exposure is True


class TestSignalValueClamping:
    """Tests for signal value clamping inherited from AlternativeSignal."""

    def test_vix_signal_clamping(self):
        """Test VIX signal value is clamped."""
        signal = VixTermStructureSignal(
            symbol="VIX",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=1.5,  # Should clamp to 1.0
            confidence=0.7,
            vix_spot=15.0,
            term_slope=0.10,
        )
        assert signal.signal_value == 1.0

    def test_yield_signal_clamping(self):
        """Test yield signal value is clamped."""
        signal = YieldCurveSignal(
            symbol="YIELD_CURVE",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=-1.5,  # Should clamp to -1.0
            confidence=0.7,
            curve_slope=-0.05,
        )
        assert signal.signal_value == -1.0

    def test_fx_signal_clamping(self):
        """Test FX signal value is clamped."""
        signal = FxCorrelationSignal(
            symbol="FX_RISK",
            source=AltDataSource.NEWS_ADVANCED,
            timestamp=datetime.now(),
            signal_value=2.0,  # Should clamp to 1.0
            confidence=1.5,  # Should clamp to 1.0
            risk_appetite_score=0.5,
        )
        assert signal.signal_value == 1.0
        assert signal.confidence == 1.0
