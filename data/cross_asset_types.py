"""
Cross-Asset Data Types - Dataclasses for cross-asset signals.

This module defines data structures for VIX term structure, yield curve,
and FX correlation signals used for cross-asset alpha generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from data.alt_data_types import AlternativeSignal, SignalDirection


class CrossAssetSource(Enum):
    """Sources of cross-asset signals."""

    VIX_STRUCTURE = "vix_structure"
    YIELD_CURVE = "yield_curve"
    FX_CORRELATION = "fx_correlation"
    COMMODITY = "commodity"


class VolatilityRegime(Enum):
    """VIX-based volatility regime classification."""

    LOW_VOL = "low_vol"  # VIX < 15
    NORMAL = "normal"  # VIX 15-20
    ELEVATED = "elevated"  # VIX 20-25
    HIGH_VOL = "high_vol"  # VIX 25-35
    CRISIS = "crisis"  # VIX > 35


class YieldCurveRegime(Enum):
    """Yield curve shape classification."""

    STEEP = "steep"  # Spread > 100 bps
    NORMAL = "normal"  # Spread 0-100 bps
    FLAT = "flat"  # Spread -25 to 0 bps
    INVERTED = "inverted"  # Spread < -25 bps


class RiskAppetiteRegime(Enum):
    """FX-based risk appetite classification."""

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


@dataclass
class VixTermStructureSignal(AlternativeSignal):
    """
    VIX term structure signal for volatility regime detection.

    The VIX term structure provides forward-looking information about
    expected volatility. Contango (upward slope) suggests low near-term
    fear, while backwardation suggests elevated near-term risk.
    """

    # VIX levels
    vix_spot: float = 0.0  # Current VIX level
    vix_1m: float = 0.0  # 1-month VIX (VIX9D or front month future)
    vix_3m: float = 0.0  # 3-month VIX (VIX3M index)

    # Term structure metrics
    term_slope: float = 0.0  # (VIX_3M - VIX_spot) / VIX_spot
    term_slope_1m: float = 0.0  # (VIX_1M - VIX_spot) / VIX_spot
    is_contango: bool = True  # True if upward sloping
    is_backwardation: bool = False  # True if downward sloping

    # Regime classification
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    contango_percentile: float = 0.5  # Historical percentile of current slope

    # Trend
    vix_change_5d: float = 0.0  # 5-day change in VIX
    vix_change_20d: float = 0.0  # 20-day change in VIX

    def __post_init__(self):
        """Validate and derive fields."""
        super().__post_init__()

        # Derive contango/backwardation from slope
        self.is_contango = self.term_slope > 0.02  # 2% threshold
        self.is_backwardation = self.term_slope < -0.02

        # Derive volatility regime from VIX level
        if self.vix_spot < 15:
            self.volatility_regime = VolatilityRegime.LOW_VOL
        elif self.vix_spot < 20:
            self.volatility_regime = VolatilityRegime.NORMAL
        elif self.vix_spot < 25:
            self.volatility_regime = VolatilityRegime.ELEVATED
        elif self.vix_spot < 35:
            self.volatility_regime = VolatilityRegime.HIGH_VOL
        else:
            self.volatility_regime = VolatilityRegime.CRISIS

    @property
    def is_fear_elevated(self) -> bool:
        """Check if fear is elevated (VIX > 20 or backwardation)."""
        return self.vix_spot > 20 or self.is_backwardation

    @property
    def is_complacent(self) -> bool:
        """Check if market is complacent (low VIX + steep contango)."""
        return self.vix_spot < 15 and self.term_slope > 0.10


@dataclass
class YieldCurveSignal(AlternativeSignal):
    """
    US Treasury yield curve signal for economic regime detection.

    The yield curve (2s10s spread) is a leading recession indicator.
    Inversion historically precedes recessions by 12-18 months.
    """

    # Yield levels (using ETF prices as proxy)
    short_rate_proxy: float = 0.0  # SHY price (1-3Y Treasury)
    mid_rate_proxy: float = 0.0  # IEF price (7-10Y Treasury)
    long_rate_proxy: float = 0.0  # TLT price (20Y Treasury)

    # Derived spreads
    curve_slope: float = 0.0  # Proxy for 2s10s spread
    curve_slope_change_5d: float = 0.0  # 5-day change
    curve_slope_change_20d: float = 0.0  # 20-day change

    # Regime
    is_inverted: bool = False
    is_steepening: bool = False
    is_flattening: bool = False
    yield_curve_regime: YieldCurveRegime = YieldCurveRegime.NORMAL

    # Economic interpretation
    recession_probability: float = 0.0  # 0-1 based on curve shape
    rate_expectation: str = "steady"  # "easing", "steady", "tightening"

    def __post_init__(self):
        """Validate and derive fields."""
        super().__post_init__()

        # Derive steepening/flattening from change
        self.is_steepening = self.curve_slope_change_5d > 0.01
        self.is_flattening = self.curve_slope_change_5d < -0.01

        # Derive regime from slope
        if self.curve_slope > 0.05:
            self.yield_curve_regime = YieldCurveRegime.STEEP
        elif self.curve_slope > 0:
            self.yield_curve_regime = YieldCurveRegime.NORMAL
        elif self.curve_slope > -0.02:
            self.yield_curve_regime = YieldCurveRegime.FLAT
        else:
            self.yield_curve_regime = YieldCurveRegime.INVERTED
            self.is_inverted = True

        # Simple recession probability based on inversion
        if self.is_inverted:
            self.recession_probability = min(0.8, 0.5 + abs(self.curve_slope) * 10)
        else:
            self.recession_probability = max(0.1, 0.3 - self.curve_slope * 5)

    @property
    def is_recessionary(self) -> bool:
        """Check if curve suggests recession risk."""
        return self.is_inverted or self.recession_probability > 0.5


@dataclass
class FxCorrelationSignal(AlternativeSignal):
    """
    FX correlation signal for risk appetite detection.

    USD strength typically indicates risk-off sentiment, while
    AUD/JPY (carry trade proxy) rising indicates risk-on.
    """

    # USD metrics
    dxy_level: float = 0.0  # USD Index level
    dxy_change_5d: float = 0.0  # 5-day change (%)
    dxy_change_20d: float = 0.0  # 20-day change (%)
    dxy_zscore: float = 0.0  # Z-score vs 60-day rolling

    # Risk sentiment metrics (AUD/JPY)
    audjpy_level: float = 0.0
    audjpy_change_5d: float = 0.0
    audjpy_change_20d: float = 0.0
    audjpy_zscore: float = 0.0

    # Combined risk metrics
    risk_appetite_score: float = 0.0  # -1 (risk-off) to +1 (risk-on)
    risk_appetite_regime: RiskAppetiteRegime = RiskAppetiteRegime.NEUTRAL
    signal_agreement: float = 0.0  # Agreement between USD and AUD/JPY

    def __post_init__(self):
        """Validate and derive fields."""
        super().__post_init__()

        # Derive risk appetite regime
        if self.risk_appetite_score > 0.3:
            self.risk_appetite_regime = RiskAppetiteRegime.RISK_ON
        elif self.risk_appetite_score < -0.3:
            self.risk_appetite_regime = RiskAppetiteRegime.RISK_OFF
        else:
            self.risk_appetite_regime = RiskAppetiteRegime.NEUTRAL

        # Calculate signal agreement
        usd_signal = -1 if self.dxy_change_5d > 0 else 1  # USD up = risk-off
        audjpy_signal = 1 if self.audjpy_change_5d > 0 else -1  # AUD/JPY up = risk-on
        self.signal_agreement = 1.0 if usd_signal == audjpy_signal else 0.0

    @property
    def is_risk_off(self) -> bool:
        """Check if risk-off environment."""
        return self.risk_appetite_regime == RiskAppetiteRegime.RISK_OFF

    @property
    def is_risk_on(self) -> bool:
        """Check if risk-on environment."""
        return self.risk_appetite_regime == RiskAppetiteRegime.RISK_ON


@dataclass
class CrossAssetAggregatedSignal:
    """
    Aggregated signal from all cross-asset sources.

    Combines VIX, yield curve, and FX signals into a unified view.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[CrossAssetSource] = field(default_factory=list)

    # Individual signals
    vix_signal: Optional[VixTermStructureSignal] = None
    yield_curve_signal: Optional[YieldCurveSignal] = None
    fx_signal: Optional[FxCorrelationSignal] = None

    # Aggregated metrics
    composite_signal: float = 0.0  # -1 to +1
    composite_confidence: float = 0.0
    agreement_ratio: float = 0.0

    # Regime summary
    overall_regime: str = "neutral"  # "risk_on", "risk_off", "neutral"
    regime_strength: float = 0.0  # 0-1

    def __post_init__(self):
        """Calculate aggregated metrics."""
        signals = []
        confidences = []

        if self.vix_signal:
            signals.append(self.vix_signal.signal_value)
            confidences.append(self.vix_signal.confidence)
            self.sources.append(CrossAssetSource.VIX_STRUCTURE)

        if self.yield_curve_signal:
            signals.append(self.yield_curve_signal.signal_value)
            confidences.append(self.yield_curve_signal.confidence)
            self.sources.append(CrossAssetSource.YIELD_CURVE)

        if self.fx_signal:
            signals.append(self.fx_signal.signal_value)
            confidences.append(self.fx_signal.confidence)
            self.sources.append(CrossAssetSource.FX_CORRELATION)

        if not signals:
            return

        # Weighted average by confidence
        total_confidence = sum(confidences)
        if total_confidence > 0:
            self.composite_signal = sum(
                s * c for s, c in zip(signals, confidences, strict=True)
            ) / total_confidence
            self.composite_confidence = sum(confidences) / len(confidences)

        # Agreement ratio
        bullish = sum(1 for s in signals if s > 0.1)
        bearish = sum(1 for s in signals if s < -0.1)
        majority = max(bullish, bearish)
        self.agreement_ratio = majority / len(signals) if signals else 0

        # Overall regime
        if self.composite_signal > 0.2 and self.agreement_ratio > 0.5:
            self.overall_regime = "risk_on"
            self.regime_strength = min(1.0, abs(self.composite_signal) * self.agreement_ratio)
        elif self.composite_signal < -0.2 and self.agreement_ratio > 0.5:
            self.overall_regime = "risk_off"
            self.regime_strength = min(1.0, abs(self.composite_signal) * self.agreement_ratio)
        else:
            self.overall_regime = "neutral"
            self.regime_strength = 0.3

    @property
    def direction(self) -> SignalDirection:
        """Overall signal direction."""
        if self.composite_signal > 0.1:
            return SignalDirection.BULLISH
        elif self.composite_signal < -0.1:
            return SignalDirection.BEARISH
        return SignalDirection.NEUTRAL

    @property
    def is_high_conviction(self) -> bool:
        """Whether this is a high-conviction signal."""
        return (
            self.composite_confidence >= 0.6
            and self.agreement_ratio >= 0.7
            and abs(self.composite_signal) >= 0.3
        )

    @property
    def should_reduce_exposure(self) -> bool:
        """Whether signals suggest reducing equity exposure."""
        # Reduce if: high VIX + inverted curve + risk-off FX
        vix_warning = (
            self.vix_signal and self.vix_signal.is_fear_elevated
        ) if self.vix_signal else False
        curve_warning = (
            self.yield_curve_signal and self.yield_curve_signal.is_recessionary
        ) if self.yield_curve_signal else False
        fx_warning = (
            self.fx_signal and self.fx_signal.is_risk_off
        ) if self.fx_signal else False

        return sum([vix_warning, curve_warning, fx_warning]) >= 2
