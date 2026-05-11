"""
Quick Indicator Analysis Functions

High-level aggregator helpers that wrap :class:`utils.indicators.TechnicalIndicators`
to deliver compact one-shot summaries:

- :func:`analyze_trend` — direction + strength via ADX/DI and SMA50/SMA200
- :func:`analyze_momentum` — overbought/oversold via RSI + Stochastic
- :func:`analyze_volatility` — Bollinger-band squeeze/expansion and ATR readout

These were split out of ``utils/indicators.py`` so that the indicator *class*
stays focused on raw calculations while the *analysis* convenience layer lives
alongside it.

Usage:
    from utils.indicator_analysis import analyze_trend, analyze_momentum, analyze_volatility

    trend = analyze_trend(close, high, low)
    momentum = analyze_momentum(close, high, low)
    volatility = analyze_volatility(close, high, low)
"""

from typing import Any, Dict

import numpy as np

# ==================== QUICK ANALYSIS FUNCTIONS ====================


def analyze_trend(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
    """
    Quick trend analysis.

    Returns trend strength, direction, and key levels.
    """
    # Local import to avoid a circular dependency with utils.indicators (which
    # re-exports the analyze_* helpers from this module for backwards compat).
    from utils.indicators import TechnicalIndicators

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
        direction = "bullish"
    elif current_minus_di > current_plus_di:
        direction = "bearish"
    else:
        direction = "neutral"

    # Trend strength
    if current_adx > 50:
        strength = "very_strong"
    elif current_adx > 25:
        strength = "strong"
    elif current_adx > 15:
        strength = "weak"
    else:
        strength = "no_trend"

    return {
        "direction": direction,
        "strength": strength,
        "adx": current_adx,
        "plus_di": current_plus_di,
        "minus_di": current_minus_di,
        "sma_50": sma_50[-1] if not np.isnan(sma_50[-1]) else None,
        "sma_200": sma_200[-1] if not np.isnan(sma_200[-1]) else None,
        "price_vs_sma_50": "above" if current_price > sma_50[-1] else "below",
        "price_vs_sma_200": "above" if current_price > sma_200[-1] else "below",
    }


def analyze_momentum(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
    """
    Quick momentum analysis.

    Returns overbought/oversold conditions and momentum strength.
    """
    from utils.indicators import TechnicalIndicators

    ind = TechnicalIndicators(high=high, low=low, close=close)

    rsi = ind.rsi(period=14)
    slowk, slowd = ind.stochastic()

    current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
    current_stoch_k = slowk[-1] if not np.isnan(slowk[-1]) else 50

    # Conditions
    if current_rsi > 70 and current_stoch_k > 80:
        condition = "overbought"
    elif current_rsi < 30 and current_stoch_k < 20:
        condition = "oversold"
    else:
        condition = "neutral"

    return {
        "condition": condition,
        "rsi": current_rsi,
        "stochastic_k": current_stoch_k,
        "stochastic_d": slowd[-1] if not np.isnan(slowd[-1]) else 50,
    }


def analyze_volatility(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, Any]:
    """
    Quick volatility analysis.

    Returns volatility level and key bands.
    """
    from utils.indicators import TechnicalIndicators

    ind = TechnicalIndicators(high=high, low=low, close=close)

    atr = ind.atr(period=14)
    upper, middle, lower = ind.bollinger_bands(period=20, std=2.0)

    current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
    current_price = close[-1]

    # BB squeeze detection
    bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] > 0 else 0

    if bb_width < 0.04:  # Less than 4% width
        volatility_state = "squeeze"  # Breakout coming
    elif bb_width > 0.12:  # More than 12% width
        volatility_state = "expansion"  # High volatility
    else:
        volatility_state = "normal"

    return {
        "state": volatility_state,
        "atr": current_atr,
        "bb_upper": upper[-1],
        "bb_middle": middle[-1],
        "bb_lower": lower[-1],
        "bb_width_pct": bb_width * 100,
        "price_position": (
            (current_price - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
        ),
    }


__all__ = ["analyze_trend", "analyze_momentum", "analyze_volatility"]
