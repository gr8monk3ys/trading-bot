"""
Factor Model Framework

Implements quantitative factor models for stock selection:
- Momentum Factor: 12-1 month price momentum (Jegadeesh-Titman)
- Value Factor: P/E, P/B ratios
- Quality Factor: ROE, earnings stability
- Volatility Factor: Low-volatility anomaly

Usage:
    from factors import FactorPortfolio

    portfolio = FactorPortfolio(broker)
    rankings = await portfolio.get_composite_rankings(symbols)

    # rankings is a dict of symbol -> composite score (0-100)
"""

from factors.base_factor import BaseFactor
from factors.factor_portfolio import FactorPortfolio
from factors.momentum_factor import MomentumFactor
from factors.volatility_factor import VolatilityFactor

__all__ = [
    "BaseFactor",
    "MomentumFactor",
    "VolatilityFactor",
    "FactorPortfolio",
]
