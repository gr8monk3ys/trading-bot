"""
Factor Portfolio - Composite Factor Ranking System

Combines multiple factors into a unified ranking:
- Momentum (15%): 12-1 month momentum (Jegadeesh-Titman)
- Relative Strength (10%): Outperformance vs SPY
- Volatility (10%): Low-vol anomaly
- Value (20%): P/E, P/B, P/S, Dividend Yield
- Quality (15%): ROE, margins, debt levels
- Growth (10%): Earnings and revenue growth
- Earnings Surprise (10%): SUE factor (Phase 3)
- Reversal (10%): 1-month reversal (Phase 3)

Advanced features:
- Factor orthogonalization: Removes correlations between factors
- Risk parity weighting: Each factor contributes equal portfolio risk
- IC-based weight adjustment: Prioritizes high-IC factors

Usage:
    portfolio = FactorPortfolio(broker, include_fundamentals=True)
    rankings = await portfolio.get_composite_rankings(symbols)

    # rankings = {'AAPL': 85.5, 'MSFT': 78.2, ...}
    # Higher score = more attractive

    # With orthogonalization and risk parity
    portfolio = FactorPortfolio(broker, use_orthogonalization=True, use_risk_parity=True)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from factors.base_factor import BaseFactor, FactorScore
from factors.momentum_factor import MomentumFactor, RelativeStrengthFactor
from factors.volatility_factor import BetaFactor, VolatilityFactor

logger = logging.getLogger(__name__)

# Lazy imports for fundamental factors (require yfinance)
ValueFactor = None
QualityFactor = None
GrowthFactor = None
EarningsSurpriseFactor = None
ReversalFactor = None
SectorRelativeMomentumFactor = None
NewsSentimentFactor = None

# Lazy imports for advanced weighting
FactorOrthogonalizer = None
RiskParityWeighter = None
AdaptiveFactorWeighter = None
OrthogonalizationMethod = None


@dataclass
class CompositeScore:
    """Composite factor score for a symbol."""

    symbol: str
    composite_score: float  # 0-100 scale
    factor_scores: Dict[str, FactorScore]
    factor_weights: Dict[str, float]
    rank: int  # 1 = best
    percentile: float  # 0-100
    timestamp: datetime


class FactorPortfolio:
    """
    Combines multiple factors into composite rankings.

    Default weights (balanced multi-factor):
    - Momentum: 15%
    - Relative Strength: 10%
    - Volatility: 10%
    - Value: 20%
    - Quality: 15%
    - Growth: 10%
    - Earnings Surprise: 10% (when available)
    - Reversal: 10% (when available)
    """

    # Weights without fundamentals (price-based only)
    PRICE_ONLY_WEIGHTS = {
        "Momentum_12M_skip1M": 0.40,
        "RelativeStrength_vs_SPY": 0.25,
        "Volatility_252D": 0.20,
        "Beta_vs_SPY": 0.15,
    }

    # Full multi-factor weights (including fundamentals and advanced alpha)
    DEFAULT_WEIGHTS = {
        # Price-based factors (30%)
        "Momentum_12M_skip1M": 0.10,
        "RelativeStrength_vs_SPY": 0.07,
        "Volatility_252D": 0.07,
        "SectorRelativeMomentum": 0.06,
        # Fundamental factors (30%)
        "Value": 0.13,
        "Quality": 0.10,
        "Growth": 0.07,
        # Advanced alpha factors (32%)
        "EarningsSurprise": 0.12,
        "Reversal": 0.12,
        # Alternative data (8%)
        "NewsSentiment": 0.08,
        "SentimentMomentum": 0.08,
    }

    def __init__(
        self,
        broker,
        weights: Optional[Dict[str, float]] = None,
        include_fundamentals: bool = True,
        use_orthogonalization: bool = False,
        use_risk_parity: bool = False,
        ic_tracker=None,
    ):
        """
        Initialize factor portfolio.

        Args:
            broker: Trading broker instance
            weights: Custom factor weights (sum should = 1.0)
            include_fundamentals: Include value/quality/growth factors (requires yfinance)
            use_orthogonalization: Apply PCA orthogonalization to remove factor correlations
            use_risk_parity: Use risk parity weighting instead of fixed weights
            ic_tracker: Optional ICTracker for dynamic weight adjustment
        """
        global ValueFactor, QualityFactor, GrowthFactor
        global FactorOrthogonalizer, RiskParityWeighter, AdaptiveFactorWeighter, OrthogonalizationMethod

        self.broker = broker
        self.include_fundamentals = include_fundamentals
        self.use_orthogonalization = use_orthogonalization
        self.use_risk_parity = use_risk_parity
        self.ic_tracker = ic_tracker

        # Historical factor returns for risk parity (populated during scoring)
        self._factor_returns_history: Dict[str, List[float]] = {}
        self._last_factor_scores: Dict[str, Dict[str, float]] = {}

        # Load orthogonalization module if needed
        if use_orthogonalization or use_risk_parity:
            try:
                from factors.factor_orthogonalization import (
                    AdaptiveFactorWeighter as AFW,
                )
                from factors.factor_orthogonalization import (
                    FactorOrthogonalizer as FO,
                )
                from factors.factor_orthogonalization import (
                    OrthogonalizationMethod as OM,
                )
                from factors.factor_orthogonalization import (
                    RiskParityWeighter as RPW,
                )
                FactorOrthogonalizer = FO
                RiskParityWeighter = RPW
                AdaptiveFactorWeighter = AFW
                OrthogonalizationMethod = OM
                logger.info("Factor orthogonalization and risk parity modules loaded")
            except ImportError as e:
                logger.warning(f"Could not load orthogonalization module: {e}")
                self.use_orthogonalization = False
                self.use_risk_parity = False

        # Initialize price-based factors
        self.factors: Dict[str, BaseFactor] = {
            "Momentum_12M_skip1M": MomentumFactor(broker, lookback_months=12, skip_months=1),
            "RelativeStrength_vs_SPY": RelativeStrengthFactor(broker, benchmark="SPY", lookback_days=63),
            "Volatility_252D": VolatilityFactor(broker, lookback_days=252),
            "Beta_vs_SPY": BetaFactor(broker, benchmark="SPY", lookback_days=252),
        }

        # Add fundamental factors if enabled
        if include_fundamentals:
            try:
                from factors.growth_factor import GrowthFactor as GF
                from factors.quality_factor import QualityFactor as QF
                from factors.value_factor import ValueFactor as VF

                ValueFactor = VF
                QualityFactor = QF
                GrowthFactor = GF

                self.factors["Value"] = ValueFactor(broker)
                self.factors["Quality"] = QualityFactor(broker)
                self.factors["Growth"] = GrowthFactor(broker)

                logger.info("Fundamental factors (Value, Quality, Growth) initialized")
            except ImportError as e:
                logger.warning(f"Could not load fundamental factors: {e}")
                logger.warning("Install yfinance: pip install yfinance")
                include_fundamentals = False

        # Add Phase 3 advanced alpha factors
        try:
            from factors.earnings_factor import EarningsSurpriseFactor as ESF
            from factors.momentum_factor import SectorRelativeMomentumFactor as SRMF
            from factors.reversal_factor import ReversalFactor as RF

            EarningsSurpriseFactor = ESF
            ReversalFactor = RF
            SectorRelativeMomentumFactor = SRMF

            self.factors["EarningsSurprise"] = EarningsSurpriseFactor(broker)
            self.factors["Reversal"] = ReversalFactor(broker)
            self.factors["SectorRelativeMomentum"] = SectorRelativeMomentumFactor(broker)

            logger.info("Advanced alpha factors (EarningsSurprise, Reversal, SectorRelativeMomentum) initialized")
        except ImportError as e:
            logger.warning(f"Could not load advanced alpha factors: {e}")

        # Add news sentiment factor (alternative data)
        try:
            from factors.sentiment_factor import NewsSentimentFactor as NSF
            from factors.sentiment_factor import SentimentMomentumFactor as SMF

            global NewsSentimentFactor
            NewsSentimentFactor = NSF

            self.factors["NewsSentiment"] = NewsSentimentFactor(broker)
            self.factors["SentimentMomentum"] = SMF(broker)

            logger.info("News sentiment factors (NewsSentiment, SentimentMomentum) initialized")
        except ImportError as e:
            logger.warning(f"Could not load news sentiment factors: {e}")

        # Set weights based on available factors
        if weights:
            self.weights = weights
        else:
            if include_fundamentals:
                # Use full multi-factor weights, but only for factors we have
                self.weights = {
                    k: v for k, v in self.DEFAULT_WEIGHTS.items()
                    if k in self.factors
                }
            else:
                # Use price-only weights
                self.weights = self.PRICE_ONLY_WEIGHTS.copy()

            # Normalize weights to sum to 1.0
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(
            f"FactorPortfolio initialized with {len(self.factors)} factors: "
            f"{', '.join(self.factors.keys())}"
        )
        logger.info(f"Factor weights: {', '.join(f'{k}={v:.1%}' for k, v in self.weights.items())}")

    async def get_factor_scores(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, FactorScore]]:
        """
        Calculate all factor scores for given symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict of factor_name -> {symbol -> FactorScore}
        """
        results = {}

        for factor_name, factor in self.factors.items():
            try:
                scores = await factor.calculate_scores_batch(symbols)
                results[factor_name] = scores
                logger.debug(
                    f"{factor_name}: calculated for {len(scores)}/{len(symbols)} symbols"
                )
            except Exception as e:
                logger.error(f"Error calculating {factor_name}: {e}")
                results[factor_name] = {}

        return results

    async def get_composite_rankings(
        self, symbols: List[str]
    ) -> Dict[str, CompositeScore]:
        """
        Get composite factor rankings for symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict of symbol -> CompositeScore
        """
        # Get all factor scores
        all_scores = await self.get_factor_scores(symbols)

        # Store for orthogonalization/risk parity
        self._last_factor_scores = {
            factor_name: {
                symbol: score.normalized_score
                for symbol, score in scores.items()
            }
            for factor_name, scores in all_scores.items()
        }

        # Determine weights to use
        effective_weights = self._get_effective_weights()

        # Apply orthogonalization if enabled
        if self.use_orthogonalization and FactorOrthogonalizer:
            orthogonalized = self._apply_orthogonalization(all_scores)
            if orthogonalized:
                all_scores = orthogonalized

        # Calculate composite scores
        composite_scores = {}

        for symbol in symbols:
            factor_scores = {}
            weighted_sum = 0.0
            total_weight = 0.0

            for factor_name, weight in effective_weights.items():
                if factor_name not in all_scores:
                    continue

                score = all_scores[factor_name].get(symbol)
                if score is None:
                    continue

                factor_scores[factor_name] = score
                weighted_sum += score.normalized_score * weight
                total_weight += weight

            if total_weight > 0:
                composite = weighted_sum / total_weight
            else:
                composite = 50.0  # Neutral if no scores available

            composite_scores[symbol] = {
                "composite": composite,
                "factor_scores": factor_scores,
                "weight_coverage": total_weight,
            }

        # Calculate ranks and percentiles
        sorted_symbols = sorted(
            composite_scores.keys(),
            key=lambda s: composite_scores[s]["composite"],
            reverse=True,
        )

        results = {}
        num_symbols = len(sorted_symbols)

        for rank, symbol in enumerate(sorted_symbols, 1):
            data = composite_scores[symbol]
            percentile = ((num_symbols - rank + 1) / num_symbols) * 100

            results[symbol] = CompositeScore(
                symbol=symbol,
                composite_score=data["composite"],
                factor_scores=data["factor_scores"],
                factor_weights=effective_weights,
                rank=rank,
                percentile=percentile,
                timestamp=datetime.now(),
            )

        logger.info(
            f"Composite rankings calculated for {len(results)} symbols. "
            f"Top 5: {', '.join(sorted_symbols[:5])}"
        )

        return results

    def _get_effective_weights(self) -> Dict[str, float]:
        """
        Get effective factor weights, applying risk parity and IC adjustments.

        Returns:
            Dict of factor_name -> weight
        """
        if not self.use_risk_parity or not RiskParityWeighter:
            # Apply IC adjustments to base weights if tracker available
            if self.ic_tracker:
                ic_adjustments = self.ic_tracker.get_weight_adjustments()
                adjusted = {}
                for factor, weight in self.weights.items():
                    mult = ic_adjustments.get(factor, 1.0)
                    adjusted[factor] = weight * mult

                # Renormalize
                total = sum(adjusted.values())
                if total > 0:
                    adjusted = {k: v / total for k, v in adjusted.items()}
                    logger.debug(f"IC-adjusted weights: {adjusted}")
                    return adjusted

            return self.weights

        # Risk parity weighting
        if not self._factor_returns_history:
            logger.debug("No factor return history for risk parity, using base weights")
            return self.weights

        try:
            rp_weighter = RiskParityWeighter(self._factor_returns_history)
            rp_result = rp_weighter.calculate_weights()

            if rp_result and rp_result.optimization_converged:
                weights = rp_result.factor_weights

                # Blend with IC weights if available
                if self.ic_tracker:
                    ic_adjustments = self.ic_tracker.get_weight_adjustments()
                    blend_factor = 0.3  # 30% IC, 70% risk parity

                    blended = {}
                    for factor in weights.keys():
                        rp_w = weights.get(factor, 0)
                        ic_mult = ic_adjustments.get(factor, 1.0)
                        blended[factor] = rp_w * (1 - blend_factor + blend_factor * ic_mult)

                    # Renormalize
                    total = sum(blended.values())
                    if total > 0:
                        blended = {k: v / total for k, v in blended.items()}
                    weights = blended

                logger.info(
                    f"Risk parity weights: {', '.join(f'{k}={v:.1%}' for k, v in weights.items())}"
                )
                return weights

        except Exception as e:
            logger.warning(f"Risk parity calculation failed: {e}")

        return self.weights

    def _apply_orthogonalization(
        self, all_scores: Dict[str, Dict[str, FactorScore]]
    ) -> Optional[Dict[str, Dict[str, FactorScore]]]:
        """
        Apply factor orthogonalization.

        Args:
            all_scores: Original factor scores

        Returns:
            Orthogonalized scores or None if failed
        """
        if not FactorOrthogonalizer:
            return None

        try:
            # Convert to normalized score dict
            score_dict = {
                factor_name: {
                    symbol: score.normalized_score
                    for symbol, score in scores.items()
                }
                for factor_name, scores in all_scores.items()
            }

            orthogonalizer = FactorOrthogonalizer(
                score_dict,
                method=OrthogonalizationMethod.PCA,
            )
            ortho_result = orthogonalizer.orthogonalize()

            if not ortho_result:
                return None

            logger.info(
                f"Factor orthogonalization applied. "
                f"Correlation reduction: {ortho_result.correlation_reduction:.1%}"
            )

            # Convert back to FactorScore format
            ortho_scores = {}
            for factor_name, scores in all_scores.items():
                ortho_name = f"{factor_name}_ortho"
                if ortho_name in ortho_result.orthogonal_scores:
                    ortho_scores[factor_name] = {}
                    for symbol, score in scores.items():
                        ortho_value = ortho_result.orthogonal_scores[ortho_name].get(symbol)
                        if ortho_value is not None:
                            # Create new FactorScore with orthogonalized value
                            ortho_scores[factor_name][symbol] = FactorScore(
                                symbol=score.symbol,
                                factor_name=score.factor_name,
                                raw_score=score.raw_score,
                                normalized_score=float(ortho_value * 50 + 50),  # Scale to 0-100
                                percentile=score.percentile,
                                timestamp=score.timestamp,
                                metadata={**score.metadata, "orthogonalized": True},
                            )

            return ortho_scores if ortho_scores else None

        except Exception as e:
            logger.warning(f"Orthogonalization failed: {e}")
            return None

    def update_factor_returns(self, date: datetime, factor_returns: Dict[str, float]):
        """
        Record factor returns for risk parity calculation.

        Args:
            date: Date of returns
            factor_returns: Dict of factor_name -> return for period
        """
        for factor_name, ret in factor_returns.items():
            if factor_name not in self._factor_returns_history:
                self._factor_returns_history[factor_name] = []
            self._factor_returns_history[factor_name].append(ret)

            # Keep last 252 days (1 year)
            if len(self._factor_returns_history[factor_name]) > 252:
                self._factor_returns_history[factor_name] = \
                    self._factor_returns_history[factor_name][-252:]

        logger.debug(f"Updated factor returns for {len(factor_returns)} factors")

    async def get_top_stocks(
        self,
        symbols: List[str],
        top_n: int = 20,
        min_score: float = 50.0,
    ) -> List[str]:
        """
        Get top-ranked stocks by composite score.

        Args:
            symbols: Universe of symbols to rank
            top_n: Number of top stocks to return
            min_score: Minimum composite score required

        Returns:
            List of top stock symbols
        """
        rankings = await self.get_composite_rankings(symbols)

        # Filter by minimum score and sort by composite
        qualified = [
            (symbol, score)
            for symbol, score in rankings.items()
            if score.composite_score >= min_score
        ]

        # Sort by score descending
        qualified.sort(key=lambda x: x[1].composite_score, reverse=True)

        top_stocks = [symbol for symbol, _ in qualified[:top_n]]

        logger.info(f"Top {len(top_stocks)} stocks selected (min_score={min_score}):")
        for i, symbol in enumerate(top_stocks[:10], 1):
            score = rankings[symbol]
            logger.info(
                f"  {i}. {symbol}: {score.composite_score:.1f} "
                f"(rank {score.rank}, {score.percentile:.0f}th percentile)"
            )

        return top_stocks

    async def get_factor_report(
        self, symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Get comprehensive factor analysis report.

        Args:
            symbols: List of symbols to analyze

        Returns:
            Dict with factor statistics and rankings
        """
        rankings = await self.get_composite_rankings(symbols)

        if not rankings:
            return {"error": "No rankings calculated"}

        # Aggregate statistics
        composite_scores = [r.composite_score for r in rankings.values()]

        # Top and bottom performers
        sorted_rankings = sorted(
            rankings.items(),
            key=lambda x: x[1].composite_score,
            reverse=True,
        )

        top_5 = [(s, r.composite_score) for s, r in sorted_rankings[:5]]
        bottom_5 = [(s, r.composite_score) for s, r in sorted_rankings[-5:]]

        # Factor breakdown for top stocks
        top_stock_factors = {}
        for symbol, _ in top_5:
            top_stock_factors[symbol] = {
                factor_name: score.normalized_score
                for factor_name, score in rankings[symbol].factor_scores.items()
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": len(rankings),
            "composite_stats": {
                "mean": float(np.mean(composite_scores)),
                "std": float(np.std(composite_scores)),
                "min": float(np.min(composite_scores)),
                "max": float(np.max(composite_scores)),
                "median": float(np.median(composite_scores)),
            },
            "factor_weights": self.weights,
            "top_5": top_5,
            "bottom_5": bottom_5,
            "top_stock_breakdown": top_stock_factors,
        }

    def add_factor(self, name: str, factor: BaseFactor, weight: float):
        """
        Add a custom factor to the portfolio.

        Args:
            name: Factor name
            factor: BaseFactor instance
            weight: Factor weight (will be normalized)
        """
        self.factors[name] = factor
        self.weights[name] = weight

        # Renormalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"Added factor '{name}' with weight {weight/total:.1%}")

    def set_weights(self, weights: Dict[str, float]):
        """
        Set custom factor weights.

        Args:
            weights: Dict of factor_name -> weight
        """
        # Normalize
        total = sum(weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}
        else:
            logger.warning("Cannot set zero total weight")

    def clear_cache(self):
        """Clear all factor caches."""
        for factor in self.factors.values():
            factor.clear_cache()
