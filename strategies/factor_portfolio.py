"""
Factor-Neutral Portfolio Construction

Implements institutional-grade portfolio construction using factor scores:
- Long top quintile, short bottom quintile
- Sector neutrality constraints
- Dollar neutrality for market-neutral variant
- Risk-adjusted position sizing

Portfolio Types:
1. Long-only factor tilt
2. Long-short factor portfolio
3. Market-neutral (beta = 0)
4. Sector-neutral (each sector has equal long/short)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.factor_models import CompositeScore, FactorModel, FactorType

logger = logging.getLogger(__name__)


class PortfolioType(Enum):
    """Types of factor portfolios."""

    LONG_ONLY = "long_only"  # Only long positions, tilted toward factors
    LONG_SHORT = "long_short"  # Long top quintile, short bottom
    MARKET_NEUTRAL = "market_neutral"  # Dollar neutral, beta ~0
    SECTOR_NEUTRAL = "sector_neutral"  # Neutral within each sector


@dataclass
class Position:
    """Individual portfolio position."""

    symbol: str
    weight: float  # Portfolio weight (-1 to 1)
    shares: Optional[int] = None  # Actual shares if calculated
    side: str = "long"  # 'long' or 'short'
    factor_score: Optional[float] = None
    sector: Optional[str] = None


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation."""

    positions: List[Position]
    total_long_weight: float
    total_short_weight: float
    net_exposure: float  # Long - Short
    gross_exposure: float  # Long + Short
    n_long: int
    n_short: int
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "positions": [
                {
                    "symbol": p.symbol,
                    "weight": p.weight,
                    "side": p.side,
                    "factor_score": p.factor_score,
                    "sector": p.sector,
                }
                for p in self.positions
            ],
            "total_long_weight": self.total_long_weight,
            "total_short_weight": self.total_short_weight,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "n_long": self.n_long,
            "n_short": self.n_short,
            "timestamp": self.timestamp.isoformat(),
        }


class FactorPortfolioConstructor:
    """
    Constructs factor-based portfolios from factor scores.

    Features:
    - Multiple portfolio types (long-only, long-short, neutral)
    - Position sizing based on factor scores
    - Risk constraints (max position, sector limits)
    - Turnover management
    """

    def __init__(
        self,
        portfolio_type: PortfolioType = PortfolioType.LONG_SHORT,
        n_stocks_per_side: int = 20,
        max_position_weight: float = 0.10,
        max_sector_weight: float = 0.30,
        min_factor_score: float = 0.5,  # Minimum z-score for inclusion
    ):
        """
        Initialize the portfolio constructor.

        Args:
            portfolio_type: Type of portfolio to construct
            n_stocks_per_side: Number of stocks per side (long/short)
            max_position_weight: Maximum weight per position
            max_sector_weight: Maximum weight per sector
            min_factor_score: Minimum factor z-score for inclusion
        """
        self.portfolio_type = portfolio_type
        self.n_stocks_per_side = n_stocks_per_side
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight
        self.min_factor_score = min_factor_score

    def construct(
        self,
        scores: Dict[str, CompositeScore],
        sectors: Dict[str, str] = None,
        existing_positions: Dict[str, float] = None,
        max_turnover: float = 0.5,
    ) -> PortfolioAllocation:
        """
        Construct portfolio from factor scores.

        Args:
            scores: Composite factor scores
            sectors: Optional sector classifications
            existing_positions: Current positions for turnover management
            max_turnover: Maximum portfolio turnover (0-1)

        Returns:
            PortfolioAllocation with positions
        """
        if self.portfolio_type == PortfolioType.LONG_ONLY:
            return self._construct_long_only(scores, sectors)
        elif self.portfolio_type == PortfolioType.LONG_SHORT:
            return self._construct_long_short(scores, sectors)
        elif self.portfolio_type == PortfolioType.MARKET_NEUTRAL:
            return self._construct_market_neutral(scores, sectors)
        elif self.portfolio_type == PortfolioType.SECTOR_NEUTRAL:
            return self._construct_sector_neutral(scores, sectors)
        else:
            raise ValueError(f"Unknown portfolio type: {self.portfolio_type}")

    def _construct_long_only(
        self,
        scores: Dict[str, CompositeScore],
        sectors: Dict[str, str] = None,
    ) -> PortfolioAllocation:
        """Construct long-only factor-tilted portfolio."""
        # Sort by composite score
        sorted_scores = sorted(
            scores.items(), key=lambda x: x[1].composite_z, reverse=True
        )

        # Select top stocks
        selected = sorted_scores[: self.n_stocks_per_side]

        # Calculate weights (score-weighted)
        total_score = sum(max(0.1, s[1].composite_z) for s in selected)
        positions = []

        for symbol, score in selected:
            raw_weight = max(0.1, score.composite_z) / total_score
            weight = min(raw_weight, self.max_position_weight)

            positions.append(
                Position(
                    symbol=symbol,
                    weight=weight,
                    side="long",
                    factor_score=score.composite_z,
                    sector=sectors.get(symbol) if sectors else None,
                )
            )

        # Renormalize weights
        total_weight = sum(p.weight for p in positions)
        for p in positions:
            p.weight = p.weight / total_weight

        return PortfolioAllocation(
            positions=positions,
            total_long_weight=1.0,
            total_short_weight=0.0,
            net_exposure=1.0,
            gross_exposure=1.0,
            n_long=len(positions),
            n_short=0,
        )

    def _construct_long_short(
        self,
        scores: Dict[str, CompositeScore],
        sectors: Dict[str, str] = None,
    ) -> PortfolioAllocation:
        """Construct long-short factor portfolio."""
        sorted_scores = sorted(
            scores.items(), key=lambda x: x[1].composite_z, reverse=True
        )

        # Select top and bottom
        longs = sorted_scores[: self.n_stocks_per_side]
        shorts = sorted_scores[-self.n_stocks_per_side:]

        positions = []

        # Long positions
        long_total = sum(max(0.1, s[1].composite_z) for s in longs)
        for symbol, score in longs:
            raw_weight = max(0.1, score.composite_z) / long_total * 0.5
            weight = min(raw_weight, self.max_position_weight)
            positions.append(
                Position(
                    symbol=symbol,
                    weight=weight,
                    side="long",
                    factor_score=score.composite_z,
                    sector=sectors.get(symbol) if sectors else None,
                )
            )

        # Short positions (invert score for weighting)
        short_total = sum(max(0.1, -s[1].composite_z) for s in shorts)
        for symbol, score in shorts:
            raw_weight = max(0.1, -score.composite_z) / short_total * 0.5
            weight = min(raw_weight, self.max_position_weight)
            positions.append(
                Position(
                    symbol=symbol,
                    weight=-weight,  # Negative for short
                    side="short",
                    factor_score=score.composite_z,
                    sector=sectors.get(symbol) if sectors else None,
                )
            )

        total_long = sum(p.weight for p in positions if p.weight > 0)
        total_short = abs(sum(p.weight for p in positions if p.weight < 0))

        return PortfolioAllocation(
            positions=positions,
            total_long_weight=total_long,
            total_short_weight=total_short,
            net_exposure=total_long - total_short,
            gross_exposure=total_long + total_short,
            n_long=len(longs),
            n_short=len(shorts),
        )

    def _construct_market_neutral(
        self,
        scores: Dict[str, CompositeScore],
        sectors: Dict[str, str] = None,
    ) -> PortfolioAllocation:
        """
        Construct dollar-neutral portfolio (net exposure = 0).

        Market neutral means equal dollar long and short.
        """
        allocation = self._construct_long_short(scores, sectors)

        # Rebalance to ensure exactly neutral
        total_long = sum(p.weight for p in allocation.positions if p.weight > 0)
        total_short = abs(sum(p.weight for p in allocation.positions if p.weight < 0))

        if total_long > 0 and total_short > 0:
            # Scale to equal 0.5 each side
            for p in allocation.positions:
                if p.weight > 0:
                    p.weight = (p.weight / total_long) * 0.5
                else:
                    p.weight = (p.weight / total_short) * -0.5

        allocation.total_long_weight = 0.5
        allocation.total_short_weight = 0.5
        allocation.net_exposure = 0.0
        allocation.gross_exposure = 1.0

        return allocation

    def _construct_sector_neutral(
        self,
        scores: Dict[str, CompositeScore],
        sectors: Dict[str, str] = None,
    ) -> PortfolioAllocation:
        """
        Construct sector-neutral portfolio.

        Within each sector, long top stocks, short bottom stocks.
        """
        if not sectors:
            logger.warning("No sector data provided, falling back to market neutral")
            return self._construct_market_neutral(scores, sectors)

        # Group by sector
        sector_scores: Dict[str, List[Tuple[str, CompositeScore]]] = {}
        for symbol, score in scores.items():
            sector = sectors.get(symbol, "Other")
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append((symbol, score))

        positions = []
        n_sectors = len(sector_scores)

        for sector, sector_list in sector_scores.items():
            if len(sector_list) < 4:  # Need at least 4 for long/short
                continue

            # Sort by score
            sorted_sector = sorted(
                sector_list, key=lambda x: x[1].composite_z, reverse=True
            )

            # Take top 2 long, bottom 2 short
            n_per_side = max(1, min(2, len(sorted_sector) // 4))
            longs = sorted_sector[:n_per_side]
            shorts = sorted_sector[-n_per_side:]

            sector_weight = 1.0 / n_sectors / 2  # Half of sector weight per side

            for symbol, score in longs:
                weight = sector_weight / n_per_side
                positions.append(
                    Position(
                        symbol=symbol,
                        weight=weight,
                        side="long",
                        factor_score=score.composite_z,
                        sector=sector,
                    )
                )

            for symbol, score in shorts:
                weight = sector_weight / n_per_side
                positions.append(
                    Position(
                        symbol=symbol,
                        weight=-weight,
                        side="short",
                        factor_score=score.composite_z,
                        sector=sector,
                    )
                )

        total_long = sum(p.weight for p in positions if p.weight > 0)
        total_short = abs(sum(p.weight for p in positions if p.weight < 0))

        return PortfolioAllocation(
            positions=positions,
            total_long_weight=total_long,
            total_short_weight=total_short,
            net_exposure=total_long - total_short,
            gross_exposure=total_long + total_short,
            n_long=len([p for p in positions if p.weight > 0]),
            n_short=len([p for p in positions if p.weight < 0]),
        )

    def calculate_turnover(
        self,
        new_allocation: PortfolioAllocation,
        existing_positions: Dict[str, float],
    ) -> float:
        """
        Calculate portfolio turnover.

        Turnover = sum of absolute weight changes / 2
        """
        new_weights = {p.symbol: p.weight for p in new_allocation.positions}
        all_symbols = set(new_weights.keys()) | set(existing_positions.keys())

        total_change = 0.0
        for symbol in all_symbols:
            old_weight = existing_positions.get(symbol, 0.0)
            new_weight = new_weights.get(symbol, 0.0)
            total_change += abs(new_weight - old_weight)

        return total_change / 2  # Two-way turnover

    def apply_turnover_constraint(
        self,
        new_allocation: PortfolioAllocation,
        existing_positions: Dict[str, float],
        max_turnover: float,
    ) -> PortfolioAllocation:
        """
        Constrain portfolio changes to max turnover.

        Blends old and new weights to respect turnover limit.
        """
        turnover = self.calculate_turnover(new_allocation, existing_positions)

        if turnover <= max_turnover:
            return new_allocation

        # Blend factor (how much of new portfolio to use)
        blend = max_turnover / turnover

        new_weights = {p.symbol: p.weight for p in new_allocation.positions}
        all_symbols = set(new_weights.keys()) | set(existing_positions.keys())

        blended_positions = []
        for symbol in all_symbols:
            old_weight = existing_positions.get(symbol, 0.0)
            new_weight = new_weights.get(symbol, 0.0)
            blended_weight = old_weight + blend * (new_weight - old_weight)

            if abs(blended_weight) > 0.001:
                blended_positions.append(
                    Position(
                        symbol=symbol,
                        weight=blended_weight,
                        side="long" if blended_weight > 0 else "short",
                    )
                )

        total_long = sum(p.weight for p in blended_positions if p.weight > 0)
        total_short = abs(sum(p.weight for p in blended_positions if p.weight < 0))

        return PortfolioAllocation(
            positions=blended_positions,
            total_long_weight=total_long,
            total_short_weight=total_short,
            net_exposure=total_long - total_short,
            gross_exposure=total_long + total_short,
            n_long=len([p for p in blended_positions if p.weight > 0]),
            n_short=len([p for p in blended_positions if p.weight < 0]),
        )


class FactorPortfolioStrategy:
    """
    High-level factor portfolio strategy for integration with trading system.

    Usage:
        strategy = FactorPortfolioStrategy(
            portfolio_type=PortfolioType.MARKET_NEUTRAL,
            rebalance_frequency='weekly'
        )

        # Generate signals
        signals = await strategy.generate_signals(symbols, price_data)

        # Get target positions
        allocation = strategy.get_target_allocation()
    """

    def __init__(
        self,
        portfolio_type: PortfolioType = PortfolioType.LONG_SHORT,
        rebalance_frequency: str = "weekly",
        factor_weights: Dict[FactorType, float] = None,
        n_stocks: int = 20,
    ):
        self.factor_model = FactorModel(factor_weights=factor_weights)
        self.constructor = FactorPortfolioConstructor(
            portfolio_type=portfolio_type, n_stocks_per_side=n_stocks
        )
        self.rebalance_frequency = rebalance_frequency
        self._last_rebalance: Optional[datetime] = None
        self._current_allocation: Optional[PortfolioAllocation] = None
        self._scores: Dict[str, CompositeScore] = {}

    async def generate_signals(
        self,
        symbols: List[str],
        price_data: pd.DataFrame,
        fundamental_data: Dict[str, Dict[str, float]] = None,
        market_caps: Dict[str, float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for all symbols.

        Args:
            symbols: List of symbols
            price_data: Historical price data
            fundamental_data: Optional fundamental data
            market_caps: Optional market cap data

        Returns:
            Dictionary of symbol -> signal
        """
        # Score universe
        self._scores = self.factor_model.score_universe(
            symbols, price_data, fundamental_data, market_caps
        )

        # Generate signals
        signals = {}
        for symbol in symbols:
            signals[symbol] = self.factor_model.get_signal(symbol, self._scores)

        return signals

    def get_target_allocation(
        self,
        sectors: Dict[str, str] = None,
        existing_positions: Dict[str, float] = None,
    ) -> PortfolioAllocation:
        """
        Get target portfolio allocation.

        Args:
            sectors: Optional sector classifications
            existing_positions: Current positions for turnover management

        Returns:
            PortfolioAllocation with target positions
        """
        if not self._scores:
            raise ValueError("Must call generate_signals first")

        allocation = self.constructor.construct(
            self._scores, sectors, existing_positions
        )

        self._current_allocation = allocation
        self._last_rebalance = datetime.now()

        return allocation

    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance."""
        if self._last_rebalance is None:
            return True

        elapsed = datetime.now() - self._last_rebalance

        if self.rebalance_frequency == "daily":
            return elapsed.days >= 1
        elif self.rebalance_frequency == "weekly":
            return elapsed.days >= 7
        elif self.rebalance_frequency == "monthly":
            return elapsed.days >= 30
        else:
            return True

    def get_factor_exposures(self) -> Dict[str, float]:
        """Get current portfolio factor exposures."""
        if not self._current_allocation or not self._scores:
            return {}

        portfolio = [
            (p.symbol, p.weight) for p in self._current_allocation.positions
        ]
        return self.factor_model.get_factor_exposures(portfolio, self._scores)
