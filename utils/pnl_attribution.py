"""
P&L Attribution System

Decomposes portfolio returns into explainable components:
1. **Alpha**: Pure strategy skill (unexplained by factors)
2. **Beta**: Market exposure (correlation with SPY)
3. **Sector**: Industry tilts (overweight tech, underweight utilities)
4. **Factor**: Factor exposures (momentum, value, size, etc.)
5. **Costs**: Transaction costs and slippage

Why this matters:
- A +20% return from 100% tech exposure during a tech rally is NOT alpha
- True alpha is return AFTER removing market, sector, and factor exposures
- Without attribution, you can't tell if strategy is skilled or lucky

Brinson Attribution (classic):
- Allocation Effect: Being in right sectors
- Selection Effect: Picking right stocks within sectors
- Interaction Effect: Combination of both

Factor Attribution (modern):
- Fama-French factors: Market, Size, Value, Profitability, Investment
- Momentum, Quality, Low-Vol anomalies
- Residual = unexplained alpha

Usage:
    attributor = PnLAttributor(portfolio_returns, benchmark_returns)

    # Daily attribution
    daily = await attributor.attribute_daily(date, positions, prices)

    # Period summary
    report = await attributor.get_attribution_report(start_date, end_date)

    print(f"Total Return: {report['total_return']:.2%}")
    print(f"  Market Beta: {report['beta_contribution']:.2%}")
    print(f"  Sector Tilts: {report['sector_contribution']:.2%}")
    print(f"  Factor Exposure: {report['factor_contribution']:.2%}")
    print(f"  Pure Alpha: {report['alpha']:.2%}")
    print(f"  Costs: {report['cost_drag']:.2%}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class AttributionComponent(Enum):
    """Components of return attribution."""
    TOTAL = "total"
    ALPHA = "alpha"  # Unexplained by any factor
    BETA = "beta"  # Market exposure
    SECTOR = "sector"  # Sector allocation
    FACTOR_MOMENTUM = "factor_momentum"
    FACTOR_VALUE = "factor_value"
    FACTOR_SIZE = "factor_size"
    FACTOR_QUALITY = "factor_quality"
    FACTOR_VOLATILITY = "factor_volatility"
    COSTS = "costs"
    RESIDUAL = "residual"  # Unexplained variance


# Sector classification (simplified GICS)
SECTOR_MAPPING = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "NVDA": "Technology", "META": "Technology", "AMD": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials",
    "VZ": "Communication Services", "T": "Communication Services", "CMCSA": "Communication Services",
}


@dataclass
class DailyAttribution:
    """Attribution for a single day."""

    date: datetime
    total_return: float
    gross_return: float  # Before costs

    # Component contributions (as returns)
    beta_contribution: float  # Market exposure
    sector_contribution: float  # Sector tilts
    factor_contributions: Dict[str, float]  # By factor
    alpha: float  # Unexplained
    cost_drag: float  # Transaction costs

    # Exposures (for analysis)
    market_beta: float
    sector_weights: Dict[str, float]
    factor_exposures: Dict[str, float]

    # R-squared (how much is explained)
    r_squared: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "total_return": self.total_return,
            "gross_return": self.gross_return,
            "beta_contribution": self.beta_contribution,
            "sector_contribution": self.sector_contribution,
            "factor_contributions": self.factor_contributions,
            "alpha": self.alpha,
            "cost_drag": self.cost_drag,
            "market_beta": self.market_beta,
            "r_squared": self.r_squared,
        }


@dataclass
class AttributionReport:
    """Attribution report for a period."""

    start_date: datetime
    end_date: datetime
    trading_days: int

    # Cumulative returns
    total_return: float
    gross_return: float
    benchmark_return: float

    # Component contributions (cumulative)
    beta_contribution: float
    sector_contribution: float
    factor_contributions: Dict[str, float]
    alpha: float
    cost_drag: float

    # Averages/statistics
    avg_beta: float
    avg_r_squared: float
    information_ratio: float  # Alpha / tracking error
    tracking_error: float

    # Sector breakdown
    sector_returns: Dict[str, float]
    sector_weights: Dict[str, float]

    # Daily attributions
    daily_attributions: List[DailyAttribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": f"{self.start_date.date()} to {self.end_date.date()}",
            "trading_days": self.trading_days,
            "total_return": self.total_return,
            "gross_return": self.gross_return,
            "benchmark_return": self.benchmark_return,
            "attribution": {
                "beta": self.beta_contribution,
                "sector": self.sector_contribution,
                "factors": self.factor_contributions,
                "alpha": self.alpha,
                "costs": self.cost_drag,
            },
            "statistics": {
                "avg_beta": self.avg_beta,
                "avg_r_squared": self.avg_r_squared,
                "information_ratio": self.information_ratio,
                "tracking_error": self.tracking_error,
            },
            "sector_breakdown": self.sector_returns,
        }


class PnLAttributor:
    """
    Decomposes portfolio P&L into attributable components.

    Uses regression-based factor attribution:
    R_p = α + β_m * R_m + β_s * R_sectors + β_f * R_factors + ε

    Where:
    - R_p = Portfolio return
    - α = Alpha (unexplained skill)
    - β_m * R_m = Market beta contribution
    - β_s * R_sectors = Sector allocation contribution
    - β_f * R_factors = Factor exposure contribution
    - ε = Residual (noise/luck)
    """

    # Minimum data for regression
    MIN_HISTORY_DAYS = 20

    def __init__(
        self,
        broker,
        benchmark_symbol: str = "SPY",
        factor_etfs: Optional[Dict[str, str]] = None,
        sector_etfs: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize P&L attributor.

        Args:
            broker: Broker instance for data fetching
            benchmark_symbol: Market benchmark (default SPY)
            factor_etfs: Map of factor name -> ETF symbol for factor returns
            sector_etfs: Map of sector name -> ETF symbol
        """
        self.broker = broker
        self.benchmark_symbol = benchmark_symbol

        # Default factor ETFs (if available)
        self.factor_etfs = factor_etfs or {
            "momentum": "MTUM",
            "value": "VLUE",
            "size": "SIZE",
            "quality": "QUAL",
            "low_vol": "USMV",
        }

        # Default sector ETFs
        self.sector_etfs = sector_etfs or {
            "Technology": "XLK",
            "Financials": "XLF",
            "Healthcare": "XLV",
            "Energy": "XLE",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC",
        }

        # History tracking
        self._portfolio_returns: List[Tuple[datetime, float]] = []
        self._benchmark_returns: List[Tuple[datetime, float]] = []
        self._factor_returns: Dict[str, List[Tuple[datetime, float]]] = {}
        self._sector_returns: Dict[str, List[Tuple[datetime, float]]] = {}
        self._positions_history: List[Tuple[datetime, Dict[str, float]]] = []
        self._costs_history: List[Tuple[datetime, float]] = []

        self._attribution_cache: List[DailyAttribution] = []

    def record_daily(
        self,
        date: datetime,
        portfolio_return: float,
        positions: Dict[str, float],
        benchmark_return: Optional[float] = None,
        costs: float = 0.0,
    ):
        """
        Record daily portfolio data for attribution.

        Args:
            date: Trading date
            portfolio_return: Portfolio return for the day
            positions: Dict of symbol -> position value
            benchmark_return: Benchmark return (fetched if not provided)
            costs: Transaction costs incurred
        """
        self._portfolio_returns.append((date, portfolio_return))
        self._positions_history.append((date, positions.copy()))
        self._costs_history.append((date, costs))

        if benchmark_return is not None:
            self._benchmark_returns.append((date, benchmark_return))

        logger.debug(
            f"Recorded daily P&L: {date.date()}, return={portfolio_return:.4f}, "
            f"positions={len(positions)}, costs=${costs:.2f}"
        )

    async def attribute_daily(
        self,
        date: datetime,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_return: float,
        costs: float = 0.0,
    ) -> DailyAttribution:
        """
        Attribute a single day's return.

        Args:
            date: Trading date
            positions: Dict of symbol -> position value
            prices: Dict of symbol -> current price
            portfolio_return: Portfolio return for the day
            costs: Transaction costs

        Returns:
            DailyAttribution with component breakdown
        """
        # Fetch benchmark return
        benchmark_return = await self._get_benchmark_return(date)

        # Calculate position weights
        total_value = sum(positions.values())
        weights = {s: v / total_value for s, v in positions.items()} if total_value > 0 else {}

        # Calculate sector weights
        sector_weights = self._calculate_sector_weights(weights)

        # Calculate factor exposures (simplified: use position characteristics)
        factor_exposures = await self._estimate_factor_exposures(weights, prices)

        # Get factor returns for the day
        factor_returns = await self._get_factor_returns(date)

        # Calculate contributions
        # Beta contribution = portfolio_beta * market_return
        # Use rolling beta from recent history
        portfolio_beta = self._calculate_rolling_beta()
        beta_contribution = portfolio_beta * benchmark_return

        # Sector contribution = sum(sector_weight * sector_return) - benchmark_return
        sector_contribution = await self._calculate_sector_contribution(
            date, sector_weights, benchmark_return
        )

        # Factor contributions
        factor_contributions = {}
        total_factor_contrib = 0.0
        for factor, exposure in factor_exposures.items():
            if factor in factor_returns:
                contrib = exposure * factor_returns[factor]
                factor_contributions[factor] = contrib
                total_factor_contrib += contrib

        # Gross return before costs
        gross_return = portfolio_return + (costs / total_value if total_value > 0 else 0)

        # Alpha = residual after removing all explained components
        explained = beta_contribution + sector_contribution + total_factor_contrib
        alpha = gross_return - explained

        # Cost drag as return
        cost_drag = costs / total_value if total_value > 0 else 0

        # R-squared (how much variance is explained)
        r_squared = self._calculate_r_squared(
            portfolio_return, beta_contribution, sector_contribution, total_factor_contrib
        )

        attribution = DailyAttribution(
            date=date,
            total_return=portfolio_return,
            gross_return=gross_return,
            beta_contribution=beta_contribution,
            sector_contribution=sector_contribution,
            factor_contributions=factor_contributions,
            alpha=alpha,
            cost_drag=cost_drag,
            market_beta=portfolio_beta,
            sector_weights=sector_weights,
            factor_exposures=factor_exposures,
            r_squared=r_squared,
        )

        self._attribution_cache.append(attribution)

        logger.debug(
            f"Daily attribution {date.date()}: "
            f"total={portfolio_return:.4f}, beta={beta_contribution:.4f}, "
            f"sector={sector_contribution:.4f}, alpha={alpha:.4f}"
        )

        return attribution

    async def get_attribution_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AttributionReport:
        """
        Generate attribution report for a period.

        Args:
            start_date: Start of period (default: earliest)
            end_date: End of period (default: latest)

        Returns:
            AttributionReport with detailed breakdown
        """
        # Filter attributions to date range
        attributions = self._attribution_cache

        if start_date:
            attributions = [a for a in attributions if a.date >= start_date]
        if end_date:
            attributions = [a for a in attributions if a.date <= end_date]

        if not attributions:
            logger.warning("No attributions available for report")
            return self._empty_report(start_date, end_date)

        # Aggregate returns (compound)
        total_return = np.prod([1 + a.total_return for a in attributions]) - 1
        gross_return = np.prod([1 + a.gross_return for a in attributions]) - 1

        # Aggregate benchmark return
        benchmark_returns = [
            r for d, r in self._benchmark_returns
            if (not start_date or d >= start_date) and (not end_date or d <= end_date)
        ]
        benchmark_return = np.prod([1 + r for r in benchmark_returns]) - 1 if benchmark_returns else 0

        # Aggregate component contributions (sum of daily contributions)
        beta_contribution = sum(a.beta_contribution for a in attributions)
        sector_contribution = sum(a.sector_contribution for a in attributions)
        alpha = sum(a.alpha for a in attributions)
        cost_drag = sum(a.cost_drag for a in attributions)

        # Aggregate factor contributions
        factor_contributions = {}
        all_factors = set()
        for a in attributions:
            all_factors.update(a.factor_contributions.keys())
        for factor in all_factors:
            factor_contributions[factor] = sum(
                a.factor_contributions.get(factor, 0) for a in attributions
            )

        # Average statistics
        avg_beta = float(np.mean([a.market_beta for a in attributions]))
        avg_r_squared = float(np.mean([a.r_squared for a in attributions]))

        # Tracking error and information ratio
        active_returns = [a.total_return - a.beta_contribution / avg_beta for a in attributions]
        tracking_error = float(np.std(active_returns) * np.sqrt(252)) if len(active_returns) > 1 else 0
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0

        # Sector breakdown
        sector_returns = self._aggregate_sector_returns(attributions)
        sector_weights = self._aggregate_sector_weights(attributions)

        report = AttributionReport(
            start_date=start_date or attributions[0].date,
            end_date=end_date or attributions[-1].date,
            trading_days=len(attributions),
            total_return=total_return,
            gross_return=gross_return,
            benchmark_return=benchmark_return,
            beta_contribution=beta_contribution,
            sector_contribution=sector_contribution,
            factor_contributions=factor_contributions,
            alpha=alpha,
            cost_drag=cost_drag,
            avg_beta=avg_beta,
            avg_r_squared=avg_r_squared,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            sector_returns=sector_returns,
            sector_weights=sector_weights,
            daily_attributions=attributions,
        )

        logger.info(
            f"Attribution report: {report.trading_days} days, "
            f"total={total_return:.2%}, alpha={alpha:.2%}, "
            f"beta={beta_contribution:.2%}, IR={information_ratio:.2f}"
        )

        return report

    def _calculate_sector_weights(self, position_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector weights from position weights."""
        sector_weights = {}

        for symbol, weight in position_weights.items():
            sector = SECTOR_MAPPING.get(symbol, "Other")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        return sector_weights

    async def _get_benchmark_return(self, date: datetime) -> float:
        """Get benchmark return for a date."""
        # Check cache
        for d, r in self._benchmark_returns:
            if d.date() == date.date():
                return r

        # Fetch from broker
        try:
            end_date = date
            start_date = date - timedelta(days=5)  # Buffer for weekends

            bars = await self.broker.get_bars(
                self.benchmark_symbol,
                timeframe="1Day",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if bars and len(bars) >= 2:
                closes = [float(b.close) for b in bars]
                return (closes[-1] / closes[-2]) - 1

        except Exception as e:
            logger.debug(f"Error fetching benchmark return: {e}")

        return 0.0

    async def _get_factor_returns(self, date: datetime) -> Dict[str, float]:
        """Get factor returns for a date."""
        factor_returns = {}

        for factor, etf in self.factor_etfs.items():
            try:
                end_date = date
                start_date = date - timedelta(days=5)

                bars = await self.broker.get_bars(
                    etf,
                    timeframe="1Day",
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )

                if bars and len(bars) >= 2:
                    closes = [float(b.close) for b in bars]
                    factor_returns[factor] = (closes[-1] / closes[-2]) - 1

            except Exception as e:
                logger.debug(f"Error fetching {factor} return: {e}")

        return factor_returns

    async def _estimate_factor_exposures(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Estimate portfolio factor exposures.

        Simplified: uses sector as proxy for some factors.
        Full implementation would use factor scores from FactorPortfolio.
        """
        exposures = {
            "momentum": 0.0,
            "value": 0.0,
            "size": 0.0,
            "quality": 0.0,
            "low_vol": 0.0,
        }

        # Simplified: use sector tilts as factor proxies
        sector_weights = self._calculate_sector_weights(weights)

        # Technology overweight -> momentum exposure
        tech_weight = sector_weights.get("Technology", 0)
        exposures["momentum"] = (tech_weight - 0.25) * 2  # Normalized

        # Financials/Energy -> value exposure
        value_sectors = sector_weights.get("Financials", 0) + sector_weights.get("Energy", 0)
        exposures["value"] = (value_sectors - 0.15) * 2

        # Utilities/Consumer Staples -> low volatility
        defensive = sector_weights.get("Utilities", 0) + sector_weights.get("Consumer Staples", 0)
        exposures["low_vol"] = (defensive - 0.10) * 2

        return exposures

    async def _calculate_sector_contribution(
        self,
        date: datetime,
        sector_weights: Dict[str, float],
        benchmark_return: float,
    ) -> float:
        """Calculate sector allocation contribution."""
        contribution = 0.0

        for sector, weight in sector_weights.items():
            if sector in self.sector_etfs:
                try:
                    etf = self.sector_etfs[sector]
                    end_date = date
                    start_date = date - timedelta(days=5)

                    bars = await self.broker.get_bars(
                        etf,
                        timeframe="1Day",
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                    )

                    if bars and len(bars) >= 2:
                        closes = [float(b.close) for b in bars]
                        sector_return = (closes[-1] / closes[-2]) - 1

                        # Brinson allocation effect
                        # (portfolio_weight - benchmark_weight) * (sector_return - benchmark_return)
                        benchmark_weight = 1.0 / len(self.sector_etfs)  # Simplified: equal weight
                        allocation_effect = (weight - benchmark_weight) * (sector_return - benchmark_return)
                        contribution += allocation_effect

                except Exception as e:
                    logger.debug(f"Error calculating sector contribution for {sector}: {e}")

        return contribution

    def _calculate_rolling_beta(self, lookback_days: int = 60) -> float:
        """Calculate rolling portfolio beta."""
        if len(self._portfolio_returns) < self.MIN_HISTORY_DAYS:
            return 1.0  # Assume market beta if insufficient data

        # Get recent returns
        port_returns = [r for _, r in self._portfolio_returns[-lookback_days:]]
        bench_returns = [r for _, r in self._benchmark_returns[-lookback_days:]]

        if len(port_returns) != len(bench_returns) or len(port_returns) < 10:
            return 1.0

        # Calculate beta
        try:
            slope, _, _, _, _ = stats.linregress(bench_returns, port_returns)
            return float(np.clip(slope, 0.0, 3.0))
        except Exception:
            return 1.0

    def _calculate_r_squared(
        self,
        actual_return: float,
        beta_contrib: float,
        sector_contrib: float,
        factor_contrib: float,
    ) -> float:
        """Calculate R-squared for single observation (simplified)."""
        predicted = beta_contrib + sector_contrib + factor_contrib

        if actual_return == 0:
            return 1.0 if predicted == 0 else 0.0

        # Simple R-squared proxy
        explained_variance = 1 - abs(actual_return - predicted) / max(abs(actual_return), 0.0001)
        return float(np.clip(explained_variance, 0, 1))

    def _aggregate_sector_returns(
        self, attributions: List[DailyAttribution]
    ) -> Dict[str, float]:
        """Aggregate sector returns from attributions."""
        sector_contribs = {}

        for a in attributions:
            for sector, weight in a.sector_weights.items():
                if sector not in sector_contribs:
                    sector_contribs[sector] = []
                sector_contribs[sector].append(weight * a.total_return)

        return {sector: sum(contribs) for sector, contribs in sector_contribs.items()}

    def _aggregate_sector_weights(
        self, attributions: List[DailyAttribution]
    ) -> Dict[str, float]:
        """Average sector weights over period."""
        if not attributions:
            return {}

        sector_sums = {}
        for a in attributions:
            for sector, weight in a.sector_weights.items():
                sector_sums[sector] = sector_sums.get(sector, 0) + weight

        n = len(attributions)
        return {sector: total / n for sector, total in sector_sums.items()}

    def _empty_report(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> AttributionReport:
        """Create empty report when no data available."""
        return AttributionReport(
            start_date=start_date or datetime.now(),
            end_date=end_date or datetime.now(),
            trading_days=0,
            total_return=0.0,
            gross_return=0.0,
            benchmark_return=0.0,
            beta_contribution=0.0,
            sector_contribution=0.0,
            factor_contributions={},
            alpha=0.0,
            cost_drag=0.0,
            avg_beta=1.0,
            avg_r_squared=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
            sector_returns={},
            sector_weights={},
        )

    def clear_history(self):
        """Clear all historical data."""
        self._portfolio_returns.clear()
        self._benchmark_returns.clear()
        self._factor_returns.clear()
        self._sector_returns.clear()
        self._positions_history.clear()
        self._costs_history.clear()
        self._attribution_cache.clear()


def print_attribution_report(report: AttributionReport):
    """Print formatted attribution report."""
    print("\n" + "=" * 65)
    print("P&L ATTRIBUTION REPORT")
    print("=" * 65)

    print(f"\nPeriod: {report.start_date.date()} to {report.end_date.date()}")
    print(f"Trading Days: {report.trading_days}")

    print("\n--- RETURNS ---")
    print(f"Total Return:     {report.total_return:+.2%}")
    print(f"Gross Return:     {report.gross_return:+.2%}")
    print(f"Benchmark Return: {report.benchmark_return:+.2%}")
    print(f"Active Return:    {report.total_return - report.benchmark_return:+.2%}")

    print("\n--- ATTRIBUTION BREAKDOWN ---")
    print(f"Market Beta:      {report.beta_contribution:+.2%}")
    print(f"Sector Tilts:     {report.sector_contribution:+.2%}")

    if report.factor_contributions:
        print("Factor Exposures:")
        for factor, contrib in report.factor_contributions.items():
            print(f"  {factor.capitalize()}: {contrib:+.2%}")

    print(f"Pure Alpha:       {report.alpha:+.2%}")
    print(f"Cost Drag:        {report.cost_drag:-.2%}")

    print("\n--- STATISTICS ---")
    print(f"Average Beta:        {report.avg_beta:.2f}")
    print(f"Tracking Error:      {report.tracking_error:.2%}")
    print(f"Information Ratio:   {report.information_ratio:.2f}")
    print(f"Avg R-Squared:       {report.avg_r_squared:.2%}")

    if report.sector_weights:
        print("\n--- SECTOR WEIGHTS ---")
        sorted_sectors = sorted(report.sector_weights.items(), key=lambda x: -x[1])
        for sector, weight in sorted_sectors[:5]:
            print(f"  {sector}: {weight:.1%}")

    print("\n" + "=" * 65)
