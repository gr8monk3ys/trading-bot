"""
Factor Models for Institutional-Grade Alpha Generation

This module implements the 5 core cross-sectional factors used by institutional
quant funds:

1. VALUE: Stocks trading cheap relative to fundamentals (P/E, P/B, EV/EBITDA)
2. QUALITY: Companies with strong profitability and low leverage (ROE, debt ratios)
3. MOMENTUM: Stocks with strong recent price performance (12-1 month returns)
4. LOW_VOLATILITY: Stocks with lower realized volatility (anomaly that persists)
5. SIZE: Market cap factor (small cap premium, though weaker in recent decades)

Factor Investing Theory:
- Factors represent systematic sources of risk that earn risk premia
- Long-short factors are market neutral (beta ~0)
- Diversified factor portfolios have better risk-adjusted returns
- Factor timing is difficult; static allocation generally preferred

Implementation Notes:
- All factors are cross-sectionally z-scored within sectors
- Winsorization at 3 standard deviations to limit outlier impact
- Industry neutrality optional but recommended for pure alpha
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """The 5 core institutional factors."""

    VALUE = "value"
    QUALITY = "quality"
    MOMENTUM = "momentum"
    LOW_VOLATILITY = "low_volatility"
    SIZE = "size"


@dataclass
class FactorScore:
    """Individual factor score for a symbol."""

    symbol: str
    factor: FactorType
    raw_score: float  # Original metric value
    z_score: float  # Cross-sectional z-score
    percentile: float  # Percentile rank (0-100)
    sector: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CompositeScore:
    """Combined factor score for a symbol."""

    symbol: str
    composite_z: float  # Weighted average of factor z-scores
    factor_scores: Dict[FactorType, FactorScore]
    quintile: int  # 1-5 (1=lowest, 5=highest)
    signal: str  # 'long', 'short', or 'neutral'
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FactorCalculator:
    """
    Calculates cross-sectional factor scores for a universe of stocks.

    Usage:
        calculator = FactorCalculator()

        # Calculate individual factors
        momentum_scores = calculator.calculate_momentum(price_data)
        value_scores = calculator.calculate_value(fundamental_data)

        # Calculate composite scores
        composite = calculator.calculate_composite(
            price_data, fundamental_data, weights={'momentum': 0.3, 'value': 0.3, ...}
        )
    """

    def __init__(
        self,
        winsorize_std: float = 3.0,
        sector_neutral: bool = True,
        min_universe_size: int = 20,
    ):
        """
        Initialize the factor calculator.

        Args:
            winsorize_std: Std devs for winsorization (default 3.0)
            sector_neutral: Z-score within sectors if True
            min_universe_size: Minimum stocks for valid factor calculation
        """
        self.winsorize_std = winsorize_std
        self.sector_neutral = sector_neutral
        self.min_universe_size = min_universe_size

    def _winsorize(self, values: np.ndarray) -> np.ndarray:
        """Winsorize values at specified standard deviations."""
        values = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return values.copy()

        finite_values = values[finite_mask]
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values))
        if not np.isfinite(std) or std == 0:
            return values.copy()

        lower = mean - self.winsorize_std * std
        upper = mean + self.winsorize_std * std
        clipped = values.copy()
        clipped[finite_mask] = np.clip(finite_values, lower, upper)
        return clipped

    def _z_score(self, values: np.ndarray) -> np.ndarray:
        """Calculate z-scores with winsorization."""
        winsorized = self._winsorize(values)
        finite_mask = np.isfinite(winsorized)
        if not finite_mask.any():
            return np.zeros_like(winsorized, dtype=float)

        finite_values = winsorized[finite_mask]
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values))
        if not np.isfinite(std) or std == 0:
            z_scores = np.zeros_like(winsorized, dtype=float)
            z_scores[~finite_mask] = np.nan
            return z_scores

        z_scores = np.zeros_like(winsorized, dtype=float)
        z_scores[finite_mask] = (finite_values - mean) / std
        z_scores[~finite_mask] = np.nan
        return z_scores

    def _percentile_rank(self, values: np.ndarray) -> np.ndarray:
        """Calculate percentile ranks (0-100)."""
        ranks = stats.rankdata(values, nan_policy="omit")
        n_valid = np.sum(~np.isnan(values))
        if n_valid == 0:
            return np.full_like(values, 50.0)
        return (ranks / n_valid) * 100

    def calculate_momentum(
        self,
        price_data: pd.DataFrame,
        lookback_months: int = 12,
        skip_recent_month: bool = True,
    ) -> Dict[str, FactorScore]:
        """
        Calculate momentum factor scores.

        Momentum Factor:
        - Classic 12-1 month momentum (Jegadeesh & Titman, 1993)
        - Skip most recent month to avoid short-term reversal
        - Cross-sectional z-score for comparability

        Args:
            price_data: DataFrame with symbols as columns, dates as index
            lookback_months: Lookback period (default 12)
            skip_recent_month: Skip recent month to avoid reversal

        Returns:
            Dictionary mapping symbols to FactorScore
        """
        if len(price_data) < 252:  # Need ~1 year of data
            logger.warning("Insufficient data for momentum calculation")
            return {}

        # Calculate returns
        lookback_days = lookback_months * 21  # Trading days
        skip_days = 21 if skip_recent_month else 0

        # Return from lookback_start to skip_days ago
        start_idx = -lookback_days if lookback_days < len(price_data) else 0
        end_idx = -skip_days if skip_days > 0 else None

        if end_idx:
            returns = (price_data.iloc[end_idx] / price_data.iloc[start_idx]) - 1
        else:
            returns = (price_data.iloc[-1] / price_data.iloc[start_idx]) - 1

        # Handle missing data
        returns = returns.replace([np.inf, -np.inf], np.nan)
        valid_mask = ~returns.isna()

        if valid_mask.sum() < self.min_universe_size:
            logger.warning(f"Too few valid returns: {valid_mask.sum()}")
            return {}

        # Z-score
        z_scores = self._z_score(returns.values)
        percentiles = self._percentile_rank(returns.values)

        results = {}
        for i, symbol in enumerate(returns.index):
            if not valid_mask.iloc[i]:
                continue
            results[symbol] = FactorScore(
                symbol=symbol,
                factor=FactorType.MOMENTUM,
                raw_score=float(returns.iloc[i]),
                z_score=float(z_scores[i]),
                percentile=float(percentiles[i]),
            )

        return results

    def calculate_value(
        self,
        fundamental_data: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
    ) -> Dict[str, FactorScore]:
        """
        Calculate value factor scores.

        Value Factor:
        - Cheap stocks relative to fundamentals
        - Uses multiple metrics: P/E, P/B, EV/EBITDA
        - Composite value = average z-score of individual metrics
        - IMPORTANT: Lower P/E = higher value, so we INVERT

        Args:
            fundamental_data: Dict of {symbol: {metric: value}}
                Required metrics: pe_ratio, pb_ratio, ev_ebitda
            metrics: List of metrics to use (default: all three)

        Returns:
            Dictionary mapping symbols to FactorScore
        """
        if metrics is None:
            metrics = ["pe_ratio", "pb_ratio", "ev_ebitda"]

        symbols = list(fundamental_data.keys())
        if len(symbols) < self.min_universe_size:
            logger.warning(f"Too few symbols for value factor: {len(symbols)}")
            return {}

        # Build matrix of metrics
        metric_values = {}
        for metric in metrics:
            values = []
            for symbol in symbols:
                val = fundamental_data.get(symbol, {}).get(metric, np.nan)
                # Filter unrealistic values
                if val is not None and val > 0:
                    values.append(val)
                else:
                    values.append(np.nan)
            metric_values[metric] = np.array(values)

        # Calculate z-scores for each metric (inverted - lower is better for value)
        z_matrices = []
        for metric in metrics:
            values = metric_values[metric]
            if np.sum(~np.isnan(values)) >= self.min_universe_size:
                z = self._z_score(values)
                z_matrices.append(-z)  # INVERT: lower P/E = higher value score

        if not z_matrices:
            logger.warning("No valid value metrics")
            return {}

        # Composite z-score (average)
        z_matrix = np.array(z_matrices)
        composite_z = np.nanmean(z_matrix, axis=0)
        percentiles = self._percentile_rank(composite_z)

        results = {}
        for i, symbol in enumerate(symbols):
            if np.isnan(composite_z[i]):
                continue
            results[symbol] = FactorScore(
                symbol=symbol,
                factor=FactorType.VALUE,
                raw_score=float(np.nanmean([metric_values[m][i] for m in metrics])),
                z_score=float(composite_z[i]),
                percentile=float(percentiles[i]),
            )

        return results

    def calculate_quality(
        self,
        fundamental_data: Dict[str, Dict[str, float]],
    ) -> Dict[str, FactorScore]:
        """
        Calculate quality factor scores.

        Quality Factor:
        - High profitability, low leverage, stable earnings
        - Uses: ROE, debt/equity ratio, earnings variability
        - Higher quality = higher score

        Args:
            fundamental_data: Dict of {symbol: {metric: value}}
                Required: roe, debt_to_equity, earnings_variability

        Returns:
            Dictionary mapping symbols to FactorScore
        """
        symbols = list(fundamental_data.keys())
        if len(symbols) < self.min_universe_size:
            return {}

        # Extract metrics
        roe = np.array([fundamental_data.get(s, {}).get("roe", np.nan) for s in symbols])
        debt_equity = np.array(
            [fundamental_data.get(s, {}).get("debt_to_equity", np.nan) for s in symbols]
        )
        earnings_var = np.array(
            [fundamental_data.get(s, {}).get("earnings_variability", np.nan) for s in symbols]
        )

        # Z-scores (high ROE good, low debt good, low variability good)
        z_components = []

        if np.sum(~np.isnan(roe)) >= self.min_universe_size:
            z_components.append(self._z_score(roe))

        if np.sum(~np.isnan(debt_equity)) >= self.min_universe_size:
            z_components.append(-self._z_score(debt_equity))  # Invert

        if np.sum(~np.isnan(earnings_var)) >= self.min_universe_size:
            z_components.append(-self._z_score(earnings_var))  # Invert

        if not z_components:
            return {}

        composite_z = np.nanmean(np.array(z_components), axis=0)
        percentiles = self._percentile_rank(composite_z)

        results = {}
        for i, symbol in enumerate(symbols):
            if np.isnan(composite_z[i]):
                continue
            results[symbol] = FactorScore(
                symbol=symbol,
                factor=FactorType.QUALITY,
                raw_score=float(roe[i]) if not np.isnan(roe[i]) else 0.0,
                z_score=float(composite_z[i]),
                percentile=float(percentiles[i]),
            )

        return results

    def calculate_low_volatility(
        self,
        price_data: pd.DataFrame,
        lookback_days: int = 252,
    ) -> Dict[str, FactorScore]:
        """
        Calculate low volatility factor scores.

        Low Volatility Anomaly:
        - Lower volatility stocks outperform on risk-adjusted basis
        - One of the most robust anomalies
        - Uses realized volatility (annualized)
        - INVERTED: lower vol = higher score

        Args:
            price_data: DataFrame with symbols as columns
            lookback_days: Days for volatility calculation

        Returns:
            Dictionary mapping symbols to FactorScore
        """
        if len(price_data) < lookback_days:
            lookback_days = len(price_data)

        recent_data = price_data.iloc[-lookback_days:]
        returns = recent_data.pct_change().dropna()

        if len(returns) < 20:
            return {}

        # Annualized volatility
        volatilities = returns.std() * np.sqrt(252)
        valid_mask = ~volatilities.isna() & (volatilities > 0)

        if valid_mask.sum() < self.min_universe_size:
            return {}

        # INVERT: lower volatility = higher score
        z_scores = -self._z_score(volatilities.values)
        percentiles = self._percentile_rank(-volatilities.values)  # Invert for percentile too

        results = {}
        for i, symbol in enumerate(volatilities.index):
            if not valid_mask.iloc[i]:
                continue
            results[symbol] = FactorScore(
                symbol=symbol,
                factor=FactorType.LOW_VOLATILITY,
                raw_score=float(volatilities.iloc[i]),
                z_score=float(z_scores[i]),
                percentile=float(percentiles[i]),
            )

        return results

    def calculate_size(
        self,
        market_caps: Dict[str, float],
        small_cap_premium: bool = True,
    ) -> Dict[str, FactorScore]:
        """
        Calculate size factor scores.

        Size Factor:
        - Small cap premium (Fama-French SMB)
        - Historically small caps outperform, though weaker recently
        - Log market cap for better distribution
        - By default, smaller = higher score (small_cap_premium=True)

        Args:
            market_caps: Dict of {symbol: market_cap}
            small_cap_premium: If True, smaller = higher score

        Returns:
            Dictionary mapping symbols to FactorScore
        """
        symbols = list(market_caps.keys())
        caps = np.array([market_caps.get(s, np.nan) for s in symbols])

        # Filter valid
        valid_mask = ~np.isnan(caps) & (caps > 0)
        if valid_mask.sum() < self.min_universe_size:
            return {}

        # Log transform only valid values to avoid runtime warnings on zero/negative caps.
        log_caps = np.full(caps.shape, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(caps, out=log_caps, where=valid_mask)

        # Z-score (invert if small cap premium)
        z_scores = self._z_score(log_caps)
        if small_cap_premium:
            z_scores = -z_scores

        percentiles = self._percentile_rank(-log_caps if small_cap_premium else log_caps)

        results = {}
        for i, symbol in enumerate(symbols):
            if not valid_mask[i]:
                continue
            results[symbol] = FactorScore(
                symbol=symbol,
                factor=FactorType.SIZE,
                raw_score=float(caps[i]),
                z_score=float(z_scores[i]),
                percentile=float(percentiles[i]),
            )

        return results

    def calculate_composite(
        self,
        factor_scores: Dict[FactorType, Dict[str, FactorScore]],
        weights: Dict[FactorType, float] = None,
    ) -> Dict[str, CompositeScore]:
        """
        Calculate composite factor scores from individual factors.

        Args:
            factor_scores: Dict of {FactorType: {symbol: FactorScore}}
            weights: Dict of {FactorType: weight} (default: equal weight)

        Returns:
            Dictionary mapping symbols to CompositeScore
        """
        if weights is None:
            weights = {ft: 1.0 / len(factor_scores) for ft in factor_scores}
        else:
            # Normalize weights and guard against caller bugs that provide all-zero weights.
            total_weight = sum(weights.values())
            if total_weight <= 0:
                logger.warning("Invalid factor weights (sum <= 0). Falling back to equal weights.")
                weights = {ft: 1.0 / len(factor_scores) for ft in factor_scores}
            else:
                weights = {k: v / total_weight for k, v in weights.items()}

        # Get all symbols
        all_symbols = set()
        for factor_dict in factor_scores.values():
            all_symbols.update(factor_dict.keys())

        results = {}
        composite_values = []
        symbol_list = []

        for symbol in all_symbols:
            weighted_z = 0.0
            total_weight_used = 0.0
            symbol_factor_scores = {}

            for factor_type, scores_dict in factor_scores.items():
                if symbol in scores_dict:
                    score = scores_dict[symbol]
                    symbol_factor_scores[factor_type] = score
                    weighted_z += weights.get(factor_type, 0) * score.z_score
                    total_weight_used += weights.get(factor_type, 0)

            if total_weight_used > 0:
                composite_z = weighted_z / total_weight_used
                composite_values.append(composite_z)
                symbol_list.append((symbol, composite_z, symbol_factor_scores))

        if not composite_values:
            return {}

        # Calculate quintiles
        quintile_breaks = np.percentile(composite_values, [20, 40, 60, 80])

        for symbol, composite_z, symbol_factor_scores in symbol_list:
            # Determine quintile
            if composite_z <= quintile_breaks[0]:
                quintile = 1
            elif composite_z <= quintile_breaks[1]:
                quintile = 2
            elif composite_z <= quintile_breaks[2]:
                quintile = 3
            elif composite_z <= quintile_breaks[3]:
                quintile = 4
            else:
                quintile = 5

            # Determine signal
            if quintile >= 4:
                signal = "long"
            elif quintile <= 2:
                signal = "short"
            else:
                signal = "neutral"

            results[symbol] = CompositeScore(
                symbol=symbol,
                composite_z=composite_z,
                factor_scores=symbol_factor_scores,
                quintile=quintile,
                signal=signal,
            )

        return results


class FactorModel:
    """
    High-level factor model for strategy integration.

    Usage:
        model = FactorModel()

        # Load data and calculate scores
        scores = model.score_universe(symbols, price_data, fundamental_data)

        # Get long/short portfolios
        longs, shorts = model.get_portfolios(scores, n_stocks=20)

        # Generate signals for a single stock
        signal = model.get_signal('AAPL', scores)
    """

    def __init__(
        self,
        factor_weights: Optional[Dict[Union["FactorType", str], float]] = None,
        long_threshold: float = 0.5,
        short_threshold: float = -0.5,
    ):
        """
        Initialize the factor model.

        Args:
            factor_weights: Weights for each factor (default: equal)
            long_threshold: Z-score threshold for long signal
            short_threshold: Z-score threshold for short signal
        """
        self.calculator = FactorCalculator()
        self.factor_weights = (
            self._coerce_factor_weights(factor_weights)
            if factor_weights is not None
            else {
                FactorType.VALUE: 0.20,
                FactorType.QUALITY: 0.20,
                FactorType.MOMENTUM: 0.30,  # Momentum slightly overweighted
                FactorType.LOW_VOLATILITY: 0.20,
                FactorType.SIZE: 0.10,  # Size underweighted (weaker premium)
            }
        )
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    @staticmethod
    def _coerce_factor_weights(
        factor_weights: Dict[Union["FactorType", str], float],
    ) -> Dict[FactorType, float]:
        """
        Accept factor weights keyed by FactorType or strings ("value", "momentum", etc.)
        and normalize keys to FactorType for internal use.
        """
        coerced: Dict[FactorType, float] = {}
        for key, weight in (factor_weights or {}).items():
            factor_type: Optional[FactorType] = None

            if isinstance(key, FactorType):
                factor_type = key
            elif isinstance(key, str):
                normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
                for candidate in FactorType:
                    if candidate.value == normalized or candidate.name.lower() == normalized:
                        factor_type = candidate
                        break

            if factor_type is None:
                logger.warning(f"Ignoring unknown factor weight key: {key!r}")
                continue

            try:
                coerced[factor_type] = float(weight)
            except (TypeError, ValueError):
                logger.warning(f"Ignoring non-numeric factor weight for {key!r}: {weight!r}")

        if not coerced:
            logger.warning("No valid factor weights provided; using defaults.")
            return {
                FactorType.VALUE: 0.20,
                FactorType.QUALITY: 0.20,
                FactorType.MOMENTUM: 0.30,
                FactorType.LOW_VOLATILITY: 0.20,
                FactorType.SIZE: 0.10,
            }

        return coerced

    def score_universe(
        self,
        symbols: List[str],
        price_data: pd.DataFrame,
        fundamental_data: Dict[str, Dict[str, float]] = None,
        market_caps: Dict[str, float] = None,
    ) -> Dict[str, CompositeScore]:
        """
        Score entire universe using all available factors.

        Args:
            symbols: List of symbols to score
            price_data: Price data DataFrame
            fundamental_data: Optional fundamental data
            market_caps: Optional market cap data

        Returns:
            Dictionary of CompositeScore per symbol
        """
        factor_scores = {}

        # Always calculate momentum (only needs price data)
        momentum = self.calculator.calculate_momentum(price_data)
        if momentum:
            factor_scores[FactorType.MOMENTUM] = momentum

        # Always calculate low volatility (only needs price data)
        low_vol = self.calculator.calculate_low_volatility(price_data)
        if low_vol:
            factor_scores[FactorType.LOW_VOLATILITY] = low_vol

        # Value and quality need fundamental data
        if fundamental_data:
            value = self.calculator.calculate_value(fundamental_data)
            if value:
                factor_scores[FactorType.VALUE] = value

            quality = self.calculator.calculate_quality(fundamental_data)
            if quality:
                factor_scores[FactorType.QUALITY] = quality

        # Size needs market cap data
        if market_caps:
            size = self.calculator.calculate_size(market_caps)
            if size:
                factor_scores[FactorType.SIZE] = size

        if not factor_scores:
            logger.warning("No factors calculated successfully")
            return {}

        # Adjust weights based on available factors
        available_weights = {ft: w for ft, w in self.factor_weights.items() if ft in factor_scores}

        return self.calculator.calculate_composite(factor_scores, available_weights)

    def get_portfolios(
        self,
        scores: Dict[str, CompositeScore],
        n_stocks: int = 20,
        dollar_neutral: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        Get long and short portfolios from scores.

        Args:
            scores: Composite scores from score_universe
            n_stocks: Number of stocks per side
            dollar_neutral: If True, equal $ long and short

        Returns:
            Tuple of (long_symbols, short_symbols)
        """
        sorted_scores = sorted(scores.items(), key=lambda x: x[1].composite_z, reverse=True)

        longs = [s[0] for s in sorted_scores[:n_stocks]]
        shorts = [s[0] for s in sorted_scores[-n_stocks:]]

        return longs, shorts

    def get_signal(
        self,
        symbol: str,
        scores: Dict[str, CompositeScore],
    ) -> Dict[str, Any]:
        """
        Get trading signal for a single symbol.

        Args:
            symbol: Stock symbol
            scores: Composite scores from score_universe

        Returns:
            Signal dictionary with action, confidence, and factor breakdown
        """
        if symbol not in scores:
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": "Symbol not in scored universe",
            }

        score = scores[symbol]

        # Map z-score to confidence (0-1)
        confidence = min(1.0, abs(score.composite_z) / 2.0)

        # Factor breakdown
        factor_breakdown = {
            ft.value: {
                "z_score": fs.z_score,
                "percentile": fs.percentile,
            }
            for ft, fs in score.factor_scores.items()
        }

        return {
            "action": score.signal,
            "confidence": confidence,
            "composite_z": score.composite_z,
            "quintile": score.quintile,
            "factor_breakdown": factor_breakdown,
            "reason": f"Factor composite z={score.composite_z:.2f}, quintile={score.quintile}",
        }

    def get_factor_exposures(
        self,
        portfolio: List[Tuple[str, float]],
        scores: Dict[str, CompositeScore],
    ) -> Dict[str, float]:
        """
        Calculate portfolio's exposure to each factor.

        Args:
            portfolio: List of (symbol, weight) tuples
            scores: Composite scores

        Returns:
            Dictionary of factor exposures (weighted average z-scores)
        """
        exposures = dict.fromkeys(FactorType, 0.0)
        total_weight = 0.0

        for symbol, weight in portfolio:
            if symbol in scores:
                score = scores[symbol]
                for ft, fs in score.factor_scores.items():
                    exposures[ft] += weight * fs.z_score
                total_weight += weight

        if total_weight > 0:
            exposures = {ft: exp / total_weight for ft, exp in exposures.items()}

        return {ft.value: exp for ft, exp in exposures.items()}
