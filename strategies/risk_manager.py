import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class RiskCalculationError(Exception):
    """Raised when risk calculations fail due to invalid data."""

    pass


class RiskManager:
    def __init__(
        self,
        max_portfolio_risk=0.02,
        max_position_risk=0.01,
        max_correlation=0.7,
        volatility_threshold=0.4,
        var_threshold=0.03,
        es_threshold=0.04,
        drawdown_threshold=0.3,
        strict_correlation_enforcement=True,  # NEW: Reject instead of just adjusting
    ):
        # P2 FIX: Validate all thresholds to prevent invalid risk calculations
        self._validate_threshold("max_portfolio_risk", max_portfolio_risk, 0, 1)
        self._validate_threshold("max_position_risk", max_position_risk, 0, 1)
        self._validate_threshold("max_correlation", max_correlation, -1, 1)
        self._validate_threshold("volatility_threshold", volatility_threshold, 0.001, 10)
        self._validate_threshold("var_threshold", var_threshold, 0.001, 1)
        self._validate_threshold("es_threshold", es_threshold, 0.001, 1)
        self._validate_threshold("drawdown_threshold", drawdown_threshold, 0.001, 1)

        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.volatility_threshold = volatility_threshold
        self.var_threshold = var_threshold
        self.es_threshold = es_threshold
        self.drawdown_threshold = drawdown_threshold
        self.strict_correlation_enforcement = strict_correlation_enforcement
        self.position_sizes = {}
        self.position_correlations = {}

    @staticmethod
    def _validate_threshold(name: str, value: float, min_val: float, max_val: float):
        """P2 FIX: Validate that threshold values are within acceptable bounds."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value)}")
        if value < min_val or value > max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
        if value == 0 and name.endswith("_threshold"):
            raise ValueError(f"{name} cannot be zero (would cause division by zero)")

    def _calculate_volatility(self, price_history):
        """Calculate annualized volatility."""
        if len(price_history) < 2:
            return 0.0
        # Avoid division by zero
        prices = price_history[:-1]
        if np.any(prices == 0):
            logger.warning("Zero prices detected in volatility calculation")
            return 1.0  # Return high volatility to signal caution
        returns = np.diff(price_history) / prices
        return np.std(returns) * np.sqrt(252)

    def _calculate_var(self, price_history):
        """Calculate Value at Risk (VaR) at 95% confidence level."""
        if len(price_history) < 2:
            return 0.0
        # Avoid division by zero
        prices = price_history[:-1]
        if np.any(prices == 0):
            logger.warning("Zero prices detected in VaR calculation")
            return -0.1  # Return large negative VaR to signal high risk
        returns = np.diff(price_history) / prices
        return np.percentile(returns, 5) * np.sqrt(252)

    def _calculate_expected_shortfall(self, price_history):
        """Calculate Expected Shortfall (ES) at 95% confidence level."""
        if len(price_history) < 2:
            return 0.0
        # Avoid division by zero
        prices = price_history[:-1]
        if np.any(prices == 0):
            logger.warning("Zero prices detected in ES calculation")
            return -0.15  # Return large negative ES to signal high risk
        returns = np.diff(price_history) / prices
        var_95 = self._calculate_var(price_history)
        tail_returns = returns[returns <= var_95]
        if len(tail_returns) == 0:
            return var_95  # Return VaR if no tail returns
        return np.mean(tail_returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, price_history):
        """Calculate maximum drawdown."""
        if len(price_history) < 2:
            return 0.0
        rolling_max = np.maximum.accumulate(price_history)
        # Avoid division by zero
        if np.any(rolling_max == 0):
            logger.warning("Zero rolling max detected in drawdown calculation")
            return -0.5  # Return large negative drawdown to signal high risk
        drawdowns = (price_history - rolling_max) / rolling_max
        return np.min(drawdowns)

    def calculate_position_risk(self, symbol: str, price_history: List[float]) -> float:
        """
        Calculate position risk using various metrics.

        Args:
            symbol: Stock symbol
            price_history: List of historical prices

        Returns:
            Risk score between 0.0 and 1.0 (higher = riskier)

        Raises:
            RiskCalculationError: If calculation fails due to invalid data
        """
        try:
            # Validate input
            if not price_history or len(price_history) < 2:
                logger.warning(
                    f"Insufficient price history for {symbol}: {len(price_history) if price_history else 0} points"
                )
                return 1.0  # Maximum risk for insufficient data

            daily_vol = self._calculate_volatility(price_history)
            var_95 = self._calculate_var(price_history)
            es_95 = self._calculate_expected_shortfall(price_history)
            max_drawdown = self._calculate_max_drawdown(price_history)

            # P2 FIX: Safe division with fallback to max risk if thresholds are invalid
            def safe_divide(numerator, denominator, default=1.0):
                """Safely divide, returning default if denominator is zero or invalid."""
                if denominator is None or denominator == 0:
                    logger.warning("Division by zero prevented in risk calculation")
                    return default
                return numerator / denominator

            # Combine metrics into a risk score (0 to 1)
            risk_score = (
                0.3 * safe_divide(daily_vol, self.volatility_threshold)
                + 0.3 * safe_divide(abs(var_95), self.var_threshold)
                + 0.2 * safe_divide(abs(es_95), self.es_threshold)
                + 0.2 * safe_divide(abs(max_drawdown), self.drawdown_threshold)
            )

            return min(risk_score, 1.0)

        except (ValueError, ZeroDivisionError, FloatingPointError) as e:
            logger.error(f"Math error calculating position risk for {symbol}: {e}")
            return 1.0  # Return maximum risk on math error
        except (TypeError, IndexError) as e:
            logger.error(f"Data error calculating position risk for {symbol}: {e}")
            return 1.0  # Return maximum risk on data error

    def calculate_position_correlation(
        self, symbol1: str, symbol2: str, price_history1: List[float], price_history2: List[float]
    ) -> float:
        """
        Calculate correlation between two positions.

        Args:
            symbol1: First stock symbol
            symbol2: Second stock symbol
            price_history1: Price history for first symbol
            price_history2: Price history for second symbol

        Returns:
            Absolute correlation coefficient (0.0 to 1.0)
        """
        try:
            # Validate inputs
            if not price_history1 or not price_history2:
                logger.warning(f"Empty price history for correlation: {symbol1} or {symbol2}")
                return 1.0

            if len(price_history1) < 2 or len(price_history2) < 2:
                logger.warning(f"Insufficient data for correlation: {symbol1}, {symbol2}")
                return 1.0

            # Align lengths
            if len(price_history1) != len(price_history2):
                min_len = min(len(price_history1), len(price_history2))
                price_history1 = price_history1[-min_len:]
                price_history2 = price_history2[-min_len:]

            # Convert to numpy arrays
            prices1 = np.array(price_history1, dtype=np.float64)
            prices2 = np.array(price_history2, dtype=np.float64)

            # Check for zero prices (avoid division by zero)
            if np.any(prices1[:-1] == 0) or np.any(prices2[:-1] == 0):
                logger.warning(
                    f"Zero prices detected in correlation calculation: {symbol1}, {symbol2}"
                )
                return 1.0

            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]

            # Check for NaN/Inf values
            if np.any(~np.isfinite(returns1)) or np.any(~np.isfinite(returns2)):
                logger.warning(f"Invalid returns in correlation calculation: {symbol1}, {symbol2}")
                return 1.0

            correlation = np.corrcoef(returns1, returns2)[0, 1]

            # Handle NaN correlation (can occur with zero variance)
            if not np.isfinite(correlation):
                logger.warning(f"Non-finite correlation between {symbol1} and {symbol2}")
                return 1.0

            return abs(correlation)

        except (ValueError, FloatingPointError) as e:
            logger.error(f"Math error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 1.0  # Return maximum correlation on error (conservative)
        except (IndexError, TypeError) as e:
            logger.error(f"Data error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 1.0  # Return maximum correlation on error

    def calculate_portfolio_risk(self, positions):
        """Calculate total portfolio risk."""
        try:
            total_risk = 0
            position_weights = []

            # Calculate position weights
            total_value = sum(pos["value"] for pos in positions.values())
            for symbol, pos in positions.items():
                weight = pos["value"] / total_value if total_value > 0 else 0
                position_weights.append(weight)

                # Add individual position risk
                total_risk += weight * pos.get("risk", self.max_position_risk)

            # Add correlation impact
            for i, (sym1, pos1) in enumerate(positions.items()):
                for j, (sym2, pos2) in enumerate(positions.items()):
                    if i < j:
                        corr = self.position_correlations.get((sym1, sym2), 0)
                        total_risk += (
                            position_weights[i]
                            * position_weights[j]
                            * pos1.get("risk", self.max_position_risk)
                            * pos2.get("risk", self.max_position_risk)
                            * corr
                        )

            return total_risk

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return self.max_portfolio_risk

    def adjust_position_size(
        self,
        symbol: str,
        desired_size: float,
        price_history: List[float],
        current_positions: Dict[str, Dict],
    ) -> float:
        """
        Adjust position size based on risk parameters.

        Args:
            symbol: Stock symbol
            desired_size: Desired position size in dollars
            price_history: Price history for the symbol
            current_positions: Dict of current positions with their data

        Returns:
            Adjusted position size (0 if rejected)
        """
        try:
            # Validate inputs
            if desired_size <= 0:
                logger.warning(f"Invalid desired_size for {symbol}: {desired_size}")
                return 0

            # Calculate position risk
            risk = self.calculate_position_risk(symbol, price_history)

            # Store risk for portfolio calculations
            if symbol in current_positions:
                current_positions[symbol]["risk"] = risk

            # Calculate correlations with existing positions
            max_correlation = 0
            for other_symbol, pos in current_positions.items():
                if other_symbol != symbol and "price_history" in pos:
                    correlation = self.calculate_position_correlation(
                        symbol, other_symbol, price_history, pos["price_history"]
                    )
                    self.position_correlations[(symbol, other_symbol)] = correlation
                    self.position_correlations[(other_symbol, symbol)] = correlation
                    max_correlation = max(max_correlation, correlation)

            # Adjust size based on risk and correlation
            risk_adjustment = 1.0
            if risk > self.max_position_risk:
                risk_adjustment *= self.max_position_risk / risk

            correlation_adjustment = 1.0
            if max_correlation > self.max_correlation:
                if self.strict_correlation_enforcement:
                    # STRICT MODE: Reject position entirely if correlation too high
                    logger.warning(
                        f"⚠️  POSITION REJECTED: {symbol} has {max_correlation:.2%} correlation "
                        f"with existing positions (max allowed: {self.max_correlation:.2%})"
                    )
                    return 0  # REJECT the position
                else:
                    # SOFT MODE: Reduce position size proportionally
                    correlation_adjustment *= self.max_correlation / max_correlation
                    logger.info(
                        f"Position size reduced for {symbol} due to {max_correlation:.2%} correlation "
                        f"(adjustment: {correlation_adjustment:.2f}x)"
                    )

            # Calculate portfolio risk impact
            portfolio_risk = self.calculate_portfolio_risk(current_positions)
            portfolio_adjustment = 1.0
            if portfolio_risk > self.max_portfolio_risk:
                portfolio_adjustment *= self.max_portfolio_risk / portfolio_risk

            # Apply all adjustments
            adjusted_size = desired_size * min(
                risk_adjustment, correlation_adjustment, portfolio_adjustment
            )

            return max(adjusted_size, 0)  # Ensure non-negative position size

        except (ZeroDivisionError, FloatingPointError) as e:
            logger.error(f"Math error adjusting position size for {symbol}: {e}")
            return 0  # Return 0 size on math error (fail safe)
        except (KeyError, TypeError) as e:
            logger.error(f"Data error adjusting position size for {symbol}: {e}")
            return 0  # Return 0 size on data error (fail safe)
