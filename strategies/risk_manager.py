import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def safe_divide(numerator: float, denominator: float, default: float = 1.0) -> float:
    """
    Safely divide, returning default if denominator is zero or invalid.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division is invalid

    Returns:
        Result of division or default value
    """
    if denominator is None or denominator == 0:
        logger.warning("Division by zero prevented in risk calculation")
        return default
    return numerator / denominator


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

    @staticmethod
    def _compute_returns(price_history) -> np.ndarray:
        """Compute simple returns without emitting runtime warnings."""
        prices = np.asarray(price_history, dtype=np.float64)
        if prices.size < 2:
            return np.array([], dtype=np.float64)

        prev = prices[:-1]
        curr = prices[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.divide(
                curr - prev,
                prev,
                out=np.full(prev.shape, np.nan, dtype=np.float64),
                where=prev != 0,
            )
        return returns[np.isfinite(returns)]

    def _calculate_volatility(self, price_history):
        """Calculate annualized volatility."""
        if len(price_history) < 2:
            return 0.0
        # Avoid division by zero
        prices = np.asarray(price_history[:-1], dtype=np.float64)
        if np.any(~np.isfinite(prices)):
            logger.warning("Invalid prices detected in volatility calculation")
            return 1.0  # Return high volatility to signal caution
        if np.any(prices == 0):
            logger.warning("Zero prices detected in volatility calculation")
            return 1.0  # Return high volatility to signal caution
        returns = self._compute_returns(price_history)
        if returns.size == 0:
            return 1.0
        return float(np.std(returns) * np.sqrt(252))

    def _calculate_var(self, price_history):
        """Calculate Value at Risk (VaR) at 95% confidence level using historical method."""
        if len(price_history) < 2:
            return 0.0
        # Avoid division by zero
        prices = np.asarray(price_history[:-1], dtype=np.float64)
        if np.any(~np.isfinite(prices)):
            logger.warning("Invalid prices detected in VaR calculation")
            return -0.1  # Return large negative VaR to signal high risk
        if np.any(prices == 0):
            logger.warning("Zero prices detected in VaR calculation")
            return -0.1  # Return large negative VaR to signal high risk
        returns = self._compute_returns(price_history)
        if returns.size == 0:
            return -0.1
        return float(np.percentile(returns, 5) * np.sqrt(252))

    def _calculate_parametric_var(self, price_history, confidence: float = 0.95):
        """
        Calculate parametric VaR assuming normal distribution.

        Args:
            price_history: List of historical prices
            confidence: Confidence level (default 95%)

        Returns:
            Parametric VaR (negative value = potential loss)
        """
        if len(price_history) < 2:
            return 0.0

        prices = np.asarray(price_history[:-1], dtype=np.float64)
        if np.any(~np.isfinite(prices)):
            logger.warning("Invalid prices detected in parametric VaR calculation")
            return -0.1
        if np.any(prices == 0):
            logger.warning("Zero prices detected in parametric VaR calculation")
            return -0.1

        returns = self._compute_returns(price_history)
        if returns.size == 0:
            return -0.1
        mean = np.mean(returns)
        std = np.std(returns)

        # Z-score for 95% confidence (1.645 for one-tailed)
        z_score = 1.645 if confidence == 0.95 else 2.326 if confidence == 0.99 else 1.645

        # Parametric VaR
        var_daily = mean - z_score * std
        var_annual = var_daily * np.sqrt(252)

        return var_annual

    def _calculate_monte_carlo_var(
        self, price_history, n_simulations: int = 10000, confidence: float = 0.95
    ):
        """
        Calculate VaR using Monte Carlo simulation.

        Simulates 10,000 1-year paths using historical mean/std.

        Args:
            price_history: List of historical prices
            n_simulations: Number of Monte Carlo paths (default 10,000)
            confidence: Confidence level (default 95%)

        Returns:
            Monte Carlo VaR (negative value = potential loss)
        """
        if len(price_history) < 20:  # Need reasonable history for MC
            return self._calculate_var(price_history)  # Fall back to historical

        prices = np.asarray(price_history[:-1], dtype=np.float64)
        if np.any(~np.isfinite(prices)):
            logger.warning("Invalid prices detected in Monte Carlo VaR calculation")
            return -0.1
        if np.any(prices == 0):
            logger.warning("Zero prices detected in Monte Carlo VaR calculation")
            return -0.1

        returns = self._compute_returns(price_history)
        if returns.size == 0:
            return -0.1
        mean = np.mean(returns)
        std = np.std(returns)

        try:
            # Simulate 1-year paths (252 trading days)
            simulated_returns = np.random.normal(mean, std, (n_simulations, 252))

            # Calculate cumulative returns for each path
            cumulative = np.cumprod(1 + simulated_returns, axis=1)

            # Get final returns (end of year)
            final_returns = cumulative[:, -1] - 1

            # VaR at confidence level
            percentile = (1 - confidence) * 100  # 5th percentile for 95% confidence
            var_mc = np.percentile(final_returns, percentile)

            return var_mc

        except Exception as e:
            logger.warning(f"Monte Carlo VaR failed, using historical: {e}")
            return self._calculate_var(price_history)

    def _calculate_cornish_fisher_var(self, price_history, confidence: float = 0.95):
        """
        Calculate VaR using Cornish-Fisher expansion for fat-tailed distributions.

        Adjusts the standard normal quantile for skewness and kurtosis.
        More accurate for non-normal return distributions.

        Args:
            price_history: List of historical prices
            confidence: Confidence level (default 95%)

        Returns:
            Cornish-Fisher adjusted VaR (negative value = potential loss)
        """
        if len(price_history) < 30:  # Need enough data for skew/kurtosis
            return self._calculate_parametric_var(price_history, confidence)

        prices = np.asarray(price_history[:-1], dtype=np.float64)
        if np.any(~np.isfinite(prices)):
            logger.warning("Invalid prices detected in Cornish-Fisher VaR calculation")
            return -0.1
        if np.any(prices == 0):
            logger.warning("Zero prices detected in Cornish-Fisher VaR calculation")
            return -0.1

        returns = self._compute_returns(price_history)
        if returns.size == 0:
            return -0.1
        mean = np.mean(returns)
        std = np.std(returns)

        try:
            from scipy import stats as scipy_stats

            skew = scipy_stats.skew(returns)
            kurt = scipy_stats.kurtosis(returns)  # Excess kurtosis

            # Standard normal quantile
            z = 1.645 if confidence == 0.95 else 2.326 if confidence == 0.99 else 1.645

            # Cornish-Fisher expansion
            # z_cf = z + (z^2 - 1)*skew/6 + (z^3 - 3z)*kurt/24 - (2z^3 - 5z)*skew^2/36
            z_cf = (
                z
                + (z**2 - 1) * skew / 6
                + (z**3 - 3 * z) * kurt / 24
                - (2 * z**3 - 5 * z) * skew**2 / 36
            )

            # Cornish-Fisher VaR
            var_cf = (mean - z_cf * std) * np.sqrt(252)

            return var_cf

        except ImportError:
            logger.warning("scipy not available, using parametric VaR")
            return self._calculate_parametric_var(price_history, confidence)
        except Exception as e:
            logger.warning(f"Cornish-Fisher VaR failed, using parametric: {e}")
            return self._calculate_parametric_var(price_history, confidence)

    def calculate_var_ensemble(self, price_history, confidence: float = 0.95) -> dict:
        """
        Calculate ensemble VaR using multiple methods.

        Returns the most conservative (lowest) VaR and all individual estimates.

        Args:
            price_history: List of historical prices
            confidence: Confidence level (default 95%)

        Returns:
            Dict with ensemble VaR and individual method results
        """
        if len(price_history) < 2:
            return {
                "ensemble_var": 0.0,
                "method": "insufficient_data",
                "historical": 0.0,
                "parametric": 0.0,
                "monte_carlo": 0.0,
                "cornish_fisher": 0.0,
            }

        # Calculate all VaR estimates
        var_hist = self._calculate_var(price_history)
        var_param = self._calculate_parametric_var(price_history, confidence)
        var_mc = self._calculate_monte_carlo_var(price_history, confidence=confidence)
        var_cf = self._calculate_cornish_fisher_var(price_history, confidence)

        # Use most conservative (most negative = worst case)
        var_estimates = {
            "historical": var_hist,
            "parametric": var_param,
            "monte_carlo": var_mc,
            "cornish_fisher": var_cf,
        }

        ensemble_var = min(var_estimates.values())
        worst_method = min(var_estimates, key=var_estimates.get)

        logger.debug(
            f"VaR Ensemble: Historical={var_hist:.2%}, Parametric={var_param:.2%}, "
            f"MC={var_mc:.2%}, CF={var_cf:.2%} -> Ensemble={ensemble_var:.2%} ({worst_method})"
        )

        return {
            "ensemble_var": ensemble_var,
            "method": worst_method,
            "historical": var_hist,
            "parametric": var_param,
            "monte_carlo": var_mc,
            "cornish_fisher": var_cf,
        }

    def _calculate_expected_shortfall(self, price_history, var_value=None):
        """Calculate Expected Shortfall (ES) at 95% confidence level.

        Args:
            price_history: List of historical prices.
            var_value: Optional pre-computed VaR value to avoid redundant calculation.
        """
        if len(price_history) < 2:
            return 0.0
        # Avoid division by zero
        prices = np.asarray(price_history[:-1], dtype=np.float64)
        if np.any(~np.isfinite(prices)):
            logger.warning("Invalid prices detected in ES calculation")
            return -0.15  # Return large negative ES to signal high risk
        if np.any(prices == 0):
            logger.warning("Zero prices detected in ES calculation")
            return -0.15  # Return large negative ES to signal high risk
        returns = self._compute_returns(price_history)
        if returns.size == 0:
            return var_value if var_value is not None else -0.15
        var_95 = var_value if var_value is not None else self._calculate_var(price_history)
        tail_returns = returns[returns <= var_95]
        if len(tail_returns) == 0:
            return var_95  # Return VaR if no tail returns
        return float(np.mean(tail_returns) * np.sqrt(252))

    def _calculate_max_drawdown(self, price_history):
        """Calculate maximum drawdown."""
        if len(price_history) < 2:
            return 0.0
        prices = np.asarray(price_history, dtype=np.float64)
        if np.any(~np.isfinite(prices)):
            logger.warning("Invalid prices detected in drawdown calculation")
            return -0.5  # Return large negative drawdown to signal high risk
        rolling_max = np.maximum.accumulate(prices)
        # Avoid division by zero
        if np.any(~np.isfinite(rolling_max)) or np.any(rolling_max == 0):
            logger.warning("Zero rolling max detected in drawdown calculation")
            return -0.5  # Return large negative drawdown to signal high risk
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns = np.divide(
                prices - rolling_max,
                rolling_max,
                out=np.full(prices.shape, np.nan, dtype=np.float64),
                where=rolling_max != 0,
            )
        finite_drawdowns = drawdowns[np.isfinite(drawdowns)]
        if finite_drawdowns.size == 0:
            return -0.5
        return float(np.min(finite_drawdowns))

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
            es_95 = self._calculate_expected_shortfall(price_history, var_value=var_95)
            max_drawdown = self._calculate_max_drawdown(price_history)

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

            if np.any(~np.isfinite(prices1)) or np.any(~np.isfinite(prices2)):
                logger.warning(f"Non-finite prices in correlation calculation: {symbol1}, {symbol2}")
                return 1.0

            with np.errstate(divide="ignore", invalid="ignore"):
                returns1 = np.divide(
                    np.diff(prices1),
                    prices1[:-1],
                    out=np.full(prices1.shape[0] - 1, np.nan, dtype=np.float64),
                    where=prices1[:-1] != 0,
                )
                returns2 = np.divide(
                    np.diff(prices2),
                    prices2[:-1],
                    out=np.full(prices2.shape[0] - 1, np.nan, dtype=np.float64),
                    where=prices2[:-1] != 0,
                )

            # Check for NaN/Inf values
            if np.any(~np.isfinite(returns1)) or np.any(~np.isfinite(returns2)):
                logger.warning(f"Invalid returns in correlation calculation: {symbol1}, {symbol2}")
                return 1.0

            with np.errstate(divide="ignore", invalid="ignore"):
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
            for _symbol, pos in positions.items():
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

    def enforce_limits(
        self,
        symbol: str,
        desired_size: float,
        price_history: List[float],
        current_positions: Dict[str, Dict],
    ) -> tuple:
        """
        Enforce risk limits and return both adjusted size and violation details.

        This is the RECOMMENDED method for OrderGateway integration.
        Unlike adjust_position_size(), this method:
        - Returns explicit violation reasons
        - Uses stricter enforcement (rejects on any violation)
        - Provides audit trail for rejected orders

        Args:
            symbol: Stock symbol
            desired_size: Desired position size in dollars
            price_history: Price history for the symbol
            current_positions: Dict of current positions with their data

        Returns:
            Tuple of (adjusted_size, violations_dict)
            - adjusted_size: Position size after limits (0 if rejected)
            - violations_dict: Dict of {limit_name: violation_message}
        """
        violations = {}

        try:
            # Validate inputs
            if desired_size <= 0:
                violations["invalid_size"] = f"Invalid desired_size: {desired_size}"
                return 0, violations

            if not price_history or len(price_history) < 2:
                violations["insufficient_data"] = (
                    f"Insufficient price history: {len(price_history) if price_history else 0} points"
                )
                return 0, violations

            # 1. VaR Check - Reject if Value at Risk too high
            var_95 = self._calculate_var(price_history)
            if abs(var_95) > self.var_threshold:
                violations["var_exceeded"] = (
                    f"VaR {abs(var_95):.2%} exceeds threshold {self.var_threshold:.2%}"
                )
                logger.warning(
                    f"⚠️  VAR LIMIT: {symbol} has VaR of {abs(var_95):.2%} "
                    f"(max: {self.var_threshold:.2%})"
                )

            # 2. Expected Shortfall Check
            es_95 = self._calculate_expected_shortfall(price_history, var_value=var_95)
            if abs(es_95) > self.es_threshold:
                violations["es_exceeded"] = (
                    f"Expected Shortfall {abs(es_95):.2%} exceeds threshold {self.es_threshold:.2%}"
                )
                logger.warning(
                    f"⚠️  ES LIMIT: {symbol} has ES of {abs(es_95):.2%} "
                    f"(max: {self.es_threshold:.2%})"
                )

            # 3. Volatility Check
            volatility = self._calculate_volatility(price_history)
            if volatility > self.volatility_threshold:
                violations["volatility_exceeded"] = (
                    f"Volatility {volatility:.2%} exceeds threshold {self.volatility_threshold:.2%}"
                )
                logger.warning(
                    f"⚠️  VOLATILITY LIMIT: {symbol} has volatility of {volatility:.2%} "
                    f"(max: {self.volatility_threshold:.2%})"
                )

            # 4. Correlation Check - Compare with existing positions
            max_correlation = 0
            correlated_with = None
            for other_symbol, pos in current_positions.items():
                if other_symbol != symbol and "price_history" in pos:
                    correlation = self.calculate_position_correlation(
                        symbol, other_symbol, price_history, pos["price_history"]
                    )
                    self.position_correlations[(symbol, other_symbol)] = correlation
                    self.position_correlations[(other_symbol, symbol)] = correlation
                    if correlation > max_correlation:
                        max_correlation = correlation
                        correlated_with = other_symbol

            if max_correlation > self.max_correlation:
                violations["correlation_exceeded"] = (
                    f"Correlation {max_correlation:.2%} with {correlated_with} "
                    f"exceeds threshold {self.max_correlation:.2%}"
                )
                logger.warning(
                    f"⚠️  CORRELATION LIMIT: {symbol} has {max_correlation:.2%} "
                    f"correlation with {correlated_with} (max: {self.max_correlation:.2%})"
                )

            # 5. Portfolio Risk Check
            # Temporarily add this position to calculate impact
            test_positions = current_positions.copy()
            risk = self.calculate_position_risk(symbol, price_history)
            test_positions[symbol] = {
                "value": desired_size,
                "risk": risk,
                "price_history": price_history,
            }
            portfolio_risk = self.calculate_portfolio_risk(test_positions)

            if portfolio_risk > self.max_portfolio_risk:
                violations["portfolio_risk_exceeded"] = (
                    f"Portfolio risk {portfolio_risk:.2%} exceeds threshold {self.max_portfolio_risk:.2%}"
                )
                logger.warning(
                    f"⚠️  PORTFOLIO RISK LIMIT: Adding {symbol} would push portfolio risk to "
                    f"{portfolio_risk:.2%} (max: {self.max_portfolio_risk:.2%})"
                )

            # 6. Max Drawdown Check
            max_drawdown = self._calculate_max_drawdown(price_history)
            if abs(max_drawdown) > self.drawdown_threshold:
                violations["drawdown_exceeded"] = (
                    f"Max drawdown {abs(max_drawdown):.2%} exceeds threshold {self.drawdown_threshold:.2%}"
                )
                logger.warning(
                    f"⚠️  DRAWDOWN LIMIT: {symbol} has max drawdown of {abs(max_drawdown):.2%} "
                    f"(max: {self.drawdown_threshold:.2%})"
                )

            # If any violations, reject the order
            if violations:
                logger.warning(
                    f"ORDER REJECTED for {symbol}: {len(violations)} limit(s) violated - "
                    f"{', '.join(violations.keys())}"
                )
                return 0, violations

            # No violations - return full size
            logger.debug(
                f"Risk limits passed for {symbol}: "
                f"VaR={abs(var_95):.2%}, ES={abs(es_95):.2%}, "
                f"Vol={volatility:.2%}, Corr={max_correlation:.2%}"
            )
            return desired_size, {}

        except Exception as e:
            logger.error(f"Error enforcing limits for {symbol}: {e}")
            violations["calculation_error"] = str(e)
            return 0, violations

    def get_risk_summary(self, symbol: str, price_history: List[float]) -> Dict:
        """
        Get a summary of all risk metrics for a symbol.

        Useful for pre-trade analysis and logging.

        Args:
            symbol: Stock symbol
            price_history: Price history for the symbol

        Returns:
            Dict with all risk metrics and their status
        """
        if not price_history or len(price_history) < 2:
            return {
                "symbol": symbol,
                "valid": False,
                "error": "Insufficient price history",
            }

        try:
            var_95 = self._calculate_var(price_history)
            es_95 = self._calculate_expected_shortfall(price_history, var_value=var_95)
            volatility = self._calculate_volatility(price_history)
            max_drawdown = self._calculate_max_drawdown(price_history)
            risk_score = self.calculate_position_risk(symbol, price_history)

            return {
                "symbol": symbol,
                "valid": True,
                "var_95": {
                    "value": abs(var_95),
                    "threshold": self.var_threshold,
                    "passed": abs(var_95) <= self.var_threshold,
                },
                "es_95": {
                    "value": abs(es_95),
                    "threshold": self.es_threshold,
                    "passed": abs(es_95) <= self.es_threshold,
                },
                "volatility": {
                    "value": volatility,
                    "threshold": self.volatility_threshold,
                    "passed": volatility <= self.volatility_threshold,
                },
                "max_drawdown": {
                    "value": abs(max_drawdown),
                    "threshold": self.drawdown_threshold,
                    "passed": abs(max_drawdown) <= self.drawdown_threshold,
                },
                "risk_score": risk_score,
            }

        except Exception as e:
            return {
                "symbol": symbol,
                "valid": False,
                "error": str(e),
            }

    # =========================================================================
    # MARGIN MONITORING - INSTITUTIONAL GRADE
    # =========================================================================

    def calculate_margin_requirement(
        self,
        positions: Dict[str, Dict],
        broker_margin_requirement: float = 0.25,
    ) -> Dict:
        """
        Calculate margin requirements for all positions.

        Args:
            positions: Dict of positions with {symbol: {value, price, quantity}}
            broker_margin_requirement: Broker's maintenance margin % (default 25%)

        Returns:
            Dict with margin analysis
        """
        total_position_value = sum(
            pos.get("value", pos.get("quantity", 0) * pos.get("price", 0))
            for pos in positions.values()
        )

        # Calculate required margin
        required_margin = total_position_value * broker_margin_requirement

        # Calculate liquidation values for each position
        position_analysis = {}
        for symbol, pos in positions.items():
            value = pos.get("value", pos.get("quantity", 0) * pos.get("price", 0))
            price = pos.get("price", 0)
            quantity = pos.get("quantity", 0)

            if quantity > 0 and price > 0:  # Long position
                # Liquidation price = entry_price * (1 - (1 - margin_req) / margin_req)
                # Simplified: price drops by (1 - margin_req) triggers liquidation
                liquidation_price = price * broker_margin_requirement
                cushion_pct = (price - liquidation_price) / price if price > 0 else 0
            else:
                liquidation_price = None
                cushion_pct = None

            position_analysis[symbol] = {
                "value": value,
                "price": price,
                "quantity": quantity,
                "liquidation_price": liquidation_price,
                "cushion_pct": cushion_pct,
            }

        return {
            "total_position_value": total_position_value,
            "required_margin": required_margin,
            "margin_requirement_pct": broker_margin_requirement,
            "position_analysis": position_analysis,
        }

    def check_margin_status(
        self,
        equity: float,
        positions: Dict[str, Dict],
        broker_margin_requirement: float = 0.25,
        warning_threshold: float = 0.35,
        halt_threshold: float = 0.30,
    ) -> Dict:
        """
        Check margin status and determine if action is needed.

        Institutional margin monitoring includes:
        - Current margin level vs requirement
        - Proximity to margin call
        - Automatic halt when too close to liquidation

        Args:
            equity: Current account equity
            positions: Dict of positions
            broker_margin_requirement: Broker's maintenance margin (default 25%)
            warning_threshold: Warn at this margin level (default 35%)
            halt_threshold: Halt trading at this level (default 30%)

        Returns:
            Dict with margin status and recommendations
        """
        margin_analysis = self.calculate_margin_requirement(
            positions, broker_margin_requirement
        )

        total_position_value = margin_analysis["total_position_value"]
        required_margin = margin_analysis["required_margin"]

        # Calculate margin percentage
        if total_position_value > 0:
            margin_pct = equity / total_position_value
        else:
            margin_pct = 1.0  # No positions = 100% margin

        # Determine margin status
        if margin_pct < halt_threshold:
            status = "CRITICAL"
            action = "HALT_TRADING"
            message = f"Margin {margin_pct:.1%} below halt threshold {halt_threshold:.1%}"
        elif margin_pct < warning_threshold:
            status = "WARNING"
            action = "REDUCE_EXPOSURE"
            message = f"Margin {margin_pct:.1%} approaching danger zone"
        else:
            status = "OK"
            action = "NONE"
            message = f"Margin {margin_pct:.1%} is healthy"

        # Calculate excess/deficit
        margin_excess = equity - required_margin
        margin_deficit = max(0, required_margin - equity)

        # Calculate how much to reduce to reach safety
        safe_position_value = equity / warning_threshold if warning_threshold > 0 else 0
        reduction_needed = max(0, total_position_value - safe_position_value)

        return {
            "status": status,
            "action": action,
            "message": message,
            "equity": equity,
            "total_position_value": total_position_value,
            "margin_pct": margin_pct,
            "required_margin": required_margin,
            "margin_excess": margin_excess,
            "margin_deficit": margin_deficit,
            "warning_threshold": warning_threshold,
            "halt_threshold": halt_threshold,
            "should_halt": margin_pct < halt_threshold,
            "should_warn": margin_pct < warning_threshold,
            "reduction_needed": reduction_needed,
            "position_analysis": margin_analysis["position_analysis"],
        }

    def get_liquidation_prices(
        self,
        positions: Dict[str, Dict],
        equity: float,
        broker_margin_requirement: float = 0.25,
    ) -> Dict[str, float]:
        """
        Calculate liquidation prices for each position.

        Returns the price at which each position would trigger margin call.

        Args:
            positions: Dict of positions
            equity: Current equity
            broker_margin_requirement: Broker's maintenance margin

        Returns:
            Dict of {symbol: liquidation_price}
        """
        liquidation_prices = {}

        total_position_value = sum(
            pos.get("value", pos.get("quantity", 0) * pos.get("price", 0))
            for pos in positions.values()
        )

        if total_position_value == 0:
            return {}

        # Current margin
        equity / total_position_value if total_position_value > 0 else 1.0

        for symbol, pos in positions.items():
            price = pos.get("price", 0)
            quantity = pos.get("quantity", 0)

            if quantity > 0 and price > 0:  # Long position
                # Calculate price that would bring margin to requirement
                # If price drops by X%, position value drops by X%
                # margin = equity / (position_value * (1 - X))
                # At liquidation: margin_req = equity / (position_value * (1 - X))
                # Solving for X: 1 - X = equity / (position_value * margin_req)

                pos.get("value", quantity * price)
                # How much can position drop before liquidation?
                # equity / (position_value * (1 - drop_pct)) = margin_req
                # (1 - drop_pct) = equity / (position_value * margin_req)
                denominator = total_position_value * broker_margin_requirement
                if denominator > 0:
                    factor = equity / denominator
                    drop_pct = 1 - (1 / factor) if factor > 0 else 1.0
                    liquidation_price = price * (1 - max(0, drop_pct))
                else:
                    liquidation_price = 0

                liquidation_prices[symbol] = max(0, liquidation_price)

        return liquidation_prices

    def should_halt_for_margin(
        self,
        equity: float,
        positions: Dict[str, Dict],
        halt_threshold: float = 0.30,
    ) -> bool:
        """
        Simple check if trading should be halted due to margin.

        This is designed to integrate with CircuitBreaker.

        Args:
            equity: Current equity
            positions: Dict of positions
            halt_threshold: Margin level to halt at

        Returns:
            True if trading should be halted
        """
        total_position_value = sum(
            pos.get("value", pos.get("quantity", 0) * pos.get("price", 0))
            for pos in positions.values()
        )

        if total_position_value == 0:
            return False

        margin_pct = equity / total_position_value
        return margin_pct < halt_threshold
