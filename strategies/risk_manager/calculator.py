"""
RiskManager calculator mixin.

Pure risk-math methods used by ``RiskManager``:

    - ``_compute_returns`` — NaN/zero-safe simple-returns helper
    - ``_calculate_volatility`` — annualized return volatility
    - ``_calculate_var`` — historical Value-at-Risk
    - ``_calculate_parametric_var`` — Gaussian-quantile VaR
    - ``_calculate_monte_carlo_var`` — Monte-Carlo simulation VaR
    - ``_calculate_cornish_fisher_var`` — fat-tailed VaR
    - ``calculate_var_ensemble`` — most-conservative ensemble of the above
    - ``_calculate_expected_shortfall`` — tail-loss expectation
    - ``_calculate_max_drawdown`` — peak-to-trough drawdown
    - ``calculate_position_risk`` — composite 0–1 risk score
    - ``calculate_position_correlation`` — absolute correlation between two
      symbols' return series
    - ``calculate_portfolio_risk`` — weighted total portfolio risk with
      correlation cross terms

These methods rely on attributes initialized by the ``RiskManager`` facade
(``self.volatility_threshold``, ``self.var_threshold``,
``self.es_threshold``, ``self.drawdown_threshold``,
``self.max_position_risk``, ``self.max_portfolio_risk``,
``self.position_correlations``) and therefore must be mixed into the same
concrete class.
"""

import logging
from typing import List

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


class RiskCalculatorMixin:
    """Pure risk-math mixin for ``RiskManager``."""

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
                logger.warning(
                    f"Non-finite prices in correlation calculation: {symbol1}, {symbol2}"
                )
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
