"""
RiskManager enforcer mixin.

Position-sizing, limit enforcement, and margin-monitoring methods used by
``RiskManager``:

    - ``adjust_position_size`` — risk-/correlation-/portfolio-aware sizing
    - ``enforce_limits`` — strict per-order limit enforcement with audit
      trail of violations
    - ``get_risk_summary`` — pre-trade summary of risk metrics vs thresholds
    - ``calculate_margin_requirement`` — required margin + per-position
      liquidation-price analysis
    - ``check_margin_status`` — overall margin level + halt/warn/ok decision
    - ``get_liquidation_prices`` — per-position liquidation price calc
    - ``should_halt_for_margin`` — boolean halt-flag for CircuitBreaker
      integration

These methods rely on calculator-mixin helpers (``_calculate_var``,
``_calculate_expected_shortfall``, ``_calculate_volatility``,
``_calculate_max_drawdown``, ``calculate_position_correlation``,
``calculate_position_risk``, ``calculate_portfolio_risk``) and on attributes
initialized by the ``RiskManager`` facade (``self.max_portfolio_risk``,
``self.max_position_risk``, ``self.max_correlation``,
``self.volatility_threshold``, ``self.var_threshold``,
``self.es_threshold``, ``self.drawdown_threshold``,
``self.strict_correlation_enforcement``, ``self.position_correlations``).
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class RiskEnforcerMixin:
    """Position-sizing, limit-enforcement, and margin-monitoring mixin."""

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
        margin_analysis = self.calculate_margin_requirement(positions, broker_margin_requirement)

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
