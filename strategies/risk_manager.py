import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

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
    ):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.volatility_threshold = volatility_threshold
        self.var_threshold = var_threshold
        self.es_threshold = es_threshold
        self.drawdown_threshold = drawdown_threshold
        self.position_sizes = {}
        self.position_correlations = {}

    def _calculate_volatility(self, price_history):
        """Calculate annualized volatility."""
        returns = np.diff(price_history) / price_history[:-1]
        return np.std(returns) * np.sqrt(252)

    def _calculate_var(self, price_history):
        """Calculate Value at Risk (VaR) at 95% confidence level."""
        returns = np.diff(price_history) / price_history[:-1]
        return np.percentile(returns, 5) * np.sqrt(252)

    def _calculate_expected_shortfall(self, price_history):
        """Calculate Expected Shortfall (ES) at 95% confidence level."""
        returns = np.diff(price_history) / price_history[:-1]
        var_95 = self._calculate_var(price_history)
        return np.mean(returns[returns <= var_95]) * np.sqrt(252)

    def _calculate_max_drawdown(self, price_history):
        """Calculate maximum drawdown."""
        rolling_max = np.maximum.accumulate(price_history)
        drawdowns = (price_history - rolling_max) / rolling_max
        return np.min(drawdowns)

    def calculate_position_risk(self, symbol, price_history):
        """Calculate position risk using various metrics."""
        try:
            daily_vol = self._calculate_volatility(price_history)
            var_95 = self._calculate_var(price_history)
            es_95 = self._calculate_expected_shortfall(price_history)
            max_drawdown = self._calculate_max_drawdown(price_history)

            # Combine metrics into a risk score (0 to 1)
            risk_score = (
                0.3 * (daily_vol / self.volatility_threshold)
                + 0.3 * (abs(var_95) / self.var_threshold)
                + 0.2 * (abs(es_95) / self.es_threshold)
                + 0.2 * (abs(max_drawdown) / self.drawdown_threshold)
            )

            return min(risk_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            return 1.0  # Return maximum risk on error

    def calculate_position_correlation(self, symbol1, symbol2, price_history1, price_history2):
        """Calculate correlation between two positions."""
        try:
            if len(price_history1) != len(price_history2):
                min_len = min(len(price_history1), len(price_history2))
                price_history1 = price_history1[-min_len:]
                price_history2 = price_history2[-min_len:]
            
            returns1 = np.diff(price_history1) / price_history1[:-1]
            returns2 = np.diff(price_history2) / price_history2[:-1]
            
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            return abs(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 1.0  # Return maximum correlation on error
            
    def calculate_portfolio_risk(self, positions):
        """Calculate total portfolio risk."""
        try:
            total_risk = 0
            position_weights = []
            
            # Calculate position weights
            total_value = sum(pos['value'] for pos in positions.values())
            for symbol, pos in positions.items():
                weight = pos['value'] / total_value if total_value > 0 else 0
                position_weights.append(weight)
                
                # Add individual position risk
                total_risk += weight * pos.get('risk', self.max_position_risk)
            
            # Add correlation impact
            for i, (sym1, pos1) in enumerate(positions.items()):
                for j, (sym2, pos2) in enumerate(positions.items()):
                    if i < j:
                        corr = self.position_correlations.get((sym1, sym2), 0)
                        total_risk += (position_weights[i] * position_weights[j] * 
                                     pos1.get('risk', self.max_position_risk) * 
                                     pos2.get('risk', self.max_position_risk) * 
                                     corr)
            
            return total_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return self.max_portfolio_risk
            
    def adjust_position_size(self, symbol, desired_size, price_history, current_positions):
        """Adjust position size based on risk parameters."""
        try:
            # Calculate position risk
            risk = self.calculate_position_risk(symbol, price_history)
            
            # Store risk for portfolio calculations
            if symbol in current_positions:
                current_positions[symbol]['risk'] = risk
            
            # Calculate correlations with existing positions
            max_correlation = 0
            for other_symbol, pos in current_positions.items():
                if other_symbol != symbol and 'price_history' in pos:
                    correlation = self.calculate_position_correlation(
                        symbol, other_symbol, price_history, pos['price_history']
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
                correlation_adjustment *= self.max_correlation / max_correlation
            
            # Calculate portfolio risk impact
            portfolio_risk = self.calculate_portfolio_risk(current_positions)
            portfolio_adjustment = 1.0
            if portfolio_risk > self.max_portfolio_risk:
                portfolio_adjustment *= self.max_portfolio_risk / portfolio_risk
            
            # Apply all adjustments
            adjusted_size = desired_size * min(
                risk_adjustment,
                correlation_adjustment,
                portfolio_adjustment
            )
            
            return max(adjusted_size, 0)  # Ensure non-negative position size
            
        except Exception as e:
            logger.error(f"Error adjusting position size for {symbol}: {e}")
            return 0  # Return 0 size on error
