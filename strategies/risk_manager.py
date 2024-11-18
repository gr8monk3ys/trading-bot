import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, max_portfolio_risk=0.02, max_position_risk=0.01, max_correlation=0.7):
        self.max_portfolio_risk = max_portfolio_risk  # 2% max portfolio risk
        self.max_position_risk = max_position_risk    # 1% max position risk
        self.max_correlation = max_correlation        # 0.7 max correlation between positions
        self.position_sizes = {}
        self.position_correlations = {}
        
    def calculate_position_risk(self, symbol, price_history):
        """Calculate position risk using various metrics."""
        try:
            returns = np.diff(price_history) / price_history[:-1]
            
            # Calculate volatility (annualized)
            daily_vol = np.std(returns) * np.sqrt(252)
            
            # Calculate Value at Risk (VaR) - 95% confidence
            var_95 = np.percentile(returns, 5) * np.sqrt(252)
            
            # Calculate Expected Shortfall (ES)
            es_95 = np.mean(returns[returns <= var_95]) * np.sqrt(252)
            
            # Calculate Maximum Drawdown
            rolling_max = np.maximum.accumulate(price_history)
            drawdowns = (price_history - rolling_max) / rolling_max
            max_drawdown = np.min(drawdowns)
            
            # Combine metrics into a risk score (0 to 1)
            risk_score = (
                0.3 * (daily_vol / 0.4) +          # Normalize vol (assuming 40% is high)
                0.3 * (abs(var_95) / 0.03) +       # Normalize VaR
                0.2 * (abs(es_95) / 0.04) +        # Normalize ES
                0.2 * (abs(max_drawdown) / 0.3)    # Normalize drawdown
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
