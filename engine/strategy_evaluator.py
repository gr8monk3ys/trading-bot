import numpy as np
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """
    Evaluator for trading strategies that scores and ranks them based on
    performance metrics, risk-adjusted returns, and consistency.
    """
    
    def __init__(self, min_backtest_days=30, evaluation_period_days=14,
                 weight_sharpe=0.25, weight_returns=0.20, weight_drawdown=0.15,
                 weight_consistency=0.20, weight_win_rate=0.10, weight_profit_factor=0.10):
        """
        Initialize strategy evaluator with weights for different metrics.
        
        Args:
            min_backtest_days: Minimum days required for a valid backtest
            evaluation_period_days: Recent period to weigh more heavily (recency bias)
            weight_sharpe: Weight for Sharpe ratio in overall score
            weight_returns: Weight for returns in overall score  
            weight_drawdown: Weight for drawdown in overall score
            weight_consistency: Weight for consistency metrics
            weight_win_rate: Weight for win rate in overall score
            weight_profit_factor: Weight for profit factor in overall score
        """
        self.min_backtest_days = min_backtest_days
        self.evaluation_period_days = evaluation_period_days
        
        # Scoring weights - must sum to 1.0
        self.weights = {
            'sharpe_ratio': weight_sharpe,
            'returns': weight_returns,
            'drawdown': weight_drawdown,
            'consistency': weight_consistency,
            'win_rate': weight_win_rate,
            'profit_factor': weight_profit_factor
        }
        
        # Validate weights
        total_weight = sum(self.weights.values())
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point errors
            logger.warning(f"Weights don't sum to 1.0: {total_weight}")
            # Normalize weights
            for k in self.weights:
                self.weights[k] /= total_weight
    
    def score_strategy(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate an overall score for a strategy based on its performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
            
        Returns:
            Score between 0 and 1, where higher is better
        """
        if not metrics:
            return 0.0
            
        try:
            # Score each component
            component_scores = {}
            
            # Sharpe Ratio: >2 is excellent, 1-2 is good, 0-1 is weak, <0 is poor
            sharpe = metrics.get('sharpe_ratio', 0)
            component_scores['sharpe_ratio'] = min(1.0, max(0, sharpe / 3.0)) if sharpe >= 0 else 0
            
            # Returns: Scale based on annualized return
            ann_return = metrics.get('annualized_return', 0)
            component_scores['returns'] = min(1.0, max(0, ann_return / 0.30))  # Cap at 30% annual return
            
            # Drawdown: Lower is better
            max_dd = metrics.get('max_drawdown', 1.0)  # Default to worst case
            component_scores['drawdown'] = max(0, 1.0 - (max_dd / 0.20))  # Scale so 20% drawdown = 0
            
            # Consistency metrics
            win_rate = metrics.get('win_rate', 0)
            component_scores['win_rate'] = min(1.0, max(0, win_rate / 0.65))  # Scale to 65% win rate = 1.0
            
            # Profit factor: >2 is good
            profit_factor = metrics.get('profit_factor', 0)
            component_scores['profit_factor'] = min(1.0, max(0, profit_factor / 2.5))
            
            # Consistency calculation (stability of returns)
            # Use Calmar ratio as a consistency proxy
            calmar = metrics.get('calmar_ratio', 0)
            component_scores['consistency'] = min(1.0, max(0, calmar / 1.5))
            
            # Calc weighted score
            final_score = sum(score * self.weights[metric] for metric, score in component_scores.items())
            
            # Apply penalties
            # Not enough trades
            if metrics.get('trade_count', 0) < 10:
                final_score *= (metrics.get('trade_count', 0) / 10.0)
                
            # Negative returns
            if metrics.get('total_return', 0) < 0:
                final_score *= 0.5  # Significant penalty for losing strategies
                
            logger.debug(f"Strategy score: {final_score:.4f}, Component scores: {component_scores}")
            return max(0, min(1.0, final_score))  # Ensure between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating strategy score: {e}", exc_info=True)
            return 0.0
    
    def evaluate_time_periods(self, metrics_by_period: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate strategy performance across different time periods.
        
        Args:
            metrics_by_period: Dictionary mapping period names to metrics dictionaries
            
        Returns:
            Evaluation results including consistency analysis and trend
        """
        if not metrics_by_period:
            return {'error': 'No data provided'}
            
        try:
            # Calculate score for each period
            scores = {}
            for period, metrics in metrics_by_period.items():
                scores[period] = self.score_strategy(metrics)
                
            # Check for performance consistency across periods
            avg_score = np.mean(list(scores.values()))
            score_std = np.std(list(scores.values()))
            consistency = 1.0 - min(1.0, score_std * 2)  # Lower std = higher consistency
            
            # Calculate performance trend
            # Assuming periods are in chronological order: 'recent' > 'medium' > 'long'
            if 'recent' in scores and 'long' in scores:
                trend = scores['recent'] - scores['long']
            else:
                trend = 0
                
            # Overall evaluation
            evaluation = {
                'scores': scores,
                'average_score': avg_score,
                'consistency': consistency,
                'trend': trend,
                'improving': trend > 0,
                'overall_rating': self._get_rating(avg_score)
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating time periods: {e}", exc_info=True)
            return {'error': str(e)}
            
    def get_optimal_parameters(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find optimal parameters based on multiple backtest results.
        
        Args:
            results: Dictionary mapping parameter sets to backtest results
            
        Returns:
            Dictionary with optimal parameters and scores
        """
        if not results:
            return {'error': 'No results provided'}
            
        try:
            # Calculate score for each parameter set
            parameter_scores = {}
            for params_key, result in results.items():
                if isinstance(result, dict) and 'metrics' in result:
                    metrics = result['metrics']
                    score = self.score_strategy(metrics)
                    parameter_scores[params_key] = {'score': score, 'metrics': metrics}
                    
            # Find best parameter set
            if not parameter_scores:
                return {'error': 'No valid results found'}
                
            best_params_key = max(parameter_scores, key=lambda k: parameter_scores[k]['score'])
            best_score = parameter_scores[best_params_key]['score']
            best_metrics = parameter_scores[best_params_key]['metrics']
            
            # Typically params_key would be a string representation of parameters
            # We assume it's in format "param1=value1,param2=value2,..."
            try:
                param_dict = {}
                param_pairs = best_params_key.split(',')
                for pair in param_pairs:
                    key, value = pair.split('=')
                    param_dict[key.strip()] = self._convert_value(value.strip())

                best_params = param_dict
            except (ValueError, AttributeError) as e:
                # If parsing fails (malformed string), just use the original key
                logger.debug(f"Could not parse params key '{best_params_key}': {e}")
                best_params = best_params_key
                
            return {
                'optimal_parameters': best_params,
                'score': best_score,
                'metrics': best_metrics,
                'all_scores': {k: v['score'] for k, v in parameter_scores.items()}
            }
            
        except Exception as e:
            logger.error(f"Error finding optimal parameters: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _convert_value(self, value_str: str) -> Union[int, float, bool, str]:
        """Convert string parameter values to appropriate types."""
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        else:
            try:
                if '.' in value_str:
                    return float(value_str)
                else:
                    return int(value_str)
            except ValueError:
                return value_str
    
    def _get_rating(self, score: float) -> str:
        """Convert numeric score to letter rating."""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C+'
        elif score >= 0.4:
            return 'C'
        elif score >= 0.3:
            return 'D'
        else:
            return 'F'
