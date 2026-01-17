import logging
import pandas as pd
from datetime import timedelta

# Set up logging
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies using historical data.
    """
    
    def __init__(self, broker=None):
        """
        Initialize the backtest engine.
        
        Args:
            broker: The broker instance to use for market data. If None, create a new one.
        """
        self.broker = broker
        self.current_date = None
        self.strategies = []
        self.results = {}
        
    async def run(self, strategies, start_date, end_date):
        """
        Run backtest for strategies over the given period.
        
        Args:
            strategies: List of strategy instances to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            List of result DataFrames, one per strategy
        """
        self.strategies = strategies
        self.current_date = start_date
        
        # Initialize result tracking for each strategy
        results = []
        for strategy in strategies:
            # Create daily results dataframe with date index
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            result_df = pd.DataFrame(index=dates, columns=[
                'equity', 'cash', 'holdings', 'returns', 'trades'
            ])
            result_df['trades'] = 0
            results.append(result_df)
        
        # Initialize strategies with broker
        for strategy in strategies:
            if not hasattr(strategy, 'broker'):
                strategy.broker = self.broker
        
        # Run backtest day by day
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() > 4:  # Saturday = 5, Sunday = 6
                current_date += timedelta(days=1)
                continue
                
            logger.debug(f"Processing date: {current_date.date()}")
            self.current_date = current_date
            
            # Process each strategy for this day
            for i, strategy in enumerate(strategies):
                try:
                    # Run one iteration of the strategy
                    await self._run_strategy_iteration(strategy, current_date)
                    
                    # Record daily results
                    result_df = results[i]
                    if current_date in result_df.index:
                        portfolio_value = self.broker.get_portfolio_value(current_date)
                        cash = self.broker.get_balance()
                        holdings = portfolio_value - cash
                        
                        result_df.loc[current_date, 'equity'] = portfolio_value
                        result_df.loc[current_date, 'cash'] = cash
                        result_df.loc[current_date, 'holdings'] = holdings
                        
                        # Calculate daily returns
                        if current_date != start_date:
                            prev_date = current_date - timedelta(days=1)
                            while prev_date not in result_df.index and prev_date >= start_date:
                                prev_date = prev_date - timedelta(days=1)
                                
                            if prev_date in result_df.index and result_df.loc[prev_date, 'equity'] > 0:
                                prev_equity = result_df.loc[prev_date, 'equity']
                                result_df.loc[current_date, 'returns'] = (portfolio_value / prev_equity) - 1
                except Exception as e:
                    logger.error(f"Error in strategy {strategy.__class__.__name__} on {current_date.date()}: {e}")
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Post-process results to fill missing values and calculate metrics
        for i, result_df in enumerate(results):
            # Forward fill equity values for non-trading days
            result_df.fillna(method='ffill', inplace=True)
            
            # Calculate cumulative returns
            result_df['cum_returns'] = (1 + result_df['returns'].fillna(0)).cumprod() - 1
            
            # Calculate additional metrics
            strategy_name = strategies[i].__class__.__name__
            self._calculate_performance_metrics(result_df, strategy_name)
        
        return results
        
    async def _run_strategy_iteration(self, strategy, current_date):
        """Run a single iteration of a strategy for the given date."""
        # Call the strategy's on_trading_iteration method if it exists
        if hasattr(strategy, 'on_trading_iteration'):
            strategy.current_date = current_date  # Set current date for the strategy
            strategy.on_trading_iteration()
            
    def _calculate_performance_metrics(self, result_df, strategy_name):
        """Calculate performance metrics for a strategy."""
        # Skip if not enough data
        if len(result_df) < 2:
            return
            
        # Calculate daily, monthly, and annual returns
        result_df['daily_returns'] = result_df['returns']
        
        # Calculate drawdowns
        result_df['peak'] = result_df['equity'].cummax()
        result_df['drawdown'] = (result_df['equity'] / result_df['peak']) - 1
        
        # Maximum drawdown
        max_drawdown = result_df['drawdown'].min()
        
        # Annualized return (assuming 252 trading days in a year)
        days = (result_df.index[-1] - result_df.index[0]).days
        if days > 0:
            years = days / 365
            total_return = result_df['cum_returns'].iloc[-1]
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
            
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        if result_df['daily_returns'].std() > 0:
            sharpe_ratio = (result_df['daily_returns'].mean() / result_df['daily_returns'].std()) * (252 ** 0.5)
        else:
            sharpe_ratio = 0
            
        # Add metrics to dataframe
        result_df.attrs['strategy'] = strategy_name
        result_df.attrs['annualized_return'] = annualized_return
        result_df.attrs['max_drawdown'] = max_drawdown
        result_df.attrs['sharpe_ratio'] = sharpe_ratio
        
        logger.info(f"Strategy {strategy_name} - Annualized Return: {annualized_return:.2%}, "
                   f"Max Drawdown: {max_drawdown:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
