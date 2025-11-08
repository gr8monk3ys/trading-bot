#!/usr/bin/env python3
"""
Strategy Tester - Simplified testing tool for trading strategies

This script allows testing of trading strategies without requiring the full broker integration.
It creates a simulated environment for evaluating strategy performance using historical data.
"""

import os
import sys
import logging
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from engine.performance_metrics import PerformanceMetrics
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy

# Import any other strategies you want to test
# from strategies.sentiment_stock_strategy import SentimentStockStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"strategy_test_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)

class MockBroker:
    """Mock broker for strategy testing without needing full broker implementation."""
    
    def __init__(self):
        self.positions = {}
        self.account_value = 100000  # Default starting capital
        self.cash = 100000
        self.orders = []
        self.subscribers = set()
        
    async def get_account(self):
        """Return mock account info."""
        class Account:
            def __init__(self, equity, cash):
                self.equity = equity
                self.cash = cash
                self.buying_power = cash
                
        return Account(self.account_value, self.cash)
    
    async def get_positions(self):
        """Return mock positions."""
        positions = []
        for symbol, pos in self.positions.items():
            class Position:
                def __init__(self, symbol, qty, price):
                    self.symbol = symbol
                    self.qty = qty
                    self.market_value = qty * price
                    self.avg_entry_price = price
                    
            positions.append(Position(symbol, pos['quantity'], pos['price']))
        return positions
    
    async def submit_order(self, order):
        """Process a mock order."""
        self.orders.append(order)
        
        # Update positions based on order
        symbol = order['symbol']
        quantity = order['quantity']
        side = order['side']
        price = order.get('price', 100.0)  # Default price for market orders
        
        if side == 'buy':
            # Add to position
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': quantity, 'price': price}
            else:
                current_qty = self.positions[symbol]['quantity']
                current_price = self.positions[symbol]['price']
                
                # Calculate new average price
                total_qty = current_qty + quantity
                new_price = (current_qty * current_price + quantity * price) / total_qty
                
                self.positions[symbol] = {'quantity': total_qty, 'price': new_price}
            
            # Update cash
            order_value = quantity * price
            self.cash -= order_value
            
        elif side == 'sell':
            # Reduce or remove position
            if symbol in self.positions:
                current_qty = self.positions[symbol]['quantity']
                
                if quantity >= current_qty:
                    # Sell entire position
                    del self.positions[symbol]
                else:
                    # Partial sell
                    self.positions[symbol]['quantity'] -= quantity
                
                # Update cash
                order_value = quantity * price
                self.cash += order_value
        
        # Update account value
        position_value = sum(pos['quantity'] * pos['price'] for pos in self.positions.values())
        self.account_value = self.cash + position_value
        
        return True
    
    async def get_historical_data(self, symbol, timeframe, start_date, end_date):
        """Get historical price data for backtesting."""
        # Generate mock data or load from CSV files
        try:
            # Try to load data from a CSV file if available
            filename = f"data/{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, parse_dates=['timestamp'])
                return df
            
            # If no file exists, generate synthetic data
            days = (end_date - start_date).days + 1
            dates = [start_date + timedelta(days=x) for x in range(days)]
            
            # Generate random price series with some trend and volatility
            base_price = 100.0
            trend = np.random.uniform(-0.2, 0.2)  # Random trend between -0.2% and 0.2% per day
            volatility = np.random.uniform(0.01, 0.03)  # Random volatility between 1% and 3%
            
            prices = [base_price]
            for i in range(1, days):
                prev_price = prices[i-1]
                change = prev_price * (trend + np.random.normal(0, volatility))
                new_price = max(prev_price + change, 0.01)  # Ensure price doesn't go negative
                prices.append(new_price)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],  # High is up to 2% above open
                'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],   # Low is up to 2% below open
                'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices], # Close is normally distributed around open
                'volume': [int(np.random.uniform(100000, 1000000)) for _ in prices]  # Random volume
            })
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save to CSV for future use
            df.to_csv(filename, index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating historical data for {symbol}: {e}")
            # Return empty DataFrame
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    async def get_market_status(self):
        """Return mock market status."""
        return {'is_open': True}
    
    def _add_subscriber(self, subscriber):
        """Add a subscriber for updates."""
        self.subscribers.add(subscriber)

class BacktestSimulator:
    """Simulator for backtesting trading strategies."""
    
    def __init__(self, strategy_class, symbols, start_date, end_date, 
                 initial_capital=100000, timeframe='1D'):
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self.broker = MockBroker()
        self.broker.account_value = initial_capital
        self.broker.cash = initial_capital
        
        # Performance tracking
        self.daily_returns = []
        self.portfolio_values = []
        self.trades = []
        self.positions = {}
        
    async def run_backtest(self, strategy_params=None):
        """Run the backtest simulation."""
        try:
            # Initialize strategy
            strategy = self.strategy_class(broker=self.broker, symbols=self.symbols)
            
            # Set strategy parameters
            params = strategy.default_parameters()
            if strategy_params:
                params.update(strategy_params)
                
            # Initialize strategy
            await strategy.initialize(**params)
            
            # Get historical data for each symbol
            historical_data = {}
            for symbol in self.symbols:
                df = await self.broker.get_historical_data(
                    symbol, self.timeframe, self.start_date, self.end_date
                )
                historical_data[symbol] = df
            
            # Set the historical data in the strategy
            strategy.current_data = historical_data
            
            # Iterate through each day in the backtest period
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            
            for date in dates:
                # Skip weekends
                if date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                    continue
                    
                # Get data for current date
                current_data = {}
                for symbol, df in historical_data.items():
                    df_date = df[df['timestamp'].dt.date == date.date()]
                    if not df_date.empty:
                        current_data[symbol] = df_date
                
                if not current_data:
                    continue
                
                # Generate signals
                await strategy.generate_signals()
                
                # Get orders from strategy
                orders = strategy.get_orders()
                
                # Execute orders
                for order in orders:
                    # Get price from current data
                    symbol = order['symbol']
                    if symbol in current_data and not current_data[symbol].empty:
                        # Use closing price for the day
                        price = float(current_data[symbol]['close'].iloc[-1])
                        order['price'] = price
                        
                    # Submit order to broker
                    await self.broker.submit_order(order)
                    
                    # Track trade
                    self.trades.append({
                        'date': date.date(),
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'quantity': order['quantity'],
                        'price': order.get('price', 0.0)
                    })
                
                # Update portfolio value
                account = await self.broker.get_account()
                self.portfolio_values.append({
                    'date': date.date(),
                    'value': account.equity
                })
                
                # Calculate daily return
                if len(self.portfolio_values) > 1:
                    prev_value = self.portfolio_values[-2]['value']
                    curr_value = self.portfolio_values[-1]['value']
                    daily_return = (curr_value - prev_value) / prev_value
                    self.daily_returns.append({
                        'date': date.date(),
                        'return': daily_return
                    })
            
            # Calculate performance metrics
            performance = self._calculate_performance()
            
            return {
                'trades': self.trades,
                'portfolio_value': self.portfolio_values,
                'daily_returns': self.daily_returns,
                'performance': performance
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}", exc_info=True)
            return None
    
    def _calculate_performance(self):
        """Calculate performance metrics from backtest results."""
        if not self.portfolio_values:
            return {}
            
        # Extract portfolio values and convert to pandas Series
        dates = [pv['date'] for pv in self.portfolio_values]
        values = [pv['value'] for pv in self.portfolio_values]
        portfolio_series = pd.Series(values, index=dates)
        
        # Extract daily returns
        daily_returns = [dr['return'] for dr in self.daily_returns]
        
        # Calculate metrics
        total_return = (portfolio_series.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return (252 trading days in a year)
        days = len(portfolio_series)
        if days > 1:
            annualized_return = ((1 + total_return) ** (252 / days)) - 1
        else:
            annualized_return = 0
            
        # Volatility (annualized)
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)
        else:
            volatility = 0
            
        # Sharpe ratio (using 0% as risk-free rate for simplicity)
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0
            
        # Maximum drawdown
        max_dd = 0
        peak = portfolio_series.iloc[0]
        for value in portfolio_series:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        # Win rate and profit factor
        wins = [t for t in self.trades if 
                (t['side'] == 'buy' and t['price'] < portfolio_series.iloc[-1]) or
                (t['side'] == 'sell' and t['price'] > portfolio_series.iloc[-1])]
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        # Calculate P&L for each trade (simplified)
        pnl = []
        for t in self.trades:
            if t['side'] == 'buy':
                pnl.append((portfolio_series.iloc[-1] - t['price']) * t['quantity'])
            else:  # sell
                pnl.append((t['price'] - portfolio_series.iloc[-1]) * t['quantity'])
                
        gross_profit = sum([p for p in pnl if p > 0])
        gross_loss = abs(sum([p for p in pnl if p < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Average trade
        avg_trade = np.mean(pnl) if pnl else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(self.trades),
            'avg_trade': avg_trade
        }

async def test_strategy(args):
    """Run strategy test with given parameters."""
    try:
        # Parse dates
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        
        # Get symbols
        symbols = args.symbols.split(',') if args.symbols else ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Define the strategy mappings
        strategy_classes = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            # 'sentiment': SentimentStockStrategy,
            # Add other strategies here
        }
        
        if args.strategy not in strategy_classes:
            print(f"Strategy '{args.strategy}' not available. Choose from: {list(strategy_classes.keys())}")
            return
            
        strategy_class = strategy_classes[args.strategy]
        print(f"\nTesting {args.strategy} strategy from {start_date} to {end_date}")
        
        # Run backtest
        simulator = BacktestSimulator(
            strategy_class=strategy_class,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital
        )
        
        # Parse strategy parameters if provided
        strategy_params = None
        if args.params:
            try:
                strategy_params = json.loads(args.params)
            except json.JSONDecodeError:
                print("Invalid JSON format for strategy parameters. Using defaults.")
        
        result = await simulator.run_backtest(strategy_params=strategy_params)
        
        if not result:
            print("Backtest failed. Check logs for details.")
            return
            
        # Print performance summary
        perf = result['performance']
        print("\nPerformance Summary:")
        print(f"Total Return: {perf['total_return']:.2%}")
        print(f"Annualized Return: {perf['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
        print(f"Win Rate: {perf['win_rate']:.2%}")
        print(f"Profit Factor: {perf['profit_factor']:.2f}")
        print(f"Number of Trades: {perf['num_trades']}")
        
        # Plot equity curve if requested
        if args.plot:
            try:
                plt.figure(figsize=(12, 6))
                
                # Convert portfolio values to DataFrame
                portfolio_df = pd.DataFrame(result['portfolio_value'])
                plt.plot(portfolio_df['date'], portfolio_df['value'])
                
                plt.title(f"{args.strategy.capitalize()} Strategy Equity Curve")
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True)
                
                # Save and show plot
                plt.savefig(f"{args.strategy}_equity_curve.png")
                print(f"\nEquity curve saved as {args.strategy}_equity_curve.png")
                
                # Also mark buy/sell points if there are trades
                if result['trades']:
                    trades_df = pd.DataFrame(result['trades'])
                    buys = trades_df[trades_df['side'] == 'buy']
                    sells = trades_df[trades_df['side'] == 'sell']
                    
                    # Get portfolio value on trade dates
                    trade_values = {}
                    for trade in result['trades']:
                        date = trade['date']
                        # Find portfolio value on this date
                        for pv in result['portfolio_value']:
                            if pv['date'] == date:
                                trade_values[date] = pv['value']
                                break
                    
                    # Plot buy and sell points
                    for _, buy in buys.iterrows():
                        if buy['date'] in trade_values:
                            plt.scatter(buy['date'], trade_values[buy['date']], 
                                       marker='^', color='green', s=100)
                    
                    for _, sell in sells.iterrows():
                        if sell['date'] in trade_values:
                            plt.scatter(sell['date'], trade_values[sell['date']], 
                                       marker='v', color='red', s=100)
                    
                    plt.savefig(f"{args.strategy}_equity_curve_with_trades.png")
                    print(f"Equity curve with trades saved as {args.strategy}_equity_curve_with_trades.png")
                
            except Exception as e:
                print(f"Error generating plot: {e}")
        
    except Exception as e:
        print(f"Error testing strategy: {e}")

def compare_strategies(args):
    """Compare multiple trading strategies."""
    async def run_comparison():
        try:
            # Parse dates
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
            
            # Get symbols
            symbols = args.symbols.split(',') if args.symbols else ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            
            # Define the strategy mappings
            strategy_classes = {
                'momentum': MomentumStrategy,
                'mean_reversion': MeanReversionStrategy,
                # 'sentiment': SentimentStockStrategy,
                # Add other strategies here
            }
            
            # Define strategies to compare
            strategies_to_compare = args.strategies.split(',') if args.strategies else list(strategy_classes.keys())
            
            # Validate strategies
            valid_strategies = [s for s in strategies_to_compare if s in strategy_classes]
            if not valid_strategies:
                print(f"No valid strategies provided. Choose from: {list(strategy_classes.keys())}")
                return
                
            # Run backtest for each strategy
            results = {}
            
            for strategy_name in valid_strategies:
                print(f"\nTesting {strategy_name} strategy...")
                
                strategy_class = strategy_classes[strategy_name]
                
                simulator = BacktestSimulator(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=args.capital
                )
                
                result = await simulator.run_backtest()
                
                if result:
                    results[strategy_name] = result
                    
                    # Print performance summary
                    perf = result['performance']
                    print(f"Total Return: {perf['total_return']:.2%}")
                    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
                    print(f"Number of Trades: {perf['num_trades']}")
            
            # Compare strategies
            if len(results) > 1:
                # Create comparison table
                comparison = pd.DataFrame({
                    name: {
                        'Total Return': result['performance']['total_return'],
                        'Annualized Return': result['performance']['annualized_return'],
                        'Sharpe Ratio': result['performance']['sharpe_ratio'],
                        'Max Drawdown': result['performance']['max_drawdown'],
                        'Win Rate': result['performance']['win_rate'],
                        'Profit Factor': result['performance']['profit_factor'],
                        'Number of Trades': result['performance']['num_trades']
                    }
                    for name, result in results.items()
                }).T
                
                # Format the table
                print("\n--- Strategy Comparison ---")
                print(comparison.to_string(float_format=lambda x: f"{x:.2%}" if x < 10 else f"{x:.2f}"))
                
                # Plot equity curves if requested
                if args.plot:
                    try:
                        plt.figure(figsize=(12, 6))
                        
                        for name, result in results.items():
                            portfolio_df = pd.DataFrame(result['portfolio_value'])
                            plt.plot(portfolio_df['date'], portfolio_df['value'], label=name)
                        
                        plt.title('Strategy Comparison - Equity Curves')
                        plt.xlabel('Date')
                        plt.ylabel('Portfolio Value ($)')
                        plt.legend()
                        plt.grid(True)
                        
                        # Save plot
                        plt.savefig('strategy_comparison.png')
                        print("\nStrategy comparison plot saved as strategy_comparison.png")
                        
                    except Exception as e:
                        print(f"Error generating plot: {e}")
                        
        except Exception as e:
            print(f"Error comparing strategies: {e}")
    
    asyncio.run(run_comparison())

def main():
    """Main entry point for the strategy tester."""
    parser = argparse.ArgumentParser(description='Trading Strategy Tester')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a single strategy')
    test_parser.add_argument('strategy', help='Strategy to test (e.g., momentum, mean_reversion)')
    test_parser.add_argument('--start-date', default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                            help='Start date for test (YYYY-MM-DD)')
    test_parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'),
                            help='End date for test (YYYY-MM-DD)')
    test_parser.add_argument('--symbols', default=None,
                            help='Comma-separated list of symbols to trade')
    test_parser.add_argument('--capital', type=float, default=100000,
                            help='Initial capital for test')
    test_parser.add_argument('--params', default=None,
                            help='JSON string with strategy parameters')
    test_parser.add_argument('--plot', action='store_true',
                            help='Generate a plot of the equity curve')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple strategies')
    compare_parser.add_argument('--strategies', default=None,
                              help='Comma-separated list of strategies to compare')
    compare_parser.add_argument('--start-date', default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                              help='Start date for test (YYYY-MM-DD)')
    compare_parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'),
                              help='End date for test (YYYY-MM-DD)')
    compare_parser.add_argument('--symbols', default=None,
                              help='Comma-separated list of symbols to trade')
    compare_parser.add_argument('--capital', type=float, default=100000,
                              help='Initial capital for test')
    compare_parser.add_argument('--plot', action='store_true',
                              help='Generate a plot comparing equity curves')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        asyncio.run(test_strategy(args))
    elif args.command == 'compare':
        compare_strategies(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
