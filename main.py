#!/usr/bin/env python3
"""
Alpaca Trading Bot - Main Entry Point

This script initializes and runs the Alpaca trading bot with multiple strategies.
It handles strategy evaluation, selection, and execution based on market conditions.
"""

import os
import sys
import logging
import asyncio
import argparse
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to path to allow imports from project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.strategy_manager import StrategyManager
from engine.performance_metrics import PerformanceMetrics
from engine.backtest_engine import BacktestEngine
from brokers.alpaca_broker import AlpacaBroker
from utils.stock_scanner import StockScanner
from config import SYMBOLS, ALPACA_CREDS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)

async def run_backtest(args):
    """Run backtest mode with selected strategies."""
    try:
        logger.info(f"Starting backtest from {args.start_date} to {args.end_date}")
        
        # Initialize broker
        broker = AlpacaBroker(paper=True)
        
        # Initialize strategy manager
        strategy_manager = StrategyManager(broker=broker)
        
        # Get available strategies
        available_strategies = strategy_manager.get_available_strategy_names()
        logger.info(f"Available strategies: {available_strategies}")
        
        # Select strategies to backtest
        strategies_to_test = []
        if args.strategy == "all":
            strategies_to_test = available_strategies
        elif args.strategy in available_strategies:
            strategies_to_test = [args.strategy]
        else:
            logger.error(f"Strategy '{args.strategy}' not found")
            return
            
        # Convert date strings to datetime objects
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        
        # Run backtests
        results = {}
        metrics = {}
        
        for strategy_name in strategies_to_test:
            try:
                logger.info(f"Backtesting strategy: {strategy_name}")
                
                # Get strategy class
                strategy_class = strategy_manager.available_strategies[strategy_name]
                
                # Run backtest
                result = await strategy_manager.backtest_engine.run_backtest(
                    strategy_class=strategy_class,
                    symbols=args.symbols.split(',') if args.symbols else SYMBOLS,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=args.capital
                )
                
                # Calculate metrics
                strategy_metrics = strategy_manager.perf_metrics.calculate_metrics(result)
                
                # Store results
                results[strategy_name] = result
                metrics[strategy_name] = strategy_metrics
                
                # Print summary
                print(f"\n--- {strategy_name} Performance Summary ---")
                print(f"Total Return: {strategy_metrics['total_return']:.2%}")
                print(f"Annualized Return: {strategy_metrics['annualized_return']:.2%}")
                print(f"Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {strategy_metrics['max_drawdown']:.2%}")
                print(f"Win Rate: {strategy_metrics['win_rate']:.2%}")
                print(f"Average Win: {strategy_metrics['avg_win']:.2%}")
                print(f"Average Loss: {strategy_metrics['avg_loss']:.2%}")
                print(f"Profit Factor: {strategy_metrics['profit_factor']:.2f}")
                print(f"Number of Trades: {strategy_metrics['num_trades']}")
                
            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {e}", exc_info=True)
        
        # Compare strategies if more than one was tested
        if len(metrics) > 1:
            print("\n--- Strategy Comparison ---")
            comparison = pd.DataFrame({
                k: {
                    'Total Return': v['total_return'],
                    'Annualized Return': v['annualized_return'],
                    'Sharpe Ratio': v['sharpe_ratio'],
                    'Max Drawdown': v['max_drawdown'],
                    'Win Rate': v['win_rate'],
                    'Profit Factor': v['profit_factor'],
                    'Number of Trades': v['num_trades']
                }
                for k, v in metrics.items()
            }).T
            
            pd.set_option('display.float_format', '{:.2%}'.format)
            numeric_cols = ['Sharpe Ratio', 'Profit Factor', 'Number of Trades']
            for col in numeric_cols:
                comparison[col] = pd.to_numeric(comparison[col])
                pd.set_option('display.float_format', '{:.2f}'.format)
            
            print(comparison)
            
            # Generate plots if requested
            if args.plot:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Plot equity curves
                    plt.figure(figsize=(12, 6))
                    for name, result in results.items():
                        plt.plot(result['portfolio_value'], label=name)
                    
                    plt.title('Equity Curves')
                    plt.xlabel('Date')
                    plt.ylabel('Portfolio Value')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig('backtest_equity_curves.png')
                    plt.close()
                    
                    logger.info("Generated equity curve plot: backtest_equity_curves.png")
                    
                except Exception as e:
                    logger.error(f"Error generating plots: {e}")
        
        logger.info("Backtest completed")
        
    except Exception as e:
        logger.error(f"Error in backtest mode: {e}", exc_info=True)

async def run_live(args):
    """Run live trading mode."""
    try:
        logger.info("Starting live trading mode")
        
        # Initialize broker
        paper = not args.real
        broker = AlpacaBroker(paper=paper)
        
        # Check if market is open
        market_status = await broker.get_market_status()
        logger.info(f"Market status: {market_status}")
        
        if not market_status.get('is_open', False) and not args.force:
            logger.warning("Market is closed. Use --force to run anyway.")
            print("Market is closed. Use --force to run anyway.")
            return
            
        # Initialize strategy manager
        strategy_manager = StrategyManager(
            broker=broker, 
            max_strategies=args.max_strategies,
            max_allocation=args.max_allocation
        )
        
        # Get available strategies
        available_strategies = strategy_manager.get_available_strategy_names()
        logger.info(f"Available strategies: {available_strategies}")
        
        # Select strategies to run
        strategies_to_run = []
        if args.strategy == "auto":
            # Auto-select the best strategies
            logger.info("Auto-selecting the best strategies...")
            await strategy_manager.evaluate_all_strategies()
            strategies_to_run = await strategy_manager.select_top_strategies(
                n=args.max_strategies,
                min_score=args.min_score
            )
        elif args.strategy == "all":
            strategies_to_run = available_strategies
        elif args.strategy in available_strategies:
            strategies_to_run = [args.strategy]
        else:
            logger.error(f"Strategy '{args.strategy}' not found")
            return
            
        logger.info(f"Selected strategies: {strategies_to_run}")
        
        # Optimize allocations if auto-allocation
        if args.auto_allocate:
            allocations = await strategy_manager.optimize_allocations(strategies_to_run)
            logger.info(f"Optimized allocations: {allocations}")
        else:
            # Equal allocation
            equal_alloc = args.max_allocation / len(strategies_to_run)
            allocations = {s: equal_alloc for s in strategies_to_run}
            logger.info(f"Equal allocations: {allocations}")
        
        # Start strategies
        started = []
        for strategy_name in strategies_to_run:
            allocation = allocations.get(strategy_name, 0.1)
            success = await strategy_manager.start_strategy(
                strategy_name=strategy_name,
                allocation=allocation,
                symbols=args.symbols.split(',') if args.symbols else SYMBOLS
            )
            if success:
                started.append(strategy_name)
                
        if not started:
            logger.error("Failed to start any strategies")
            return
            
        logger.info(f"Started {len(started)} strategies: {started}")
        
        # Set up periodic tasks
        async def periodic_evaluation():
            """Run periodic strategy evaluation and rebalancing."""
            try:
                while True:
                    # Wait for the next evaluation period
                    await asyncio.sleep(args.evaluation_interval * 3600)  # Convert hours to seconds
                    
                    logger.info("Running periodic strategy evaluation and rebalancing")
                    
                    # Re-evaluate strategies
                    if args.strategy == "auto":
                        scores = await strategy_manager.evaluate_all_strategies()
                        logger.info(f"Updated strategy scores: {scores}")
                        
                        # Re-optimize allocations
                        if args.auto_allocate:
                            allocations = await strategy_manager.optimize_allocations()
                            logger.info(f"Updated allocations: {allocations}")
                            
                            # Rebalance
                            await strategy_manager.rebalance_strategies()
                    
                    # Generate performance report
                    report = await strategy_manager.generate_performance_report(days=7)
                    logger.info(f"Weekly performance report: {report}")
                    
            except Exception as e:
                logger.error(f"Error in periodic evaluation: {e}", exc_info=True)
        
        # Start periodic evaluation task
        evaluation_task = asyncio.create_task(periodic_evaluation())
        
        # Keep running until interrupted
        try:
            logger.info("Trading bot running. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        finally:
            # Cancel evaluation task
            evaluation_task.cancel()
            
            # Stop all strategies
            await strategy_manager.stop_all_strategies(liquidate=args.liquidate_on_exit)
            logger.info("All strategies stopped")
            
    except Exception as e:
        logger.error(f"Error in live trading mode: {e}", exc_info=True)

async def optimize_parameters(args):
    """Optimize strategy parameters."""
    try:
        logger.info(f"Starting parameter optimization for strategy: {args.strategy}")
        
        # Initialize broker
        broker = AlpacaBroker(paper=True)
        
        # Initialize strategy manager
        strategy_manager = StrategyManager(broker=broker)
        
        # Get available strategies
        available_strategies = strategy_manager.get_available_strategy_names()
        
        if args.strategy not in available_strategies:
            logger.error(f"Strategy '{args.strategy}' not found")
            return
            
        # Get strategy class
        strategy_class = strategy_manager.available_strategies[args.strategy]
        
        # Create strategy instance to get default parameters
        temp_strategy = strategy_class(broker=broker, symbols=[])
        default_params = temp_strategy.default_parameters()
        
        # Parse parameter ranges
        param_ranges = {}
        if args.param_ranges:
            try:
                param_ranges = json.loads(args.param_ranges)
            except json.JSONDecodeError:
                logger.error("Invalid JSON format for parameter ranges")
                return
        
        # If no ranges specified, use defaults with some variation
        if not param_ranges:
            for param, value in default_params.items():
                # Only optimize numeric parameters
                if isinstance(value, (int, float)) and param != 'allocation':
                    if isinstance(value, int):
                        param_ranges[param] = {
                            'min': max(1, int(value * 0.5)),
                            'max': int(value * 1.5),
                            'step': 1
                        }
                    else:
                        param_ranges[param] = {
                            'min': value * 0.5,
                            'max': value * 1.5,
                            'step': value * 0.1
                        }
        
        logger.info(f"Parameter ranges to optimize: {param_ranges}")
        
        # Convert date strings to datetime objects
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        
        symbols = args.symbols.split(',') if args.symbols else SYMBOLS
        
        # Generate parameter combinations
        import itertools
        
        param_values = {}
        for param, range_info in param_ranges.items():
            min_val = range_info['min']
            max_val = range_info['max']
            step = range_info['step']
            
            if isinstance(min_val, int) and isinstance(max_val, int):
                values = list(range(min_val, max_val + 1, step))
            else:
                # Generate float range
                values = []
                val = min_val
                while val <= max_val:
                    values.append(val)
                    val += step
                    
            param_values[param] = values
        
        # Generate combinations
        param_names = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[p] for p in param_names]))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        # Run backtests for each combination
        results = []
        
        for i, combo in enumerate(combinations):
            params = default_params.copy()
            for j, param in enumerate(param_names):
                params[param] = combo[j]
                
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")
            
            # Run backtest
            result = await strategy_manager.backtest_engine.run_backtest(
                strategy_class=strategy_class,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=args.capital,
                strategy_params=params
            )
            
            # Calculate metrics
            metrics = strategy_manager.perf_metrics.calculate_metrics(result)
            
            # Store results
            results.append({
                'params': params,
                'metrics': metrics
            })
        
        # Find best parameters
        if args.optimize_for == 'sharpe':
            best_result = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
        elif args.optimize_for == 'return':
            best_result = max(results, key=lambda x: x['metrics']['total_return'])
        elif args.optimize_for == 'drawdown':
            best_result = min(results, key=lambda x: x['metrics']['max_drawdown'])
        else:
            best_result = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
        
        # Print results
        print("\n--- Parameter Optimization Results ---")
        print(f"Best parameters for {args.strategy} optimized for {args.optimize_for}:")
        for param, value in best_result['params'].items():
            if param in param_ranges:
                print(f"{param}: {value}")
                
        print("\nPerformance with optimized parameters:")
        print(f"Total Return: {best_result['metrics']['total_return']:.2%}")
        print(f"Annualized Return: {best_result['metrics']['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {best_result['metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {best_result['metrics']['max_drawdown']:.2%}")
        print(f"Win Rate: {best_result['metrics']['win_rate']:.2%}")
        
        # Save results to file
        output_file = f"{args.strategy}_optimized_params.json"
        with open(output_file, 'w') as f:
            json.dump({
                'optimized_params': best_result['params'],
                'performance': best_result['metrics']
            }, f, indent=4, default=str)
            
        logger.info(f"Saved optimized parameters to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in parameter optimization: {e}", exc_info=True)

def main():
    """Entry point for the trading bot."""
    parser = argparse.ArgumentParser(description='Alpaca Trading Bot')
    
    # Mode selection
    parser.add_argument('mode', choices=['live', 'backtest', 'optimize'],
                       help='Operation mode: live trading, backtesting, or parameter optimization')
    
    # Strategy selection
    parser.add_argument('--strategy', default='auto',
                       help='Strategy to use (name, "all", or "auto" for automatic selection)')
    
    # Symbol selection
    parser.add_argument('--symbols', default=None,
                       help='Comma-separated list of symbols to trade')
    
    # Live trading options
    parser.add_argument('--real', action='store_true',
                       help='Use real trading instead of paper trading')
    parser.add_argument('--force', action='store_true',
                       help='Force execution even if market is closed')
    parser.add_argument('--max-strategies', type=int, default=3,
                       help='Maximum number of strategies to run simultaneously')
    parser.add_argument('--max-allocation', type=float, default=0.9,
                       help='Maximum capital allocation (0.0 to 1.0)')
    parser.add_argument('--min-score', type=float, default=0.5,
                       help='Minimum strategy score for selection')
    parser.add_argument('--auto-allocate', action='store_true',
                       help='Automatically allocate capital based on performance')
    parser.add_argument('--evaluation-interval', type=int, default=24,
                       help='Hours between strategy evaluations')
    parser.add_argument('--liquidate-on-exit', action='store_true',
                       help='Liquidate all positions on exit')
    
    # Backtest options
    parser.add_argument('--start-date', default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital for backtest')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots for backtest results')
    
    # Optimization options
    parser.add_argument('--param-ranges', default=None,
                       help='JSON string with parameter ranges to optimize')
    parser.add_argument('--optimize-for', choices=['sharpe', 'return', 'drawdown'], default='sharpe',
                       help='Metric to optimize for')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'live':
            asyncio.run(run_live(args))
        elif args.mode == 'backtest':
            asyncio.run(run_backtest(args))
        elif args.mode == 'optimize':
            asyncio.run(optimize_parameters(args))
    except Exception as e:
        logger.error(f"Error running {args.mode} mode: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
