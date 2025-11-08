import os
import logging
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from strategies.base_strategy import BaseStrategy
from engine.performance_metrics import PerformanceMetrics
from engine.strategy_evaluator import StrategyEvaluator
from engine.backtest_engine import BacktestEngine
from config import SYMBOLS, ALPACA_CREDS
from brokers.alpaca_broker import AlpacaBroker

# Set up logging
logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Strategy Manager for managing multiple trading strategies, 
    evaluating their performance, and optimizing allocation.
    """
    
    def __init__(self, broker=None, max_strategies=5, max_allocation=0.9, min_backtest_days=30, 
                 evaluation_period_days=14, strategy_path='strategies', broker_type='alpaca'):
        """
        Initialize the Strategy Manager.
        
        Args:
            broker: The broker instance to use. If None, one will be created.
            max_strategies: Maximum number of active strategies to run simultaneously.
            max_allocation: Maximum capital allocation across all strategies (0.0 to 1.0).
            min_backtest_days: Minimum days for backtesting.
            evaluation_period_days: Days to look back for strategy evaluation.
            strategy_path: Path to strategy modules.
            broker_type: Type of broker to use if broker is not provided.
        """
        self.max_strategies = max_strategies
        self.max_allocation = max_allocation
        self.min_backtest_days = min_backtest_days
        self.evaluation_period_days = evaluation_period_days
        self.strategy_path = strategy_path
        
        # Initialize broker
        self.broker = broker
        if self.broker is None:
            if broker_type.lower() == 'alpaca':
                logger.info("Creating Alpaca broker instance")
                paper = ALPACA_CREDS.get("PAPER", True)
                self.broker = AlpacaBroker(paper=paper)
            else:
                raise ValueError(f"Unsupported broker type: {broker_type}")
        
        # Initialize component managers
        self.perf_metrics = PerformanceMetrics()
        self.evaluator = StrategyEvaluator(
            min_backtest_days=self.min_backtest_days,
            evaluation_period_days=self.evaluation_period_days
        )
        self.backtest_engine = BacktestEngine(self.broker)
        
        # Strategy tracking
        self.available_strategies = {}  # name -> class
        self.active_strategies = {}  # name -> instance
        self.strategy_performances = {}  # name -> perf metrics
        self.strategy_allocations = {}  # name -> allocation %
        self.strategy_status = {}  # name -> status
        
        # Load available strategies
        self._load_available_strategies()
        
    def _load_available_strategies(self):
        """Load all available strategy classes from the strategy directory."""
        try:
            import importlib
            import inspect
            
            # Convert relative path to absolute if needed
            if not os.path.isabs(self.strategy_path):
                self.strategy_path = os.path.join(os.getcwd(), self.strategy_path)
                
            logger.info(f"Loading strategies from: {self.strategy_path}")
            
            # Get all Python files in the strategy directory
            strategy_files = [f for f in os.listdir(self.strategy_path) 
                             if f.endswith('.py') and f != '__init__.py' and f != 'base_strategy.py']
            
            for file in strategy_files:
                module_name = file[:-3]  # Remove .py extension
                
                try:
                    # Import the module
                    module_path = f"strategies.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Find all classes in the module that inherit from BaseStrategy
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseStrategy) and obj != BaseStrategy):
                            strategy_name = obj.NAME if hasattr(obj, 'NAME') else name
                            self.available_strategies[strategy_name] = obj
                            logger.info(f"Loaded strategy: {strategy_name}")
                            
                except Exception as e:
                    logger.error(f"Error loading strategy module {module_name}: {e}", exc_info=True)
                    
            logger.info(f"Loaded {len(self.available_strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}", exc_info=True)
    
    async def evaluate_all_strategies(self, symbols=None, lookback_days=None):
        """
        Evaluate all available strategies for the given symbols.
        
        Args:
            symbols: List of symbols to evaluate. If None, use config symbols.
            lookback_days: Days to look back for evaluation. If None, use min_backtest_days.
            
        Returns:
            Dict mapping strategy names to evaluation scores.
        """
        if symbols is None:
            symbols = SYMBOLS
            
        if lookback_days is None:
            lookback_days = self.min_backtest_days
            
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Evaluating all strategies from {start_date} to {end_date} for {len(symbols)} symbols")
        
        scores = {}
        
        for name, strategy_class in self.available_strategies.items():
            try:
                # Run backtest
                result = await self.backtest_engine.run_backtest(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Calculate metrics
                metrics = self.perf_metrics.calculate_metrics(result)
                
                # Score the strategy
                score = self.evaluator.score_strategy(metrics)
                
                # Store results
                self.strategy_performances[name] = metrics
                scores[name] = score
                self.strategy_status[name] = "evaluated"
                
                logger.info(f"Strategy {name} evaluation: Score={score:.4f}, "
                           f"Return={metrics['total_return']:.2%}, "
                           f"Sharpe={metrics['sharpe_ratio']:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluating strategy {name}: {e}", exc_info=True)
                scores[name] = -1.0
                self.strategy_status[name] = "error"
                
        return scores
    
    async def select_top_strategies(self, n=None, min_score=0.5):
        """
        Select top N strategies based on evaluation scores.
        
        Args:
            n: Number of strategies to select. If None, use max_strategies.
            min_score: Minimum score for a strategy to be selected.
            
        Returns:
            List of selected strategy names.
        """
        if n is None:
            n = self.max_strategies
            
        # Get scores
        if not self.strategy_performances:
            await self.evaluate_all_strategies()
            
        # Score all strategies
        scores = {}
        for name, metrics in self.strategy_performances.items():
            score = self.evaluator.score_strategy(metrics)
            scores[name] = score
            
        # Filter by minimum score
        valid_strategies = {name: score for name, score in scores.items() if score >= min_score}
        
        # Sort by score (descending)
        sorted_strategies = sorted(valid_strategies.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N
        selected = [name for name, _ in sorted_strategies[:n]]
        
        logger.info(f"Selected top {len(selected)} strategies: {', '.join(selected)}")
        return selected
    
    async def optimize_allocations(self, strategies=None):
        """
        Optimize capital allocation across selected strategies.
        
        Args:
            strategies: List of strategy names to allocate capital to.
                      If None, use the top strategies.
                      
        Returns:
            Dict of strategy names to allocation percentages.
        """
        if strategies is None:
            strategies = await self.select_top_strategies()
            
        if not strategies:
            logger.warning("No strategies to allocate capital to")
            return {}
            
        # Get performance metrics for selected strategies
        metrics = {name: self.strategy_performances.get(name, {}) for name in strategies}
        
        # Filter out strategies with missing metrics
        valid_metrics = {name: m for name, m in metrics.items() if m}
        
        if not valid_metrics:
            logger.warning("No valid metrics for allocation optimization")
            return {name: self.max_allocation / len(strategies) for name in strategies}
            
        # Simple allocation based on Sharpe ratio
        total_sharpe = sum(m.get('sharpe_ratio', 0.5) for m in valid_metrics.values())
        
        if total_sharpe <= 0:
            # Equal allocation if total Sharpe is negative or zero
            allocations = {name: self.max_allocation / len(valid_metrics) for name in valid_metrics}
        else:
            # Weighted allocation based on Sharpe ratio
            allocations = {
                name: (m.get('sharpe_ratio', 0.5) / total_sharpe) * self.max_allocation 
                for name, m in valid_metrics.items()
            }
            
        # Add zero allocation for strategies without metrics
        for name in strategies:
            if name not in allocations:
                allocations[name] = 0.0
                
        logger.info(f"Optimized allocations: {allocations}")
        self.strategy_allocations = allocations
        return allocations
    
    async def start_strategy(self, strategy_name, parameters=None, symbols=None, allocation=None):
        """
        Start a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to start.
            parameters: Strategy parameters. If None, use defaults.
            symbols: Symbols to trade. If None, use config symbols.
            allocation: Capital allocation (0.0 to 1.0).
                      If None, use optimized allocation.
                      
        Returns:
            Boolean indicating success.
        """
        if strategy_name not in self.available_strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return False
            
        if strategy_name in self.active_strategies:
            logger.warning(f"Strategy {strategy_name} already running")
            return True
            
        # Get allocation
        if allocation is None:
            allocation = self.strategy_allocations.get(strategy_name, 0.1)
            
        # Get symbols
        if symbols is None:
            symbols = SYMBOLS
            
        # Create strategy instance
        try:
            strategy_class = self.available_strategies[strategy_name]
            strategy = strategy_class(broker=self.broker, symbols=symbols)
            
            # Merge default parameters with provided parameters
            merged_params = strategy.default_parameters()
            if parameters:
                merged_params.update(parameters)
                
            # Add allocation to parameters
            merged_params['allocation'] = allocation
            
            # Initialize the strategy
            logger.info(f"Initializing strategy {strategy_name} with allocation {allocation:.2%}")
            success = await strategy.initialize(**merged_params)
            
            if not success:
                logger.error(f"Failed to initialize strategy {strategy_name}")
                return False
                
            # Store the active strategy
            self.active_strategies[strategy_name] = strategy
            self.strategy_status[strategy_name] = "running"
            
            logger.info(f"Started strategy {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting strategy {strategy_name}: {e}", exc_info=True)
            return False
    
    async def stop_strategy(self, strategy_name, liquidate=False):
        """
        Stop a running strategy.
        
        Args:
            strategy_name: Name of the strategy to stop.
            liquidate: Whether to liquidate all positions.
            
        Returns:
            Boolean indicating success.
        """
        if strategy_name not in self.active_strategies:
            logger.warning(f"Strategy {strategy_name} not running")
            return False
            
        try:
            strategy = self.active_strategies[strategy_name]
            
            # Liquidate positions if requested
            if liquidate:
                await strategy.liquidate_all_positions()
                
            # Stop the strategy
            await strategy.shutdown()
            
            # Update tracking
            del self.active_strategies[strategy_name]
            self.strategy_status[strategy_name] = "stopped"
            
            logger.info(f"Stopped strategy {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping strategy {strategy_name}: {e}", exc_info=True)
            return False
    
    async def start_selected_strategies(self, n=None, min_score=0.5):
        """
        Start the top N strategies.
        
        Args:
            n: Number of strategies to start. If None, use max_strategies.
            min_score: Minimum score for a strategy to be selected.
            
        Returns:
            List of started strategy names.
        """
        # Select top strategies
        selected = await self.select_top_strategies(n=n, min_score=min_score)
        
        # Optimize allocations
        allocations = await self.optimize_allocations(selected)
        
        # Start strategies
        started = []
        for name in selected:
            if await self.start_strategy(name, allocation=allocations.get(name)):
                started.append(name)
                
        logger.info(f"Started {len(started)} strategies: {', '.join(started)}")
        return started
    
    async def stop_all_strategies(self, liquidate=False):
        """
        Stop all running strategies.
        
        Args:
            liquidate: Whether to liquidate all positions.
            
        Returns:
            Number of strategies stopped.
        """
        stopped = 0
        running_strategies = list(self.active_strategies.keys())
        
        for name in running_strategies:
            if await self.stop_strategy(name, liquidate=liquidate):
                stopped += 1
                
        logger.info(f"Stopped {stopped} strategies")
        return stopped
    
    async def rebalance_strategies(self):
        """
        Rebalance capital allocation across active strategies.
        
        Returns:
            Boolean indicating success.
        """
        if not self.active_strategies:
            logger.warning("No active strategies to rebalance")
            return False
            
        # Re-optimize allocations
        allocations = await self.optimize_allocations(list(self.active_strategies.keys()))
        
        # Update allocations
        for name, strategy in self.active_strategies.items():
            new_allocation = allocations.get(name, 0.0)
            strategy.update_parameters(allocation=new_allocation)
            logger.info(f"Updated allocation for {name}: {new_allocation:.2%}")
            
        return True
    
    async def get_strategy_info(self, strategy_name=None):
        """
        Get information about strategies.
        
        Args:
            strategy_name: Specific strategy to get info for. If None, get all.
            
        Returns:
            Dict with strategy information.
        """
        if strategy_name:
            # Get info for specific strategy
            if strategy_name not in self.available_strategies:
                return {"error": f"Strategy {strategy_name} not found"}
                
            status = self.strategy_status.get(strategy_name, "unknown")
            allocation = self.strategy_allocations.get(strategy_name, 0.0)
            performance = self.strategy_performances.get(strategy_name, {})
            
            return {
                "name": strategy_name,
                "status": status,
                "allocation": allocation,
                "performance": performance,
                "is_active": strategy_name in self.active_strategies
            }
        else:
            # Get info for all strategies
            all_info = {}
            for name in self.available_strategies:
                status = self.strategy_status.get(name, "unknown")
                allocation = self.strategy_allocations.get(name, 0.0)
                
                all_info[name] = {
                    "status": status,
                    "allocation": allocation,
                    "is_active": name in self.active_strategies
                }
                
            return all_info
    
    def get_available_strategy_names(self):
        """Get list of all available strategy names."""
        return list(self.available_strategies.keys())
    
    def get_active_strategy_names(self):
        """Get list of active strategy names."""
        return list(self.active_strategies.keys())
    
    async def get_portfolio_stats(self):
        """
        Get overall portfolio statistics.
        
        Returns:
            Dict with portfolio statistics.
        """
        try:
            # Get account info
            account = await self.broker.get_account()
            
            # Get positions
            positions = await self.broker.get_positions()
            
            # Calculate total value
            portfolio_value = float(account.equity)
            cash = float(account.cash)
            
            # Calculate position values by strategy
            position_values = {}
            for name, strategy in self.active_strategies.items():
                tracked_positions = await self.broker.get_tracked_positions(strategy)
                strategy_value = sum(float(p.market_value) for p in tracked_positions)
                position_values[name] = strategy_value
                
            # Calculate realized and unrealized P&L
            today_date = datetime.now().date()
            yesterday_date = today_date - timedelta(days=1)
            
            # Get positions and trades
            daily_pnl = {}
            
            return {
                "portfolio_value": portfolio_value,
                "cash": cash,
                "equity": portfolio_value,
                "buying_power": float(account.buying_power),
                "position_count": len(positions),
                "strategy_allocations": self.strategy_allocations,
                "active_strategies": len(self.active_strategies),
                "position_values": position_values
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {e}", exc_info=True)
            return {"error": str(e)}
            
    async def generate_performance_report(self, days=30):
        """
        Generate a performance report.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            Dict with performance report data.
        """
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        report = {
            "period": f"{start_date} to {end_date}",
            "days": days,
            "strategies": {},
            "portfolio": {}
        }
        
        # Get current portfolio stats
        portfolio_stats = await self.get_portfolio_stats()
        report["portfolio"]["current"] = portfolio_stats
        
        # Get strategy performance
        for name, metrics in self.strategy_performances.items():
            report["strategies"][name] = {
                "metrics": metrics,
                "allocation": self.strategy_allocations.get(name, 0.0),
                "status": self.strategy_status.get(name, "unknown")
            }
            
        return report
