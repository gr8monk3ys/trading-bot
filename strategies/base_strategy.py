from abc import ABC, abstractmethod
import logging
import asyncio
from lumibot.strategies import Strategy
import numpy as np

logger = logging.getLogger(__name__)

class BaseStrategy(Strategy):
    def __init__(self, name=None, broker=None, parameters=None):
        """Initialize the strategy."""
        name = name or self.__class__.__name__
        parameters = parameters or {}
        
        # Initialize parent class with broker
        super().__init__(name=name, broker=broker)
        
        # Initialize our parameters
        self.parameters = parameters
        self.interval = parameters.get('interval', 60)  # Default to 60 seconds
        self.symbols = parameters.get('symbols', [])
        self._shutdown_event = asyncio.Event()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.price_history = {} #added

    async def initialize(self, **kwargs):
        """Initialize strategy parameters."""
        try:
            # Update parameters
            self.parameters.update(kwargs)
            
            # Set up strategy parameters
            self.interval = self.parameters.get('interval', 60)
            self.symbols = self.parameters.get('symbols', [])
            
            # Initialize any other strategy-specific parameters
            await self._initialize_parameters()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}", exc_info=True)
            return False

    async def _initialize_parameters(self):
        """Initialize strategy-specific parameters. Override in subclass."""
        self.sentiment_threshold = self.parameters.get('sentiment_threshold', 0.6)
        self.position_size = self.parameters.get('position_size', 0.1)
        self.max_position_size = self.parameters.get('max_position_size', 0.25)
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.05)
        self.portfolio_risk_limit = self.parameters.get('portfolio_risk_limit', 0.02)
        self.position_risk_limit = self.parameters.get('position_risk_limit', 0.01)
        self.max_correlation = self.parameters.get('max_correlation', 0.7)
        self.var_confidence = self.parameters.get('var_confidence', 0.95)
        self.price_history_window = self.parameters.get('price_history_window', 30)
        self.volatility_threshold = self.parameters.get('volatility_threshold', 0.4)
        self.var_threshold = self.parameters.get('var_threshold', 0.03)
        self.es_threshold = self.parameters.get('es_threshold', 0.04)
        self.drawdown_threshold = self.parameters.get('drawdown_threshold', 0.3)


    async def on_trading_iteration(self):
        """Main trading logic. Must be implemented by subclasses."""
        raise NotImplementedError

    def before_market_opens(self):
        """Actions to take before market opens."""
        pass

    def before_starting(self):
        """Initialize strategy before starting."""
        pass

    def after_market_closes(self):
        """Actions to take after market closes."""
        pass

    def on_abrupt_closing(self):
        """Handle abrupt closings."""
        pass

    def trace_stats(self, context, snapshot_before):
        """Record strategy statistics."""
        pass

    def get_parameters(self):
        """Get strategy parameters."""
        return self.parameters

    def set_parameters(self, parameters):
        """Set strategy parameters."""
        self.parameters = parameters
        self._initialize_parameters()

    def on_bot_crash(self, error):
        """Called when the bot crashes."""
        self.logger.error(f"Bot crashed: {error}")

    async def cleanup(self):
        """Cleanup resources."""
        self.running = False
        tasks = [t for t in self.tasks if not t.done()]
        if tasks:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self):
        """Run the strategy."""
        try:
            while not self._shutdown_event.is_set():
                # Get current positions
                positions = await self.get_positions()
                
                # Update stop losses for existing positions
                for position in positions:
                    await self._update_stop_loss(position)

                # Get trading signals for each symbol
                for symbol in self.symbols:
                    try:
                        signal = await self.get_signal(symbol)
                        if signal:
                            await self.execute_trade(symbol, signal)
                    except Exception as e:
                        logger.error(f"Error processing signal for {symbol}: {e}", exc_info=True)
                
                # Sleep before next iteration
                await asyncio.sleep(self.interval)
                
        except Exception as e:
            logger.error(f"Error in strategy {self.__class__.__name__}: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def backtest(self, *args, **kwargs):
        """Run backtesting."""
        try:
            self.running = True
            await super().backtest(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in backtesting {self.name}: {e}")
            raise
        finally:
            await self.cleanup()

    def initialize(self, symbols=None, cash_at_risk=0.5, max_positions=3, 
                  stop_loss_pct=0.05, take_profit_pct=0.20, max_drawdown=0.15):
        self.symbols = symbols or []
        self.cash_at_risk = cash_at_risk
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.positions_dict = {}

        # Performance tracking
        self.trades_made = 0
        self.successful_trades = 0
        self.total_profit_loss = 0
        self.peak_portfolio_value = self.portfolio_value
        self.current_drawdown = 0

    @abstractmethod
    async def analyze_symbol(self, symbol):
        """Analyze a symbol and return trading signals."""
        pass

    @abstractmethod
    def execute_trade(self, symbol, signal):
        """Execute a trade based on the signal."""
        pass

    def create_order(self, symbol, quantity, side, type="market", limit_price=None, stop_price=None):
        """
        Create an order object.

        Args:
            symbol (str): The symbol to trade.
            quantity (float): The quantity to trade.
            side (str): 'buy' or 'sell'.
            type (str): 'market', 'limit', or 'stop'.
            limit_price (float, optional): The limit price for limit orders.
            stop_price (float, optional): The stop price for stop orders.

        Returns:
            dict: The order object.
        """
        order = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "type": type,
        }
        if limit_price:
            order["limit_price"] = limit_price
        if stop_price:
            order["stop_price"] = stop_price
        return order

    def check_risk_limits(self):
        """Check if any risk limits have been breached."""
        try:
            current_value = self.portfolio_value
            self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
            self.current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value

            if self.current_drawdown > self.max_drawdown:
                self.logger.warning(f"Maximum drawdown limit reached: {self.current_drawdown:.2%}")
                return False

            current_positions = len(self.get_positions())
            if current_positions >= self.max_positions:
                self.logger.warning(f"Maximum positions limit reached: {current_positions}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in risk limit check: {str(e)}")
            return False

    def position_sizing(self, symbol, strategy_type="stock"):
        """Calculate position size using Kelly Criterion and volatility."""
        try:
            cash = self.get_cash()
            last_price = self.get_last_price(symbol)

            # Calculate historical volatility
            historical_data = self.get_historical_prices(symbol, 30)
            returns = np.diff(np.log(historical_data))
            volatility = np.std(returns) * np.sqrt(252)

            # Adjust position size based on volatility
            volatility_scalar = 1 / (1 + volatility)

            # Apply Kelly Criterion with safety factor
            win_rate = self.successful_trades / max(1, self.trades_made)
            kelly_fraction = max(0.0, (win_rate - (1 - win_rate)) / 1)
            safe_kelly = kelly_fraction * 0.5  # Half-Kelly for safety

            # Adjust sizing based on strategy type
            if strategy_type == "option":
                # More conservative sizing for options
                safe_kelly *= 0.5

            risk_adjusted_cash = cash * self.cash_at_risk * safe_kelly * volatility_scalar
            quantity = round(risk_adjusted_cash / last_price, 0)

            self.logger.info(f"Position sizing for {symbol} - Cash: {cash}, Quantity: {quantity}, Kelly: {safe_kelly:.2f}")
            return cash, last_price, quantity

        except Exception as e:
            self.logger.error(f"Error in position sizing for {symbol}: {str(e)}")
            return cash, last_price, 0
    async def _update_stop_loss(self, position):
        """Update the stop-loss level for a position based on volatility."""
        try:
            symbol = position.symbol
            current_price = position.current_price  # Assuming Position object has current_price
            volatility = self._calculate_volatility(symbol)

            # Example: Set stop-loss at 2 standard deviations below the current price
            stop_loss = current_price - (2 * volatility * current_price)

            #TODO: compare to the parameter and take the greater of the two.

            self.logger.info(f"Updating stop-loss for {symbol} to {stop_loss:.2f}")

            #TODO: Implement logic to update the stop-loss order with the broker

        except Exception as e:
            self.logger.error(f"Error updating stop-loss for {symbol}: {e}", exc_info=True)

    def _calculate_volatility(self, symbol):
        """Calculate the historical volatility for a symbol."""
        try:
            # Assuming self.price_history is available and populated by the strategy
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.price_history_window:
                self.logger.warning(f"Insufficient price history for {symbol} to calculate volatility")
                return 0  # Or some default value

            prices = np.array(self.price_history[symbol])
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualize
            return volatility

        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}", exc_info=True)
            return 0  # Or some default value

    def update_performance_metrics(self, trade_result, symbol):
        """Update performance tracking metrics after each trade."""
        try:
            self.trades_made += 1
            if trade_result > 0:
                self.successful_trades += 1
            self.total_profit_loss += trade_result

            win_rate = self.successful_trades / self.trades_made
            avg_profit_loss = self.total_profit_loss / self.trades_made

            self.logger.info(f"Performance metrics for {symbol} - Win rate: {win_rate:.2%}, Avg P/L: {avg_profit_loss:.2f}")

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    async def shutdown(self):
        """Shutdown the strategy."""
        self._shutdown_event.set()
        await self.cleanup()
