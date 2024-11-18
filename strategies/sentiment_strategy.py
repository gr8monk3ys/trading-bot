"""
Sentiment-based stock trading strategy.
"""

import logging
from datetime import datetime, timedelta
import numpy as np
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SentimentStockStrategy(BaseStrategy):
    """Trading strategy based on sentiment analysis."""
    
    def __init__(self, name=None, broker=None, budget=None, parameters=None):
        """Initialize sentiment strategy."""
        super().__init__(name=name, broker=broker, budget=budget, parameters=parameters)
        self.symbols = parameters.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'])
        self.position_data = {}
        self.last_analysis_time = {}
        
    def before_market_opens(self):
        """Prepare for market open."""
        logger.info("Preparing for market open...")
        for symbol in self.symbols:
            self.last_analysis_time[symbol] = datetime.min
            
    def before_starting(self):
        """Initialize before starting."""
        logger.info("Initializing sentiment strategy...")
        self.update_position_data()
        
    def update_position_data(self):
        """Update position data for all symbols."""
        for symbol in self.symbols:
            position = self.get_position(symbol)
            self.position_data[symbol] = {
                'position': position,
                'quantity': float(position['quantity']) if position else 0,
                'entry_price': float(position['avg_price']) if position else 0
            }
            
    async def on_trading_iteration(self):
        """Main trading logic."""
        try:
            self.update_position_data()
            
            for symbol in self.symbols:
                # Skip if analyzed recently
                if (datetime.now() - self.last_analysis_time.get(symbol, datetime.min)) < timedelta(minutes=5):
                    continue
                    
                # TODO: Implement sentiment analysis
                sentiment_score = 0.5  # Placeholder
                
                position_data = self.position_data.get(symbol, {})
                current_position = position_data.get('position')
                
                if sentiment_score > self.sentiment_threshold and not current_position:
                    # Bullish signal - consider buying
                    self._handle_buy_signal(symbol, sentiment_score)
                elif sentiment_score < (1 - self.sentiment_threshold) and current_position:
                    # Bearish signal - consider selling
                    self._handle_sell_signal(symbol, sentiment_score)
                    
                self.last_analysis_time[symbol] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}", exc_info=True)
            
    def _handle_buy_signal(self, symbol, sentiment_score):
        """Handle buy signal."""
        try:
            # Calculate position size based on sentiment strength
            sentiment_strength = (sentiment_score - self.sentiment_threshold) / (1 - self.sentiment_threshold)
            position_size = min(self.position_size * sentiment_strength, self.max_position_size)
            
            # Calculate quantity based on budget and position size
            quote = self.get_last_price(symbol)
            if not quote:
                return
                
            cash = self.get_cash()
            max_shares = int((cash * position_size) / quote)
            
            if max_shares > 0:
                logger.info(f"Placing buy order for {max_shares} shares of {symbol}")
                self.create_order(
                    symbol,
                    max_shares,
                    "buy",
                    type="market"
                )
        except Exception as e:
            logger.error(f"Error handling buy signal for {symbol}: {e}", exc_info=True)
            
    def _handle_sell_signal(self, symbol, sentiment_score):
        """Handle sell signal."""
        try:
            position_data = self.position_data.get(symbol, {})
            quantity = position_data.get('quantity', 0)
            
            if quantity > 0:
                logger.info(f"Placing sell order for {quantity} shares of {symbol}")
                self.create_order(
                    symbol,
                    quantity,
                    "sell",
                    type="market"
                )
        except Exception as e:
            logger.error(f"Error handling sell signal for {symbol}: {e}", exc_info=True)
            
    def after_market_closes(self):
        """Clean up after market closes."""
        logger.info("Market closed, performing cleanup...")
        self.position_data = {}
        self.last_analysis_time = {}
