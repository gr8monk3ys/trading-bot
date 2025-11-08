from datetime import timedelta
import logging
import asyncio
import numpy as np
import pandas as pd
from utils.sentiment_analysis import analyze_sentiment
from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
from brokers.order_builder import OrderBuilder

logger = logging.getLogger(__name__)

class SentimentStockStrategy(BaseStrategy):
    """
    ⚠️  WARNING: DO NOT USE IN PRODUCTION ⚠️

    This strategy is DISABLED and should NOT be used for live or paper trading.

    CRITICAL ISSUES:
    1. Falls back to FAKE news data when API fails (lines 354-359)
    2. Fake headlines produce false sentiment signals
    3. Will generate trades based on fabricated market sentiment
    4. GUARANTEED TO LOSE MONEY with fake data

    This class is kept for reference only. Any attempt to initialize it will raise an error.

    To fix this strategy:
    1. Remove the fake news fallback (lines 354-359)
    2. Properly handle news API failures (return neutral sentiment, skip trade)
    3. Add validation that news data is real before making trading decisions
    4. Test with real Alpaca news API to ensure it works
    """
    async def initialize(self, **kwargs):
        """Initialize the sentiment strategy."""
        # CRITICAL SAFETY: Prevent this strategy from being used
        if not kwargs.get('_allow_broken_strategy', False):
            error_msg = (
                "\n" + "="*80 + "\n"
                "🚨 STRATEGY DISABLED - SentimentStockStrategy 🚨\n"
                "="*80 + "\n"
                "This strategy is DISABLED due to critical bugs:\n"
                "  • Uses FAKE news data as fallback (lines 354-359)\n"
                "  • Generates false sentiment signals\n"
                "  • Will lose money trading on fabricated data\n"
                "\n"
                "This strategy cannot be used until the fake news fallback is removed.\n"
                "See class docstring for details.\n"
                "="*80
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Initialize the base strategy
            await super().initialize(**kwargs)
                       
            # Initialize tracking dictionaries
            self.last_trade_dict = {symbol: None for symbol in self.symbols}
            self.sentiment_history = {symbol: [] for symbol in self.symbols}
            
            # Strategy parameters
            self.sentiment_window = self.parameters.get('sentiment_window', 5)
            self.price_history_window = self.parameters.get('price_history_window', 30)
            self.sentiment_threshold = self.parameters.get('sentiment_threshold', 0.6)
            self.position_size = self.parameters.get('position_size', 0.1)
            self.max_position_size = self.parameters.get('max_position_size', 0.25)
            self.stop_loss = self.parameters.get('stop_loss', 0.02)
            self.take_profit = self.parameters.get('take_profit', 0.05)
            
            # Initialize RSI and SMA parameters
            self.rsi_period = int(self.parameters.get('rsi_period', 14))
            self.rsi_overbought = int(self.parameters.get('rsi_overbought', 70))
            self.rsi_oversold = int(self.parameters.get('rsi_oversold', 30))
            self.short_sma = int(self.parameters.get('short_sma', 20))
            self.long_sma = int(self.parameters.get('long_sma', 50))
            
            # Risk manager initialization
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters.get('max_portfolio_risk', 0.02),
                max_position_risk=self.parameters.get('max_position_risk', 0.01),
                max_correlation=self.parameters.get('max_correlation', 0.7)
            )

            # Add strategy as subscriber to broker
            self.broker._add_subscriber(self)

            # Initialize current prices
            self.current_prices = {}
            
            # Initialize price history
            self.price_history = {symbol: [] for symbol in self.symbols}
            
            # Initialize technical indicators
            self.technical_indicators = {symbol: {} for symbol in self.symbols}

            logger.info("Strategy initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}", exc_info=True)
            return False

    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return
                
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(close_price)
            if len(self.price_history[symbol]) > self.price_history_window:
                self.price_history[symbol].pop(0)

            # Update current price
            self.current_prices[symbol] = close_price

            # Update technical indicators
            self._update_technical_indicators(symbol)

            # Log the symbol and close price
            logger.info(f"Received bar data for {symbol}: Close price = {close_price}")

            # Only proceed with analysis if we have enough data
            if len(self.price_history[symbol]) < self.short_sma:
                logger.debug(f"Not enough price history for {symbol} to calculate indicators")
                return

            # Get sentiment score
            sentiment_prob, sentiment = await self._get_sentiment(symbol)
            if sentiment_prob is None:
                logger.error(f"Could not get sentiment for {symbol}")
                return

            # Analyze if we should make a trade
            await self._analyze_and_trade(symbol, sentiment_prob, sentiment)
            
        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    async def _analyze_and_trade(self, symbol, sentiment_prob, sentiment):
        """Analyze indicators and execute trades if signals align."""
        try:
            # Get current position and account info
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)
            account = await self.broker.get_account()
            buying_power = float(account.buying_power)

            logger.info(f"Current buying power: ${buying_power:.2f}")
            if current_position:
                logger.info(f"Current position in {symbol}: {current_position.qty} shares")

            # Check technical indicators
            rsi_value = self.technical_indicators[symbol].get('rsi', 50)
            sma_signal = self.technical_indicators[symbol].get('sma_signal', 'neutral')
            
            logger.info(f"Technical indicators for {symbol}: RSI = {rsi_value}, SMA Signal = {sma_signal}")

            # Buy signal: positive sentiment + technical confirmation
            buy_signal = (
                sentiment == "positive" and 
                sentiment_prob > self.sentiment_threshold and
                (sma_signal == 'bullish' or (rsi_value < self.rsi_oversold and sma_signal != 'bearish'))
            )
            
            # Sell signal: negative sentiment + technical confirmation
            sell_signal = (
                (sentiment == "negative" and sentiment_prob > self.sentiment_threshold) or
                (rsi_value > self.rsi_overbought and sma_signal == 'bearish')
            )

            # Execute trade based on signals
            if buy_signal and not current_position:
                await self._execute_buy(symbol, buying_power, self.current_prices[symbol])
            elif sell_signal and current_position:
                await self._execute_sell(symbol, current_position)
            else:
                logger.info(f"No trading action for {symbol}")
                
        except Exception as e:
            logger.error(f"Error analyzing and trading {symbol}: {e}", exc_info=True)

    async def _execute_buy(self, symbol, buying_power, current_price):
        """Execute a buy order with proper position sizing."""
        try:
            # Calculate position size based on available buying power
            position_value = min(buying_power * self.position_size, buying_power * 0.95)
            if position_value <= 0:
                logger.warning(f"Insufficient buying power (${buying_power:.2f}) to open position in {symbol}")
                return

            # Calculate quantity (allow fractional shares)
            quantity = position_value / current_price
            if quantity < 0.01:
                logger.warning(f"Calculated quantity ({quantity:.4f}) is less than 0.01 shares for {symbol}")
                return

            # Adjust position size based on risk
            if len(self.price_history[symbol]) >= self.price_history_window:
                # Get current positions for risk calculations
                positions = await self.broker.get_positions()
                current_positions = {}
                for pos in positions:
                    pos_symbol = pos.symbol
                    if pos_symbol in self.price_history:
                        current_positions[pos_symbol] = {
                            'value': float(pos.market_value),
                            'price_history': self.price_history[pos_symbol],
                            'risk': None  # Will be calculated by risk manager
                        }
                
                # Use risk manager to adjust position size
                adjusted_value = self.risk_manager.adjust_position_size(
                    symbol, 
                    position_value, 
                    self.price_history[symbol],
                    current_positions
                )
                
                adjusted_quantity = adjusted_value / current_price
                if adjusted_quantity > 0:
                    quantity = adjusted_quantity
                    position_value = adjusted_value
                    logger.info(f"Risk-adjusted quantity for {symbol}: {quantity:.2f} shares")
                else:
                    logger.warning(f"Risk manager reduced position size to zero for {symbol}")
                    return

            # CRITICAL SAFETY: Enforce maximum position size limit (5% of portfolio)
            position_value, quantity = await self.enforce_position_size_limit(symbol, position_value, current_price)

            # Allow fractional shares (Alpaca minimum is typically 0.01)
            if quantity < 0.01:
                logger.warning(f"Position size too small after limit enforcement for {symbol}")
                return

            # Calculate bracket order levels
            entry_price = current_price
            stop_price = entry_price * (1 - self.stop_loss)      # 2% stop loss
            target_price = entry_price * (1 + self.take_profit)  # 5% take profit

            # Create and submit bracket order for automatic risk management
            # Use fractional shares for precise position sizing
            order = (OrderBuilder(symbol, 'buy', quantity)
                     .market()
                     .bracket(take_profit=target_price, stop_loss=stop_price)
                     .gtc()
                     .build())
            result = await self.broker.submit_order_advanced(order)

            logger.info(f"BUY bracket order placed for {quantity:.4f} shares of {symbol}: {result.id}")
            logger.info(f"  Entry: ${entry_price:.2f}")
            logger.info(f"  Take Profit: ${target_price:.2f} (+{self.take_profit*100:.1f}%)")
            logger.info(f"  Stop Loss: ${stop_price:.2f} (-{self.stop_loss*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error executing buy for {symbol}: {e}", exc_info=True)

    async def _execute_sell(self, symbol, position):
        """Execute a sell order to exit position."""
        try:
            # Create and submit market sell order
            quantity = float(position.qty)
            order = (OrderBuilder(symbol, 'sell', quantity)
                     .market()
                     .day()
                     .build())

            # Submit the order
            result = await self.broker.submit_order_advanced(order)
            logger.info(f"SELL order placed for {quantity} shares of {symbol}: {result.id}")
            
        except Exception as e:
            logger.error(f"Error executing sell for {symbol}: {e}", exc_info=True)

    def _update_technical_indicators(self, symbol):
        """Update technical indicators for a symbol."""
        try:
            if len(self.price_history[symbol]) < self.short_sma:
                return
                
            prices = self.price_history[symbol]
            
            # Calculate RSI
            if len(prices) >= self.rsi_period + 1:
                price_array = np.array(prices)
                delta = np.diff(price_array)
                
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                # First average gain and loss
                avg_gain = np.mean(gain[:self.rsi_period])
                avg_loss = np.mean(loss[:self.rsi_period])
                
                # Rest of the data using smoothed averages
                for i in range(self.rsi_period, len(delta)):
                    avg_gain = (avg_gain * (self.rsi_period - 1) + gain[i]) / self.rsi_period
                    avg_loss = (avg_loss * (self.rsi_period - 1) + loss[i]) / self.rsi_period
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                self.technical_indicators[symbol]['rsi'] = rsi
            
            # Calculate SMAs
            if len(prices) >= self.long_sma:
                short_sma_value = np.mean(prices[-self.short_sma:])
                long_sma_value = np.mean(prices[-self.long_sma:])
                
                self.technical_indicators[symbol]['short_sma'] = short_sma_value
                self.technical_indicators[symbol]['long_sma'] = long_sma_value
                
                # Determine SMA signal
                if short_sma_value > long_sma_value:
                    self.technical_indicators[symbol]['sma_signal'] = 'bullish'
                elif short_sma_value < long_sma_value:
                    self.technical_indicators[symbol]['sma_signal'] = 'bearish'
                else:
                    self.technical_indicators[symbol]['sma_signal'] = 'neutral'
            
        except Exception as e:
            logger.error(f"Error updating technical indicators for {symbol}: {e}", exc_info=True)

    async def _get_sentiment(self, symbol):
        """Get sentiment analysis for a symbol."""
        try:
            today = self.get_datetime()
            three_days_prior = today - timedelta(days=3)

            # Get news from Alpaca API
            news = await self.get_news(symbol, three_days_prior, today)
            
            if not news:
                logger.info(f"No news found for {symbol}")
                return 0, "neutral"

            headlines = [article.get('headline', '') for article in news]
            logger.info(f"Analyzing {len(headlines)} headlines for {symbol}")

            # Analyze sentiment
            probability, sentiment = analyze_sentiment(headlines)

            # Store sentiment history
            self.sentiment_history[symbol].append((probability, sentiment))
            if len(self.sentiment_history[symbol]) > self.sentiment_window:
                self.sentiment_history[symbol].pop(0)

            logger.info(f"Sentiment for {symbol} - Probability: {probability:.3f}, Sentiment: {sentiment}")
            
            # Get aggregated sentiment over time
            agg_prob, agg_sentiment = self.get_aggregated_sentiment(symbol)
            logger.info(f"Aggregated sentiment for {symbol} - Probability: {agg_prob:.3f}, Sentiment: {agg_sentiment}")
            
            return agg_prob, agg_sentiment

        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {str(e)}")
            return None, None
            
    async def get_news(self, symbol, start_date, end_date):
        """Get news for a symbol using Alpaca API."""
        try:
            # Format dates for API request
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Call broker's API to get news
            news_items = await self.broker.get_news(symbol, start_str, end_str)
            
            # Process news items into a consistent format
            processed_news = []
            for item in news_items:
                processed_news.append({
                    'headline': item.headline,
                    'summary': getattr(item, 'summary', ''),
                    'url': getattr(item, 'url', ''),
                    'source': getattr(item, 'source', ''),
                    'timestamp': getattr(item, 'created_at', None)
                })
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            
            # Fallback to sample news for testing purposes
            logger.info("Using fallback sample news for testing")
            return [
                {'headline': f"{symbol} is showing strong performance", 'source': 'test'},
                {'headline': f"Analysts suggest caution on {symbol}", 'source': 'test'}
            ]

    def get_aggregated_sentiment(self, symbol):
        """Calculate aggregated sentiment over the sentiment window."""
        if not self.sentiment_history[symbol]:
            return 0, "neutral"

        recent_sentiments = self.sentiment_history[symbol]

        # Calculate weighted average of probabilities (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(recent_sentiments))
        weighted_probs = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for (prob, sent), weight in zip(recent_sentiments, weights):
            weighted_probs.append(prob * weight)
            sentiment_counts[sent] += 1

        avg_probability = np.mean(weighted_probs) if weighted_probs else 0
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else "neutral"

        return avg_probability, dominant_sentiment

    def create_order(self, symbol, quantity, side, **kwargs):
        """Create an order object for submission."""
        order = {
            "symbol": symbol,
            "quantity": float(quantity),
            "side": side
        }
        
        # Add additional parameters
        order.update(kwargs)
        
        return order
