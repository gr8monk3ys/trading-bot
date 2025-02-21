from datetime import timedelta
from sentiment_analysis import analyze_sentiment
from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
import os
import numpy as np

class SentimentStockStrategy(BaseStrategy):
    async def initialize(self, **kwargs):
        """Initialize the sentiment strategy."""
        try:
            # Initialize the base strategy
            await super().initialize(**kwargs)
                       
            # Initialize tracking dictionaries
            self.last_trade_dict = {symbol: None for symbol in self.symbols}
            self.sentiment_history = {symbol: [] for symbol in self.symbols}
            
            # Strategy parameters
            self.sentiment_window = self.parameters.get('sentiment_window', 5)

            # Add strategy as subscriber to broker
            self.broker._add_subscriber(self)

            # Initialize current prices
            self.current_prices = {}
            
            # Initialize price history
            self.price_history = {symbol: [] for symbol in self.symbols}

            self.logger.info("Strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}", exc_info=True)
            return False

    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data."""
        try:
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(close_price)
            if len(self.price_history[symbol]) > self.price_history_window:
                self.price_history[symbol].pop(0)

            # Update current price
            self.current_prices[symbol] = close_price

            # Log the symbol and close price
            self.logger.info(f"Received bar data for {symbol}: Close price = {close_price}")

            # Get sentiment score
            sentiment_prob, sentiment = self.get_sentiment(symbol)
            if sentiment_prob is None:
                self.logger.error(f"Could not get sentiment for {symbol}")
                return

            # Get current position and account info
            try:
                positions = await self.broker.get_positions()  # Use self.broker, not self.api
                current_position = next((p for p in positions if p.symbol == symbol), None)
                account = await self.broker.get_account()  # Use self.broker, not self.api
                buying_power = float(account.buying_power)

                self.logger.info(f"Current buying power: ${buying_power:.2f}")

                if current_position:
                    self.logger.info(f"Current position in {symbol}: {current_position.qty} shares")

            except Exception as e:
                self.logger.error(f"Error getting account/position info: {e}")
                return

            # Generate signals based on sentiment and price
            if sentiment == "positive" and sentiment_prob > self.sentiment_threshold:
                if not current_position:  # Only buy if we don't have a position
                    try:
                        # Calculate position size based on available buying power
                        position_value = min(buying_power * self.position_size, buying_power * 0.95)  # Use 95% of buying power max
                        if position_value <= 0:
                            self.logger.warning(f"Insufficient buying power (${buying_power:.2f}) to open position in {symbol}")
                            return

                        quantity = position_value / close_price  # Use close_price from bar data

                        # Create market buy order
                        order = self.create_order(
                            symbol,
                            quantity,
                            'buy',
                            type='market'
                        )

                        await self.broker.submit_order(order) # Use self.broker
                        self.logger.info(f"Placed buy order for {quantity:.2f} shares of {symbol}")
                    except Exception as e:
                        self.logger.error(f"Error placing buy order for {symbol}: {e}")
                else:
                    self.logger.info(f"Already have position in {symbol}, skipping buy")

            elif sentiment == "negative" and sentiment_prob > self.sentiment_threshold and current_position:
                try:
                    # Create market sell order
                    order = self.create_order(
                        symbol,
                        float(current_position.qty),
                        'sell',
                        type='market'
                    )
                    await self.broker.submit_order(order) # Use self.broker
                    self.logger.info(f"Placed sell order for {current_position.qty} shares of {symbol}")
                except Exception as e:
                    self.logger.error(f"Error placing sell order for {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    def get_sentiment(self, symbol):
        """Get sentiment analysis for a symbol."""
        try:
            today = self.get_datetime()
            three_days_prior = today - timedelta(days=3)

            news = self.api.get_news(
                symbol=symbol,
                start=three_days_prior.strftime('%Y-%m-%d'),
                end=today.strftime('%Y-%m-%d')
            )

            if not news:
                self.logger.info(f"No news found for {symbol}")
                return 0, "neutral"

            headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
            self.logger.info(f"Analyzing {len(headlines)} headlines for {symbol}")

            probability, sentiment = analyze_sentiment(headlines)

            # Store sentiment history
            self.sentiment_history[symbol].append((probability, sentiment))
            if len(self.sentiment_history[symbol]) > self.sentiment_window:
                self.sentiment_history[symbol].pop(0)

            self.logger.info(f"Sentiment for {symbol} - Probability: {probability:.3f}, Sentiment: {sentiment}")
            return probability, sentiment

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis for {symbol}: {str(e)}")
            return 0, "neutral"
    
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

        avg_probability = np.mean(weighted_probs)
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]

        return avg_probability, dominant_sentiment

    def calculate_position_size(self, symbol, sentiment_prob):
        """Calculate position size based on sentiment probability, risk metrics, and current portfolio."""
        try:
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)

            # Get current positions for risk calculations
            current_positions = {}
            for pos_symbol in self.symbols:
                position = self.get_position(pos_symbol)
                if position:
                    current_positions[pos_symbol] = {
                        'value': float(position.market_value),
                        'price_history': self.price_history.get(pos_symbol, []),
                        'risk': None  # Will be calculated by risk manager
                    }

            # Base position size
            base_size = portfolio_value * self.position_size

            # Adjust size based on sentiment probability
            sentiment_multiplier = min(sentiment_prob / self.sentiment_threshold, 1.5)
            desired_size = base_size * sentiment_multiplier

            # Get price history for risk calculations
            price_history = np.array(self.price_history.get(symbol, []))
            if len(price_history) < 2:
                self.logger.warning(f"Insufficient price history for {symbol}")
                return None

            # Adjust position size based on risk parameters
            adjusted_size = self.risk_manager.adjust_position_size(
                symbol, desired_size, price_history, current_positions
            )

            return min(adjusted_size, portfolio_value * self.max_position_size)

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return None
    
    async def get_positions(self):
        """Get current positions."""
        try:
            return await self.broker.get_tracked_positions(self)
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}", exc_info=True)
            return []
    
    async def get_last_price(self, symbol):
        """Get the last price for a symbol."""
        try:
            price = await self.broker.get_last_price(symbol)
            if price is None:
                self.logger.error(f"Could not get price for {symbol}")
            return price
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol):
        """Analyze a symbol and generate trading signals.  Now a placeholder."""
        pass

    async def execute_trade(self, symbol, signal):
        """Execute a trade based on the signal. Now a placeholder."""
        pass

    async def on_trading_iteration(self):
        """Main trading logic. Now a placeholder."""
        pass

    def before_market_opens(self):
        """Called before the market opens."""
        self.logger.info("Market is about to open. Preparing for trading...")

        # Update risk parameters
        #self.risk_manager.update_market_conditions() #Removed

        # Clear old data
        for symbol in self.symbols:
            self.sentiment_history[symbol] = []
            #self.price_history[symbol] = [] #Removed

    def before_starting(self):
        """Called before the strategy starts."""
        self.logger.info("Strategy is starting...")

        # Initialize risk manager
        #self.risk_manager.initialize() #Removed

        # Verify trading parameters
        if not self.symbols:
            raise ValueError("No symbols specified for trading")

        # Verify API connection
        try:
            account = self.broker.get_account()
            self.logger.info(f"Connected to account with ${account.cash:.2f} in cash")
        except Exception as e:
            self.logger.error(f"Failed to connect to broker: {e}")
            raise

    def on_abrupt_closing(self):
        """Called when the strategy is abruptly closed."""
        self.logger.warning("Strategy is being abruptly closed...")

        # Close all positions
        for symbol in self.symbols:
            position = self.get_position(symbol)
            if position is not None:
                self.submit_order(
                    symbol=symbol,
                    side="sell",
                    type="market",
                    qty=position.quantity
                )
                self.logger.info(f"Emergency closing position in {symbol}")

    def on_bot_crash(self, error):
        """Called when the bot crashes."""
        self.logger.error(f"Bot crashed: {error}")

        # Attempt to close all positions
        try:
            for symbol in self.symbols:
                position = self.get_position(symbol)
                if position is not None:
                    self.submit_order(
                        symbol=symbol,
                        side="sell",
                        type="market",
                        qty=position.quantity
                    )
                    self.logger.info(f"Emergency closing position in {symbol} after crash")
        except Exception as e:
            self.logger.error(f"Failed to close positions after crash: {e}")

    def get_position(self, symbol):
        """Get current position for a symbol."""
        try:
            return self.api.get_position(symbol)
        except:
            return None
