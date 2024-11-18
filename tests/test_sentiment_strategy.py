import os
import logging
from datetime import datetime, timedelta
import yfinance as yf
from sentiment_analysis import analyze_sentiment
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSentimentStrategy:
    def __init__(self, symbols, sentiment_threshold=0.6):
        self.symbols = symbols
        self.sentiment_threshold = sentiment_threshold
        self.sentiment_history = {symbol: [] for symbol in symbols}
    
    def get_market_data(self, symbol):
        """Get current market data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return hist.iloc[-1]['Close']
            return None
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def analyze_symbol(self, symbol, headlines):
        """Analyze a symbol using sentiment analysis"""
        try:
            probability, sentiment = analyze_sentiment(headlines)
            
            # Store sentiment
            self.sentiment_history[symbol].append((probability, sentiment))
            if len(self.sentiment_history[symbol]) > 5:  # Keep last 5 readings
                self.sentiment_history[symbol].pop(0)
            
            return probability, sentiment
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return 0, "neutral"

def test_sentiment_strategy():
    # Test symbols - major tech companies that often have news coverage
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'META']
    
    # Initialize strategy
    strategy = SimpleSentimentStrategy(symbols)
    
    logger.info("Starting sentiment strategy test...")
    logger.info(f"Testing with symbols: {symbols}")
    
    # Test sentiment analysis for each symbol
    for symbol in symbols:
        logger.info(f"\nAnalyzing {symbol}...")
        
        # Get current market data
        current_price = strategy.get_market_data(symbol)
        if current_price:
            logger.info(f"Current price for {symbol}: ${current_price:.2f}")
        
        # Get some recent headlines for testing
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if news:
            headlines = [item['title'] for item in news[:5]]  # Get last 5 headlines
            logger.info(f"Found {len(headlines)} recent headlines for {symbol}")
            
            # Analyze sentiment
            probability, sentiment = strategy.analyze_symbol(symbol, headlines)
            logger.info(f"Sentiment for {symbol}: {sentiment} (probability: {probability:.3f})")
            
            # Determine trading action
            if probability >= strategy.sentiment_threshold:
                action = "buy" if sentiment == "positive" else "sell" if sentiment == "negative" else "hold"
                logger.info(f"Recommended action for {symbol}: {action}")
                
                # Calculate theoretical position metrics
                if current_price and action in ["buy", "sell"]:
                    # Example position size calculation (10% of $100,000 portfolio)
                    position_value = 10000  # $10,000 position size
                    shares = int(position_value / current_price)
                    
                    stop_loss = current_price * 0.98  # 2% stop loss
                    take_profit = current_price * 1.05  # 5% take profit
                    
                    logger.info(f"Theoretical position: {shares} shares at ${current_price:.2f}")
                    logger.info(f"Stop Loss: ${stop_loss:.2f}")
                    logger.info(f"Take Profit: ${take_profit:.2f}")
            else:
                logger.info("No action recommended (low confidence)")
        else:
            logger.info(f"No recent news found for {symbol}")

def main():
    try:
        test_sentiment_strategy()
        logger.info("\nStrategy test completed successfully!")
    except Exception as e:
        logger.error(f"Error during strategy test: {str(e)}")

if __name__ == "__main__":
    main()
