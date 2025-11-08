import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpaca credentials configuration
ALPACA_CREDS = {
    "API_KEY": os.environ.get("ALPACA_API_KEY", ""),
    "API_SECRET": os.environ.get("ALPACA_SECRET_KEY", ""),
    "PAPER": str(os.environ.get("PAPER", "True")).lower() == "true"  # Ensure string comparison
}

# Trading symbols - Reduced for testing
SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta
    "TSLA",  # Tesla
]

# Trading parameters
TRADING_PARAMS = {
    "INTERVAL": 60,  # 60 seconds between checks
    "SENTIMENT_THRESHOLD": 0.6,
    "POSITION_SIZE": 0.10,  # 10% of portfolio
    "MAX_POSITION_SIZE": 0.25,  # 25% maximum position size
    "STOP_LOSS": 0.02,  # 2% stop loss
    "TAKE_PROFIT": 0.05,  # 5% take profit
}

# Risk management parameters
RISK_PARAMS = {
    "MAX_PORTFOLIO_RISK": 0.02,  # 2% portfolio-wide risk limit
    "MAX_POSITION_RISK": 0.01,   # 1% individual position risk limit
    "MAX_CORRELATION": 0.7,        # Maximum position correlation
    "VAR_CONFIDENCE": 0.95,        # Value at Risk confidence level
}

# Technical analysis parameters
TECHNICAL_PARAMS = {
    "SENTIMENT_WINDOW": 5,       # Number of sentiment data points to consider
    "PRICE_HISTORY_WINDOW": 30,  # Number of price data points to retain
    "SHORT_SMA": 20,             # Short Simple Moving Average period
    "LONG_SMA": 50,              # Long Simple Moving Average period
    "RSI_PERIOD": 14,            # Relative Strength Index period
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
}
