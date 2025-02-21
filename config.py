import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpaca credentials configuration
ALPACA_CREDS = {
   "API_KEY": os.getenv("ALPACA_API_KEY_ID", "AKF8LBJTECAUWTFSXQAY"),
    "API_SECRET": os.getenv("ALPACA_SECRET_KEY", "YmuQW3vIwDQ9hjM7JbLbz8dUqzH7UAeNdqbeCbBW"),
    "PAPER": str(os.getenv("PAPER", "True")).lower() == "true"  # Ensure string comparison
}

# Trading symbols - Reduced for testing
SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
]

# Trading parameters
TRADING_PARAMS = {
    "SENTIMENT_THRESHOLD": 0.6,
    "POSITION_SIZE": 0.10,  # 10% of portfolio
    "MAX_POSITION_SIZE": 0.25,  # 25% maximum position size
    "STOP_LOSS": 0.02,  # 2% stop loss
    "TAKE_PROFIT": 0.05,  # 5% take profit
}

# Risk management parameters
RISK_PARAMS = {
    "PORTFOLIO_RISK_LIMIT": 0.02,  # 2% portfolio-wide risk limit
    "POSITION_RISK_LIMIT": 0.01,   # 1% individual position risk limit
    "MAX_CORRELATION": 0.7,        # Maximum position correlation
    "VAR_CONFIDENCE": 0.95,        # Value at Risk confidence level
}

# Technical analysis parameters
TECHNICAL_PARAMS = {
    "SHORT_SMA": 20,  # Short Simple Moving Average period
    "LONG_SMA": 50,   # Long Simple Moving Average period
    "RSI_PERIOD": 14, # Relative Strength Index period
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
}
