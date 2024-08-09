"""
Trading Strategies Package

Contains trading strategies for the Alpaca trading bot, including:
- Momentum Strategy: Uses MACD, RSI, and ADX to identify trend strength
- Mean Reversion Strategy: Uses Bollinger Bands and RSI for overbought/oversold conditions
- Sentiment Strategy: Uses news sentiment analysis to make trading decisions
"""

from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy

# NOTE: Commenting out broken/experimental strategies per TODO.md Priority 1
# These strategies need to be deleted:
# - SentimentStockStrategy: Uses fake news data (line 347-352)
# - OptionsStrategy: 8 TODOs, not implemented
# from strategies.sentiment_stock_strategy import SentimentStockStrategy
# from strategies.options_strategy import OptionsStrategy
from strategies.risk_manager import RiskManager

__all__ = [
    "BaseStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    # 'SentimentStockStrategy',
    # 'OptionsStrategy',
    "RiskManager",
]
