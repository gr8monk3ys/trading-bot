"""
Trading Strategies Package

Contains trading strategies for the Alpaca trading bot, including:
- Momentum Strategy: Uses MACD, RSI, and ADX to identify trend strength
- Mean Reversion Strategy: Uses Bollinger Bands and RSI for overbought/oversold conditions
- Sentiment Strategy: Uses news sentiment analysis to make trading decisions
"""

from strategies.base_strategy import BaseStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.sentiment_strategy import SentimentStockStrategy as SentimentStrategy
from strategies.sentiment_stock_strategy import SentimentStockStrategy
from strategies.options_strategy import OptionsStrategy
from strategies.risk_manager import RiskManager

__all__ = [
    'BaseStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'SentimentStrategy',
    'SentimentStockStrategy',
    'OptionsStrategy',
    'RiskManager'
]
