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

# Trading symbols - Default list (used if dynamic selection is disabled)
SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta
    "TSLA",  # Tesla
]

# Dynamic Symbol Selection (NEW - WORKING!)
SYMBOL_SELECTION = {
    "USE_DYNAMIC_SELECTION": False,  # DISABLED for initial paper trading - use static list
    "TOP_N_SYMBOLS": 20,  # Number of stocks to trade (more = more diversification)
    "MIN_MOMENTUM_SCORE": 1.0,  # Minimum 5-day price movement % to consider
    "RESCAN_INTERVAL_HOURS": 24,  # Rescan market every N hours for new opportunities
    # SimpleSymbolSelector finds:
    # - High volume stocks (>1M shares/day) - easier to trade
    # - Reasonable prices ($10-$500) - not penny stocks, not too expensive
    # - Active movers - stocks with momentum
    # Result: ~20 liquid, tradable stocks with real price action
}

# Trading parameters
TRADING_PARAMS = {
    "INTERVAL": 60,  # 60 seconds between checks
    "SENTIMENT_THRESHOLD": 0.6,
    "POSITION_SIZE": 0.10,  # 10% of portfolio (used if Kelly is disabled)
    "MAX_POSITION_SIZE": 0.25,  # 25% maximum position size
    "STOP_LOSS": 0.02,  # 2% stop loss
    "TAKE_PROFIT": 0.05,  # 5% take profit

    # Kelly Criterion position sizing (optimal leverage)
    "USE_KELLY_CRITERION": False,  # Set to True to enable Kelly-based position sizing
    "KELLY_FRACTION": 0.5,  # 0.5 = Half Kelly (conservative), 0.25 = Quarter Kelly, 1.0 = Full Kelly
    "KELLY_MIN_TRADES": 30,  # Minimum trades before trusting Kelly calculation
    "KELLY_LOOKBACK": 50,  # Use last N trades for Kelly calculation
    "MIN_POSITION_SIZE": 0.01,  # Minimum 1% position size

    # Volatility Regime Detection (adaptive risk management)
    "USE_VOLATILITY_REGIME": False,  # Set to True to enable VIX-based position/stop adjustments
    # Regimes: Very Low (VIX<12), Low (12-15), Normal (15-20), Elevated (20-30), High (>30)
    # Very Low: +40% position, -30% stop | High: -60% position, +50% stop

    # Streak-Based Position Sizing (dynamic sizing based on recent performance)
    "USE_STREAK_SIZING": False,  # Set to True to enable streak-based position adjustments
    "STREAK_LOOKBACK": 10,  # Number of recent trades to analyze for streak detection
    "HOT_STREAK_THRESHOLD": 7,  # Wins needed (out of lookback) for hot streak
    "COLD_STREAK_THRESHOLD": 3,  # Max wins (out of lookback) for cold streak
    "HOT_MULTIPLIER": 1.2,  # Position size multiplier during hot streaks (+20%)
    "COLD_MULTIPLIER": 0.7,  # Position size multiplier during cold streaks (-30%)
    "STREAK_RESET_AFTER": 5,  # Reset to baseline after N trades in same streak

    # Multi-Timeframe Analysis (trend confirmation across timeframes)
    "USE_MULTI_TIMEFRAME": False,  # Set to True to enable multi-timeframe analysis
    "MTF_MIN_CONFIDENCE": 0.70,  # Minimum confidence (0.0-1.0) to enter trade
    "MTF_REQUIRE_DAILY_ALIGNMENT": True,  # Daily timeframe must not contradict signal
    # Analyzes 5Min, 15Min, 1Hour, 1Day timeframes
    # Expected improvement: +8-12% win rate, -30-40% reduction in false signals
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
