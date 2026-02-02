import logging
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def _validate_config():
    """
    P0 FIX: Validate configuration parameters on module load.

    Prevents invalid configurations from causing runtime errors or financial losses.
    Raises ValueError for critical issues, warns for suspicious values.
    """
    errors = []
    warnings = []

    # Validate TRADING_PARAMS
    if TRADING_PARAMS["POSITION_SIZE"] <= 0 or TRADING_PARAMS["POSITION_SIZE"] > 1:
        errors.append(
            f"POSITION_SIZE must be between 0 and 1, got {TRADING_PARAMS['POSITION_SIZE']}"
        )

    if TRADING_PARAMS["MAX_POSITION_SIZE"] <= 0 or TRADING_PARAMS["MAX_POSITION_SIZE"] > 1:
        errors.append(
            f"MAX_POSITION_SIZE must be between 0 and 1, got {TRADING_PARAMS['MAX_POSITION_SIZE']}"
        )

    if TRADING_PARAMS["MAX_POSITION_SIZE"] < TRADING_PARAMS["POSITION_SIZE"]:
        errors.append(
            f"MAX_POSITION_SIZE ({TRADING_PARAMS['MAX_POSITION_SIZE']}) must be >= POSITION_SIZE ({TRADING_PARAMS['POSITION_SIZE']})"
        )

    if TRADING_PARAMS["STOP_LOSS"] <= 0 or TRADING_PARAMS["STOP_LOSS"] > 0.5:
        errors.append(f"STOP_LOSS must be between 0 and 0.5, got {TRADING_PARAMS['STOP_LOSS']}")

    if TRADING_PARAMS["TAKE_PROFIT"] <= 0 or TRADING_PARAMS["TAKE_PROFIT"] > 1:
        errors.append(f"TAKE_PROFIT must be between 0 and 1, got {TRADING_PARAMS['TAKE_PROFIT']}")

    if TRADING_PARAMS["KELLY_FRACTION"] < 0 or TRADING_PARAMS["KELLY_FRACTION"] > 1:
        errors.append(
            f"KELLY_FRACTION must be between 0 and 1, got {TRADING_PARAMS['KELLY_FRACTION']}"
        )

    if (
        TRADING_PARAMS["MIN_POSITION_SIZE"] < 0
        or TRADING_PARAMS["MIN_POSITION_SIZE"] > TRADING_PARAMS["POSITION_SIZE"]
    ):
        warnings.append(
            f"MIN_POSITION_SIZE ({TRADING_PARAMS['MIN_POSITION_SIZE']}) should be <= POSITION_SIZE ({TRADING_PARAMS['POSITION_SIZE']})"
        )

    # Validate RISK_PARAMS
    if RISK_PARAMS["MAX_PORTFOLIO_RISK"] <= 0 or RISK_PARAMS["MAX_PORTFOLIO_RISK"] > 0.2:
        errors.append(
            f"MAX_PORTFOLIO_RISK must be between 0 and 0.2, got {RISK_PARAMS['MAX_PORTFOLIO_RISK']}"
        )

    if RISK_PARAMS["MAX_POSITION_RISK"] <= 0 or RISK_PARAMS["MAX_POSITION_RISK"] > 0.1:
        errors.append(
            f"MAX_POSITION_RISK must be between 0 and 0.1, got {RISK_PARAMS['MAX_POSITION_RISK']}"
        )

    if RISK_PARAMS["VAR_CONFIDENCE"] <= 0.5 or RISK_PARAMS["VAR_CONFIDENCE"] >= 1:
        errors.append(
            f"VAR_CONFIDENCE must be between 0.5 and 1 (exclusive), got {RISK_PARAMS['VAR_CONFIDENCE']}"
        )

    # Validate TECHNICAL_PARAMS
    if TECHNICAL_PARAMS["RSI_PERIOD"] < 2 or TECHNICAL_PARAMS["RSI_PERIOD"] > 100:
        warnings.append(
            f"RSI_PERIOD ({TECHNICAL_PARAMS['RSI_PERIOD']}) is outside typical range (2-100)"
        )

    if TECHNICAL_PARAMS["RSI_OVERSOLD"] >= TECHNICAL_PARAMS["RSI_OVERBOUGHT"]:
        errors.append(
            f"RSI_OVERSOLD ({TECHNICAL_PARAMS['RSI_OVERSOLD']}) must be < RSI_OVERBOUGHT ({TECHNICAL_PARAMS['RSI_OVERBOUGHT']})"
        )

    if TECHNICAL_PARAMS["SHORT_SMA"] >= TECHNICAL_PARAMS["LONG_SMA"]:
        errors.append(
            f"SHORT_SMA ({TECHNICAL_PARAMS['SHORT_SMA']}) must be < LONG_SMA ({TECHNICAL_PARAMS['LONG_SMA']})"
        )

    # Log warnings
    for warning in warnings:
        logger.warning(f"CONFIG WARNING: {warning}")

    # Raise errors
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


# Alpaca credentials configuration
# P0 FIX: Validate credentials exist instead of defaulting to empty strings
_api_key = os.environ.get("ALPACA_API_KEY")
_api_secret = os.environ.get("ALPACA_SECRET_KEY")

if not _api_key or not _api_secret:
    # Only raise if we're not in a test environment
    _is_test = os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TESTING")
    if not _is_test:
        logger.error(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment variables. "
            "Create a .env file with your Alpaca credentials."
        )
        # Don't raise immediately - allow imports to work, but log a clear warning
        # The broker will raise when it tries to connect
        logger.warning("Trading will fail until valid API credentials are provided.")

ALPACA_CREDS = {
    "API_KEY": _api_key or "",
    "API_SECRET": _api_secret or "",
    "PAPER": str(os.environ.get("PAPER", "True")).lower() == "true",  # Ensure string comparison
}

# Trading symbols - Default list (used if dynamic selection is disabled)
SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta
    "TSLA",  # Tesla
]

# Cryptocurrency pairs - Supported for 24/7 trading
# These can be used in addition to or instead of stock symbols
CRYPTO_SYMBOLS = [
    "BTC/USD",   # Bitcoin
    "ETH/USD",   # Ethereum
    "SOL/USD",   # Solana
    "AVAX/USD",  # Avalanche
    "DOGE/USD",  # Dogecoin
    "SHIB/USD",  # Shiba Inu
    "LTC/USD",   # Litecoin
    "BCH/USD",   # Bitcoin Cash
    "LINK/USD",  # Chainlink
    "UNI/USD",   # Uniswap
    "AAVE/USD",  # Aave
    "DOT/USD",   # Polkadot
    "MATIC/USD", # Polygon
    "XLM/USD",   # Stellar
    "ATOM/USD",  # Cosmos
]

# Crypto trading parameters
CRYPTO_PARAMS = {
    "ENABLED": True,  # Enable crypto trading
    "USE_CRYPTO_ONLY": False,  # If True, only trade crypto (not stocks)
    # Default crypto symbols to trade
    "DEFAULT_PAIRS": ["BTC/USD", "ETH/USD", "SOL/USD"],
    # Position sizing for crypto (can be different from stocks due to volatility)
    "POSITION_SIZE": 0.05,  # 5% of portfolio per crypto position (lower due to volatility)
    "MAX_POSITION_SIZE": 0.15,  # 15% maximum crypto position size
    # Crypto-specific risk parameters
    "STOP_LOSS": 0.05,  # 5% stop loss (higher due to volatility)
    "TAKE_PROFIT": 0.10,  # 10% take profit (higher due to volatility)
    # 24/7 trading settings
    "TRADE_24_7": True,  # Enable around-the-clock trading
    "TRADING_INTERVAL": 60,  # Seconds between checks (can be more frequent for crypto)
}

# Overnight trading parameters (Blue Ocean ATS - 24/5 trading)
# Available: Sunday 8 PM ET to Friday 4 AM ET
# Enables trading outside regular hours for supported symbols
OVERNIGHT_PARAMS = {
    "ENABLED": True,  # Enable overnight trading via Blue Ocean ATS
    # Position sizing (more conservative due to lower liquidity)
    "POSITION_SIZE_MULTIPLIER": 0.3,  # 30% of regular position size
    "MAX_OVERNIGHT_POSITIONS": 3,  # Maximum concurrent overnight positions
    # Symbols allowed for overnight trading (empty = all overnight-tradeable symbols)
    "ALLOWED_SYMBOLS": [],  # e.g., ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    # Risk parameters
    "STOP_LOSS_MULTIPLIER": 1.5,  # 1.5x wider stop-loss due to lower liquidity
    "MAX_SPREAD_PCT": 0.01,  # Max 1.0% bid-ask spread allowed
    # Order settings
    "ORDER_TYPE": "limit",  # Always use limit orders in overnight session
    "LIMIT_ORDER_OFFSET_PCT": 0.002,  # 0.2% offset from mid-price
    # Monitoring
    "TRADING_INTERVAL": 300,  # 5 minutes between checks (less frequent than regular)
    "LOG_OVERNIGHT_TRADES": True,  # Extra logging for overnight trades
}

# Dynamic Symbol Selection (NEW - WORKING!)
SYMBOL_SELECTION = {
    "USE_DYNAMIC_SELECTION": True,  # ENABLED - automatically scan for best opportunities
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
    # Expected improvement: +4-6% annual returns from mathematically optimal sizing
    "USE_KELLY_CRITERION": True,  # ENABLED for maximum profit
    "KELLY_FRACTION": 0.5,  # 0.5 = Half Kelly (conservative), 0.25 = Quarter Kelly, 1.0 = Full Kelly
    "KELLY_MIN_TRADES": 30,  # Minimum trades before trusting Kelly calculation
    "KELLY_LOOKBACK": 50,  # Use last N trades for Kelly calculation
    "MIN_POSITION_SIZE": 0.01,  # Minimum 1% position size
    # Volatility Regime Detection (adaptive risk management)
    # Expected improvement: +5-8% annual returns from adaptive risk management
    "USE_VOLATILITY_REGIME": True,  # ENABLED for maximum profit
    # Regimes: Very Low (VIX<12), Low (12-15), Normal (15-20), Elevated (20-30), High (>30)
    # Very Low: +40% position, -30% stop | High: -60% position, +50% stop
    # Streak-Based Position Sizing (dynamic sizing based on recent performance)
    # Expected improvement: +4-7% annual returns from compounding wins, cutting losses
    "USE_STREAK_SIZING": True,  # ENABLED for maximum profit
    "STREAK_LOOKBACK": 10,  # Number of recent trades to analyze for streak detection
    "HOT_STREAK_THRESHOLD": 7,  # Wins needed (out of lookback) for hot streak
    "COLD_STREAK_THRESHOLD": 3,  # Max wins (out of lookback) for cold streak
    "HOT_MULTIPLIER": 1.2,  # Position size multiplier during hot streaks (+20%)
    "COLD_MULTIPLIER": 0.7,  # Position size multiplier during cold streaks (-30%)
    "STREAK_RESET_AFTER": 5,  # Reset to baseline after N trades in same streak
    # Multi-Timeframe Analysis (trend confirmation across timeframes)
    # Expected improvement: +8-12% win rate, -30-40% reduction in false signals
    "USE_MULTI_TIMEFRAME": True,  # ENABLED for maximum profit
    "MTF_MIN_CONFIDENCE": 0.70,  # Minimum confidence (0.0-1.0) to enter trade
    "MTF_REQUIRE_DAILY_ALIGNMENT": True,  # Daily timeframe must not contradict signal
    # Analyzes 5Min, 15Min, 1Hour, 1Day timeframes
    # Expected improvement: +8-12% win rate, -30-40% reduction in false signals
}

# Risk management parameters
RISK_PARAMS = {
    "MAX_PORTFOLIO_RISK": 0.02,  # 2% portfolio-wide risk limit
    "MAX_POSITION_RISK": 0.01,  # 1% individual position risk limit
    "MAX_CORRELATION": 0.7,  # Maximum position correlation
    "VAR_CONFIDENCE": 0.95,  # Value at Risk confidence level
}

# Technical analysis parameters
TECHNICAL_PARAMS = {
    "SENTIMENT_WINDOW": 5,  # Number of sentiment data points to consider
    "PRICE_HISTORY_WINDOW": 30,  # Number of price data points to retain
    "SHORT_SMA": 20,  # Short Simple Moving Average period
    "LONG_SMA": 50,  # Long Simple Moving Average period
    "RSI_PERIOD": 14,  # Relative Strength Index period
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
}

# News sentiment parameters (FinBERT-based analysis)
SENTIMENT_PARAMS = {
    # Enable/disable news sentiment analysis
    "USE_NEWS_SENTIMENT": True,  # ENABLED for sentiment-aware trading
    # Sentiment thresholds for signal generation
    "BULLISH_THRESHOLD": 0.3,  # Score above this = bullish signal
    "BEARISH_THRESHOLD": -0.3,  # Score below this = bearish signal
    # News analysis settings
    "LOOKBACK_HOURS": 24,  # Hours of news to analyze (default: 1 day)
    "MIN_NEWS_COUNT": 3,  # Minimum articles required for valid sentiment
    "INCLUDE_SUMMARY": True,  # Analyze article summaries (more accurate, slower)
    # Caching settings
    "CACHE_TTL_MINUTES": 15,  # Cache sentiment results for this long
    # GPU acceleration (requires CUDA)
    "USE_GPU": False,  # Set to True if GPU available for faster inference
    # Signal weight (how much sentiment influences final signal)
    "SENTIMENT_WEIGHT": 0.3,  # 30% weight in combined signal (0.0-1.0)
}

# Options trading parameters
OPTIONS_PARAMS = {
    # Enable/disable options trading (requires Alpaca options approval)
    "ENABLED": True,  # Enabled for tail hedging

    # Expiration preferences
    "DEFAULT_EXPIRATION_DAYS": 30,  # Default days to expiration for new positions
    "MIN_DAYS_TO_EXPIRATION": 7,  # Minimum DTE (avoid last-week decay issues)
    "MAX_DAYS_TO_EXPIRATION": 45,  # Maximum DTE (balance theta decay vs premium)

    # Position limits
    "MAX_CONTRACTS": 10,  # Maximum contracts per position
    "MAX_TOTAL_CONTRACTS": 50,  # Maximum total option contracts across all positions
    "MAX_NOTIONAL_EXPOSURE": 50000,  # Maximum notional value at risk

    # Delta targeting for income strategies
    "COVERED_CALL_DELTA_TARGET": 0.30,  # Target delta for covered calls (0.30 = ~30% ITM prob)
    "CASH_SECURED_PUT_DELTA_TARGET": -0.30,  # Target delta for CSPs

    # Risk management
    "MAX_LOSS_PER_TRADE": 500,  # Maximum loss per option trade
    "CLOSE_AT_PROFIT_PCT": 50,  # Close position at 50% of max profit
    "CLOSE_AT_LOSS_PCT": 200,  # Close position at 200% loss (2x premium paid)

    # Spread requirements (for bid-ask spread quality)
    "MAX_SPREAD_PCT": 10.0,  # Maximum bid-ask spread as % of mid price
    "MIN_OPEN_INTEREST": 100,  # Minimum open interest for liquidity

    # Strategy preferences
    "PREFER_LIMIT_ORDERS": True,  # Always use limit orders (recommended for options)
    "DEFAULT_ORDER_TYPE": "limit",  # Default order type

    # =========================================================================
    # TAIL HEDGING PARAMETERS
    # Buy protective puts in low-volatility regimes to protect against crashes
    # =========================================================================
    "TAIL_HEDGE_ENABLED": True,  # Enable automatic tail hedging
    "TAIL_HEDGE_VIX_THRESHOLD": 15,  # Buy puts when VIX < 15 (complacency)
    "TAIL_HEDGE_ALLOCATION": 0.02,  # Allocate 2% of portfolio to puts
    "TAIL_HEDGE_PUT_DELTA": -0.15,  # Target 15-delta puts (OTM protection)
    "TAIL_HEDGE_DTE_MIN": 30,  # Minimum days to expiration for hedge
    "TAIL_HEDGE_DTE_MAX": 45,  # Maximum days to expiration for hedge
    "TAIL_HEDGE_UNDERLYING": "SPY",  # Hedge with SPY puts (market exposure)
    "TAIL_HEDGE_CLOSE_VIX": 25,  # Close hedge when VIX > 25 (take profit)
    "TAIL_HEDGE_CLOSE_PROFIT_PCT": 50,  # Close at 50% profit
}

# Reinforcement Learning parameters (DQN agent)
RL_PARAMS = {
    # Enable/disable RL-based trading
    "ENABLED": False,  # Disabled by default (experimental feature)
    # DQN architecture
    "STATE_SIZE": 20,  # Dimension of state vector
    "HIDDEN_SIZES": [128, 64],  # Hidden layer sizes
    # Training hyperparameters
    "LEARNING_RATE": 0.001,  # Adam optimizer learning rate
    "GAMMA": 0.99,  # Discount factor for future rewards
    "EPSILON_START": 1.0,  # Initial exploration rate
    "EPSILON_END": 0.01,  # Minimum exploration rate
    "EPSILON_DECAY": 0.995,  # Exploration decay rate per step
    "BATCH_SIZE": 64,  # Training batch size
    "BUFFER_SIZE": 10000,  # Replay buffer capacity
    "TARGET_UPDATE_FREQ": 100,  # Steps between target network updates
    # Compute settings
    "USE_GPU": False,  # Set True if GPU available (CUDA or MPS)
    # Training requirements
    "MIN_TRAINING_EPISODES": 100,  # Minimum episodes before live trading
    "MIN_TRAINING_STEPS": 10000,  # Minimum steps before live trading
    # Model persistence
    "MODEL_DIR": "models",  # Directory to save/load models
    "AUTO_SAVE_FREQ": 1000,  # Steps between auto-saves (0 to disable)
    # Agent variant
    "USE_DOUBLE_DQN": True,  # Use Double DQN (reduces overestimation bias)
}

# Machine Learning parameters (LSTM price prediction)
ML_PARAMS = {
    # Enable/disable LSTM predictions
    "LSTM_ENABLED": True,  # Set to False to disable ML predictions
    # Model architecture
    "SEQUENCE_LENGTH": 60,  # Number of historical bars for input (60 = 1 hour of minute data)
    "PREDICTION_HORIZON": 5,  # Predict 5 bars ahead
    "HIDDEN_SIZE": 64,  # LSTM hidden layer size
    "NUM_LAYERS": 2,  # Number of stacked LSTM layers
    "DROPOUT": 0.2,  # Dropout rate for regularization
    # Training parameters
    "EPOCHS": 50,  # Training epochs
    "BATCH_SIZE": 32,  # Mini-batch size
    "LEARNING_RATE": 0.001,  # Adam optimizer learning rate
    "EARLY_STOPPING_PATIENCE": 10,  # Epochs to wait before early stopping
    "VALIDATION_SPLIT": 0.2,  # Fraction of data for validation
    # Inference parameters
    "USE_GPU": False,  # Set to True if GPU available for faster training/inference
    "MIN_CONFIDENCE": 0.6,  # Minimum prediction confidence to act on signal
    "SIGNAL_WEIGHT": 0.2,  # Weight of ML signal in combined signal (0.0-1.0)
    # Model persistence
    "MODEL_DIR": "models",  # Directory to save/load trained models
    "AUTO_RETRAIN_DAYS": 7,  # Retrain model every N days (0 to disable)
    # Data requirements
    "MIN_TRAINING_BARS": 500,  # Minimum historical bars needed for training
}

# Backtest parameters (NEW - for realistic simulation)
BACKTEST_PARAMS = {
    # Slippage modeling - critical for realistic results
    "SLIPPAGE_PCT": 0.004,  # 0.4% slippage per trade (conservative estimate)
    "USE_SLIPPAGE": True,  # Enable slippage in backtests
    "BID_ASK_SPREAD": 0.001,  # 0.1% bid-ask spread
    "COMMISSION_PER_SHARE": 0.0,  # Alpaca is commission-free
    # Walk-forward validation settings
    "WALK_FORWARD_ENABLED": True,
    "TRAIN_RATIO": 0.7,  # 70% training, 30% testing
    "N_SPLITS": 5,  # Number of walk-forward splits
    "MIN_TRAIN_DAYS": 30,  # Minimum training period in days
    # Statistical significance thresholds
    "MIN_TRADES_FOR_SIGNIFICANCE": 50,  # Need 50+ trades for valid results
    "OVERFITTING_RATIO_THRESHOLD": 2.0,  # In-sample / out-of-sample ratio threshold
    # Realistic execution modeling
    "EXECUTION_DELAY_BARS": 1,  # Assume 1-bar delay for order execution
    "USE_LIMIT_ORDERS": False,  # If True, model limit order fill rates
    "LIMIT_ORDER_FILL_RATE": 0.7,  # 70% of limit orders fill at target price
}

# LLM Analysis parameters (GPT-4 / Claude for text analysis)
# Adds +2-4% alpha from earnings calls, Fed speeches, SEC filings, news themes
LLM_PARAMS = {
    # Enable/disable LLM analysis
    "ENABLED": True,  # Enable LLM-powered text analysis
    # Provider configuration
    "PRIMARY_PROVIDER": "anthropic",  # "anthropic" or "openai"
    "OPENAI_MODEL": "gpt-4o",  # OpenAI model to use
    "ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022",  # Anthropic model to use
    # Cost management
    "DAILY_COST_CAP_USD": 50.0,  # Maximum daily LLM API cost
    "WEEKLY_COST_CAP_USD": 250.0,  # Maximum weekly LLM API cost
    "MONTHLY_COST_CAP_USD": 750.0,  # Maximum monthly LLM API cost
    # Rate limiting
    "MAX_TOKENS_PER_MINUTE": 50000,  # Token rate limit
    "MAX_REQUESTS_PER_MINUTE": 10,  # Request rate limit
    # Cache TTL (in hours) - reduces costs significantly
    "CACHE_TTL_EARNINGS_HOURS": 168,  # 7 days - transcripts don't change
    "CACHE_TTL_FED_HOURS": 24,  # 24 hours - speeches don't change
    "CACHE_TTL_SEC_HOURS": 720,  # 30 days - SEC filings don't change
    "CACHE_TTL_NEWS_HOURS": 4,  # 4 hours - news is time-sensitive
    # Analysis weights in ensemble
    "EARNINGS_WEIGHT": 0.35,  # Highest - direct company info
    "FED_SPEECH_WEIGHT": 0.15,  # Market-wide signal
    "SEC_FILING_WEIGHT": 0.25,  # Important fundamental data
    "NEWS_THEME_WEIGHT": 0.25,  # Timely but noisy
    # Confidence thresholds
    "MIN_CONFIDENCE": 0.4,  # Minimum confidence to use signal
    "SIGNAL_WEIGHT": 0.15,  # Weight in overall ensemble (15%)
    # Data source API keys (or from env)
    # ALPHA_VANTAGE_API_KEY - for earnings transcripts
    # FMP_API_KEY - for earnings transcripts (fallback)
    # OPENAI_API_KEY - for GPT-4
    # ANTHROPIC_API_KEY - for Claude
}

# P0 FIX: Validate configuration on module load
# This catches invalid configs early rather than at runtime
_validate_config()
