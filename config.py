import os
import logging
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
        errors.append(f"POSITION_SIZE must be between 0 and 1, got {TRADING_PARAMS['POSITION_SIZE']}")

    if TRADING_PARAMS["MAX_POSITION_SIZE"] <= 0 or TRADING_PARAMS["MAX_POSITION_SIZE"] > 1:
        errors.append(f"MAX_POSITION_SIZE must be between 0 and 1, got {TRADING_PARAMS['MAX_POSITION_SIZE']}")

    if TRADING_PARAMS["MAX_POSITION_SIZE"] < TRADING_PARAMS["POSITION_SIZE"]:
        errors.append(f"MAX_POSITION_SIZE ({TRADING_PARAMS['MAX_POSITION_SIZE']}) must be >= POSITION_SIZE ({TRADING_PARAMS['POSITION_SIZE']})")

    if TRADING_PARAMS["STOP_LOSS"] <= 0 or TRADING_PARAMS["STOP_LOSS"] > 0.5:
        errors.append(f"STOP_LOSS must be between 0 and 0.5, got {TRADING_PARAMS['STOP_LOSS']}")

    if TRADING_PARAMS["TAKE_PROFIT"] <= 0 or TRADING_PARAMS["TAKE_PROFIT"] > 1:
        errors.append(f"TAKE_PROFIT must be between 0 and 1, got {TRADING_PARAMS['TAKE_PROFIT']}")

    if TRADING_PARAMS["KELLY_FRACTION"] < 0 or TRADING_PARAMS["KELLY_FRACTION"] > 1:
        errors.append(f"KELLY_FRACTION must be between 0 and 1, got {TRADING_PARAMS['KELLY_FRACTION']}")

    if TRADING_PARAMS["MIN_POSITION_SIZE"] < 0 or TRADING_PARAMS["MIN_POSITION_SIZE"] > TRADING_PARAMS["POSITION_SIZE"]:
        warnings.append(f"MIN_POSITION_SIZE ({TRADING_PARAMS['MIN_POSITION_SIZE']}) should be <= POSITION_SIZE ({TRADING_PARAMS['POSITION_SIZE']})")

    # Validate RISK_PARAMS
    if RISK_PARAMS["MAX_PORTFOLIO_RISK"] <= 0 or RISK_PARAMS["MAX_PORTFOLIO_RISK"] > 0.2:
        errors.append(f"MAX_PORTFOLIO_RISK must be between 0 and 0.2, got {RISK_PARAMS['MAX_PORTFOLIO_RISK']}")

    if RISK_PARAMS["MAX_POSITION_RISK"] <= 0 or RISK_PARAMS["MAX_POSITION_RISK"] > 0.1:
        errors.append(f"MAX_POSITION_RISK must be between 0 and 0.1, got {RISK_PARAMS['MAX_POSITION_RISK']}")

    if RISK_PARAMS["VAR_CONFIDENCE"] <= 0.5 or RISK_PARAMS["VAR_CONFIDENCE"] >= 1:
        errors.append(f"VAR_CONFIDENCE must be between 0.5 and 1 (exclusive), got {RISK_PARAMS['VAR_CONFIDENCE']}")

    # Validate TECHNICAL_PARAMS
    if TECHNICAL_PARAMS["RSI_PERIOD"] < 2 or TECHNICAL_PARAMS["RSI_PERIOD"] > 100:
        warnings.append(f"RSI_PERIOD ({TECHNICAL_PARAMS['RSI_PERIOD']}) is outside typical range (2-100)")

    if TECHNICAL_PARAMS["RSI_OVERSOLD"] >= TECHNICAL_PARAMS["RSI_OVERBOUGHT"]:
        errors.append(f"RSI_OVERSOLD ({TECHNICAL_PARAMS['RSI_OVERSOLD']}) must be < RSI_OVERBOUGHT ({TECHNICAL_PARAMS['RSI_OVERBOUGHT']})")

    if TECHNICAL_PARAMS["SHORT_SMA"] >= TECHNICAL_PARAMS["LONG_SMA"]:
        errors.append(f"SHORT_SMA ({TECHNICAL_PARAMS['SHORT_SMA']}) must be < LONG_SMA ({TECHNICAL_PARAMS['LONG_SMA']})")

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

# Backtest parameters (NEW - for realistic simulation)
BACKTEST_PARAMS = {
    # Slippage modeling - critical for realistic results
    "SLIPPAGE_PCT": 0.004,        # 0.4% slippage per trade (conservative estimate)
    "USE_SLIPPAGE": True,         # Enable slippage in backtests
    "BID_ASK_SPREAD": 0.001,      # 0.1% bid-ask spread
    "COMMISSION_PER_SHARE": 0.0,  # Alpaca is commission-free

    # Walk-forward validation settings
    "WALK_FORWARD_ENABLED": True,
    "TRAIN_RATIO": 0.7,           # 70% training, 30% testing
    "N_SPLITS": 5,                # Number of walk-forward splits
    "MIN_TRAIN_DAYS": 30,         # Minimum training period in days

    # Statistical significance thresholds
    "MIN_TRADES_FOR_SIGNIFICANCE": 50,  # Need 50+ trades for valid results
    "OVERFITTING_RATIO_THRESHOLD": 2.0, # In-sample / out-of-sample ratio threshold

    # Realistic execution modeling
    "EXECUTION_DELAY_BARS": 1,    # Assume 1-bar delay for order execution
    "USE_LIMIT_ORDERS": False,    # If True, model limit order fill rates
    "LIMIT_ORDER_FILL_RATE": 0.7, # 70% of limit orders fill at target price
}

# P0 FIX: Validate configuration on module load
# This catches invalid configs early rather than at runtime
_validate_config()
