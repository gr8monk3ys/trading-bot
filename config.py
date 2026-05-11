import logging
import os
from typing import TypedDict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def _validate_config():
    """
    Validate configuration parameters on module load.

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
    if RISK_PARAMS["MAX_PORTFOLIO_RISK"] <= 0 or RISK_PARAMS["MAX_PORTFOLIO_RISK"] > 1:
        errors.append(
            f"MAX_PORTFOLIO_RISK must be between 0 and 1, got {RISK_PARAMS['MAX_PORTFOLIO_RISK']}"
        )

    if RISK_PARAMS["MAX_POSITION_RISK"] <= 0 or RISK_PARAMS["MAX_POSITION_RISK"] > 1:
        errors.append(
            f"MAX_POSITION_RISK must be between 0 and 1, got {RISK_PARAMS['MAX_POSITION_RISK']}"
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


def _parse_bool_env(name: str, default: bool = True) -> bool:
    """Parse a boolean environment variable using permissive truthy values."""
    raw = str(os.environ.get(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _parse_float_env(name: str, default: float) -> float:
    """Parse a float environment variable with fallback."""
    raw = str(os.environ.get(name, default)).strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


class AlpacaCreds(TypedDict):
    API_KEY: str
    API_SECRET: str
    PAPER: bool


def _read_alpaca_creds_from_env() -> AlpacaCreds:
    """
    Read Alpaca credentials from environment variables.

    Supports both explicit names (ALPACA_API_KEY/ALPACA_SECRET_KEY) and
    compatibility aliases (API_KEY/API_SECRET).
    """
    return {
        "API_KEY": os.environ.get("ALPACA_API_KEY") or os.environ.get("API_KEY") or "",
        "API_SECRET": os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("API_SECRET") or "",
        "PAPER": _parse_bool_env("PAPER", default=True),
    }


# Snapshot of credentials for modules that rely on constant-style imports.
# This is intentionally non-strict to avoid import-time side effects.
ALPACA_CREDS = _read_alpaca_creds_from_env()


def get_alpaca_creds(refresh: bool = False) -> AlpacaCreds:
    """
    Return Alpaca credentials.

    Args:
        refresh: Re-read from environment before returning.
    """
    if refresh:
        ALPACA_CREDS.update(_read_alpaca_creds_from_env())
    return {
        "API_KEY": ALPACA_CREDS["API_KEY"],
        "API_SECRET": ALPACA_CREDS["API_SECRET"],
        "PAPER": ALPACA_CREDS["PAPER"],
    }


def require_alpaca_credentials(context: str = "trading") -> AlpacaCreds:
    """
    Return Alpaca credentials and raise when missing required keys.

    Use this in runtime paths that need broker connectivity.
    """
    creds = get_alpaca_creds(refresh=True)
    if creds["API_KEY"] and creds["API_SECRET"]:
        return creds

    raise ValueError(
        "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
        "(or API_KEY/API_SECRET) before running "
        f"{context}."
    )


# Trading symbols - Default list (used if dynamic selection is disabled)
SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta
    "TSLA",  # Tesla
]

# Dynamic Symbol Selection
SYMBOL_SELECTION = {
    "USE_DYNAMIC_SELECTION": True,  # ENABLED - automatically scan for best opportunities
    "TOP_N_SYMBOLS": 20,  # Number of stocks to trade (more = more diversification)
    "MIN_MOMENTUM_SCORE": 1.0,  # Minimum 5-day price movement % to consider
    "RESCAN_INTERVAL_HOURS": 24,  # Rescan market every N hours for new opportunities
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
    "USE_KELLY_CRITERION": True,
    "KELLY_FRACTION": 0.5,  # 0.5 = Half Kelly (conservative)
    "KELLY_MIN_TRADES": 30,  # Minimum trades before trusting Kelly calculation
    "KELLY_LOOKBACK": 50,  # Use last N trades for Kelly calculation
    "MIN_POSITION_SIZE": 0.01,  # Minimum 1% position size
    # Volatility Regime Detection (adaptive risk management)
    "USE_VOLATILITY_REGIME": True,
    # Streak-Based Position Sizing (dynamic sizing based on recent performance)
    "USE_STREAK_SIZING": True,
    "STREAK_LOOKBACK": 10,  # Number of recent trades to analyze for streak detection
    "HOT_STREAK_THRESHOLD": 7,  # Wins needed (out of lookback) for hot streak
    "COLD_STREAK_THRESHOLD": 3,  # Max wins (out of lookback) for cold streak
    "HOT_MULTIPLIER": 1.2,  # Position size multiplier during hot streaks (+20%)
    "COLD_MULTIPLIER": 0.7,  # Position size multiplier during cold streaks (-30%)
    "STREAK_RESET_AFTER": 5,  # Reset to baseline after N trades in same streak
    # Multi-Timeframe Analysis (trend confirmation across timeframes)
    "USE_MULTI_TIMEFRAME": True,
    "MTF_MIN_CONFIDENCE": 0.70,  # Minimum confidence (0.0-1.0) to enter trade
    "MTF_REQUIRE_DAILY_ALIGNMENT": True,  # Daily timeframe must not contradict signal
}

# Risk management parameters
RISK_PARAMS = {
    "MAX_PORTFOLIO_RISK": _parse_float_env("MAX_PORTFOLIO_RISK", 0.02),
    "MAX_POSITION_RISK": _parse_float_env("MAX_POSITION_RISK", 0.01),
    "MAX_CORRELATION": _parse_float_env("MAX_CORRELATION", 0.7),
    "VAR_CONFIDENCE": _parse_float_env("VAR_CONFIDENCE", 0.95),
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

# Backtest parameters (realistic simulation)
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

# Validate configuration on module load
_validate_config()
