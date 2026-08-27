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

# Risk management parameters
RISK_PARAMS = {
    "MAX_PORTFOLIO_RISK": _parse_float_env("MAX_PORTFOLIO_RISK", 0.02),
    "MAX_POSITION_RISK": _parse_float_env("MAX_POSITION_RISK", 0.01),
    "MAX_CORRELATION": _parse_float_env("MAX_CORRELATION", 0.7),
    "VAR_CONFIDENCE": _parse_float_env("VAR_CONFIDENCE", 0.95),
}

# Validate configuration on module load
_validate_config()
