"""
Crypto symbol utilities for trading bot.

Provides centralized crypto symbol detection and normalization.
This module eliminates duplication between alpaca_broker.py and order_builder.py.
"""

from typing import List, Optional

# Supported cryptocurrency pairs (Alpaca format with forward slash)
CRYPTO_PAIRS: List[str] = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "AVAX/USD",
    "DOGE/USD",
    "SHIB/USD",
    "LTC/USD",
    "BCH/USD",
    "LINK/USD",
    "UNI/USD",
    "AAVE/USD",
    "DOT/USD",
    "MATIC/USD",
    "XLM/USD",
    "ATOM/USD",
    "SUSHI/USD",
    "YFI/USD",
    "MKR/USD",
    "CRV/USD",
    "BAT/USD",
]

# Base crypto symbols (without /USD) - derived from CRYPTO_PAIRS
CRYPTO_BASE_SYMBOLS: List[str] = [pair.split("/")[0] for pair in CRYPTO_PAIRS]


def is_crypto_symbol(symbol: str) -> bool:
    """
    Check if a symbol is a cryptocurrency pair.

    Handles multiple formats:
    - Slash format: BTC/USD, ETH/USD
    - Compact format: BTCUSD, ETHUSD
    - Dash format: BTC-USD, ETH-USD
    - Base symbol only: BTC, ETH

    Args:
        symbol: Symbol to check (e.g., "BTC/USD", "BTCUSD", "BTC-USD", "AAPL")

    Returns:
        True if symbol is a recognized crypto pair, False otherwise

    Examples:
        >>> is_crypto_symbol("BTC/USD")
        True
        >>> is_crypto_symbol("BTCUSD")
        True
        >>> is_crypto_symbol("BTC-USD")
        True
        >>> is_crypto_symbol("AAPL")
        False
    """
    if not symbol:
        return False

    # Normalize: uppercase and remove dashes/underscores
    normalized = symbol.upper().strip().replace("-", "").replace("_", "")

    # Check with slash (explicit crypto format)
    if "/" in normalized:
        return normalized in CRYPTO_PAIRS

    # Check without slash (e.g., BTCUSD)
    for pair in CRYPTO_PAIRS:
        if normalized == pair.replace("/", ""):
            return True

    return False


def normalize_crypto_symbol(symbol: str) -> str:
    """
    Normalize a crypto symbol to Alpaca format (BASE/USD).

    Converts various formats to the standard Alpaca format with forward slash.

    Args:
        symbol: Crypto symbol in any format (BTC, BTCUSD, BTC-USD, BTC/USD)

    Returns:
        Normalized symbol in Alpaca format (e.g., "BTC/USD")

    Raises:
        ValueError: If symbol is empty or not a recognized crypto pair

    Examples:
        >>> normalize_crypto_symbol("BTC")
        'BTC/USD'
        >>> normalize_crypto_symbol("BTCUSD")
        'BTC/USD'
        >>> normalize_crypto_symbol("btc-usd")
        'BTC/USD'
        >>> normalize_crypto_symbol("BTC/USD")
        'BTC/USD'
    """
    if not symbol:
        raise ValueError("Symbol cannot be empty")

    # Normalize: uppercase, strip whitespace, remove dashes/underscores
    normalized = symbol.upper().strip().replace("-", "").replace("_", "")

    # Already in correct format
    if "/" in normalized and normalized in CRYPTO_PAIRS:
        return normalized

    # Convert from BTCUSD format to BTC/USD
    for pair in CRYPTO_PAIRS:
        if normalized == pair.replace("/", ""):
            return pair

    raise ValueError(
        f"Unrecognized crypto symbol: {symbol}. "
        f"Supported pairs: {', '.join(CRYPTO_PAIRS[:5])}..."
    )


def get_crypto_base(symbol: str) -> Optional[str]:
    """
    Get the base currency from a crypto symbol.

    Args:
        symbol: Crypto symbol (e.g., "BTC/USD", "BTCUSD", "BTC")

    Returns:
        Base currency (e.g., "BTC") or None if not a recognized crypto symbol

    Examples:
        >>> get_crypto_base("BTC/USD")
        'BTC'
        >>> get_crypto_base("ETHUSD")
        'ETH'
        >>> get_crypto_base("AAPL")
        None
    """
    if not is_crypto_symbol(symbol):
        return None

    try:
        normalized = normalize_crypto_symbol(symbol)
        return normalized.split("/")[0]
    except ValueError:
        return None


def get_crypto_quote(symbol: str) -> Optional[str]:
    """
    Get the quote currency from a crypto symbol.

    For Alpaca, all crypto pairs are quoted in USD.

    Args:
        symbol: Crypto symbol (e.g., "BTC/USD", "BTCUSD")

    Returns:
        Quote currency (always "USD" for Alpaca) or None if not crypto

    Examples:
        >>> get_crypto_quote("BTC/USD")
        'USD'
        >>> get_crypto_quote("AAPL")
        None
    """
    if not is_crypto_symbol(symbol):
        return None
    return "USD"


def format_crypto_symbol(base: str, quote: str = "USD") -> str:
    """
    Format a crypto symbol from base and quote currencies.

    Args:
        base: Base currency (e.g., "BTC", "ETH")
        quote: Quote currency (default: "USD")

    Returns:
        Formatted symbol (e.g., "BTC/USD")

    Raises:
        ValueError: If the resulting symbol is not supported

    Examples:
        >>> format_crypto_symbol("BTC")
        'BTC/USD'
        >>> format_crypto_symbol("ETH", "USD")
        'ETH/USD'
    """
    symbol = f"{base.upper()}/{quote.upper()}"
    if symbol not in CRYPTO_PAIRS:
        raise ValueError(f"Unsupported crypto pair: {symbol}")
    return symbol
