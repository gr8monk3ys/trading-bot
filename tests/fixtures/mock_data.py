"""
Mock data generators for testing
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


def generate_price_series(
    start_price: float = 100.0, days: int = 100, trend: str = "neutral", volatility: float = 0.02
) -> pd.Series:
    """
    Generate a price series for testing

    Args:
        start_price: Starting price
        days: Number of days
        trend: "up", "down", or "neutral"
        volatility: Daily volatility (std dev)

    Returns:
        pandas Series with datetime index
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Drift based on trend
    drift_map = {"up": 0.001, "down": -0.001, "neutral": 0.0}
    drift = drift_map.get(trend, 0.0)

    # Generate returns
    returns = np.random.normal(drift, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))

    return pd.Series(prices, index=dates)


def generate_ohlcv_data(
    symbol: str = "TEST",
    days: int = 100,
    start_price: float = 100.0,
    trend: str = "neutral",
    volatility: float = 0.02,
) -> pd.DataFrame:
    """
    Generate OHLCV data for testing

    Args:
        symbol: Stock symbol
        days: Number of days
        start_price: Starting price
        trend: "up", "down", or "neutral"
        volatility: Daily volatility

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Generate close prices
    close_series = generate_price_series(start_price, days, trend, volatility)

    data = []
    for i, (date, close) in enumerate(zip(dates, close_series)):
        # Intraday range
        daily_range = close * np.random.uniform(0.01, 0.03)

        open_price = float(close * (1 + np.random.uniform(-0.01, 0.01)))
        high = float(max(open_price, close) + daily_range * 0.5)
        low = float(min(open_price, close) - daily_range * 0.5)
        volume = float(np.random.uniform(1_000_000, 10_000_000))

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "symbol": symbol,
            }
        )

    return pd.DataFrame(data)


def generate_technical_indicators(
    prices: pd.Series, periods: List[int] = [20, 50, 200]
) -> Dict[str, pd.Series]:
    """
    Generate common technical indicators for testing

    Args:
        prices: Price series
        periods: SMA periods to calculate

    Returns:
        Dictionary of indicator series
    """
    indicators = {}

    # Simple Moving Averages
    for period in periods:
        indicators[f"sma_{period}"] = prices.rolling(window=period).mean()

    # RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = prices.rolling(window=20).mean()
    std20 = prices.rolling(window=20).std()
    indicators["bb_upper"] = sma20 + (std20 * 2)
    indicators["bb_lower"] = sma20 - (std20 * 2)

    # Volume-weighted average price (simplified)
    indicators["vwap"] = prices.rolling(window=20).mean()

    return indicators


def generate_momentum_scenario(days: int = 100) -> pd.DataFrame:
    """
    Generate a momentum scenario (strong uptrend)

    Returns:
        OHLCV DataFrame with momentum characteristics
    """
    return generate_ohlcv_data(
        symbol="MOMENTUM_TEST", days=days, start_price=100.0, trend="up", volatility=0.015
    )


def generate_mean_reversion_scenario(days: int = 100) -> pd.DataFrame:
    """
    Generate a mean reversion scenario (oscillating around mean)

    Returns:
        OHLCV DataFrame with mean-reverting characteristics
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Create oscillating pattern
    mean_price = 100.0
    amplitude = 10.0
    frequency = 0.1

    closes = (
        mean_price + amplitude * np.sin(np.arange(days) * frequency) + np.random.normal(0, 2, days)
    )

    data = []
    for i, (date, close) in enumerate(zip(dates, closes)):
        daily_range = close * 0.02
        open_price = float(close * (1 + np.random.uniform(-0.005, 0.005)))
        high = float(max(open_price, close) + daily_range * 0.5)
        low = float(min(open_price, close) - daily_range * 0.5)
        volume = float(np.random.uniform(1_000_000, 10_000_000))

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "symbol": "MEANREV_TEST",
            }
        )

    return pd.DataFrame(data)


def generate_volatile_scenario(days: int = 100) -> pd.DataFrame:
    """
    Generate a highly volatile scenario

    Returns:
        OHLCV DataFrame with high volatility
    """
    return generate_ohlcv_data(
        symbol="VOLATILE_TEST",
        days=days,
        start_price=100.0,
        trend="neutral",
        volatility=0.05,  # High volatility
    )


def generate_sideways_scenario(days: int = 100) -> pd.DataFrame:
    """
    Generate a sideways/ranging market scenario

    Returns:
        OHLCV DataFrame with sideways pattern
    """
    return generate_ohlcv_data(
        symbol="SIDEWAYS_TEST",
        days=days,
        start_price=100.0,
        trend="neutral",
        volatility=0.01,  # Low volatility
    )


def generate_multi_symbol_data(
    symbols: List[str] = ["AAPL", "TSLA", "SPY"], days: int = 100
) -> Dict[str, pd.DataFrame]:
    """
    Generate OHLCV data for multiple symbols

    Args:
        symbols: List of symbols
        days: Number of days

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    data = {}

    for symbol in symbols:
        # Vary characteristics by symbol
        if "tech" in symbol.lower() or symbol in ["AAPL", "TSLA", "NVDA"]:
            volatility = 0.025
            trend = "up"
        elif symbol in ["SPY", "QQQ"]:
            volatility = 0.015
            trend = "neutral"
        else:
            volatility = 0.02
            trend = "neutral"

        data[symbol] = generate_ohlcv_data(
            symbol=symbol,
            days=days,
            start_price=np.random.uniform(50, 200),
            trend=trend,
            volatility=volatility,
        )

    return data


def generate_backtest_data(
    start_date: str = "2024-01-01", end_date: str = "2024-03-01", symbols: List[str] = ["TEST"]
) -> Dict[str, pd.DataFrame]:
    """
    Generate data suitable for backtesting

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: List of symbols

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days

    data = {}
    for symbol in symbols:
        df = generate_ohlcv_data(symbol=symbol, days=days)
        # Adjust dates to match requested range
        df["timestamp"] = pd.date_range(start=start, end=end, periods=len(df))
        data[symbol] = df

    return data


def generate_account_data(
    equity: float = 100000.0, cash: float = 50000.0, positions: int = 5
) -> dict:
    """
    Generate mock account data

    Args:
        equity: Total equity
        cash: Available cash
        positions: Number of positions

    Returns:
        Dictionary with account data
    """
    return {
        "id": "test_account_123",
        "equity": equity,
        "cash": cash,
        "buying_power": cash * 4,  # 4x margin
        "portfolio_value": equity,
        "position_count": positions,
        "long_market_value": equity - cash,
        "short_market_value": 0.0,
    }


def generate_position_data(
    symbol: str = "TEST", qty: float = 100, entry_price: float = 100.0, current_price: float = 105.0
) -> dict:
    """
    Generate mock position data

    Args:
        symbol: Stock symbol
        qty: Quantity
        entry_price: Entry price
        current_price: Current price

    Returns:
        Dictionary with position data
    """
    cost_basis = qty * entry_price
    market_value = qty * current_price
    unrealized_pl = market_value - cost_basis
    unrealized_plpc = (current_price - entry_price) / entry_price

    return {
        "symbol": symbol,
        "qty": qty,
        "side": "long",
        "avg_entry_price": entry_price,
        "current_price": current_price,
        "market_value": market_value,
        "cost_basis": cost_basis,
        "unrealized_pl": unrealized_pl,
        "unrealized_plpc": unrealized_plpc,
    }
