"""
Test helper functions and utilities
"""

from datetime import datetime
from typing import Any, Coroutine

import pandas as pd


def assert_approximately_equal(
    actual: float, expected: float, tolerance: float = 0.01, message: str = None
):
    """
    Assert that two floats are approximately equal

    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Tolerance (default 1%)
        message: Custom error message
    """
    diff = abs(actual - expected)
    max_diff = abs(expected * tolerance)

    if diff > max_diff:
        msg = message or f"Expected {expected}, got {actual} (tolerance: {tolerance})"
        raise AssertionError(msg)


def assert_in_range(value: float, min_val: float, max_val: float, message: str = None):
    """
    Assert that a value is within a range

    Args:
        value: Value to check
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        message: Custom error message
    """
    if not (min_val <= value <= max_val):
        msg = message or f"Expected {value} to be between {min_val} and {max_val}"
        raise AssertionError(msg)


def assert_all_positive(values: list, message: str = None):
    """Assert all values in list are positive"""
    if not all(v > 0 for v in values):
        msg = message or f"Expected all positive values, got: {values}"
        raise AssertionError(msg)


def assert_series_increasing(series: pd.Series, message: str = None):
    """Assert pandas Series is monotonically increasing"""
    if not series.is_monotonic_increasing:
        msg = message or "Expected series to be increasing"
        raise AssertionError(msg)


def assert_series_decreasing(series: pd.Series, message: str = None):
    """Assert pandas Series is monotonically decreasing"""
    if not series.is_monotonic_decreasing:
        msg = message or "Expected series to be decreasing"
        raise AssertionError(msg)


async def run_async_test(coro: Coroutine) -> Any:
    """
    Helper to run async tests

    Args:
        coro: Async coroutine to run

    Returns:
        Result of coroutine
    """
    return await coro


def create_mock_strategy_params(**kwargs) -> dict:
    """
    Create mock strategy parameters with defaults

    Args:
        **kwargs: Override specific parameters

    Returns:
        Dictionary of strategy parameters
    """
    defaults = {
        "position_size": 0.05,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "sentiment_threshold": 0.1,
        "price_history_window": 30,
        "allocation": 1.0,
    }

    defaults.update(kwargs)
    return defaults


def create_mock_risk_params(**kwargs) -> dict:
    """
    Create mock risk management parameters

    Args:
        **kwargs: Override specific parameters

    Returns:
        Dictionary of risk parameters
    """
    defaults = {
        "max_position_size": 0.1,
        "max_portfolio_risk": 0.2,
        "max_drawdown": 0.15,
        "max_correlation": 0.7,
        "var_confidence": 0.95,
        "position_limit": 10,
    }

    defaults.update(kwargs)
    return defaults


def assert_valid_order(order: dict):
    """
    Assert order has required fields and valid values

    Args:
        order: Order dictionary to validate
    """
    required_fields = ["symbol", "qty", "side", "type"]

    for field in required_fields:
        assert field in order, f"Order missing required field: {field}"

    assert order["qty"] > 0, "Order quantity must be positive"
    assert order["side"] in ["buy", "sell"], f"Invalid side: {order['side']}"
    assert order["type"] in [
        "market",
        "limit",
        "stop",
        "stop_limit",
    ], f"Invalid order type: {order['type']}"


def assert_valid_position(position: dict):
    """
    Assert position has required fields and valid values

    Args:
        position: Position dictionary to validate
    """
    required_fields = ["symbol", "qty", "avg_entry_price", "current_price"]

    for field in required_fields:
        assert field in position, f"Position missing required field: {field}"

    assert position["qty"] > 0, "Position quantity must be positive"
    assert position["avg_entry_price"] > 0, "Entry price must be positive"
    assert position["current_price"] > 0, "Current price must be positive"


def assert_valid_account(account: dict):
    """
    Assert account has required fields and valid values

    Args:
        account: Account dictionary to validate
    """
    required_fields = ["equity", "cash", "buying_power"]

    for field in required_fields:
        assert field in account, f"Account missing required field: {field}"

    assert account["equity"] >= 0, "Equity must be non-negative"
    assert account["cash"] >= 0, "Cash must be non-negative"
    assert account["buying_power"] >= 0, "Buying power must be non-negative"


def assert_valid_signal(signal: dict):
    """
    Assert trading signal has required fields and valid values

    Args:
        signal: Signal dictionary to validate
    """
    required_fields = ["action", "confidence"]

    for field in required_fields:
        assert field in signal, f"Signal missing required field: {field}"

    assert signal["action"] in ["buy", "sell", "hold"], f"Invalid action: {signal['action']}"

    assert (
        0.0 <= signal["confidence"] <= 1.0
    ), f"Confidence must be between 0 and 1, got: {signal['confidence']}"


def calculate_expected_pnl(
    qty: float, entry_price: float, exit_price: float, side: str = "long"
) -> float:
    """
    Calculate expected P&L for a trade

    Args:
        qty: Quantity
        entry_price: Entry price
        exit_price: Exit price
        side: 'long' or 'short'

    Returns:
        Expected P&L
    """
    if side == "long":
        return qty * (exit_price - entry_price)
    else:  # short
        return qty * (entry_price - exit_price)


def calculate_expected_return(entry_price: float, exit_price: float, side: str = "long") -> float:
    """
    Calculate expected return percentage

    Args:
        entry_price: Entry price
        exit_price: Exit price
        side: 'long' or 'short'

    Returns:
        Return percentage (e.g., 0.05 for 5%)
    """
    if side == "long":
        return (exit_price - entry_price) / entry_price
    else:  # short
        return (entry_price - exit_price) / entry_price


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return (excess_returns.mean() / excess_returns.std()) * (252**0.5)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Maximum drawdown (negative value)
    """
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return drawdowns.min()


def mock_market_hours(is_open: bool = True) -> dict:
    """
    Create mock market hours data

    Args:
        is_open: Whether market is open

    Returns:
        Dictionary with market status
    """
    now = datetime.now()
    return {
        "is_open": is_open,
        "next_open": now if is_open else now.replace(hour=9, minute=30),
        "next_close": now.replace(hour=16, minute=0),
        "timestamp": now,
    }


class AsyncMock:
    """Mock class for async functions"""

    def __init__(self, return_value=None):
        self.return_value = return_value
        self.called = False
        self.call_count = 0
        self.call_args_list = []

    async def __call__(self, *args, **kwargs):
        self.called = True
        self.call_count += 1
        self.call_args_list.append((args, kwargs))
        return self.return_value


def create_test_logger():
    """Create a test logger that doesn't write to disk"""
    import logging

    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)

    # Add null handler to prevent output
    logger.addHandler(logging.NullHandler())

    return logger


def freeze_time(frozen_datetime: datetime):
    """
    Context manager to freeze time for testing

    Args:
        frozen_datetime: Datetime to freeze at

    Example:
        with freeze_time(datetime(2024, 1, 1)):
            # Code here will see datetime.now() as 2024-01-01
    """
    import unittest.mock as mock

    class FreezeTime:
        def __enter__(self):
            self.patcher = mock.patch("datetime.datetime")
            mock_datetime = self.patcher.start()
            mock_datetime.now.return_value = frozen_datetime
            return mock_datetime

        def __exit__(self, *args):
            self.patcher.stop()

    return FreezeTime()
