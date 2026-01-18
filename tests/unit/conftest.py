"""
Shared pytest fixtures and helpers for unit tests.

This conftest.py provides common fixtures used across multiple test files,
following DRY principles to avoid duplicate setup code.

Available Fixtures:
- mock_broker: AsyncMock broker with default account values
- mock_account: MagicMock account with configurable equity/cash
- mock_position: Factory for creating mock positions

Helper Functions:
- create_mock_account: Create account with specific values
- create_mock_position: Create position with symbol and quantity
- generate_price_history: Generate price history for testing
"""

from typing import Optional
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock


# =============================================================================
# SHARED CONSTANTS
# =============================================================================

# Default account values (used across risk manager, circuit breaker, etc.)
DEFAULT_STARTING_EQUITY = 100000.0
DEFAULT_STARTING_CASH = 50000.0
DEFAULT_BUYING_POWER = 200000.0

# Common test symbols
TEST_SYMBOL_AAPL = "AAPL"
TEST_SYMBOL_MSFT = "MSFT"
TEST_SYMBOL_GOOGL = "GOOGL"
TEST_SYMBOL_TSLA = "TSLA"

# Price generation defaults
DEFAULT_PRICE_START = 100.0
DEFAULT_PRICE_POINTS = 30


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_account(
    equity: float = DEFAULT_STARTING_EQUITY,
    cash: float = DEFAULT_STARTING_CASH,
    buying_power: float = DEFAULT_BUYING_POWER
) -> MagicMock:
    """
    Create a mock account object with specified values.

    Args:
        equity: Account equity value
        cash: Account cash value
        buying_power: Account buying power

    Returns:
        MagicMock with account attributes as strings (Alpaca format)
    """
    account = MagicMock()
    account.equity = str(equity)
    account.cash = str(cash)
    account.buying_power = str(buying_power)
    return account


def create_mock_position(
    symbol: str,
    qty: str,
    unrealized_pl: str = "0.00",
    unrealized_plpc: str = "0.00"
) -> MagicMock:
    """
    Create a mock position object.

    Args:
        symbol: Stock symbol
        qty: Position quantity as string
        unrealized_pl: Unrealized P/L as string
        unrealized_plpc: Unrealized P/L percent as string

    Returns:
        MagicMock with position attributes
    """
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.unrealized_pl = unrealized_pl
    pos.unrealized_plpc = unrealized_plpc
    return pos


def generate_price_history(
    start_price: float = DEFAULT_PRICE_START,
    num_points: int = DEFAULT_PRICE_POINTS,
    volatility: float = 0.02,
    trend: float = 0.0,
    seed: Optional[int] = None
) -> list[float]:
    """
    Generate synthetic price history for testing.

    Args:
        start_price: Starting price
        num_points: Number of price points to generate
        volatility: Daily volatility (standard deviation of returns)
        trend: Daily trend (drift in returns)
        seed: Random seed for reproducibility

    Returns:
        List of prices
    """
    if seed is not None:
        np.random.seed(seed)

    prices = [start_price]
    for _ in range(num_points - 1):
        # Generate log-normal returns
        ret = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))  # Ensure positive prices

    return prices


def generate_correlated_price_histories(
    correlation: float,
    start_price1: float = DEFAULT_PRICE_START,
    start_price2: float = DEFAULT_PRICE_START,
    num_points: int = DEFAULT_PRICE_POINTS,
    seed: Optional[int] = None
) -> tuple[list[float], list[float]]:
    """
    Generate two correlated price histories for correlation testing.

    Args:
        correlation: Target correlation between the two series
        start_price1: Starting price for first series
        start_price2: Starting price for second series
        num_points: Number of price points
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prices1, prices2)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate correlated random returns
    returns1 = np.random.normal(0, 0.02, num_points - 1)

    # Generate second series with target correlation
    noise = np.random.normal(0, 0.02, num_points - 1)
    returns2 = correlation * returns1 + np.sqrt(1 - correlation**2) * noise

    # Convert returns to prices
    prices1 = [start_price1]
    prices2 = [start_price2]

    for r1, r2 in zip(returns1, returns2):
        prices1.append(prices1[-1] * (1 + r1))
        prices2.append(prices2[-1] * (1 + r2))

    return prices1, prices2


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_broker() -> AsyncMock:
    """
    Create a mock broker with default account values.

    Returns:
        AsyncMock broker with get_account configured to return default values
    """
    broker = AsyncMock()
    broker.get_account.return_value = create_mock_account()
    broker.get_positions.return_value = []
    return broker


@pytest.fixture
def mock_account() -> MagicMock:
    """
    Create a mock account with default values.

    Returns:
        MagicMock account with default equity/cash values
    """
    return create_mock_account()


@pytest.fixture
def sample_price_history() -> list[float]:
    """
    Create a sample price history for testing.

    Returns:
        List of 30 prices starting at 100.0 with moderate volatility
    """
    return generate_price_history(
        start_price=DEFAULT_PRICE_START,
        num_points=DEFAULT_PRICE_POINTS,
        volatility=0.02,
        seed=42  # Fixed seed for reproducibility
    )


@pytest.fixture
def stable_price_history() -> list[float]:
    """
    Create a stable price history (low volatility) for testing.

    Returns:
        List of prices with very low volatility
    """
    return generate_price_history(
        start_price=DEFAULT_PRICE_START,
        num_points=DEFAULT_PRICE_POINTS,
        volatility=0.005,
        trend=0.001,
        seed=42
    )


@pytest.fixture
def volatile_price_history() -> list[float]:
    """
    Create a volatile price history (high volatility) for testing.

    Returns:
        List of prices with high volatility
    """
    return generate_price_history(
        start_price=DEFAULT_PRICE_START,
        num_points=DEFAULT_PRICE_POINTS,
        volatility=0.05,
        seed=42
    )


@pytest.fixture
def downtrend_price_history() -> list[float]:
    """
    Create a downtrending price history for testing.

    Returns:
        List of prices with negative trend
    """
    return generate_price_history(
        start_price=DEFAULT_PRICE_START,
        num_points=DEFAULT_PRICE_POINTS,
        volatility=0.02,
        trend=-0.01,
        seed=42
    )


@pytest.fixture
def correlated_price_histories() -> tuple[list[float], list[float]]:
    """
    Create two highly correlated price histories for correlation testing.

    Returns:
        Tuple of (prices1, prices2) with ~0.8 correlation
    """
    return generate_correlated_price_histories(
        correlation=0.8,
        num_points=DEFAULT_PRICE_POINTS,
        seed=42
    )


@pytest.fixture
def uncorrelated_price_histories() -> tuple[list[float], list[float]]:
    """
    Create two uncorrelated price histories for correlation testing.

    Returns:
        Tuple of (prices1, prices2) with ~0.2 correlation
    """
    return generate_correlated_price_histories(
        correlation=0.2,
        num_points=DEFAULT_PRICE_POINTS,
        seed=42
    )
