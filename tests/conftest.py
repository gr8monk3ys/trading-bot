import os
import sys
import asyncio
import contextlib
import gc
from datetime import datetime, timedelta

import pandas as pd
import pytest

if "numpy" in sys.modules:
    np = sys.modules["numpy"]
else:
    import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_policy_loop():
    """Return the current event loop bound to policy without creating one."""
    policy = asyncio.get_event_loop_policy()
    local_state = getattr(policy, "_local", None)
    if local_state is None:
        return None
    return getattr(local_state, "_loop", None)


def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
    if loop.is_closed() or loop.is_running():
        return
    with contextlib.suppress(Exception):
        loop.run_until_complete(loop.shutdown_asyncgens())
    with contextlib.suppress(Exception):
        loop.run_until_complete(loop.shutdown_default_executor())
    with contextlib.suppress(Exception):
        loop.close()


def _close_lingering_event_loops() -> None:
    loop = _get_policy_loop()
    if loop is not None:
        _close_loop(loop)
    with contextlib.suppress(Exception):
        asyncio.get_event_loop_policy().set_event_loop(None)

    for obj in gc.get_objects():
        if isinstance(obj, asyncio.AbstractEventLoop):
            _close_loop(obj)


def pytest_sessionfinish(session, exitstatus):
    """Best-effort close of lingering loops before plugin cleanup callbacks."""
    _close_lingering_event_loops()


def pytest_unconfigure(config):
    """Final loop cleanup before pytest cleanup stack raises unraisable warnings."""
    _close_lingering_event_loops()


@pytest.fixture
def mock_broker():
    """Mock broker for testing"""

    class MockBroker:
        def __init__(self):
            self.positions = {}
            self.orders = []
            self.account = {"cash": 100000.0, "portfolio_value": 100000.0}

        async def get_position(self, symbol):
            return self.positions.get(symbol)

        async def get_positions(self):
            return list(self.positions.values())

        async def create_order(self, symbol, qty, side, type, time_in_force):
            order = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": type,
                "time_in_force": time_in_force,
                "id": len(self.orders) + 1,
            }
            self.orders.append(order)
            return order

        async def get_account(self):
            return self.account

        async def get_bars(self, symbols, timeframe, start, end):
            # Generate mock price data
            mock_data = {}
            for symbol in symbols:
                dates = pd.date_range(start=start, end=end, freq="D")
                close_prices = np.random.normal(100, 5, len(dates))

                # Ensure prices have a trend for testing
                for i in range(1, len(close_prices)):
                    close_prices[i] = close_prices[i - 1] * (1 + np.random.normal(0.001, 0.02))

                data = []
                for i, date in enumerate(dates):
                    bar = {
                        "t": date,
                        "o": close_prices[i] * 0.99,
                        "h": close_prices[i] * 1.02,
                        "l": close_prices[i] * 0.98,
                        "c": close_prices[i],
                        "v": np.random.randint(1000, 100000),
                    }
                    data.append(bar)
                mock_data[symbol] = data
            return mock_data

    return MockBroker()


@pytest.fixture
def test_symbols():
    """Test symbols for strategies"""
    return ["AAPL", "MSFT", "AMZN"]


@pytest.fixture
def momentum_strategy(mock_broker):
    """Momentum strategy fixture"""
    from strategies.momentum_strategy import MomentumStrategy

    strategy = MomentumStrategy(broker=mock_broker)
    strategy.set_parameters(
        {
            "position_size": 0.1,
            "max_positions": 3,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
        }
    )
    return strategy


@pytest.fixture
def mean_reversion_strategy(mock_broker):
    """Mean reversion strategy fixture"""
    from strategies.mean_reversion_strategy import MeanReversionStrategy

    strategy = MeanReversionStrategy(broker=mock_broker)
    strategy.set_parameters(
        {
            "position_size": 0.1,
            "max_positions": 3,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "bb_period": 20,
            "bb_std_dev": 2,
        }
    )
    return strategy


@pytest.fixture
def test_period():
    """Test period for backtesting"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date, end_date
