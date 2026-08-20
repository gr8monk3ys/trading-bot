"""Bar-subscription wiring for AdaptiveStrategy.

MomentumStrategy.initialize() and MeanReversionStrategy.initialize() each
subscribe themselves to the broker's bar feed. AdaptiveStrategy builds both as
sub-strategies, so in live mode BOTH arms received every bar and traded
independently through their own gateways, while AdaptiveStrategy — never
subscribed itself — never ran on_bar, never called _update_regime(), and left
current_regime at None forever.

The net effect was the opposite of what the class advertises: no regime
switching at all, and two strategies designed to take opposing positions
(momentum buys breakouts, mean reversion buys dips) trading the same account
at the same time.

These tests pin the fixed wiring: the coordinator owns the bar feed, and its
arms are driven only through its routing.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from strategies.adaptive_strategy import AdaptiveStrategy
from tests.unit.conftest import create_mock_account

SYMBOL = "AAPL"


def _make_broker():
    """AsyncMock broker with a real subscriber set, mirroring AlpacaBroker's API."""
    broker = AsyncMock()
    broker.get_account.return_value = create_mock_account()
    broker.get_positions.return_value = []
    broker.get_bars = AsyncMock(return_value=None)
    subscribers = set()
    broker._subscribers = subscribers
    broker._add_subscriber = lambda s: subscribers.add(s)
    broker._remove_subscriber = lambda s: subscribers.discard(s)
    return broker


async def _initialized_strategy():
    broker = _make_broker()
    strategy = AdaptiveStrategy(
        broker=broker,
        parameters={"symbols": [SYMBOL]},
        order_gateway=MagicMock(),
    )
    assert await strategy.initialize() is True
    return strategy, broker


async def test_adaptive_owns_the_bar_subscription():
    strategy, broker = await _initialized_strategy()

    assert strategy in broker._subscribers


async def test_arms_are_not_independently_subscribed():
    strategy, broker = await _initialized_strategy()

    # If either arm stays subscribed it trades on its own, bypassing regime
    # routing entirely — that is the bug this pins.
    assert strategy.momentum_strategy not in broker._subscribers
    assert strategy.mean_reversion_strategy not in broker._subscribers
    assert len(broker._subscribers) == 1


async def test_bar_reaches_only_the_active_arm():
    strategy, _ = await _initialized_strategy()
    strategy._update_regime = AsyncMock()  # isolate routing from regime detection
    strategy.active_strategy = strategy.momentum_strategy
    strategy.momentum_strategy.on_bar = AsyncMock()
    strategy.mean_reversion_strategy.on_bar = AsyncMock()

    await strategy.on_bar(SYMBOL, 100.0, 101.0, 99.0, 100.5, 1_000_000, datetime.now())

    strategy.momentum_strategy.on_bar.assert_awaited_once()
    strategy.mean_reversion_strategy.on_bar.assert_not_awaited()
