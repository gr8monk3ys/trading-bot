"""Bar-subscription wiring for AdaptiveStrategy.

Before the session runner, MomentumStrategy.initialize() and
MeanReversionStrategy.initialize() each subscribed themselves to the broker's
bar feed, so under AdaptiveStrategy both arms traded independently while the
coordinator never saw a bar (#89). Now no strategy subscribes to anything:
the LiveSession owns the feed and routes bars through the coordinator's
prepare()/decide().
"""

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
        order_submission=MagicMock(),
    )
    assert await strategy.initialize() is True
    return strategy, broker


async def test_no_strategy_subscribes_to_bars_itself():
    """Strategies decide; the LiveSession that drives them owns the bar feed (ADR 0001).

    The #89 bug (both arms self-subscribed and traded independently while the
    coordinator never saw a bar) is unrepresentable now: nothing in strategies/
    touches the subscriber set.
    """
    strategy, broker = await _initialized_strategy()

    assert broker._subscribers == set()
    assert strategy.momentum_strategy is not None and strategy.mean_reversion_strategy is not None


async def test_live_session_is_the_only_subscriber_and_routes_to_the_coordinator():
    from engine.session import LiveSession

    strategy, broker = await _initialized_strategy()
    session = LiveSession(strategy, broker, MagicMock(), [SYMBOL])
    session.subscribe()

    assert broker._subscribers == {session}
    assert strategy.execution_mode == "live"
    session.unsubscribe()
    assert broker._subscribers == set()
