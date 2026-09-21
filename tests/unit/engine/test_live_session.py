"""LiveSession: the websocket clock. One bar in, one session run, intents submitted."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from brokers.protocol import Position
from engine.order_submission import OrderIntent, OrderOutcome, OrderStatus
from engine.session import LiveSession, PortfolioView
from strategies.momentum_strategy import MomentumStrategy

WHEN = datetime(2024, 3, 1, 10, 30)


def _broker(positions=(), cash=100_000.0):
    broker = MagicMock()
    broker.get_positions = AsyncMock(return_value=list(positions))
    broker.get_account = AsyncMock(
        return_value=MagicMock(cash=str(cash), equity=str(cash), buying_power=str(cash))
    )
    broker.get_latest_quote = AsyncMock(return_value=MagicMock(ask_price=100.0))
    subscribers = set()
    broker._subscribers = subscribers
    broker._add_subscriber = lambda s: subscribers.add(s)
    broker._remove_subscriber = lambda s: subscribers.discard(s)
    return broker


def _submission():
    sub = MagicMock()
    sub.submit = AsyncMock(
        side_effect=lambda i: OrderOutcome(OrderStatus.FILLED, i.symbol, i.side, i.qty, i.qty, "id")
    )
    return sub


class _Recorder:
    def __init__(self):
        self.seen = []

    async def prepare(self, when, histories):
        self.seen.append(("prepare", when, {k: len(v) for k, v in histories.items()}))

    async def decide(self, symbol, when, view):
        return [OrderIntent(symbol, "buy", 1)]


async def test_each_bar_runs_a_session_with_the_growing_history():
    strategy = _Recorder()
    broker = _broker()
    session = LiveSession(strategy, broker, _submission(), ["AAPL"], history=3)
    session.subscribe()
    assert broker._subscribers == {session}
    assert strategy.execution_mode == "live"

    for i in range(4):
        report = await session.on_bar("AAPL", 1, 2, 0.5, 1.5, 10, WHEN + timedelta(minutes=i))
    assert report.intents == 1 and report.outcomes[0].status is OrderStatus.FILLED
    assert [s[2]["AAPL"] for s in strategy.seen] == [1, 2, 3, 3]  # deque keeps `history` bars


async def test_bars_for_unknown_symbols_are_ignored():
    strategy = _Recorder()
    session = LiveSession(strategy, _broker(), _submission(), ["AAPL"])
    assert await session.on_bar("MSFT", 1, 1, 1, 1, 1, WHEN) is None
    assert strategy.seen == []


# --- the live rules, through decide() ------------------------------------------


def _live_momentum(**params):
    strategy = MomentumStrategy(broker=AsyncMock(), parameters={"symbols": ["AAPL"], **params})
    strategy.symbols = ["AAPL"]
    strategy.signals, strategy.current_prices, strategy.price_history = {}, {}, {"AAPL": []}
    strategy.stop_prices, strategy.target_prices, strategy.entry_prices = {}, {}, {}
    strategy.peak_prices, strategy.last_signal_time = {}, {}
    strategy._initialize_parameters()
    for key, default in (
        ("use_trailing_stop", False),
        ("trailing_stop_pct", 0.02),
        ("trailing_activation_pct", 0.02),
        ("position_size", 0.1),
        ("max_positions", 5),
        ("stop_loss", 0.03),
        ("take_profit", 0.05),
    ):
        setattr(strategy, key, params.get(key, default))
    strategy.execution_mode = "live"
    strategy.risk_manager = None
    strategy.enforce_position_size_limit = AsyncMock(side_effect=lambda s, v, p, **k: (v, v / p))
    return strategy


def _view(positions=(), cash=100_000.0):
    return PortfolioView(
        equity=cash,
        cash=cash,
        positions={p.symbol: p for p in positions},
        buying_power=cash,
        _ask=AsyncMock(return_value=100.0),
    )


async def test_live_buy_is_a_bracket_intent_and_records_tracking():
    strategy = _live_momentum(
        use_kelly_criterion=False, position_size=0.1, stop_loss=0.03, take_profit=0.05
    )
    strategy.signals["AAPL"] = "buy"
    strategy.current_prices["AAPL"] = 100.0

    (intent,) = await strategy.decide("AAPL", WHEN, _view())

    assert intent.side == "buy" and intent.qty == 100 and intent.time_in_force == "gtc"
    assert intent.stop_loss == 97.0 and intent.take_profit == 105.0
    assert strategy.entry_prices["AAPL"] == 100.0 and strategy.last_signal_time["AAPL"] == WHEN


async def test_live_cooldown_and_max_positions_block_entries():
    strategy = _live_momentum(use_kelly_criterion=False, max_positions=1)
    strategy.signals["AAPL"] = "buy"
    strategy.current_prices["AAPL"] = 100.0
    strategy.last_signal_time["AAPL"] = WHEN - timedelta(minutes=5)
    assert await strategy.decide("AAPL", WHEN, _view()) == []
    strategy.last_signal_time.clear()
    full = _view(positions=[Position("MSFT", 10, 50.0)])
    assert await strategy.decide("AAPL", WHEN, full) == []


async def test_live_sell_signal_exits_the_long():
    strategy = _live_momentum(use_kelly_criterion=False)
    strategy.signals["AAPL"] = "sell"
    strategy.current_prices["AAPL"] = 100.0
    (intent,) = await strategy.decide("AAPL", WHEN, _view(positions=[Position("AAPL", 40, 90.0)]))
    assert intent.is_exit and intent.side == "sell" and intent.qty == 40


async def test_trailing_stop_exit_fires_after_giving_back_from_the_peak():
    strategy = _live_momentum(
        use_kelly_criterion=False,
        use_trailing_stop=True,
        trailing_stop_pct=0.02,
        trailing_activation_pct=0.02,
    )
    strategy.signals["AAPL"] = "neutral"
    strategy.entry_prices["AAPL"] = 100.0
    held = _view(positions=[Position("AAPL", 10, 100.0)])
    strategy.current_prices["AAPL"] = 110.0
    assert await strategy.decide("AAPL", WHEN, held) == []  # sets the peak
    strategy.current_prices["AAPL"] = 107.0  # > 2% below the 110 peak
    (intent,) = await strategy.decide("AAPL", WHEN, held)
    assert intent.is_exit and intent.reason == "trailing_stop_long" and intent.qty == 10
    assert "AAPL" not in strategy.entry_prices


async def test_daily_mode_ignores_trailing_stops_unless_asked():
    strategy = _live_momentum(
        use_trailing_stop=True, trailing_stop_pct=0.02, trailing_activation_pct=0.02
    )
    strategy.execution_mode = "daily"
    strategy.signals["AAPL"] = "neutral"
    strategy.entry_prices["AAPL"] = 100.0
    strategy.peak_prices["AAPL"] = 110.0
    strategy.current_prices["AAPL"] = 105.0  # in profit, but 4.5% below the peak
    held = _view(positions=[Position("AAPL", 10, 100.0)])
    assert await strategy.decide("AAPL", WHEN, held) == []
    strategy.parameters["daily_exits"] = True
    (intent,) = await strategy.decide("AAPL", WHEN, held)
    assert intent.is_exit
