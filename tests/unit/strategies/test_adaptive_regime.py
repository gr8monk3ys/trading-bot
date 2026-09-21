"""One regime per bar; switching arms flattens the outgoing arm (ADR 0009)."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from brokers.protocol import Position
from engine.order_submission import OrderIntent
from engine.session import PortfolioView
from strategies.adaptive_strategy import AdaptiveStrategy

WHEN = datetime(2024, 3, 1)


def _view(positions=()):
    return PortfolioView(
        100_000.0,
        100_000.0,
        {p.symbol: p for p in positions},
        100_000.0,
        AsyncMock(return_value=100.0),
    )


async def _strategy():
    with (
        patch("strategies.adaptive_strategy.MomentumStrategy") as Mom,
        patch("strategies.adaptive_strategy.MeanReversionStrategy") as Mr,
        patch("strategies.adaptive_strategy.MarketRegimeDetector"),
    ):
        mom, mr = AsyncMock(), AsyncMock()
        mom.initialize = AsyncMock(return_value=True)
        mr.initialize = AsyncMock(return_value=True)
        mom.decide = AsyncMock(
            return_value=[OrderIntent("SPY", "sell", 5), OrderIntent("SPY", "buy", 5)]
        )
        mr.decide = AsyncMock(return_value=[OrderIntent("SPY", "buy", 7)])
        Mom.return_value, Mr.return_value = mom, mr
        s = AdaptiveStrategy(
            broker=AsyncMock(), parameters={"symbols": ["SPY", "QQQ"]}, order_submission=MagicMock()
        )
        assert await s.initialize() is True
        return s, mom, mr


def _regime(kind, mult, confidence=0.9):
    return {
        "type": kind,
        "confidence": confidence,
        "position_multiplier": mult,
        "recommended_strategy": kind,
    }


async def test_arms_read_one_multiplier_and_are_never_mutated():
    s, mom, mr = await _strategy()
    assert mom.regime_multiplier == s._regime_multiplier == mr.regime_multiplier
    assert s._regime_multiplier("SPY") == 1.0
    await s._switch_strategy(_regime("volatile", 0.5))
    assert s._regime_multiplier("SPY") == 0.5
    assert "position_size" not in [c[0] for c in mom.method_calls]


async def test_bull_regime_drops_short_entries_without_touching_the_arm():
    s, mom, mr = await _strategy()
    await s._switch_strategy(_regime("bull", 1.2))
    s.current_regime = "bull"
    intents = await s.decide("SPY", WHEN, _view())
    assert [i.side for i in intents] == ["buy"]
    s.current_regime = "bear"
    await s._switch_strategy(_regime("bear", 0.8))
    assert [i.side for i in await s.decide("SPY", WHEN, _view())] == ["sell", "buy"]


async def test_switching_arms_flattens_the_outgoing_arms_positions_first():
    s, mom, mr = await _strategy()
    assert await s._switch_strategy(_regime("bull", 1.0)) is False  # first arm, nothing to flatten
    assert await s._switch_strategy(_regime("sideways", 1.0)) is True
    s.current_regime = "sideways"
    held = _view(positions=[Position("SPY", 40, 90.0)])
    (exit_,) = await s.decide("SPY", WHEN, held)
    assert exit_.is_exit and exit_.side == "sell" and exit_.qty == 40 and "flatten" in exit_.reason
    mr.decide.assert_not_called()
    (mr_intent,) = await s.decide("SPY", WHEN, held)  # next session: the new arm decides
    assert mr_intent.qty == 7
    assert await s.decide("QQQ", WHEN, _view()) == [
        OrderIntent("SPY", "buy", 7)
    ]  # flat symbol: nothing to flatten


async def test_low_confidence_keeps_the_arm_and_the_multiplier():
    s, mom, mr = await _strategy()
    await s._switch_strategy(_regime("bull", 1.2))
    assert await s._switch_strategy(_regime("sideways", 0.5, confidence=0.1)) is False
    assert s.active_strategy is mom and s._regime_multiplier("SPY") == 1.2
