"""
Parity tests: the live MomentumStrategy can run inside BacktestEngine.

Covers:
- Regression: orders flow through OrderGateway when MomentumStrategy is driven
  by BacktestEngine (the gap fixed in this PR; signals previously generated
  but execute_trade was a `pass`).
- Cooldown: _execute_signal honors the engine-supplied
  `_current_simulated_time` instead of wall-clock datetime.now().
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from engine.backtest_engine import BacktestEngine
from strategies.momentum_strategy import MomentumStrategy


# ---------------------------------------------------------------------------
# Regression: live MomentumStrategy executes orders via BacktestEngine
# ---------------------------------------------------------------------------


@dataclass
class _Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _build_trending_bars(n_days: int = 80, seed: int = 42) -> list[_Bar]:
    """Synthetic OHLCV tuned to produce at least one buy signal in
    MomentumStrategy's standard mode.

    Phase 1 (40 days): mild drift down — lets RSI cool below overbought.
    Phase 2 (40 days): gentle linear uptrend — RSI stays mostly < 70 so
        the score is not penalized.
    Volume: low baseline (~800k) with periodic 2.5M spikes — satisfies
        `volume > 1.5 * volume_ma` only on spike days.

    The combination yields:
      - MACD bullish (MACD > signal, histogram > 0) → +1
      - fast > medium > slow MA alignment → +1
      - ADX > 25 (sustained directional move) → trend_strength satisfied
      - Spike-day volume > 1.5 * 20-day MA → volume_confirmation satisfied

    A trade fires when these align — typically once or twice in the second
    half of the series. Tuning is fragile by design; the test only requires
    `total_trades > 0` so we don't pin a specific day.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2)

    phase1 = 100.0 - np.linspace(0, 5, 40) + rng.normal(0, 0.3, 40)
    phase2 = phase1[-1] + np.linspace(0.5, 10, 40) + rng.normal(0, 0.3, 40)
    closes = np.concatenate([phase1, phase2])

    volumes = np.full(80, 800_000.0)
    # Spike every 4th day during phase 2
    for i in range(40, 80, 4):
        volumes[i] = 2_500_000.0
    volumes = volumes + rng.normal(0, 30_000, 80)

    bars: list[_Bar] = []
    for i in range(len(closes)):
        prev = closes[i - 1] if i > 0 else closes[0]
        c = float(closes[i])
        o = float(prev)
        h = float(max(o, c) + 0.5)
        low = float(min(o, c) - 0.5)
        bars.append(
            _Bar(
                timestamp=start + timedelta(days=i),
                open=o,
                high=h,
                low=low,
                close=c,
                volume=float(volumes[i]),
            )
        )
    return bars[:n_days] if n_days <= len(bars) else bars


class _FakeDataBroker:
    """Stub for AlpacaBroker that yields a deterministic synthetic series."""

    def __init__(self, bars: list[_Bar], *args, **kwargs):
        self._bars = bars

    async def get_bars(self, symbol, start, end, timeframe="1Day"):
        return list(self._bars)


class _FakeHistoricalUniverse:
    def __init__(self, broker=None):
        self.broker = broker

    async def initialize(self):
        return None

    def get_statistics(self):
        return {"total_symbols": 1}

    def get_tradeable_symbols(self, _date, symbols):
        return symbols


@pytest.mark.asyncio
async def test_momentum_strategy_runs_in_backtest_engine_and_places_orders(monkeypatch):
    """
    Regression for the gap fixed by this PR.

    Before the fix, MomentumStrategy.execute_trade was a no-op, so running it
    through BacktestEngine produced signals but zero orders. After the fix,
    execute_trade dispatches to _execute_signal and _check_exit_conditions,
    and orders flow through OrderGateway → BacktestBroker.

    Asserts that at least one trade is recorded by the engine.
    """
    bars = _build_trending_bars()

    def _fake_alpaca(*_args, **_kwargs):
        return _FakeDataBroker(bars)

    monkeypatch.setattr("brokers.alpaca_broker.AlpacaBroker", _fake_alpaca)
    monkeypatch.setattr("engine.backtest_engine.HistoricalUniverse", _FakeHistoricalUniverse)

    engine = BacktestEngine()
    result = await engine.run_backtest(
        strategy_class=MomentumStrategy,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 4, 30),
        initial_capital=100_000,
        # Disable expensive features that need additional infra in unit tests
        strategy_params={
            "use_multi_timeframe": False,
            "use_volatility_regime": False,
            "use_kelly_criterion": False,
        },
    )

    assert result["total_trades"] > 0, (
        "MomentumStrategy produced zero trades in BacktestEngine — "
        "execute_trade is bypassing the order path again."
    )

    # Bracket orders must reach the broker with their actual fractional qty
    # intact. Before the fix in fix/backtest-broker-fractional-qty, bracket
    # orders with sub-1 qty (common with Kelly/volatility sizing) were
    # truncated to 0 inside BacktestBroker.submit_order_advanced — orders
    # nominally "happened" but the trade record showed quantity=0 and the
    # equity curve stayed flat. This assertion locks the contract closed.
    first_trade = result["trades"][0]
    assert first_trade["quantity"] > 0, (
        f"Bracket order reached the broker but was sized to "
        f"{first_trade['quantity']} — BacktestBroker.submit_order_advanced "
        "is truncating fractional qty (see fix/backtest-broker-fractional-qty)."
    )


# ---------------------------------------------------------------------------
# Cooldown: simulated time, not wall clock, gates _execute_signal
# ---------------------------------------------------------------------------


def _make_strategy_for_cooldown_test() -> MomentumStrategy:
    """Construct a MomentumStrategy with the minimum state for _execute_signal."""
    broker = AsyncMock()
    broker.get_positions = AsyncMock(return_value=[])
    broker.get_account = AsyncMock(
        return_value=SimpleNamespace(buying_power="100000", equity="100000")
    )

    strategy = MomentumStrategy(
        broker=broker,
        parameters={"symbols": ["AAPL"]},
    )
    strategy.symbols = ["AAPL"]
    strategy.last_signal_time = {"AAPL": None}
    strategy.current_prices = {"AAPL": 100.0}
    strategy.price_history = {"AAPL": []}
    strategy.parameters = {"use_kelly_criterion": False}
    return strategy


@pytest.mark.asyncio
async def test_cooldown_uses_simulated_time_not_wall_clock(monkeypatch):
    """
    The cooldown gate in _execute_signal must read `_current_simulated_time`
    when the engine has set it, otherwise a fast historical replay (where
    datetime.now() advances in microseconds) erroneously blocks every signal
    after the first.

    Strategy: stamp last_signal_time at simulated T0, then call _execute_signal
    with simulated T0 + 30min — must short-circuit (cooldown). Then call again
    with simulated T0 + 2h — must proceed past the cooldown gate.
    """
    strategy = _make_strategy_for_cooldown_test()

    sentinel_buy = AsyncMock()
    monkeypatch.setattr(strategy, "_execute_buy_signal", sentinel_buy)

    t0 = datetime(2024, 6, 3, 14, 0, 0)
    strategy.last_signal_time["AAPL"] = t0

    # Inside cooldown window — must NOT dispatch
    strategy._current_simulated_time = t0 + timedelta(minutes=30)
    await strategy._execute_signal("AAPL", "buy")
    assert sentinel_buy.await_count == 0, (
        "Cooldown gate failed under simulated time: dispatched within 1h window."
    )

    # Past cooldown window — MUST dispatch
    strategy._current_simulated_time = t0 + timedelta(hours=2)
    await strategy._execute_signal("AAPL", "buy")
    assert sentinel_buy.await_count == 1, (
        "Cooldown gate over-blocked under simulated time: did not dispatch after 2h."
    )


@pytest.mark.asyncio
async def test_cooldown_falls_back_to_wall_clock_when_simulated_time_absent(monkeypatch):
    """When `_current_simulated_time` is unset (live mode), fall back to
    datetime.now() — preserving existing live behavior."""
    strategy = _make_strategy_for_cooldown_test()

    sentinel_buy = AsyncMock()
    monkeypatch.setattr(strategy, "_execute_buy_signal", sentinel_buy)

    # No `_current_simulated_time` attribute set → wall-clock path.
    # last_signal_time well in the past → cooldown should not block.
    strategy.last_signal_time["AAPL"] = datetime.now() - timedelta(hours=24)

    await strategy._execute_signal("AAPL", "buy")
    assert sentinel_buy.await_count == 1, (
        "Wall-clock cooldown fallback rejected a signal that should have passed."
    )
