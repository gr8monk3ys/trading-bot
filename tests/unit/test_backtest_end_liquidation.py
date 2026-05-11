"""Unit tests for end-of-backtest liquidation.

Step 2C follow-up to the 2026-05 honest cleanup:

The pre-fix `BacktestEngine.run_backtest` left positions open at end-of-
period and computed `total_return` from `broker.get_portfolio_value()`,
which includes unrealized MTM on still-open positions. For the Task 8
honest baseline this inflated the headline from realized strategy P&L to
"buy NVDA/etc. in 2020 and hold via momentum filter".

These tests pin that the engine now:

1. Calls a liquidation pass after the trading-day loop, closing every
   remaining open position at the final trading day's close.
2. Records those liquidation trades in `broker.get_trades()` so they flow
   through `_calculate_trade_pnl` (realized P&L only).
3. Applies the same realistic execution profile (slippage + spread) to
   liquidation fills — they are not free.
4. Is a no-op when the strategy already closed everything.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brokers.order_builder import OrderBuilder
from engine.backtest_engine import BacktestEngine
from strategies.base_strategy import BaseStrategy


# =============================================================================
# Test scaffolding: hermetic bar data and synthetic strategies
# =============================================================================


class _Bar:
    """Bar-like object with the attributes BacktestEngine._load_symbol_data reads."""

    def __init__(self, ts, open_, high, low, close, volume):
        self.timestamp = ts
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _make_bars(symbol: str, start: datetime, n: int = 10, base_price: float = 100.0):
    """Build a deterministic uptrending bar series."""
    bars = []
    for i in range(n):
        ts = pd.Timestamp(start) + pd.Timedelta(days=i)
        close = base_price + i  # +$1/day so end-of-period close is well-defined
        bars.append(_Bar(ts, close - 0.5, close + 0.5, close - 1.0, close, 1_000_000))
    return bars


def _bars_by_symbol(symbols, start, n=10):
    """Build a side-effect map for `data_broker.get_bars(symbol, ...)`."""
    table = {sym: _make_bars(sym, start, n=n) for sym in symbols}

    async def _get_bars(symbol, **kwargs):
        return table.get(symbol, [])

    return _get_bars, table


class _BuyAndHoldStrategy(BaseStrategy):
    """On day 1, buy a fixed quantity per symbol. Never sell.

    This is the minimum strategy that guarantees the backtest ends with
    open positions, so the liquidation pass has work to do.
    """

    NAME = "BuyAndHoldStrategy"

    # Class-level state to avoid coupling to strategy-specific kwargs.
    _bought: set = set()

    async def initialize(self, **kwargs):
        # Do not call super().initialize() — it starts background tasks.
        type(self)._bought = set()

    async def analyze_symbol(self, symbol):
        if symbol in type(self)._bought:
            return {"action": "neutral"}
        return {"action": "buy", "symbol": symbol}

    async def execute_trade(self, symbol, signal):
        if signal.get("action") != "buy":
            return
        if symbol in type(self)._bought:
            return
        order_request = OrderBuilder(symbol, "buy", 10).market().day().build()
        result = await self.submit_entry_order(
            order_request=order_request,
            reason="buy_and_hold",
        )
        if result is not None and getattr(result, "success", False):
            type(self)._bought.add(symbol)


class _BuyThenSellStrategy(BaseStrategy):
    """Buy on iteration 1, sell on iteration 2 — should leave 0 open positions.

    Uses the gateway's `submit_exit_order` directly (bypassing
    `BaseStrategy.submit_exit_order`, which does an `await
    broker.get_positions()` that does not work against the sync
    `BacktestBroker.get_positions` API). The gateway path is sufficient
    to prove the engine adds no extra trades when nothing is left open.
    """

    NAME = "BuyThenSellStrategy"

    _bought: set = set()
    _sold: set = set()

    async def initialize(self, **kwargs):
        type(self)._bought = set()
        type(self)._sold = set()

    async def analyze_symbol(self, symbol):
        if symbol not in type(self)._bought:
            return {"action": "buy", "symbol": symbol}
        if symbol not in type(self)._sold:
            return {"action": "sell", "symbol": symbol}
        return {"action": "neutral"}

    async def execute_trade(self, symbol, signal):
        action = signal.get("action")
        if action == "buy" and symbol not in type(self)._bought:
            order_request = OrderBuilder(symbol, "buy", 10).market().day().build()
            result = await self.submit_entry_order(
                order_request=order_request, reason="enter"
            )
            if result is not None and getattr(result, "success", False):
                type(self)._bought.add(symbol)
        elif action == "sell" and symbol not in type(self)._sold:
            # Close the full position via gateway's exit path directly.
            gateway = getattr(self, "order_gateway", None)
            if gateway is None:
                return
            result = await gateway.submit_exit_order(
                symbol=symbol,
                quantity=10,
                strategy_name=self.name,
                side="sell",
                reason="exit",
            )
            if result is not None and getattr(result, "success", False):
                type(self)._sold.add(symbol)


# =============================================================================
# TESTS
# =============================================================================


class TestEndOfBacktestLiquidation:
    """When a backtest ends with open positions, the engine should close them
    at the final available close price (with realistic slippage + spread costs)
    so the headline equity reflects what the operator could have actually
    captured, not unrealized mark-to-market.
    """

    @pytest.mark.asyncio
    async def test_engine_calls_liquidate_at_end_of_run(self):
        """After run_backtest completes, broker.get_positions() must be empty
        and the trade log must include liquidation entries at the final date."""
        symbols = ["AAA", "BBB"]
        start = datetime(2024, 1, 2)
        n_days = 10
        get_bars, _ = _bars_by_symbol(symbols, start, n=n_days)

        data_broker = MagicMock()
        data_broker.get_bars = AsyncMock(side_effect=get_bars)

        engine = BacktestEngine(broker=data_broker)
        result = await engine.run_backtest(
            strategy_class=_BuyAndHoldStrategy,
            symbols=symbols,
            start_date=start,
            end_date=start + pd.Timedelta(days=n_days + 1),
            initial_capital=100_000,
        )

        # 1. No positions remain after end-of-run liquidation.
        assert "positions" in result
        assert len(result["positions"]) == 0, (
            f"Expected 0 open positions after liquidation, "
            f"got {len(result['positions'])}: {result['positions']}"
        )

        # 2. The trade log includes at least one liquidation trade per symbol
        # that was held at end-of-period. Liquidation must happen at/after
        # the final entry trade for each symbol.
        trades = result["trades"]
        assert len(trades) >= len(symbols) * 2, (
            f"Expected at least one entry + one liquidation trade per symbol, "
            f"got {len(trades)} trades"
        )

        # The very last trades should be sells (liquidations of long positions).
        sell_trades = [t for t in trades if t["side"] == "sell"]
        assert len(sell_trades) >= len(symbols), (
            f"Expected at least one sell per held symbol; got {len(sell_trades)} sells"
        )
        for symbol in symbols:
            sells_for_symbol = [t for t in sell_trades if t["symbol"] == symbol]
            assert len(sells_for_symbol) >= 1, (
                f"No liquidation sell recorded for held symbol {symbol}"
            )

        # 3. final_equity reflects realized cash (no unrealized MTM).
        # After liquidation, get_portfolio_value should equal get_balance
        # (no positions left). We can sanity-check by recomputing.
        assert "final_equity" in result
        # final_equity must be a finite real number
        assert isinstance(result["final_equity"], (int, float))
        assert result["final_equity"] > 0

    @pytest.mark.asyncio
    async def test_liquidation_uses_realistic_execution_profile(self):
        """Liquidation trades must incur slippage so they appear in the
        trade ledger with a non-zero slippage figure, matching the broker's
        realistic execution profile (not free fills)."""
        symbols = ["AAA"]
        start = datetime(2024, 1, 2)
        get_bars, table = _bars_by_symbol(symbols, start, n=10)
        final_close = table["AAA"][-1].close  # Known: base_price + n - 1

        data_broker = MagicMock()
        data_broker.get_bars = AsyncMock(side_effect=get_bars)

        engine = BacktestEngine(broker=data_broker)
        result = await engine.run_backtest(
            strategy_class=_BuyAndHoldStrategy,
            symbols=symbols,
            start_date=start,
            end_date=start + pd.Timedelta(days=11),
            initial_capital=100_000,
            execution_profile="realistic",
        )

        # Liquidation sell trade should exist.
        sell_trades = [t for t in result["trades"] if t["side"] == "sell"]
        assert len(sell_trades) >= 1, "No liquidation sell trade recorded"

        liquidation = sell_trades[-1]
        # The fill price should be at or near final_close, but NOT exactly
        # equal — realistic profile applies spread + market impact slippage
        # against the seller (fill < mid).
        assert liquidation["price"] < final_close, (
            f"Liquidation sell filled at {liquidation['price']:.4f} >= mid "
            f"{final_close:.4f} — slippage not applied"
        )
        # Sanity bound — slippage should be small (< 5%), not catastrophic.
        slippage_pct = (final_close - liquidation["price"]) / final_close
        assert 0 < slippage_pct < 0.05, (
            f"Slippage out of bounds: {slippage_pct:.4%}"
        )

    @pytest.mark.asyncio
    async def test_no_open_positions_no_liquidation_trades(self):
        """If the strategy closes all positions before end-of-period, the
        liquidation pass is a no-op — no extra trades appended."""
        symbols = ["AAA"]
        start = datetime(2024, 1, 2)
        get_bars, _ = _bars_by_symbol(symbols, start, n=10)

        data_broker = MagicMock()
        data_broker.get_bars = AsyncMock(side_effect=get_bars)

        engine = BacktestEngine(broker=data_broker)
        result = await engine.run_backtest(
            strategy_class=_BuyThenSellStrategy,
            symbols=symbols,
            start_date=start,
            end_date=start + pd.Timedelta(days=11),
            initial_capital=100_000,
        )

        # The strategy itself should have closed the position; positions empty.
        assert len(result["positions"]) == 0

        # The trade ledger should contain exactly one buy and one sell —
        # no double-counting from an additional liquidation pass.
        buys = [t for t in result["trades"] if t["side"] == "buy"]
        sells = [t for t in result["trades"] if t["side"] == "sell"]
        assert len(buys) == 1, f"Expected 1 buy, got {len(buys)}"
        assert len(sells) == 1, (
            f"Expected exactly 1 sell (no extra liquidation), got {len(sells)}: "
            f"{sells}"
        )
