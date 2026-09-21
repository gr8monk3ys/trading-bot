"""OrderSubmission is tested through its interface against the backtest adapter."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from brokers.backtest import BacktestBroker, ExecutionProfile
from engine.order_submission import OrderIntent, OrderOutcome, OrderStatus, OrderSubmission
from utils.circuit_breaker import TradingHaltedException


def _broker(**kw):
    broker = BacktestBroker(
        initial_balance=100_000, execution_profile="idealistic", random_seed=7, **kw
    )
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    broker.set_price_data(
        "SPY",
        pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=dates),
    )
    broker.advance_to(datetime(2024, 1, 2))
    return broker


async def test_market_entry_fills():
    submission = OrderSubmission(_broker())
    outcome = await submission.submit(OrderIntent("SPY", "buy", 10))
    assert outcome.status is OrderStatus.FILLED and outcome.ok
    assert outcome.qty_filled == 10 and outcome.fill_price == pytest.approx(100.0, rel=0.01)
    assert submission.history == [outcome]


async def test_fractional_quantity_survives():
    broker = _broker()
    outcome = await OrderSubmission(broker).submit(OrderIntent("SPY", "buy", 0.4))
    assert outcome.status is OrderStatus.FILLED and outcome.qty_filled == pytest.approx(0.4)
    (pos,) = await broker.get_positions()
    assert pos.qty == pytest.approx(0.4)


async def test_zero_quantity_is_rejected_without_touching_the_broker():
    broker = _broker()
    outcome = await OrderSubmission(broker).submit(OrderIntent("SPY", "buy", 0))
    assert outcome.status is OrderStatus.REJECTED and outcome.reason == "zero_quantity"
    assert await broker.get_orders() == []


async def test_simulated_broker_rejection_is_an_outcome_not_a_fill():
    broker = _broker()
    broker.set_execution_profile(
        ExecutionProfile("always_reject", 1.0, 1.0, 1, 1, reject_probability=1.0)
    )
    outcome = await OrderSubmission(broker).submit(OrderIntent("SPY", "buy", 10))
    assert outcome.status is OrderStatus.REJECTED and not outcome.ok
    assert outcome.qty_filled == 0 and outcome.reason == "simulated_liquidity_reject"
    assert await broker.get_positions() == []


async def test_exit_goes_through_the_same_path():
    broker = _broker()
    submission = OrderSubmission(broker)
    await submission.submit(OrderIntent("SPY", "buy", 10))
    outcome = await submission.submit(OrderIntent("SPY", "sell", 10, is_exit=True))
    assert outcome.status is OrderStatus.FILLED
    assert await broker.get_positions() == []
    assert len(await broker.get_orders()) == 2


async def test_protective_levels_reach_the_backtest_broker():
    broker = _broker()
    await OrderSubmission(broker).submit(
        OrderIntent("SPY", "buy", 10, stop_loss=95.0, take_profit=110.0)
    )
    assert broker._stop_orders["SPY"]["stop_price"] == 95.0
    assert broker._stop_orders["SPY"]["quantity"] == 10
    assert broker._stop_orders["SPY"]["side"] == "sell"


async def test_exit_never_registers_a_stop():
    broker = _broker()
    submission = OrderSubmission(broker)
    await submission.submit(OrderIntent("SPY", "buy", 10))
    await submission.submit(OrderIntent("SPY", "sell", 10, is_exit=True, stop_loss=1.0))
    assert "SPY" not in broker._stop_orders


async def test_halted_entry_never_reaches_the_broker_but_exits_pass():
    broker = _broker()
    breaker = MagicMock()

    async def enforce(is_exit_order=False):
        if not is_exit_order:
            raise TradingHaltedException("Trading halted: daily loss", reason="daily_loss_limit")

    breaker.enforce_before_order = AsyncMock(side_effect=enforce)
    audit = MagicMock()
    submission = OrderSubmission(broker, circuit_breaker=breaker, audit_log=audit)
    entry = await submission.submit(OrderIntent("SPY", "buy", 10, strategy_name="s"))
    assert entry.status is OrderStatus.HALTED and "halted" in entry.reason
    assert await broker.get_orders() == []
    audit.log.assert_called_once()
    broker.positions["SPY"] = {"symbol": "SPY", "quantity": 10, "entry_price": 100.0}
    exit_ = await submission.submit(OrderIntent("SPY", "sell", 10, is_exit=True))
    assert exit_.status is OrderStatus.FILLED


async def test_live_shaped_reply_that_is_resting_is_accepted():
    broker = MagicMock()
    broker.enable_gateway_requirement = MagicMock(return_value="tok")
    broker._internal_submit_order = AsyncMock(
        return_value=MagicMock(id="abc", status="new", filled_qty="0", qty="1", side="buy")
    )
    outcome = await OrderSubmission(broker).submit(
        OrderIntent("AAPL", "buy", 1, order_type="limit", limit_price=50.0)
    )
    assert outcome.status is OrderStatus.ACCEPTED and outcome.order_id == "abc"
    broker._internal_submit_order.assert_awaited_once()
    assert broker._internal_submit_order.await_args.kwargs["gateway_token"] == "tok"


async def test_broker_exception_is_a_rejected_outcome():
    broker = MagicMock(spec=["submit_order_advanced"])
    broker.submit_order_advanced = AsyncMock(side_effect=RuntimeError("boom"))
    outcome = await OrderSubmission(broker).submit(OrderIntent("AAPL", "buy", 1))
    assert outcome == OrderOutcome(OrderStatus.REJECTED, "AAPL", "buy", 1.0, 0.0, "", None, "boom")


async def test_builder_market_orders_are_priced_as_market_orders():
    """Regression: the enum's str() used to miss the 'market' slippage branch."""
    broker = _broker()
    await OrderSubmission(broker).submit(OrderIntent("SPY", "buy", 10))
    (order,) = await broker.get_orders()
    assert order["type"] == "market"
    assert order["side"] == "buy"


async def test_limit_intent_reaches_the_backtest_broker_as_a_limit_order():
    broker = _broker()
    await OrderSubmission(broker).submit(
        OrderIntent("SPY", "buy", 10, order_type="limit", limit_price=99.5)
    )
    (order,) = await broker.get_orders()
    assert order["type"] == "limit" and order["price"] == 99.5
