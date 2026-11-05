"""
Tests for the order-time gate of utils/circuit_breaker.py.

`enforce_before_order` is the check OrderSubmission runs before every order, so
these tests drive it (and the calendar plumbing it consults) with a stubbed
broker account and a stubbed economic calendar. The breaker's own threshold,
true-peak, caching and fail-safe logic runs unmocked.
"""

import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

import utils.circuit_breaker as circuit_breaker_module
from utils.circuit_breaker import CircuitBreaker, TradingHaltedException

STARTING_EQUITY = 100_000.0
MAX_DAILY_LOSS = 0.03  # rapid drawdown threshold = 0.03 * 0.67 = 2.01%


def _account(equity: float) -> MagicMock:
    account = MagicMock()
    account.equity = str(equity)
    return account


class StubCalendar:
    """Stands in for EconomicEventCalendar; records how often it is asked."""

    def __init__(self, is_safe: bool, info: dict):
        self.is_safe = is_safe
        self.info = info
        self.calls = 0

    def is_safe_to_trade(self):
        self.calls += 1
        return self.is_safe, self.info


@pytest.fixture
def broker():
    broker = AsyncMock()
    broker.get_account.return_value = _account(STARTING_EQUITY)
    return broker


@pytest.fixture
def cb(broker):
    cb = CircuitBreaker(
        max_daily_loss=MAX_DAILY_LOSS,
        auto_close_positions=False,
        use_economic_calendar=False,
    )
    cb.broker = broker
    cb.starting_equity = STARTING_EQUITY
    cb.peak_equity_today = STARTING_EQUITY
    cb.last_reset_date = datetime.now().date()
    return cb


def _expire_order_cache(cb: CircuitBreaker) -> None:
    """Age the 1-second order-check cache so the next call hits the broker."""
    cb._order_check_time -= cb._order_check_ttl + 1


class TestEnforceBeforeOrder:
    async def test_requires_initialized_broker(self):
        cb = CircuitBreaker(use_economic_calendar=False)
        with pytest.raises(RuntimeError, match="not initialized"):
            await cb.enforce_before_order()

    async def test_allows_order_within_limits_and_caches_result(self, cb, broker):
        broker.get_account.return_value = _account(99_000.0)  # -1%

        await cb.enforce_before_order()
        await cb.enforce_before_order()  # within 1s TTL: served from cache

        assert broker.get_account.await_count == 1
        assert cb._order_check_cache is False
        assert cb.trading_halted is False

    async def test_daily_loss_limit_raises_with_reason_and_loss(self, cb, broker):
        broker.get_account.return_value = _account(96_900.0)  # -3.1%

        with pytest.raises(TradingHaltedException) as exc_info:
            await cb.enforce_before_order()

        exc = exc_info.value
        assert exc.reason == "daily_loss_limit"
        assert exc.loss_pct == pytest.approx(0.031)
        assert cb.trading_halted is True
        assert cb._halt_reason == "daily_loss_limit"

    async def test_rapid_drawdown_measured_from_true_peak(self, cb, broker):
        # Run up to a 110k peak, then give some back without tripping anything.
        broker.get_account.return_value = _account(110_000.0)
        await cb.enforce_before_order()
        _expire_order_cache(cb)
        broker.get_account.return_value = _account(108_500.0)  # -1.36% from peak
        await cb.enforce_before_order()
        assert cb._true_peak_equity == 110_000.0

        # Still up on the day (+7.7%), but 2.09% off the true peak: halt.
        _expire_order_cache(cb)
        broker.get_account.return_value = _account(107_700.0)
        with pytest.raises(TradingHaltedException) as exc_info:
            await cb.enforce_before_order()

        assert exc_info.value.reason == "rapid_drawdown"
        assert exc_info.value.loss_pct == pytest.approx(2_300 / 110_000)
        assert cb.trading_halted is True

    async def test_already_halted_raises_without_broker_call(self, cb, broker):
        cb.trading_halted = True
        cb._halt_reason = "rapid_drawdown"
        cb._halt_loss_pct = 0.025

        with pytest.raises(TradingHaltedException) as exc_info:
            await cb.enforce_before_order(is_exit_order=True)

        assert exc_info.value.reason == "rapid_drawdown"
        assert exc_info.value.loss_pct == 0.025
        broker.get_account.assert_not_awaited()

    async def test_halted_without_recorded_reason_defaults_to_daily_loss(self, cb):
        cb.trading_halted = True

        with pytest.raises(TradingHaltedException) as exc_info:
            await cb.enforce_before_order()

        assert exc_info.value.reason == "daily_loss_limit"
        assert exc_info.value.loss_pct is None

    async def test_cached_halt_within_ttl_raises(self, cb, broker):
        cb._order_check_cache = True
        cb._order_check_time = circuit_breaker_module.time.monotonic()

        with pytest.raises(TradingHaltedException) as exc_info:
            await cb.enforce_before_order()

        assert exc_info.value.reason == "cached_halt"
        broker.get_account.assert_not_awaited()

    async def test_broker_error_fails_safe_and_blocks_order(self, cb, broker):
        boom = ConnectionError("api down")
        broker.get_account.side_effect = boom

        with pytest.raises(TradingHaltedException) as exc_info:
            await cb.enforce_before_order()

        assert exc_info.value.reason == "check_error"
        assert exc_info.value.__cause__ is boom

    async def test_new_day_resets_baseline_and_allows_order(self, cb, broker):
        cb.last_reset_date = datetime.now().date() - timedelta(days=1)
        broker.get_account.return_value = _account(90_000.0)  # -10% vs yesterday

        await cb.enforce_before_order()

        assert cb.starting_equity == 90_000.0
        assert cb._true_peak_equity == 90_000.0
        assert cb.last_reset_date == datetime.now().date()
        assert cb.trading_halted is False

    async def test_high_impact_event_blocks_entries_only(self, cb, broker):
        cb.economic_calendar = StubCalendar(
            is_safe=False, info={"blocking_event": "FOMC", "hours_until_safe": 1.5}
        )

        with pytest.raises(TradingHaltedException) as exc_info:
            await cb.enforce_before_order(is_exit_order=False)

        assert exc_info.value.reason == "economic_event"
        assert exc_info.value.event_name == "FOMC"
        assert "1.5h" in str(exc_info.value)
        broker.get_account.assert_not_awaited()

        # Exits bypass event blocking and fall through to the equity check.
        await cb.enforce_before_order(is_exit_order=True)
        broker.get_account.assert_awaited_once()


class TestCheckBeforeOrder:
    async def test_returns_false_when_allowed(self, cb):
        assert await cb.check_before_order() is False

    async def test_returns_true_instead_of_raising_when_halted(self, cb, broker):
        broker.get_account.return_value = _account(96_000.0)  # -4%
        assert await cb.check_before_order() is True
        assert cb.trading_halted is True


class TestQuickCheck:
    def test_halted_flag_wins(self, cb):
        cb.trading_halted = True
        assert cb.quick_check() is True

    def test_not_halted(self, cb):
        assert cb.quick_check() is False

    def test_stale_day_is_permissive(self, cb):
        cb.last_reset_date = datetime.now().date() - timedelta(days=1)
        assert cb.quick_check() is False


class TestEconomicCalendarPlumbing:
    def test_without_calendar_everything_is_neutral(self, cb):
        assert cb.check_economic_events() == {
            "is_safe": True,
            "blocking_event": None,
            "hours_until_safe": 0,
            "position_multiplier": 1.0,
            "events_today": [],
        }
        assert cb.get_position_multiplier() == 1.0
        assert cb.is_blocked_by_event() == (False, None, 0)

    def test_calendar_result_is_normalized_and_cached(self, cb):
        calendar = StubCalendar(is_safe=True, info={"position_multiplier": 0.5})
        cb.economic_calendar = calendar

        first = cb.check_economic_events()
        second = cb.check_economic_events()

        assert first == {
            "is_safe": True,
            "blocking_event": None,
            "hours_until_safe": 0,
            "position_multiplier": 0.5,
            "events_today": [],
        }
        assert second is first
        assert calendar.calls == 1
        assert cb.get_position_multiplier() == 0.5
        assert cb.is_blocked_by_event() == (False, None, 0)

    def test_cache_expires_after_ttl(self, cb):
        calendar = StubCalendar(is_safe=True, info={})
        cb.economic_calendar = calendar

        cb.check_economic_events()
        cb._calendar_cache_time -= cb._calendar_cache_ttl + 1
        cb.check_economic_events()

        assert calendar.calls == 2

    def test_get_status_includes_calendar_when_configured(self, cb):
        cb.economic_calendar = StubCalendar(
            is_safe=False, info={"blocking_event": "CPI", "hours_until_safe": 2.0}
        )

        status = cb.get_status()

        assert status["economic_calendar"]["is_safe"] is False
        assert status["economic_calendar"]["blocking_event"] == "CPI"

    def test_missing_calendar_module_disables_feature(self, cb, monkeypatch):
        monkeypatch.setattr(circuit_breaker_module, "EconomicEventCalendar", None)
        monkeypatch.setitem(sys.modules, "utils.economic_calendar", None)
        cb.use_economic_calendar = True

        cb._init_economic_calendar()

        assert cb.use_economic_calendar is False
        assert cb.economic_calendar is None


class TestManualResetAccountError:
    async def test_reset_still_happens_when_loss_lookup_fails(self, cb, broker):
        cb.trading_halted = True
        broker.get_account.side_effect = [ConnectionError("api down"), _account(97_000.0)]

        assert await cb.manual_reset(confirmation_token="CONFIRM_RESET") is True

        assert cb.trading_halted is False
        assert cb.starting_equity == 97_000.0


def test_set_order_submission_attaches_gateway(cb):
    gateway = object()
    cb.set_order_submission(gateway)
    assert cb.order_submission is gateway
