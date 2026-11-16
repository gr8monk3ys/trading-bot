"""
Tests for utils/order_lifecycle.py

The tracker is a small state machine: SUBMITTED and PARTIAL can move on,
FILLED / CANCELED / REJECTED are terminal, and an illegal transition is
dropped (state unchanged) rather than raised. The transition table is finite,
so the invariants are checked exhaustively over every state pair and every
three-step update sequence instead of with hand-picked examples.
"""

import itertools
import logging
from datetime import datetime

import pytest

from utils.order_lifecycle import (
    ALLOWED_TRANSITIONS,
    OrderLifecycleRecord,
    OrderLifecycleTracker,
    OrderState,
)

ORDER_ID = "order-1"
SYMBOL = "AAPL"
TERMINAL_STATES = {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED}
ALL_STATES = list(OrderState)


def _tracker_in_state(state: OrderState) -> OrderLifecycleTracker:
    """Build a tracker holding one order already in ``state``."""
    tracker = OrderLifecycleTracker()
    tracker.register_order(ORDER_ID, SYMBOL, "buy", 10)
    if state is OrderState.PARTIAL or state in TERMINAL_STATES:
        tracker.update_state(ORDER_ID, state)
    assert tracker.get_state(ORDER_ID) is state
    return tracker


def _expected_after(start: OrderState, sequence) -> OrderState:
    """Reference model: apply a transition only if the table allows it."""
    state = start
    for target in sequence:
        if target is state or target in ALLOWED_TRANSITIONS[state]:
            state = target
    return state


class TestTransitionTable:
    def test_every_state_has_an_entry(self):
        assert set(ALLOWED_TRANSITIONS) == set(OrderState)

    def test_terminal_states_allow_nothing(self):
        for state in TERMINAL_STATES:
            assert ALLOWED_TRANSITIONS[state] == set()

    def test_no_transition_back_to_submitted(self):
        for targets in ALLOWED_TRANSITIONS.values():
            assert OrderState.SUBMITTED not in targets


class TestRegisterOrder:
    def test_new_order_starts_submitted_with_fields(self):
        tracker = OrderLifecycleTracker()
        tracker.register_order(
            ORDER_ID, SYMBOL, "sell", 5.5, strategy_name="momentum", client_order_id="c-1"
        )

        exported = tracker.export_state()[ORDER_ID]
        assert tracker.get_state(ORDER_ID) is OrderState.SUBMITTED
        assert exported["symbol"] == SYMBOL
        assert exported["side"] == "sell"
        assert exported["quantity"] == 5.5
        assert exported["strategy_name"] == "momentum"
        assert exported["client_order_id"] == "c-1"

    def test_re_registering_resets_state_to_submitted(self):
        tracker = _tracker_in_state(OrderState.FILLED)

        tracker.register_order(ORDER_ID, SYMBOL, "buy", 10)

        assert tracker.get_state(ORDER_ID) is OrderState.SUBMITTED

    def test_unknown_order_has_no_state(self):
        assert OrderLifecycleTracker().get_state("missing") is None


class TestUpdateState:
    @pytest.mark.parametrize(("start", "target"), list(itertools.product(ALL_STATES, ALL_STATES)))
    def test_every_state_pair_follows_the_table(self, start, target):
        tracker = _tracker_in_state(start)

        tracker.update_state(ORDER_ID, target)

        legal = target is start or target in ALLOWED_TRANSITIONS[start]
        assert tracker.get_state(ORDER_ID) is (target if legal else start)

    @pytest.mark.parametrize("start", ALL_STATES)
    def test_every_three_step_sequence_matches_reference_model(self, start):
        for sequence in itertools.product(ALL_STATES, repeat=3):
            tracker = _tracker_in_state(start)
            for target in sequence:
                tracker.update_state(ORDER_ID, target)
            assert tracker.get_state(ORDER_ID) is _expected_after(start, sequence), sequence

    @pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES, key=lambda s: s.value))
    def test_terminal_state_is_never_left(self, terminal):
        tracker = _tracker_in_state(terminal)

        for target in ALL_STATES:
            tracker.update_state(ORDER_ID, target)
            assert tracker.get_state(ORDER_ID) is terminal

    def test_invalid_transition_warns_and_keeps_timestamp(self, caplog):
        tracker = _tracker_in_state(OrderState.FILLED)
        before = tracker.export_state()[ORDER_ID]["updated_at"]

        with caplog.at_level(logging.WARNING, logger="utils.order_lifecycle"):
            tracker.update_state(ORDER_ID, OrderState.CANCELED)

        assert "Invalid order transition filled -> canceled" in caplog.text
        assert tracker.export_state()[ORDER_ID]["updated_at"] == before

    def test_valid_transition_advances_updated_at(self):
        tracker = _tracker_in_state(OrderState.SUBMITTED)
        record_created = datetime.fromisoformat(tracker.export_state()[ORDER_ID]["created_at"])

        tracker.update_state(ORDER_ID, OrderState.PARTIAL)

        exported = tracker.export_state()[ORDER_ID]
        assert datetime.fromisoformat(exported["updated_at"]) >= record_created
        assert datetime.fromisoformat(exported["created_at"]) == record_created

    def test_unknown_order_is_a_logged_no_op(self, caplog):
        tracker = _tracker_in_state(OrderState.SUBMITTED)

        with caplog.at_level(logging.WARNING, logger="utils.order_lifecycle"):
            tracker.update_state("ghost", OrderState.FILLED)

        assert "unknown order ghost" in caplog.text
        assert tracker.get_state("ghost") is None
        assert set(tracker.export_state()) == {ORDER_ID}


class TestPersistence:
    def test_export_import_round_trip_preserves_every_field(self):
        source = OrderLifecycleTracker()
        source.register_order("a", "AAPL", "buy", 3, strategy_name="s1", client_order_id="c-a")
        source.register_order("b", "MSFT", "sell", 7.25)
        source.update_state("a", OrderState.PARTIAL)
        source.update_state("b", OrderState.REJECTED)
        exported = source.export_state()

        restored = OrderLifecycleTracker()
        restored.import_state(exported)

        assert restored.export_state() == exported
        assert restored.get_state("a") is OrderState.PARTIAL
        assert restored.get_state("b") is OrderState.REJECTED

    def test_restored_order_still_enforces_transitions(self):
        source = _tracker_in_state(OrderState.CANCELED)
        restored = OrderLifecycleTracker()
        restored.import_state(source.export_state())

        restored.update_state(ORDER_ID, OrderState.FILLED)

        assert restored.get_state(ORDER_ID) is OrderState.CANCELED

    def test_import_replaces_existing_orders(self):
        tracker = _tracker_in_state(OrderState.SUBMITTED)
        other = OrderLifecycleTracker()
        other.register_order("other", "SPY", "buy", 1)

        tracker.import_state(other.export_state())

        assert tracker.get_state(ORDER_ID) is None
        assert tracker.get_state("other") is OrderState.SUBMITTED

    def test_import_empty_state_clears_tracker(self):
        tracker = _tracker_in_state(OrderState.PARTIAL)

        tracker.import_state({})

        assert tracker.export_state() == {}

    def test_import_coerces_quantity_and_defaults_optional_fields(self):
        tracker = OrderLifecycleTracker()
        tracker.import_state(
            {
                "x": {
                    "order_id": "x",
                    "symbol": "QQQ",
                    "side": "buy",
                    "quantity": "12",
                    "state": "partial",
                    "created_at": "2026-01-02T09:30:00",
                    "updated_at": "2026-01-02T09:31:00",
                }
            }
        )

        exported = tracker.export_state()["x"]
        assert exported["quantity"] == 12.0
        assert isinstance(exported["quantity"], float)
        assert exported["strategy_name"] is None
        assert exported["client_order_id"] is None
        assert exported["created_at"] == "2026-01-02T09:30:00"
        assert tracker.get_state("x") is OrderState.PARTIAL

    def test_import_rejects_unknown_state_value(self):
        tracker = OrderLifecycleTracker()
        payload = _tracker_in_state(OrderState.SUBMITTED).export_state()
        payload[ORDER_ID]["state"] = "expired"

        with pytest.raises(ValueError):
            tracker.import_state(payload)

    def test_import_rejects_missing_required_field(self):
        tracker = OrderLifecycleTracker()
        payload = _tracker_in_state(OrderState.SUBMITTED).export_state()
        del payload[ORDER_ID]["symbol"]

        with pytest.raises(KeyError):
            tracker.import_state(payload)


def test_record_defaults_to_submitted():
    record = OrderLifecycleRecord(
        order_id=ORDER_ID, symbol=SYMBOL, side="buy", quantity=1.0, strategy_name=None
    )

    assert record.state is OrderState.SUBMITTED
    assert record.client_order_id is None
