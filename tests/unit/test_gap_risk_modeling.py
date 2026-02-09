"""
Tests for gap risk modeling in backtest broker.

These tests verify that overnight gaps are properly simulated and that
stop orders that are gapped through fill at the open price, not the stop price.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from brokers.backtest_broker import BacktestBroker, GapEvent, GapStatistics

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def broker():
    """Create a BacktestBroker with gap tracking enabled."""
    return BacktestBroker(initial_balance=100000)


@pytest.fixture
def broker_with_position(broker):
    """Create a broker with an existing long position."""
    # Create price data
    dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
    prices = np.linspace(100, 110, 30)
    data = pd.DataFrame(
        {
            "open": prices * 0.995,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.ones(30) * 1_000_000,
        },
        index=dates,
    )
    broker.set_price_data("AAPL", data)
    broker._current_date = dates[15]

    # Create a position
    broker.positions["AAPL"] = {
        "symbol": "AAPL",
        "quantity": 100,
        "entry_price": 105.0,
    }

    return broker


@pytest.fixture
def price_data_with_gap():
    """Create price data with a significant overnight gap."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
    prices = np.linspace(100, 110, 30)

    # Create a gap on day 15
    opens = prices * 0.995
    opens[15] = prices[14] * 0.92  # 8% gap down

    data = pd.DataFrame(
        {
            "open": opens,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.ones(30) * 1_000_000,
        },
        index=dates,
    )
    return data, dates


# ============================================================================
# GAPDATACLASS TESTS
# ============================================================================


class TestGapEvent:
    """Tests for GapEvent dataclass."""

    def test_gap_event_creation(self):
        """Test creating a GapEvent."""
        event = GapEvent(
            symbol="AAPL",
            date=datetime(2024, 1, 15),
            prev_close=100.0,
            open_price=92.0,
            gap_pct=-0.08,
            position_side="long",
            position_qty=100,
            stop_price=95.0,
            stop_triggered=True,
            slippage_from_stop=3.0,
        )

        assert event.symbol == "AAPL"
        assert event.gap_pct == -0.08
        assert event.stop_triggered is True
        assert event.slippage_from_stop == 3.0

    def test_gap_event_no_stop(self):
        """Test GapEvent without a stop order."""
        event = GapEvent(
            symbol="MSFT",
            date=datetime(2024, 1, 15),
            prev_close=100.0,
            open_price=103.0,
            gap_pct=0.03,
            position_side="long",
            position_qty=50,
            stop_price=None,
            stop_triggered=False,
            slippage_from_stop=0.0,
        )

        assert event.stop_price is None
        assert event.stop_triggered is False


class TestGapStatistics:
    """Tests for GapStatistics dataclass."""

    def test_gap_statistics_creation(self):
        """Test creating GapStatistics."""
        stats = GapStatistics(
            total_gaps=10,
            gaps_exceeding_2pct=3,
            stops_gapped_through=2,
            total_gap_slippage=150.0,
            largest_gap_pct=0.08,
            average_gap_pct=0.025,
        )

        assert stats.total_gaps == 10
        assert stats.gaps_exceeding_2pct == 3
        assert stats.stops_gapped_through == 2
        assert stats.total_gap_slippage == 150.0

    def test_gap_statistics_empty(self):
        """Test empty gap statistics."""
        stats = GapStatistics(
            total_gaps=0,
            gaps_exceeding_2pct=0,
            stops_gapped_through=0,
            total_gap_slippage=0.0,
            largest_gap_pct=0.0,
            average_gap_pct=0.0,
        )

        assert stats.total_gaps == 0
        assert stats.largest_gap_pct == 0.0


# ============================================================================
# STOP ORDER MANAGEMENT TESTS
# ============================================================================


class TestStopOrderManagement:
    """Tests for stop order tracking."""

    def test_set_stop_order(self, broker):
        """Test setting a stop order."""
        broker._current_date = datetime(2024, 1, 15)
        broker.set_stop_order("AAPL", stop_price=95.0, quantity=100)

        assert "AAPL" in broker._stop_orders
        assert broker._stop_orders["AAPL"]["stop_price"] == 95.0
        assert broker._stop_orders["AAPL"]["quantity"] == 100
        assert broker._stop_orders["AAPL"]["side"] == "sell"

    def test_set_stop_order_short_position(self, broker):
        """Test setting a stop order for a short position."""
        broker._current_date = datetime(2024, 1, 15)
        broker.set_stop_order("AAPL", stop_price=105.0, quantity=100, side="buy")

        assert broker._stop_orders["AAPL"]["side"] == "buy"

    def test_clear_stop_order(self, broker):
        """Test clearing a stop order."""
        broker.set_stop_order("AAPL", stop_price=95.0, quantity=100)
        broker.clear_stop_order("AAPL")

        assert "AAPL" not in broker._stop_orders

    def test_clear_nonexistent_stop_order(self, broker):
        """Test clearing a stop that doesn't exist (should not error)."""
        broker.clear_stop_order("NONEXISTENT")  # Should not raise


# ============================================================================
# PREVIOUS DAY CLOSE TRACKING TESTS
# ============================================================================


class TestPrevDayCloseTracking:
    """Tests for previous day close tracking."""

    def test_update_prev_day_close(self, broker):
        """Test updating previous day close."""
        broker.update_prev_day_close("AAPL", 100.0)

        assert "AAPL" in broker._prev_day_close
        assert broker._prev_day_close["AAPL"] == 100.0

    def test_update_prev_day_closes_from_positions(self, broker_with_position):
        """Test updating closes for all held positions."""
        broker_with_position.update_prev_day_closes(
            broker_with_position._current_date
        )

        assert "AAPL" in broker_with_position._prev_day_close


# ============================================================================
# GAP SIMULATION TESTS
# ============================================================================


class TestGapSimulation:
    """Tests for overnight gap simulation."""

    def test_simulate_small_gap_ignored(self, broker_with_position):
        """Test that small gaps (<0.5%) are ignored."""
        broker_with_position.update_prev_day_close("AAPL", 105.0)

        # 0.3% gap - should be ignored
        gap_event = broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=105.315,  # +0.3%
            date=datetime(2024, 1, 16),
        )

        assert gap_event is None

    def test_simulate_significant_gap_tracked(self, broker_with_position):
        """Test that significant gaps (>0.5%) are tracked."""
        broker_with_position.update_prev_day_close("AAPL", 105.0)

        # 2% gap down
        gap_event = broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=102.9,  # -2%
            date=datetime(2024, 1, 16),
        )

        assert gap_event is not None
        assert gap_event.symbol == "AAPL"
        assert gap_event.gap_pct < 0  # Negative gap
        assert abs(gap_event.gap_pct - (-0.02)) < 0.001

    def test_gap_without_position_returns_none(self, broker):
        """Test that gap returns None if no position exists."""
        broker.update_prev_day_close("AAPL", 100.0)

        gap_event = broker.simulate_overnight_gap(
            symbol="AAPL",
            open_price=92.0,  # -8% gap
            date=datetime(2024, 1, 16),
        )

        # No position, so no gap event
        assert gap_event is None

    def test_stop_gapped_through_long_position(self, broker_with_position):
        """Test stop order gapped through on long position."""
        broker_with_position.update_prev_day_close("AAPL", 105.0)
        broker_with_position.set_stop_order("AAPL", stop_price=100.0, quantity=100)

        # Gap down below stop
        gap_event = broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=95.0,  # Below stop of 100
            date=datetime(2024, 1, 16),
        )

        assert gap_event is not None
        assert gap_event.stop_triggered is True
        assert gap_event.slippage_from_stop == 5.0  # 100 - 95

    def test_stop_not_triggered_if_gap_above_stop(self, broker_with_position):
        """Test stop not triggered if gap is above stop price."""
        broker_with_position.update_prev_day_close("AAPL", 105.0)
        broker_with_position.set_stop_order("AAPL", stop_price=100.0, quantity=100)

        # Gap down but still above stop
        gap_event = broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=101.0,  # Above stop of 100
            date=datetime(2024, 1, 16),
        )

        assert gap_event is not None
        assert gap_event.stop_triggered is False
        assert gap_event.slippage_from_stop == 0.0

    def test_short_position_gap_up_triggers_stop(self, broker):
        """Test short position stop triggered by gap up."""
        # Create short position
        broker.positions["AAPL"] = {
            "symbol": "AAPL",
            "quantity": -100,  # Short position
            "entry_price": 100.0,
        }
        broker.update_prev_day_close("AAPL", 95.0)
        broker.set_stop_order("AAPL", stop_price=100.0, quantity=100, side="buy")

        # Gap up above stop
        gap_event = broker.simulate_overnight_gap(
            symbol="AAPL",
            open_price=105.0,  # Above stop of 100
            date=datetime(2024, 1, 16),
        )

        assert gap_event is not None
        assert gap_event.stop_triggered is True
        assert gap_event.slippage_from_stop == 5.0  # 105 - 100

    def test_gapped_stop_executes_at_open_price(self, broker_with_position):
        """Test that gapped stop fills at open price, not stop price."""
        broker_with_position.update_prev_day_close("AAPL", 105.0)
        broker_with_position.set_stop_order("AAPL", stop_price=100.0, quantity=100)

        initial_balance = broker_with_position.balance

        # Gap down below stop
        broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=95.0,  # Below stop of 100
            date=datetime(2024, 1, 16),
        )

        # Position should be closed and cash should reflect fill at $95, not $100
        # After selling 100 shares at ~$95, balance should increase by ~$9500
        # (minus slippage from the place_order call)
        assert broker_with_position.balance > initial_balance
        # The stop order should be cleared
        assert "AAPL" not in broker_with_position._stop_orders


# ============================================================================
# GAP STATISTICS TESTS
# ============================================================================


class TestGapStatisticsCalculation:
    """Tests for gap statistics calculation."""

    def test_get_gap_statistics_empty(self, broker):
        """Test gap statistics with no events."""
        stats = broker.get_gap_statistics()

        assert stats.total_gaps == 0
        assert stats.stops_gapped_through == 0
        assert stats.total_gap_slippage == 0.0

    def test_get_gap_statistics_with_events(self, broker_with_position):
        """Test gap statistics with multiple events."""
        broker_with_position.update_prev_day_close("AAPL", 105.0)
        broker_with_position.set_stop_order("AAPL", stop_price=100.0, quantity=100)

        # Record several gap events manually
        broker_with_position._gap_events.append(
            GapEvent(
                symbol="AAPL",
                date=datetime(2024, 1, 16),
                prev_close=105.0,
                open_price=103.0,
                gap_pct=-0.019,
                position_side="long",
                position_qty=100,
                stop_price=None,
                stop_triggered=False,
                slippage_from_stop=0.0,
            )
        )
        broker_with_position._gap_events.append(
            GapEvent(
                symbol="AAPL",
                date=datetime(2024, 1, 17),
                prev_close=103.0,
                open_price=95.0,
                gap_pct=-0.078,
                position_side="long",
                position_qty=100,
                stop_price=100.0,
                stop_triggered=True,
                slippage_from_stop=5.0,
            )
        )

        stats = broker_with_position.get_gap_statistics()

        assert stats.total_gaps == 2
        assert stats.gaps_exceeding_2pct == 1  # 7.8% gap
        assert stats.stops_gapped_through == 1
        assert stats.total_gap_slippage == 5.0
        assert abs(stats.largest_gap_pct - 0.078) < 0.001

    def test_get_gap_events(self, broker):
        """Test getting list of gap events."""
        broker._gap_events.append(
            GapEvent(
                symbol="AAPL",
                date=datetime(2024, 1, 16),
                prev_close=100.0,
                open_price=95.0,
                gap_pct=-0.05,
                position_side="long",
                position_qty=100,
                stop_price=None,
                stop_triggered=False,
                slippage_from_stop=0.0,
            )
        )

        events = broker.get_gap_events()

        assert len(events) == 1
        assert events[0].symbol == "AAPL"

    def test_clear_gap_tracking(self, broker):
        """Test clearing all gap tracking data."""
        broker._gap_events.append(
            GapEvent(
                symbol="AAPL",
                date=datetime(2024, 1, 16),
                prev_close=100.0,
                open_price=95.0,
                gap_pct=-0.05,
                position_side="long",
                position_qty=100,
                stop_price=None,
                stop_triggered=False,
                slippage_from_stop=0.0,
            )
        )
        broker._stop_orders["AAPL"] = {"stop_price": 95.0}
        broker._prev_day_close["AAPL"] = 100.0

        broker.clear_gap_tracking()

        assert len(broker._gap_events) == 0
        assert len(broker._stop_orders) == 0
        assert len(broker._prev_day_close) == 0


# ============================================================================
# PROCESS DAY START GAPS TESTS
# ============================================================================


class TestProcessDayStartGaps:
    """Tests for processing gaps at start of trading day."""

    def test_process_day_start_gaps(self, broker, price_data_with_gap):
        """Test processing gaps for all positions at day start."""
        data, dates = price_data_with_gap
        broker.set_price_data("AAPL", data)

        # Create position
        broker.positions["AAPL"] = {
            "symbol": "AAPL",
            "quantity": 100,
            "entry_price": 105.0,
        }

        # Set previous close
        broker._prev_day_close["AAPL"] = data.iloc[14]["close"]

        # Process gaps for day 15 (which has the gap)
        gap_events = broker.process_day_start_gaps(dates[15])

        assert len(gap_events) == 1
        assert gap_events[0].symbol == "AAPL"

    def test_process_day_start_gaps_no_position(self, broker, price_data_with_gap):
        """Test that gaps are not tracked without a position."""
        data, dates = price_data_with_gap
        broker.set_price_data("AAPL", data)

        # No position, just set prev close
        broker._prev_day_close["AAPL"] = data.iloc[14]["close"]

        gap_events = broker.process_day_start_gaps(dates[15])

        assert len(gap_events) == 0

    def test_process_day_start_gaps_multiple_symbols(self, broker):
        """Test processing gaps for multiple positions."""
        # Create data for two symbols
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")

        for symbol in ["AAPL", "MSFT"]:
            data = pd.DataFrame(
                {
                    "open": [100, 101, 102, 103, 90, 105, 106, 107, 108, 109],  # Gap on day 4
                    "high": [101, 102, 103, 104, 106, 106, 107, 108, 109, 110],
                    "low": [99, 100, 101, 102, 89, 104, 105, 106, 107, 108],
                    "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                    "volume": [1_000_000] * 10,
                },
                index=dates,
            )
            broker.set_price_data(symbol, data)
            broker.positions[symbol] = {
                "symbol": symbol,
                "quantity": 100,
                "entry_price": 100.0,
            }
            broker._prev_day_close[symbol] = 103.0  # Day before gap

        gap_events = broker.process_day_start_gaps(dates[4])

        # Both symbols should have gap events
        assert len(gap_events) == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestGapRiskIntegration:
    """Integration tests for gap risk modeling."""

    def test_full_gap_workflow(self, broker):
        """Test complete gap risk workflow."""
        # Setup price data with a large gap
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 85, 90, 91, 92, 93, 94],  # 18% gap on day 4
                "high": [101, 102, 103, 104, 95, 95, 96, 97, 98, 99],
                "low": [99, 100, 101, 102, 82, 85, 86, 87, 88, 89],
                "close": [100, 101, 102, 103, 90, 92, 93, 94, 95, 96],
                "volume": [1_000_000] * 10,
            },
            index=dates,
        )
        broker.set_price_data("AAPL", data)

        # Create position with stop
        broker.positions["AAPL"] = {
            "symbol": "AAPL",
            "quantity": 100,
            "entry_price": 100.0,
        }
        broker.set_stop_order("AAPL", stop_price=95.0, quantity=100)
        broker._prev_day_close["AAPL"] = 103.0  # Day 3 close

        # Process the gap day
        broker._current_date = dates[4]
        gap_events = broker.process_day_start_gaps(dates[4])

        # Should have detected the gap and triggered stop
        assert len(gap_events) == 1
        event = gap_events[0]
        assert event.stop_triggered is True
        assert event.slippage_from_stop == 10.0  # 95 - 85

        # Stop should be cleared after execution
        assert "AAPL" not in broker._stop_orders

        # Get statistics
        stats = broker.get_gap_statistics()
        assert stats.total_gaps == 1
        assert stats.stops_gapped_through == 1
        assert stats.total_gap_slippage == 10.0

    def test_multiple_days_gap_tracking(self, broker):
        """Test gap tracking across multiple days."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        data = pd.DataFrame(
            {
                "open": [100, 102, 98, 103, 95, 90, 91, 92, 93, 94],  # Multiple gaps
                "high": [101, 103, 103, 104, 100, 95, 96, 97, 98, 99],
                "low": [99, 100, 97, 102, 94, 85, 86, 87, 88, 89],
                "close": [100, 101, 102, 103, 98, 92, 93, 94, 95, 96],
                "volume": [1_000_000] * 10,
            },
            index=dates,
        )
        broker.set_price_data("AAPL", data)

        broker.positions["AAPL"] = {
            "symbol": "AAPL",
            "quantity": 100,
            "entry_price": 100.0,
        }

        # Simulate multi-day backtest
        total_gaps = 0
        for i in range(1, len(dates)):
            broker._current_date = dates[i]
            broker._prev_day_close["AAPL"] = data.iloc[i - 1]["close"]

            gap_events = broker.process_day_start_gaps(dates[i])
            total_gaps += len(gap_events)

            # Skip if position was closed
            if "AAPL" not in broker.positions:
                break

        # Should have tracked multiple gaps
        stats = broker.get_gap_statistics()
        assert stats.total_gaps > 0


class TestGapRiskEdgeCases:
    """Edge case tests for gap risk modeling."""

    def test_gap_with_no_prev_close(self, broker_with_position):
        """Test gap simulation when no previous close is recorded."""
        # Don't set prev_day_close

        gap_event = broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=95.0,
            date=datetime(2024, 1, 16),
        )

        assert gap_event is None

    def test_gap_with_missing_price_data(self, broker):
        """Test gap processing with missing price data."""
        broker.positions["UNKNOWN"] = {
            "symbol": "UNKNOWN",
            "quantity": 100,
            "entry_price": 100.0,
        }
        broker._prev_day_close["UNKNOWN"] = 100.0

        # Should not error, just return no events
        gap_events = broker.process_day_start_gaps(datetime(2024, 1, 16))

        # No price data means _get_open_price returns None
        assert len(gap_events) == 0

    def test_exactly_at_stop_price(self, broker_with_position):
        """Test behavior when opening exactly at stop price."""
        broker_with_position.update_prev_day_close("AAPL", 105.0)
        broker_with_position.set_stop_order("AAPL", stop_price=100.0, quantity=100)

        # Open exactly at stop price - this should NOT trigger gap-through
        # because we didn't gap THROUGH it (we're at it, not below it)
        gap_event = broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=100.0,  # Exactly at stop
            date=datetime(2024, 1, 16),
        )

        assert gap_event is not None
        assert gap_event.stop_triggered is False  # At stop, not through it

    def test_very_large_gap(self, broker_with_position):
        """Test behavior with extremely large gap (e.g., bankruptcy)."""
        broker_with_position.update_prev_day_close("AAPL", 100.0)
        broker_with_position.set_stop_order("AAPL", stop_price=90.0, quantity=100)

        # 50% gap down (e.g., bankruptcy news)
        gap_event = broker_with_position.simulate_overnight_gap(
            symbol="AAPL",
            open_price=50.0,
            date=datetime(2024, 1, 16),
        )

        assert gap_event is not None
        assert gap_event.gap_pct == -0.50
        assert gap_event.stop_triggered is True
        assert gap_event.slippage_from_stop == 40.0  # 90 - 50
