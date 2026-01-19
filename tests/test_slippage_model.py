"""
Tests for SlippageModel in simple_backtest.py

Tests cover:
- Slippage calculation for buy orders
- Slippage calculation for sell orders
- Cost tracking and aggregation
- Disabled slippage mode
- Edge cases (zero price, large orders)
"""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_backtest import SlippageModel


class TestSlippageModelInit:
    """Test SlippageModel initialization."""

    def test_default_initialization(self):
        """Test default values are loaded from config."""
        model = SlippageModel()
        assert model.slippage_pct > 0
        assert model.bid_ask_spread > 0
        assert model.enabled is True

    def test_custom_initialization(self):
        """Test custom values override defaults."""
        model = SlippageModel(
            slippage_pct=0.01, bid_ask_spread=0.005, commission_per_share=0.01, enabled=False
        )
        assert model.slippage_pct == 0.01
        assert model.bid_ask_spread == 0.005
        assert model.commission_per_share == 0.01
        assert model.enabled is False

    def test_disabled_mode(self):
        """Test that disabled mode returns original price."""
        model = SlippageModel(enabled=False)
        exec_price, slip, spread, comm = model.apply_slippage(100.0, "buy", 10)
        assert exec_price == 100.0
        assert slip == 0.0
        assert spread == 0.0
        assert comm == 0.0


class TestSlippageCalculation:
    """Test slippage calculations for different order types."""

    def test_buy_order_increases_price(self):
        """Buy orders should execute at higher price due to slippage."""
        model = SlippageModel(slippage_pct=0.004, bid_ask_spread=0.001)
        quoted_price = 100.0
        exec_price, slip, spread, comm = model.apply_slippage(quoted_price, "buy", 10)

        # Price should be higher for buy
        assert exec_price > quoted_price
        # Should be approximately 100 * 1.004 * 1.0005 = 100.45
        assert 100.4 < exec_price < 100.6

    def test_sell_order_decreases_price(self):
        """Sell orders should execute at lower price due to slippage."""
        model = SlippageModel(slippage_pct=0.004, bid_ask_spread=0.001)
        quoted_price = 100.0
        exec_price, slip, spread, comm = model.apply_slippage(quoted_price, "sell", 10)

        # Price should be lower for sell
        assert exec_price < quoted_price
        # Should be approximately 100 * 0.996 * 0.9995 = 99.55
        assert 99.4 < exec_price < 99.6

    def test_slippage_cost_calculation(self):
        """Slippage cost should be correctly calculated."""
        model = SlippageModel(slippage_pct=0.01, bid_ask_spread=0.0, commission_per_share=0.0)
        _, slip, _, _ = model.apply_slippage(100.0, "buy", 10)

        # Slippage = 100 * 0.01 * 10 = 10
        assert slip == 10.0

    def test_spread_cost_calculation(self):
        """Spread cost should be correctly calculated."""
        model = SlippageModel(slippage_pct=0.0, bid_ask_spread=0.01, commission_per_share=0.0)
        _, _, spread, _ = model.apply_slippage(100.0, "buy", 10)

        # Spread = 100 * 0.005 * 10 = 5 (half spread per side)
        assert spread == 5.0

    def test_commission_calculation(self):
        """Commission should be calculated per share."""
        model = SlippageModel(slippage_pct=0.0, bid_ask_spread=0.0, commission_per_share=0.01)
        _, _, _, comm = model.apply_slippage(100.0, "buy", 100)

        # Commission = 0.01 * 100 = 1.0
        assert comm == 1.0


class TestCostTracking:
    """Test cumulative cost tracking."""

    def test_costs_accumulate_over_trades(self):
        """Costs should accumulate across multiple trades."""
        model = SlippageModel(slippage_pct=0.01, bid_ask_spread=0.01, commission_per_share=0.01)

        # Execute 3 trades
        model.apply_slippage(100.0, "buy", 10)
        model.apply_slippage(100.0, "sell", 10)
        model.apply_slippage(100.0, "buy", 10)

        costs = model.get_total_costs()

        # Each trade: slippage=10, spread=5, commission=0.1
        assert costs["slippage"] == 30.0
        assert costs["spread"] == 15.0
        assert costs["commission"] == pytest.approx(0.30, rel=1e-9)
        assert costs["total"] == pytest.approx(45.30, rel=1e-9)

    def test_reset_clears_costs(self):
        """Reset should clear all accumulated costs."""
        model = SlippageModel(slippage_pct=0.01)
        model.apply_slippage(100.0, "buy", 10)

        model.reset()

        costs = model.get_total_costs()
        assert costs["total"] == 0.0

    def test_get_total_costs_structure(self):
        """get_total_costs should return dict with expected keys."""
        model = SlippageModel()
        costs = model.get_total_costs()

        assert "slippage" in costs
        assert "spread" in costs
        assert "commission" in costs
        assert "total" in costs


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_quantity(self):
        """Zero quantity should return zero costs."""
        model = SlippageModel(slippage_pct=0.01)
        exec_price, slip, spread, comm = model.apply_slippage(100.0, "buy", 0)

        assert slip == 0.0
        assert spread == 0.0
        assert comm == 0.0

    def test_very_small_price(self):
        """Very small prices should still work correctly."""
        model = SlippageModel(slippage_pct=0.01)
        exec_price, slip, spread, comm = model.apply_slippage(0.01, "buy", 1000)

        assert exec_price > 0.01
        assert slip > 0

    def test_large_order(self):
        """Large orders should calculate proportional costs."""
        model = SlippageModel(slippage_pct=0.01, bid_ask_spread=0.0, commission_per_share=0.0)
        _, slip_small, _, _ = model.apply_slippage(100.0, "buy", 100)
        model.reset()
        _, slip_large, _, _ = model.apply_slippage(100.0, "buy", 1000)

        # Slippage should scale linearly with quantity
        assert slip_large == slip_small * 10

    def test_symmetry_of_buy_sell(self):
        """Buy and sell slippage should be symmetric (opposite direction)."""
        model = SlippageModel(slippage_pct=0.01, bid_ask_spread=0.001)

        buy_price, _, _, _ = model.apply_slippage(100.0, "buy", 10)
        sell_price, _, _, _ = model.apply_slippage(100.0, "sell", 10)

        # Both should be equidistant from 100 (approximately)
        buy_diff = buy_price - 100.0
        sell_diff = 100.0 - sell_price

        # Should be approximately equal
        assert abs(buy_diff - sell_diff) < 0.01


class TestIntegrationScenarios:
    """Test realistic trading scenarios."""

    def test_round_trip_trade_cost(self):
        """
        Test total cost of a round-trip trade (buy then sell).

        A realistic scenario: buy 100 shares at $100, sell at $100.
        With slippage, you should lose money even if price unchanged.
        """
        model = SlippageModel(
            slippage_pct=0.004, bid_ask_spread=0.001, commission_per_share=0.0  # 0.4%  # 0.1%
        )

        # Buy
        buy_price, _, _, _ = model.apply_slippage(100.0, "buy", 100)

        # Sell at same quoted price
        sell_price, _, _, _ = model.apply_slippage(100.0, "sell", 100)

        # P/L should be negative (transaction costs)
        pnl = (sell_price - buy_price) * 100
        assert pnl < 0

        # Total transaction cost should be approximately 1% of position
        costs = model.get_total_costs()
        position_value = 100 * 100.0
        cost_ratio = costs["total"] / position_value
        assert 0.008 < cost_ratio < 0.012  # ~1% total cost

    def test_realistic_backtest_scenario(self):
        """
        Simulate a mini backtest with multiple trades.

        5 trades: 3 winners (+5% each), 2 losers (-3% each)
        Gross P/L: 3*5% - 2*3% = 9%
        With transaction costs, net should be lower.
        """
        model = SlippageModel(slippage_pct=0.004, bid_ask_spread=0.001, commission_per_share=0.0)

        initial_capital = 100000
        position_size = 0.1  # 10% per trade

        # Simulate 5 trades
        trades = [
            ("buy", 100.0, "sell", 105.0),  # +5%
            ("buy", 100.0, "sell", 105.0),  # +5%
            ("buy", 100.0, "sell", 97.0),  # -3%
            ("buy", 100.0, "sell", 105.0),  # +5%
            ("buy", 100.0, "sell", 97.0),  # -3%
        ]

        gross_pnl = 0
        for entry_side, entry_price, exit_side, exit_price in trades:
            qty = int((initial_capital * position_size) / entry_price)

            buy_exec, _, _, _ = model.apply_slippage(entry_price, entry_side, qty)
            sell_exec, _, _, _ = model.apply_slippage(exit_price, exit_side, qty)

            trade_pnl = (sell_exec - buy_exec) * qty
            gross_pnl += trade_pnl

        costs = model.get_total_costs()

        # With 5 round-trip trades, should have measurable cost drag
        # Each trade has slippage + spread costs
        assert costs["total"] > 0
        # Note: With 0.4% slippage + 0.1% spread = ~0.5% per trade, 10 sides = ~0.5% total
        assert costs["total"] / initial_capital > 0.004  # At least 0.4% in costs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
