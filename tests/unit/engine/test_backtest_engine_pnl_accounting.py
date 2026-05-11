"""Unit tests for _calculate_trade_pnl covering both long and short legs.

Step 2B follow-up: previously _calculate_trade_pnl only handled long state.
Sell-to-open shorts (and their buy-to-cover legs) were silently recorded
with pnl=0, biasing profit_factor and win-rate framing.
"""

from engine.backtest_engine import BacktestEngine


def _engine():
    # _calculate_trade_pnl is a pure synchronous helper that does not touch
    # broker/data state, so bypass __init__ to avoid coupling.
    return BacktestEngine.__new__(BacktestEngine)


class TestLongPnL:
    def test_long_open_then_close_at_profit(self):
        engine = _engine()
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 100.0, "timestamp": "2024-01-01"},
            {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 110.0, "timestamp": "2024-01-15"},
        ]
        records = engine._calculate_trade_pnl(trades)
        assert records[0]["pnl"] == 0
        assert records[1]["pnl"] == 100.0  # (110 - 100) * 10

    def test_long_open_then_close_at_loss(self):
        engine = _engine()
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 5, "price": 200.0, "timestamp": "2024-01-01"},
            {"symbol": "AAPL", "side": "sell", "quantity": 5, "price": 180.0, "timestamp": "2024-01-15"},
        ]
        records = engine._calculate_trade_pnl(trades)
        assert records[1]["pnl"] == -100.0  # (180 - 200) * 5


class TestShortPnL:
    def test_short_open_then_cover_at_profit(self):
        """Sell-to-open at 100, buy-to-cover at 90 -> +100 profit on 10 shares."""
        engine = _engine()
        trades = [
            {"symbol": "TSLA", "side": "sell", "quantity": 10, "price": 100.0, "timestamp": "2024-01-01"},
            {"symbol": "TSLA", "side": "buy", "quantity": 10, "price": 90.0, "timestamp": "2024-01-15"},
        ]
        records = engine._calculate_trade_pnl(trades)
        assert records[0]["pnl"] == 0
        assert records[1]["pnl"] == 100.0  # (100 - 90) * 10

    def test_short_open_then_cover_at_loss(self):
        """Sell-to-open at 100, buy-to-cover at 115 -> -150 loss on 10 shares."""
        engine = _engine()
        trades = [
            {"symbol": "TSLA", "side": "sell", "quantity": 10, "price": 100.0, "timestamp": "2024-01-01"},
            {"symbol": "TSLA", "side": "buy", "quantity": 10, "price": 115.0, "timestamp": "2024-01-15"},
        ]
        records = engine._calculate_trade_pnl(trades)
        assert records[1]["pnl"] == -150.0


class TestMixedAndPartials:
    def test_long_then_short_separate_legs(self):
        """Buy 10 @ 100, sell 10 @ 110 (close long, +100). Then sell 10 @ 105, buy 10 @ 100 (close short, +50)."""
        engine = _engine()
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 100.0, "timestamp": "2024-01-01"},
            {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 110.0, "timestamp": "2024-01-15"},
            {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 105.0, "timestamp": "2024-02-01"},
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 100.0, "timestamp": "2024-02-15"},
        ]
        records = engine._calculate_trade_pnl(trades)
        assert records[1]["pnl"] == 100.0
        assert records[3]["pnl"] == 50.0

    def test_partial_long_close(self):
        """Buy 10, sell 5 -> PnL on 5 shares; remaining 5 still tracked."""
        engine = _engine()
        trades = [
            {"symbol": "NVDA", "side": "buy", "quantity": 10, "price": 500.0, "timestamp": "2024-01-01"},
            {"symbol": "NVDA", "side": "sell", "quantity": 5, "price": 600.0, "timestamp": "2024-01-15"},
            {"symbol": "NVDA", "side": "sell", "quantity": 5, "price": 700.0, "timestamp": "2024-02-01"},
        ]
        records = engine._calculate_trade_pnl(trades)
        assert records[1]["pnl"] == 500.0  # (600 - 500) * 5
        assert records[2]["pnl"] == 1000.0  # (700 - 500) * 5

    def test_partial_short_cover(self):
        """Sell 10 @ 100, buy 5 @ 90 -> +50 partial cover. Then buy 5 @ 110 -> -50."""
        engine = _engine()
        trades = [
            {"symbol": "META", "side": "sell", "quantity": 10, "price": 100.0, "timestamp": "2024-01-01"},
            {"symbol": "META", "side": "buy", "quantity": 5, "price": 90.0, "timestamp": "2024-01-15"},
            {"symbol": "META", "side": "buy", "quantity": 5, "price": 110.0, "timestamp": "2024-02-01"},
        ]
        records = engine._calculate_trade_pnl(trades)
        assert records[1]["pnl"] == 50.0  # (100 - 90) * 5
        assert records[2]["pnl"] == -50.0  # (100 - 110) * 5
