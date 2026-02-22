from __future__ import annotations

from collections import deque

from live_trader import LiveTrader


class _DummyStrategy:
    def __init__(self) -> None:
        self.price_history = {
            "BTC/USD": deque(maxlen=3),
            "ETH/USD": deque(maxlen=3),
        }
        self.current_prices = {}
        self.signals = {}
        self.indicators = {}


def test_restore_internal_strategy_state_preserves_active_symbol_history() -> None:
    trader = LiveTrader(strategy_name="momentum", symbols=["BTC/USD", "ETH/USD"], parameters={})
    strategy = _DummyStrategy()

    trader._restore_internal_strategy_state(
        strategy,
        {
            "price_history": {
                "AAPL": [{"close": 150.0}],
                "BTC/USD": [{"close": 50000.0}],
            }
        },
    )

    assert "BTC/USD" in strategy.price_history
    assert "ETH/USD" in strategy.price_history
    assert list(strategy.price_history["BTC/USD"]) == [{"close": 50000.0}]
    assert list(strategy.price_history["ETH/USD"]) == []
    # Legacy symbols can be retained without breaking active symbol buffers.
    assert "AAPL" in strategy.price_history
