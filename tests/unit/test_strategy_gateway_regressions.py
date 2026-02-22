"""Regression tests for strategy gateway routing and constructor compatibility."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


class _FakeOrderBuilder:
    """Minimal order builder stub used to avoid external dependency details in tests."""

    def __init__(self, symbol: str, side: str, qty: int):
        self.symbol = symbol
        self.side = side
        self.qty = qty

    def market(self):
        return self

    def day(self):
        return self

    def build(self):
        return SimpleNamespace(
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            type="market",
        )


class _FakeLSTMPredictor:
    """Lightweight predictor stub for constructor wiring tests."""

    def __init__(
        self,
        sequence_length=60,
        prediction_horizon=5,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        model_dir="models",
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_dir = model_dir


class _ConcreteBaseStrategy:
    """Factory mixin for lightweight BaseStrategy concrete classes in tests."""

    @staticmethod
    def build():
        from strategies.base_strategy import BaseStrategy

        class _TestStrategy(BaseStrategy):
            async def analyze_symbol(self, symbol):
                return {"action": "hold"}

            async def execute_trade(self, symbol, signal):
                return None

        return _TestStrategy


@pytest.mark.asyncio
async def test_momentum_backtest_entry_order_uses_strategy_gateway_helpers(monkeypatch):
    """Backtest entry path should not call broker.submit_order_advanced directly."""
    from strategies import momentum_strategy_backtest as module

    monkeypatch.setattr(module, "OrderBuilder", _FakeOrderBuilder)

    broker = AsyncMock()
    broker.submit_order_advanced = AsyncMock()
    strategy = module.MomentumStrategyBacktest(broker=broker, parameters={})
    strategy.submit_entry_order = AsyncMock(return_value=SimpleNamespace(success=True))
    strategy.submit_exit_order = AsyncMock()

    await strategy._place_backtest_order("AAPL", 10, "buy", is_exit=False)

    strategy.submit_entry_order.assert_awaited_once()
    strategy.submit_exit_order.assert_not_awaited()
    broker.submit_order_advanced.assert_not_awaited()

    kwargs = strategy.submit_entry_order.await_args.kwargs
    assert kwargs["reason"] == "backtest_entry"
    assert kwargs["max_positions"] is None
    assert kwargs["order_request"].symbol == "AAPL"
    assert kwargs["order_request"].side == "buy"


@pytest.mark.asyncio
async def test_momentum_backtest_exit_order_uses_submit_exit_order(monkeypatch):
    """Backtest exit path should route through submit_exit_order for safety checks."""
    from strategies import momentum_strategy_backtest as module

    monkeypatch.setattr(module, "OrderBuilder", _FakeOrderBuilder)

    strategy = module.MomentumStrategyBacktest(broker=AsyncMock(), parameters={})
    strategy.submit_entry_order = AsyncMock()
    strategy.submit_exit_order = AsyncMock(return_value=SimpleNamespace(success=True))

    await strategy._place_backtest_order("MSFT", 7, "sell", is_exit=True)

    strategy.submit_exit_order.assert_awaited_once_with(
        symbol="MSFT",
        qty=7,
        side="sell",
        reason="backtest_exit",
    )
    strategy.submit_entry_order.assert_not_awaited()


def test_lstm_constructor_accepts_strategy_manager_signature(monkeypatch):
    """LSTM strategy must support broker/parameters/order_gateway constructor path."""
    from strategies import lstm_enhanced_strategy as module

    monkeypatch.setattr(module, "LSTMPredictor", _FakeLSTMPredictor)

    gateway = Mock()
    strategy = module.LSTMEnhancedStrategy(
        broker=AsyncMock(),
        parameters={"symbols": ["AAPL", "MSFT"], "lstm_sequence_length": 21},
        order_gateway=gateway,
    )

    assert strategy.order_gateway is gateway
    assert strategy.parameters["symbols"] == ["AAPL", "MSFT"]
    assert strategy.lstm.sequence_length == 21


def test_lstm_constructor_keeps_legacy_inputs_and_prefers_parameters(monkeypatch):
    """Legacy symbols/config remain supported while modern parameters take precedence."""
    from strategies import lstm_enhanced_strategy as module

    monkeypatch.setattr(module, "LSTMPredictor", _FakeLSTMPredictor)

    strategy = module.LSTMEnhancedStrategy(
        broker=AsyncMock(),
        symbols=["AAPL"],
        config={"lstm_sequence_length": 13, "from_config": True},
        parameters={"symbols": ["TSLA"], "lstm_sequence_length": 34, "from_parameters": True},
    )

    assert strategy.parameters["symbols"] == ["TSLA"]
    assert strategy.parameters["from_config"] is True
    assert strategy.parameters["from_parameters"] is True
    assert strategy.lstm.sequence_length == 34


@pytest.mark.asyncio
async def test_base_strategy_entry_order_blocks_when_gateway_missing():
    """Entry orders must fail closed if no gateway is configured."""
    TestStrategy = _ConcreteBaseStrategy.build()

    broker = AsyncMock()
    broker.submit_order_advanced = AsyncMock(return_value=SimpleNamespace(id="direct-order"))

    strategy = TestStrategy(broker=broker, parameters={})
    order_request = SimpleNamespace(symbol="AAPL", qty=1, side="buy")

    result = await strategy.submit_entry_order(order_request=order_request, reason="test-entry")

    assert result is None
    broker.submit_order_advanced.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_strategy_exit_order_blocks_when_gateway_missing():
    """Exit orders must fail closed if no gateway is configured."""
    TestStrategy = _ConcreteBaseStrategy.build()

    broker = AsyncMock()
    broker.get_positions.return_value = [SimpleNamespace(symbol="AAPL", qty="5")]
    broker.submit_order_advanced = AsyncMock(return_value=SimpleNamespace(id="direct-order"))

    strategy = TestStrategy(broker=broker, parameters={})

    result = await strategy.submit_exit_order(symbol="AAPL", qty=3, side="sell", reason="test-exit")

    assert result is None
    broker.submit_order_advanced.assert_not_awaited()
