from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from brokers.broker_interface import (
    AccountInfo,
    Bar,
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerStatus,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from brokers.multi_broker import MultiBrokerManager


@dataclass
class _CompatOrder:
    id: str
    symbol: str
    qty: float
    side: str = "buy"
    filled_qty: float = 0.0
    status: str = "accepted"


class _RuntimeCompatBroker(BrokerInterface):
    def __init__(self, name: str, *, should_fail: bool = False):
        self._name = name
        self._should_fail = should_fail
        self._connected = True
        self.ws_started = False
        self.ws_symbols = None
        self.ws_stopped = False
        self._gateway_token = "tok_runtime_compat"

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_paper(self) -> bool:
        return True

    async def connect(self) -> bool:
        if self._should_fail:
            raise BrokerConnectionError("connect failed")
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_status(self) -> BrokerStatus:
        return BrokerStatus.CONNECTED if self._connected else BrokerStatus.DISCONNECTED

    async def health_check(self) -> bool:
        return not self._should_fail

    async def get_account(self) -> AccountInfo:
        if self._should_fail:
            raise BrokerError("account failed")
        return AccountInfo(
            broker_name=self.name,
            account_id=f"{self.name}-acct",
            equity=100000,
            cash=50000,
            buying_power=150000,
            portfolio_value=100000,
        )

    async def get_positions(self) -> list[Position]:
        if self._should_fail:
            raise BrokerError("positions failed")
        return [
            Position(
                symbol="AAPL",
                quantity=5,
                avg_entry_price=150.0,
                market_value=750.0,
                unrealized_pnl=10.0,
                unrealized_pnl_pct=0.013,
                current_price=152.0,
            )
        ]

    async def get_position(self, symbol: str) -> Optional[Position]:
        for pos in await self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    async def submit_order(self, request: OrderRequest) -> Order:
        if self._should_fail:
            raise BrokerError("submit failed")
        return Order(
            order_id=f"{self.name}-ord",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            filled_quantity=0,
            status=OrderStatus.ACCEPTED,
            broker_name=self.name,
        )

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_order(self, order_id: str) -> Optional[Order]:
        return Order(
            order_id=order_id,
            client_order_id="cid",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
            status=OrderStatus.ACCEPTED,
            broker_name=self.name,
        )

    async def get_orders(self, status=None, symbols=None, limit=100) -> list[Order]:
        return [
            Order(
                order_id=f"{self.name}-ord",
                client_order_id="cid",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1,
                status=OrderStatus.ACCEPTED,
                broker_name=self.name,
            )
        ]

    async def cancel_all_orders(self) -> int:
        return 0

    async def get_bars(self, symbol, timeframe, start, end=None, limit=1000) -> list[Bar]:
        return [
            Bar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=1.0,
                high=2.0,
                low=1.0,
                close=1.5,
                volume=1000,
            )
        ]

    async def get_latest_quote(self, symbol: str) -> dict[str, float]:
        return {"bid": 10.0, "ask": 10.2, "last": 10.1}

    async def get_clock(self) -> dict[str, Any]:
        return {"is_open": True, "next_open": "09:30", "next_close": "16:00"}

    async def close_position(self, symbol: str) -> Optional[Order]:
        return None

    async def close_all_positions(self) -> list[Order]:
        return []

    def enable_gateway_requirement(self) -> str:
        return self._gateway_token

    async def _internal_submit_order(
        self,
        order_request: Any,
        gateway_token: str,
        check_impact: bool = True,
    ) -> _CompatOrder:
        if self._should_fail:
            raise BrokerError("internal submit failed")
        if gateway_token != self._gateway_token:
            raise BrokerError("invalid gateway token")
        qty = float(getattr(order_request, "qty", 0) or 0)
        return _CompatOrder(id=f"{self.name}-internal", symbol=order_request.symbol, qty=qty)

    async def submit_order_advanced(self, order_request, check_impact: bool = True) -> _CompatOrder:
        if self._should_fail:
            raise BrokerError("advanced submit failed")
        qty = float(getattr(order_request, "qty", 0) or 0)
        return _CompatOrder(id=f"{self.name}-advanced", symbol=order_request.symbol, qty=qty)

    async def start_websocket(self, symbols: Optional[list[str]] = None) -> None:
        self.ws_started = True
        self.ws_symbols = symbols

    async def stop_websocket(self) -> None:
        self.ws_stopped = True


@pytest.mark.asyncio
async def test_multi_broker_runtime_shape_normalization_for_backup_account_and_positions():
    primary = _RuntimeCompatBroker("primary", should_fail=True)
    backup = _RuntimeCompatBroker("backup", should_fail=False)
    manager = MultiBrokerManager(
        primary=primary,
        backups=[backup],
        auto_start_monitoring=False,
    )

    account = await manager.get_account()
    assert hasattr(account, "id")
    assert str(account.id) == "backup-acct"

    positions = await manager.get_positions()
    assert positions
    assert hasattr(positions[0], "qty")
    assert float(positions[0].qty) == 5.0


@pytest.mark.asyncio
async def test_multi_broker_internal_submit_fails_over_to_backup():
    primary = _RuntimeCompatBroker("primary", should_fail=True)
    backup = _RuntimeCompatBroker("backup", should_fail=False)
    manager = MultiBrokerManager(
        primary=primary,
        backups=[backup],
        auto_start_monitoring=False,
    )

    token = manager.enable_gateway_requirement()
    order_request = SimpleNamespace(
        symbol="AAPL",
        qty=2,
        side="buy",
        order_type="market",
        time_in_force="day",
    )
    result = await manager._internal_submit_order(order_request, gateway_token=token)

    assert result is not None
    assert str(getattr(result, "id", "")) == "backup-ord"
    assert manager.is_failed_over is True
    assert manager.active_broker == backup


@pytest.mark.asyncio
async def test_multi_broker_start_stop_websocket_passthrough():
    primary = _RuntimeCompatBroker("primary", should_fail=False)
    manager = MultiBrokerManager(primary=primary, backups=[], auto_start_monitoring=False)

    await manager.start_websocket(["BTC/USD", "ETH/USD"])
    await manager.stop_websocket()

    assert primary.ws_started is True
    assert primary.ws_symbols == ["BTC/USD", "ETH/USD"]
    assert primary.ws_stopped is True
