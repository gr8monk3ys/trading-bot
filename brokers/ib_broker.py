"""
Interactive Brokers Broker - IB TWS/Gateway Implementation

Backup broker implementation for Interactive Brokers API.
Requires IB TWS or IB Gateway running locally or via cloud.

Prerequisites:
    pip install ib_insync

Configuration:
    IB_HOST=127.0.0.1  # TWS/Gateway host
    IB_PORT=7497       # Paper: 7497, Live: 7496
    IB_CLIENT_ID=1     # Client ID for this connection

Usage:
    broker = InteractiveBrokersBroker(
        host="127.0.0.1",
        port=7497,  # Paper trading
        client_id=1,
    )
    await broker.connect()
    account = await broker.get_account()

Note: This is a skeleton implementation. Full IB integration requires:
    1. IB account with API access enabled
    2. TWS or IB Gateway running
    3. ib_insync library installed
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from brokers.broker_interface import (
    AccountInfo,
    Bar,
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerOrderError,
    BrokerStatus,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

logger = logging.getLogger(__name__)

# Try to import ib_insync (optional dependency)
try:
    from ib_insync import IB, Contract, MarketOrder, LimitOrder, StopOrder, Trade
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. IB broker will not function.")


class InteractiveBrokersBroker(BrokerInterface):
    """
    Interactive Brokers implementation of BrokerInterface.

    Uses ib_insync for async IB API communication.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # Paper: 7497, Live: 7496
        client_id: int = 1,
        readonly: bool = False,
        account: Optional[str] = None,  # Specific account (for multi-account)
    ):
        """
        Initialize IB broker connection.

        Args:
            host: TWS/Gateway host address
            port: TWS/Gateway port (7497=paper, 7496=live)
            client_id: Unique client ID for this connection
            readonly: If True, only allow read operations
            account: Specific account ID (for multi-account setups)
        """
        self._host = host
        self._port = port
        self._client_id = client_id
        self._readonly = readonly
        self._account = account

        self._ib: Optional["IB"] = None
        self._connected = False
        self._is_paper = port == 7497

        if not IB_AVAILABLE:
            logger.error(
                "ib_insync not installed. Install with: pip install ib_insync"
            )

    @property
    def name(self) -> str:
        return "InteractiveBrokers"

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    async def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        if not IB_AVAILABLE:
            raise BrokerConnectionError(
                "ib_insync not installed. Install with: pip install ib_insync"
            )

        try:
            self._ib = IB()
            await self._ib.connectAsync(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                readonly=self._readonly,
                account=self._account or "",
            )
            self._connected = True
            logger.info(f"Connected to IB at {self._host}:{self._port}")
            return True

        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to IB: {e}")
            raise BrokerConnectionError(f"IB connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    async def get_status(self) -> BrokerStatus:
        """Get connection status."""
        if not self._ib:
            return BrokerStatus.DISCONNECTED

        if self._ib.isConnected():
            return BrokerStatus.CONNECTED
        return BrokerStatus.DISCONNECTED

    async def health_check(self) -> bool:
        """Check if IB connection is healthy."""
        if not self._ib or not self._connected:
            return False

        try:
            # Request server time as health check
            await asyncio.wait_for(
                self._ib.reqCurrentTimeAsync(),
                timeout=5.0,
            )
            return True
        except Exception as e:
            logger.warning(f"IB health check failed: {e}")
            return False

    async def get_account(self) -> AccountInfo:
        """Get account information."""
        self._ensure_connected()

        try:
            # Get account summary
            account_values = self._ib.accountSummary(self._account or "")

            # Parse account values into dict
            values = {}
            for av in account_values:
                values[av.tag] = float(av.value) if av.value else 0.0

            return AccountInfo(
                broker_name=self.name,
                account_id=self._account or self._ib.managedAccounts()[0],
                equity=values.get("NetLiquidation", 0),
                cash=values.get("TotalCashValue", 0),
                buying_power=values.get("BuyingPower", 0),
                portfolio_value=values.get("GrossPositionValue", 0),
                maintenance_margin=values.get("MaintMarginReq", 0),
                initial_margin=values.get("InitMarginReq", 0),
                currency="USD",
                status="active",
            )

        except Exception as e:
            logger.error(f"Failed to get IB account: {e}")
            raise BrokerError(f"Failed to get account: {e}")

    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        self._ensure_connected()

        try:
            ib_positions = self._ib.positions(self._account or "")

            positions = []
            for pos in ib_positions:
                # Calculate unrealized P&L (requires market data)
                current_price = pos.avgCost  # Placeholder
                market_value = pos.position * current_price

                positions.append(Position(
                    symbol=pos.contract.symbol,
                    quantity=pos.position,
                    avg_entry_price=pos.avgCost,
                    market_value=market_value,
                    unrealized_pnl=0.0,  # Would need market data
                    unrealized_pnl_pct=0.0,
                    current_price=current_price,
                    cost_basis=abs(pos.position * pos.avgCost),
                    asset_class=self._ib_sec_type_to_asset_class(pos.contract.secType),
                    broker_name=self.name,
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get IB positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def submit_order(self, request: OrderRequest) -> Order:
        """Submit an order to IB."""
        self._ensure_connected()

        if self._readonly:
            raise BrokerOrderError("Broker is in read-only mode")

        try:
            # Create contract
            contract = Contract()
            contract.symbol = request.symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"

            # Create order
            if request.order_type == OrderType.MARKET:
                ib_order = MarketOrder(
                    action="BUY" if request.side == OrderSide.BUY else "SELL",
                    totalQuantity=request.quantity,
                )
            elif request.order_type == OrderType.LIMIT:
                ib_order = LimitOrder(
                    action="BUY" if request.side == OrderSide.BUY else "SELL",
                    totalQuantity=request.quantity,
                    lmtPrice=request.limit_price,
                )
            elif request.order_type == OrderType.STOP:
                ib_order = StopOrder(
                    action="BUY" if request.side == OrderSide.BUY else "SELL",
                    totalQuantity=request.quantity,
                    stopPrice=request.stop_price,
                )
            else:
                raise BrokerOrderError(f"Unsupported order type: {request.order_type}")

            # Set time in force
            ib_order.tif = self._tif_to_ib(request.time_in_force)

            # Submit order
            trade = self._ib.placeOrder(contract, ib_order)

            # Wait briefly for order to be acknowledged
            await asyncio.sleep(0.5)

            return self._trade_to_order(trade)

        except Exception as e:
            logger.error(f"Failed to submit IB order: {e}")
            raise BrokerOrderError(f"Order submission failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        self._ensure_connected()

        try:
            # Find the trade
            for trade in self._ib.trades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return True

            logger.warning(f"Order {order_id} not found")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel IB order: {e}")
            return False

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        self._ensure_connected()

        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                return self._trade_to_order(trade)
        return None

    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbols: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get orders."""
        self._ensure_connected()

        orders = []
        for trade in self._ib.trades()[:limit]:
            order = self._trade_to_order(trade)

            # Filter by status
            if status and order.status != status:
                continue

            # Filter by symbol
            if symbols and order.symbol not in symbols:
                continue

            orders.append(order)

        return orders

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        self._ensure_connected()

        open_trades = [t for t in self._ib.trades() if not t.isDone()]
        for trade in open_trades:
            self._ib.cancelOrder(trade.order)

        return len(open_trades)

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Bar]:
        """Get historical bars."""
        self._ensure_connected()

        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"

            # Convert timeframe to IB format
            bar_size = self._timeframe_to_ib(timeframe)

            # Calculate duration
            if end is None:
                end = datetime.now()

            duration_days = (end - start).days + 1
            duration_str = f"{min(duration_days, 365)} D"

            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
            )

            return [
                Bar(
                    symbol=symbol,
                    timestamp=bar.date,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                for bar in bars[-limit:]
            ]

        except Exception as e:
            logger.error(f"Failed to get IB bars: {e}")
            raise BrokerError(f"Failed to get bars: {e}")

    async def get_latest_quote(self, symbol: str) -> Dict[str, float]:
        """Get latest quote."""
        self._ensure_connected()

        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"

            ticker = self._ib.reqMktData(contract)
            await asyncio.sleep(1)  # Wait for data

            return {
                "bid": ticker.bid or 0,
                "ask": ticker.ask or 0,
                "last": ticker.last or 0,
                "volume": ticker.volume or 0,
            }

        except Exception as e:
            logger.error(f"Failed to get IB quote: {e}")
            raise BrokerError(f"Failed to get quote: {e}")

    async def get_clock(self) -> Dict[str, Any]:
        """Get market clock (IB doesn't have direct equivalent)."""
        # IB doesn't have a direct market clock API
        # Use contract details or reqCurrentTime
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        is_open = market_open <= now <= market_close and now.weekday() < 5

        return {
            "is_open": is_open,
            "next_open": market_open.isoformat(),
            "next_close": market_close.isoformat(),
        }

    async def close_position(self, symbol: str) -> Optional[Order]:
        """Close position for symbol."""
        position = await self.get_position(symbol)
        if not position or position.quantity == 0:
            return None

        # Create closing order
        request = OrderRequest(
            symbol=symbol,
            side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
            quantity=abs(position.quantity),
            order_type=OrderType.MARKET,
        )

        return await self.submit_order(request)

    async def close_all_positions(self) -> List[Order]:
        """Close all positions."""
        positions = await self.get_positions()
        orders = []

        for pos in positions:
            if pos.quantity != 0:
                order = await self.close_position(pos.symbol)
                if order:
                    orders.append(order)

        return orders

    # === Helper Methods ===

    def _ensure_connected(self):
        """Ensure broker is connected."""
        if not self._ib or not self._connected:
            raise BrokerConnectionError("Not connected to IB")

    def _trade_to_order(self, trade: "Trade") -> Order:
        """Convert IB Trade to Order."""
        return Order(
            order_id=str(trade.order.orderId),
            client_order_id=str(trade.order.clientId),
            symbol=trade.contract.symbol,
            side=OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL,
            order_type=self._ib_order_type(trade.order.orderType),
            quantity=trade.order.totalQuantity,
            filled_quantity=trade.orderStatus.filled,
            status=self._ib_status(trade.orderStatus.status),
            limit_price=getattr(trade.order, "lmtPrice", None),
            stop_price=getattr(trade.order, "auxPrice", None),
            avg_fill_price=trade.orderStatus.avgFillPrice,
            time_in_force=self._ib_tif(trade.order.tif),
            broker_name=self.name,
        )

    def _ib_order_type(self, ib_type: str) -> OrderType:
        """Convert IB order type to standard."""
        mapping = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP,
            "STP LMT": OrderType.STOP_LIMIT,
            "TRAIL": OrderType.TRAILING_STOP,
        }
        return mapping.get(ib_type, OrderType.MARKET)

    def _ib_status(self, ib_status: str) -> OrderStatus:
        """Convert IB status to standard."""
        mapping = {
            "PendingSubmit": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.PENDING,
            "Submitted": OrderStatus.ACCEPTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "ApiCancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
        }
        return mapping.get(ib_status, OrderStatus.PENDING)

    def _ib_tif(self, ib_tif: str) -> TimeInForce:
        """Convert IB TIF to standard."""
        mapping = {
            "DAY": TimeInForce.DAY,
            "GTC": TimeInForce.GTC,
            "IOC": TimeInForce.IOC,
            "FOK": TimeInForce.FOK,
            "OPG": TimeInForce.OPG,
        }
        return mapping.get(ib_tif, TimeInForce.DAY)

    def _tif_to_ib(self, tif: TimeInForce) -> str:
        """Convert standard TIF to IB."""
        mapping = {
            TimeInForce.DAY: "DAY",
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
            TimeInForce.OPG: "OPG",
        }
        return mapping.get(tif, "DAY")

    def _timeframe_to_ib(self, timeframe: str) -> str:
        """Convert timeframe to IB bar size."""
        mapping = {
            "1Min": "1 min",
            "5Min": "5 mins",
            "15Min": "15 mins",
            "1Hour": "1 hour",
            "1Day": "1 day",
        }
        return mapping.get(timeframe, "1 day")

    def _ib_sec_type_to_asset_class(self, sec_type: str) -> str:
        """Convert IB security type to asset class."""
        mapping = {
            "STK": "equity",
            "OPT": "option",
            "FUT": "futures",
            "CASH": "forex",
            "CRYPTO": "crypto",
        }
        return mapping.get(sec_type, "equity")
