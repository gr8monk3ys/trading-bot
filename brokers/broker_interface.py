"""
Broker Interface - Abstract Base Class for All Brokers

Defines the standard interface that all broker implementations must follow.
This enables multi-broker failover and strategy portability.

Usage:
    class MyBroker(BrokerInterface):
        async def get_account(self) -> AccountInfo:
            ...

Implementations:
    - AlpacaBroker: Primary broker (Alpaca Trading API)
    - InteractiveBrokersBroker: Backup broker (IB TWS/Gateway)
    - SimulatedBroker: Paper trading / backtesting
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BrokerStatus(Enum):
    """Broker connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"  # Partial functionality
    UNKNOWN = "unknown"


class OrderStatus(Enum):
    """Standard order status across all brokers."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Standard order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At open
    CLS = "cls"  # At close


@dataclass
class AccountInfo:
    """Standardized account information."""
    broker_name: str
    account_id: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trading_buying_power: Optional[float] = None
    maintenance_margin: Optional[float] = None
    initial_margin: Optional[float] = None
    currency: str = "USD"
    status: str = "active"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "broker_name": self.broker_name,
            "account_id": self.account_id,
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "portfolio_value": self.portfolio_value,
            "currency": self.currency,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Position:
    """Standardized position information."""
    symbol: str
    quantity: float  # Positive = long, negative = short
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    current_price: Optional[float] = None
    cost_basis: Optional[float] = None
    asset_class: str = "equity"  # equity, option, crypto, forex
    broker_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "current_price": self.current_price,
            "asset_class": self.asset_class,
        }


@dataclass
class Order:
    """Standardized order information."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    broker_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "status": self.status.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "avg_fill_price": self.avg_fill_price,
            "time_in_force": self.time_in_force.value,
        }


@dataclass
class Bar:
    """OHLCV bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


@dataclass
class OrderRequest:
    """Order request to submit to broker."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None
    extended_hours: bool = False

    # Bracket order legs (optional)
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    trailing_stop_percent: Optional[float] = None


class BrokerInterface(ABC):
    """
    Abstract base class for all broker implementations.

    All methods are async to support non-blocking I/O.
    Implementations should handle their own connection management.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name identifier."""
        pass

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Whether this is a paper trading account."""
        pass

    # === Account & Connection ===

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    async def get_status(self) -> BrokerStatus:
        """Get current connection status."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if broker is healthy and responsive
        """
        pass

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        pass

    # === Positions ===

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all current positions."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        pass

    # === Orders ===

    @abstractmethod
    async def submit_order(self, request: OrderRequest) -> Order:
        """
        Submit an order.

        Args:
            request: OrderRequest with order details

        Returns:
            Order object with order_id

        Raises:
            BrokerError on failure
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancel request accepted
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        pass

    @abstractmethod
    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbols: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get orders with optional filters."""
        pass

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        pass

    # === Market Data ===

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,  # e.g., "1Day", "1Hour", "5Min"
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Bar]:
        """Get historical bars."""
        pass

    @abstractmethod
    async def get_latest_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get latest quote for symbol.

        Returns:
            Dict with 'bid', 'ask', 'last', 'volume'
        """
        pass

    @abstractmethod
    async def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock status.

        Returns:
            Dict with 'is_open', 'next_open', 'next_close'
        """
        pass

    # === Position Management ===

    @abstractmethod
    async def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close entire position for a symbol.

        Returns:
            Order object for the closing trade
        """
        pass

    @abstractmethod
    async def close_all_positions(self) -> List[Order]:
        """
        Close all positions.

        Returns:
            List of closing orders
        """
        pass


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class BrokerConnectionError(BrokerError):
    """Connection-related error."""
    pass


class BrokerOrderError(BrokerError):
    """Order submission/cancellation error."""
    pass


class BrokerDataError(BrokerError):
    """Market data error."""
    pass
