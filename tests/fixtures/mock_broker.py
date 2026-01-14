"""
Mock broker for testing - simulates Alpaca API behavior
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class MockBar:
    """Mock price bar data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __getattribute__(self, name):
        # Handle both attribute and dict-style access
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None


@dataclass
class MockPosition:
    """Mock position"""
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_plpc: float
    cost_basis: float


@dataclass
class MockAccount:
    """Mock account"""
    id: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


@dataclass
class MockOrder:
    """Mock order"""
    id: str
    client_order_id: str
    symbol: str
    qty: float
    side: str
    type: str
    time_in_force: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    created_at: datetime
    updated_at: datetime


class MockAlpacaBroker:
    """
    Mock Alpaca broker for testing

    Simulates realistic market behavior including:
    - Price movements
    - Slippage
    - Order fills
    - Position tracking
    - Account updates
    """

    def __init__(self, paper: bool = True, initial_capital: float = 100000.0):
        self.paper = paper
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, MockPosition] = {}
        self.orders: List[MockOrder] = []
        self.order_counter = 0
        self.price_data: Dict[str, List[MockBar]] = {}
        self.slippage = 0.001  # 0.1% slippage

    async def get_account(self) -> MockAccount:
        """Get account info"""
        position_value = sum(
            p.qty * p.current_price
            for p in self.positions.values()
        )
        equity = self.cash + position_value

        return MockAccount(
            id="mock_account",
            equity=equity,
            cash=self.cash,
            buying_power=self.cash * 4,  # 4x margin
            portfolio_value=equity
        )

    async def get_positions(self) -> List[MockPosition]:
        """Get all positions"""
        return list(self.positions.values())

    async def get_position(self, symbol: str) -> Optional[MockPosition]:
        """Get position for symbol"""
        return self.positions.get(symbol)

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = "market",
        time_in_force: str = "day"
    ) -> MockOrder:
        """Submit order (simplified)"""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"

        # Get current price
        current_price = await self._get_current_price(symbol)

        # Apply slippage
        if side == "buy":
            fill_price = current_price * (1 + self.slippage)
        else:
            fill_price = current_price * (1 - self.slippage)

        # Create order
        order = MockOrder(
            id=order_id,
            client_order_id=order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force,
            status="filled",
            filled_qty=qty,
            filled_avg_price=fill_price,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.orders.append(order)

        # Update positions and cash
        await self._execute_order(order)

        return order

    async def submit_order_advanced(self, order_request) -> MockOrder:
        """Submit advanced order (bracket, etc.)"""
        # Simplified - just execute main order
        # Handle both OrderBuilder (with .type) and dict
        if hasattr(order_request, 'symbol'):
            # Pydantic model from Alpaca
            symbol = order_request.symbol
            qty = order_request.qty
            side = order_request.side
            order_type = getattr(order_request, 'type', 'market')
            time_in_force = getattr(order_request, 'time_in_force', 'day')
        else:
            # Dict
            symbol = order_request['symbol']
            qty = order_request['qty']
            side = order_request['side']
            order_type = order_request.get('type', 'market')
            time_in_force = order_request.get('time_in_force', 'day')

        return await self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )

    async def cancel_order(self, order_id: str):
        """Cancel order"""
        for order in self.orders:
            if order.id == order_id:
                order.status = "canceled"
                return True
        return False

    async def get_orders(self, status: str = None) -> List[MockOrder]:
        """Get orders"""
        if status:
            return [o for o in self.orders if o.status == status]
        return self.orders

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: datetime = None,
        end: datetime = None,
        limit: int = 100
    ) -> List[MockBar]:
        """Get historical bars"""
        # Generate or return cached data
        if symbol not in self.price_data:
            self.price_data[symbol] = await self._generate_price_data(
                symbol, start or datetime.now() - timedelta(days=365), end or datetime.now()
            )

        return self.price_data[symbol]

    async def get_last_price(self, symbol: str) -> float:
        """Get last price"""
        return await self._get_current_price(symbol)

    async def get_market_status(self) -> Dict:
        """Get market status"""
        now = datetime.now()
        return {
            'is_open': True,  # Always open for testing
            'next_open': now,
            'next_close': now + timedelta(hours=6),
            'timestamp': now
        }

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        if symbol in self.price_data and self.price_data[symbol]:
            return self.price_data[symbol][-1].close

        # Default price if no data
        return 100.0

    async def _execute_order(self, order: MockOrder):
        """Execute order and update positions/cash"""
        symbol = order.symbol
        qty = order.filled_qty
        price = order.filled_avg_price
        side = order.side

        if side == "buy":
            # Deduct cash
            cost = qty * price
            self.cash -= cost

            # Update or create position
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_qty = pos.qty + qty
                new_avg_price = (pos.avg_entry_price * pos.qty + price * qty) / new_qty
                pos.qty = new_qty
                pos.avg_entry_price = new_avg_price
                pos.current_price = price
                pos.cost_basis = new_qty * new_avg_price
                pos.unrealized_pl = new_qty * (price - new_avg_price)
                pos.unrealized_plpc = (price - new_avg_price) / new_avg_price if new_avg_price > 0 else 0
            else:
                self.positions[symbol] = MockPosition(
                    symbol=symbol,
                    qty=qty,
                    side="long",
                    avg_entry_price=price,
                    current_price=price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                    cost_basis=qty * price
                )

        elif side == "sell":
            # Add cash
            proceeds = qty * price
            self.cash += proceeds

            # Update or close position
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.qty -= qty

                if pos.qty <= 0:
                    del self.positions[symbol]
                else:
                    pos.current_price = price
                    pos.unrealized_pl = pos.qty * (price - pos.avg_entry_price)
                    pos.unrealized_plpc = (price - pos.avg_entry_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0

    async def _generate_price_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        market_regime: str = "normal"
    ) -> List[MockBar]:
        """
        Generate realistic price data for testing

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            market_regime: "bull", "bear", "sideways", "normal", "volatile"
        """
        dates = pd.date_range(start=start, end=end, freq='D')
        num_bars = len(dates)

        # Starting price
        base_price = 100.0

        # Market regime parameters
        regime_params = {
            "bull": {"drift": 0.001, "volatility": 0.015},
            "bear": {"drift": -0.001, "volatility": 0.020},
            "sideways": {"drift": 0.0, "volatility": 0.010},
            "normal": {"drift": 0.0005, "volatility": 0.015},
            "volatile": {"drift": 0.0, "volatility": 0.035}
        }

        params = regime_params.get(market_regime, regime_params["normal"])

        # Generate returns using geometric Brownian motion
        returns = np.random.normal(
            params["drift"],
            params["volatility"],
            num_bars
        )

        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        bars = []
        for i, date in enumerate(dates):
            close = prices[i]
            intraday_range = close * np.random.uniform(0.01, 0.03)

            open_price = float(close * (1 + np.random.uniform(-0.01, 0.01)))
            high = float(max(open_price, close) + intraday_range * 0.5)
            low = float(min(open_price, close) - intraday_range * 0.5)
            volume = float(np.random.uniform(1_000_000, 10_000_000))

            bars.append(MockBar(
                timestamp=date,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            ))

        return bars

    def set_market_regime(self, symbol: str, regime: str):
        """Set market regime for symbol (for testing different conditions)"""
        # Clear cached data and regenerate
        if symbol in self.price_data:
            start = self.price_data[symbol][0].timestamp
            end = self.price_data[symbol][-1].timestamp
            asyncio.create_task(self._generate_price_data(symbol, start, end, regime))
