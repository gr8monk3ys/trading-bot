"""
Mock broker for backtesting purposes with institutional-grade gap risk modeling.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GapEvent:
    """Record of an overnight gap event affecting a position."""
    symbol: str
    date: datetime
    prev_close: float
    open_price: float
    gap_pct: float
    position_side: str  # 'long' or 'short'
    position_qty: int
    stop_price: Optional[float]
    stop_triggered: bool
    slippage_from_stop: float  # Difference between stop and actual fill


@dataclass
class GapStatistics:
    """Aggregate gap statistics for analysis."""
    total_gaps: int
    gaps_exceeding_2pct: int
    stops_gapped_through: int
    total_gap_slippage: float
    largest_gap_pct: float
    average_gap_pct: float


class BacktestBroker:
    """Simple broker for backtesting purposes with realistic slippage modeling"""

    def __init__(
        self,
        api_key=None,
        api_secret=None,
        paper=True,
        initial_balance=10000,
        slippage_bps=5.0,
        spread_bps=3.0,
        enable_partial_fills=True,
    ):
        """
        Initialize broker with starting balance and slippage parameters.

        Args:
            api_key: API key (unused in backtest)
            api_secret: API secret (unused in backtest)
            paper: Paper trading mode (unused in backtest)
            initial_balance: Starting cash balance
            slippage_bps: Slippage in basis points (default 5.0 = 0.05%)
            spread_bps: Bid-ask spread in basis points (default 3.0 = 0.03%)
            enable_partial_fills: Whether to simulate partial fills on large orders
        """
        # Note: api_key, api_secret, paper are accepted for API compatibility
        # but not stored (unused in backtesting, avoids accidental credential exposure)
        self.balance = initial_balance
        self.positions = {}
        self.orders = []
        self.trades = []
        self.price_data = {}

        # Slippage parameters (basis points = 1/100th of 1%)
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.enable_partial_fills = enable_partial_fills

        # Current date for backtesting (set by BacktestEngine)
        self._current_date = None

        # Gap risk tracking
        self._gap_events: List[GapEvent] = []
        self._stop_orders: Dict[str, Dict] = {}  # symbol -> stop order details
        self._prev_day_close: Dict[str, float] = {}  # symbol -> previous close

    def _get_actual_daily_volume(self, symbol: str, date) -> float:
        """
        Get actual historical daily volume for realistic slippage calculation.

        Uses rolling 20-day average volume to smooth out anomalies.
        Falls back to conservative estimate if data unavailable.

        Args:
            symbol: Stock symbol
            date: Date to get volume for

        Returns:
            Average daily volume (float)
        """
        if symbol not in self.price_data:
            return 1000000.0  # Fallback default

        df = self.price_data[symbol]

        try:
            # Normalize date for comparison
            import pytz
            if hasattr(date, 'tzinfo') and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            # Get volume data up to (but not including) current date to avoid look-ahead
            try:
                historical_volume = df[df.index < date]['volume']
            except TypeError:
                # Handle timezone issues
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') else date
                historical_volume = df_naive[df_naive.index < date_naive]['volume']

            if len(historical_volume) == 0:
                return 1000000.0

            # Use rolling 20-day average for stability
            recent_volumes = historical_volume.tail(20)
            avg_volume = float(recent_volumes.mean()) if len(recent_volumes) > 0 else 1000000.0

            # Enforce minimum to prevent extreme slippage on data gaps
            return max(avg_volume, 100000.0)

        except Exception as e:
            logger.debug(f"Error getting volume for {symbol}: {e}")
            return 1000000.0  # Fallback on any error

    def _calculate_dynamic_spread(self, symbol: str, date, base_spread_bps: float = 3.0) -> float:
        """
        Calculate realistic bid-ask spread based on liquidity and volatility.

        Factors:
        - Volume: Lower volume = wider spread
        - Price: Lower price stocks have wider spreads (as % of price)
        - Volatility: Higher volatility = wider spread

        Args:
            symbol: Stock symbol
            date: Current date
            base_spread_bps: Base spread in basis points

        Returns:
            Dynamic spread in basis points
        """
        if symbol not in self.price_data:
            return base_spread_bps

        df = self.price_data[symbol]

        try:
            # Get historical data (avoid look-ahead)
            import pytz
            if hasattr(date, 'tzinfo') and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            try:
                historical = df[df.index < date]
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') else date
                historical = df_naive[df_naive.index < date_naive]

            if len(historical) < 10:
                return base_spread_bps

            # Factor 1: Volume-based liquidity (lower volume = wider spread)
            avg_volume = historical['volume'].tail(20).mean()
            volume_factor = 1.0
            if avg_volume < 500000:
                volume_factor = 2.0  # Low liquidity
            elif avg_volume < 1000000:
                volume_factor = 1.5
            elif avg_volume > 10000000:
                volume_factor = 0.7  # Very liquid

            # Factor 2: Price-based (lower price = wider spread as % of price)
            price = historical['close'].iloc[-1]
            price_factor = 1.0
            if price < 10:
                price_factor = 2.0  # Penny stock territory
            elif price < 50:
                price_factor = 1.3
            elif price > 500:
                price_factor = 0.8  # High-price stocks often more liquid

            # Factor 3: Volatility (higher volatility = wider spread)
            returns = historical['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            vol_factor = 1.0
            if volatility > 0.50:
                vol_factor = 2.0  # Very volatile
            elif volatility > 0.30:
                vol_factor = 1.5
            elif volatility < 0.15:
                vol_factor = 0.8  # Low volatility

            # Combine factors
            dynamic_spread = base_spread_bps * volume_factor * price_factor * vol_factor

            # Cap at reasonable bounds (0.01% to 0.5%)
            return max(1.0, min(dynamic_spread, 50.0))

        except Exception as e:
            logger.debug(f"Error calculating spread for {symbol}: {e}")
            return base_spread_bps

    def _get_stock_volatility(self, symbol: str, date) -> float:
        """Calculate annualized volatility for a stock."""
        if symbol not in self.price_data:
            return 0.30  # Default 30% volatility

        df = self.price_data[symbol]

        try:
            import pytz
            if hasattr(date, 'tzinfo') and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            try:
                historical = df[df.index < date]
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') else date
                historical = df_naive[df_naive.index < date_naive]

            if len(historical) < 20:
                return 0.30

            returns = historical['close'].pct_change().tail(20).dropna()
            return float(returns.std() * np.sqrt(252))

        except Exception:
            return 0.30

    def set_price_data(self, symbol, data):
        """Set historical price data for a symbol"""
        self.price_data[symbol] = data

    def get_price(self, symbol, date):
        """Get price for a symbol at given date"""
        if symbol not in self.price_data:
            raise ValueError(f"No price data for {symbol}")

        # Get closest date
        df = self.price_data[symbol]

        # Handle timezone comparison
        try:
            import pytz

            if df.index.tz is not None and (not hasattr(date, "tzinfo") or date.tzinfo is None):
                date = date.replace(tzinfo=pytz.UTC) if hasattr(date, "replace") else date
        except (ImportError, AttributeError):
            pass

        # Try exact match first
        try:
            if date in df.index:
                return df.loc[date, "close"]
        except TypeError:
            # Fallback: normalize to naive timestamps
            df_naive = df.copy()
            try:
                df_naive.index = df_naive.index.tz_localize(None)
            except TypeError:
                pass  # Already naive
            date_naive = (
                date.replace(tzinfo=None) if hasattr(date, "tzinfo") and date.tzinfo else date
            )
            if date_naive in df_naive.index:
                return df_naive.loc[date_naive, "close"]

        # Get closest previous date
        try:
            idx = df.index.get_indexer([date], method="pad")[0]
            if idx >= 0:
                return df.iloc[idx]["close"]
        except TypeError:
            # Fallback for timezone issues
            df_naive = df.copy()
            try:
                df_naive.index = df_naive.index.tz_localize(None)
            except TypeError:
                pass
            date_naive = (
                date.replace(tzinfo=None) if hasattr(date, "tzinfo") and date.tzinfo else date
            )
            idx = df_naive.index.get_indexer([date_naive], method="pad")[0]
            if idx >= 0:
                return df_naive.iloc[idx]["close"]

        raise ValueError(f"No price data for {symbol} at {date}")

    def get_historical_prices(self, symbol, days=30, end_date=None):
        """Get historical price data for a symbol"""
        if symbol not in self.price_data:
            # Create dummy data for testing
            end = end_date or datetime.now()
            start = end - timedelta(days=days)
            dates = pd.date_range(start=start, end=end, freq="B")

            # Generate random prices with upward trend
            np.random.seed(42 + hash(symbol) % 100)  # Consistent but different for each symbol
            price = 100 + np.random.rand() * 100  # Random start price between 100-200
            daily_returns = np.random.normal(0.0005, 0.015, len(dates))  # Slight upward bias
            prices = price * (1 + pd.Series(daily_returns)).cumprod()

            # Create OHLC data
            data = pd.DataFrame(
                {
                    "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
                    "high": prices * (1 + np.random.uniform(0.001, 0.02, len(dates))),
                    "low": prices * (1 - np.random.uniform(0.001, 0.02, len(dates))),
                    "close": prices,
                    "volume": np.random.randint(100000, 10000000, len(dates)),
                },
                index=dates,
            )

            # Ensure high is always the highest
            data["high"] = data[["open", "close", "high"]].max(axis=1)
            data["low"] = data[["open", "close", "low"]].min(axis=1)

            self.price_data[symbol] = data
            return data

        return self.price_data[symbol].tail(days)

    def _calculate_slippage(self, symbol, quantity, side, base_price, order_type):
        """
        Calculate realistic slippage based on order characteristics.

        Uses Almgren-Chriss inspired market impact model with:
        - Dynamic spread based on liquidity/volatility
        - Actual historical volume (not hardcoded)
        - Temporary + permanent impact components

        Args:
            symbol: Stock symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            base_price: Base price (mid price)
            order_type: 'market' or 'limit'

        Returns:
            Execution price after slippage
        """
        current_date = self._current_date if self._current_date else datetime.now()

        # 1. Dynamic bid-ask spread based on liquidity and volatility
        dynamic_spread_bps = self._calculate_dynamic_spread(symbol, current_date, self.spread_bps)
        spread_cost = base_price * (dynamic_spread_bps / 10000.0)

        # 2. Get ACTUAL daily volume (not hardcoded 1M)
        avg_daily_volume = self._get_actual_daily_volume(symbol, current_date)
        participation_rate = quantity / avg_daily_volume

        # 3. Almgren-Chriss inspired market impact model
        # Temporary impact: I_temp = c * sigma * sqrt(participation_rate)
        # Permanent impact: I_perm = d * sigma * participation_rate
        volatility = self._get_stock_volatility(symbol, current_date)

        c_temp = 0.6  # Temporary impact coefficient
        d_perm = 0.15  # Permanent impact coefficient

        temporary_impact_pct = c_temp * volatility * np.sqrt(participation_rate)
        permanent_impact_pct = d_perm * volatility * participation_rate

        # Total impact (capped at 10%)
        total_impact_pct = min(temporary_impact_pct + permanent_impact_pct, 0.10)
        market_impact_cost = base_price * total_impact_pct

        # 4. Market orders pay spread + impact; limit orders pay reduced impact
        if order_type == "market":
            total_slippage = spread_cost + market_impact_cost
        else:  # limit orders pay less (assuming they get filled at limit)
            total_slippage = market_impact_cost * 0.3  # 30% of market order slippage

        # 5. Apply slippage direction
        if side == "buy":
            execution_price = base_price + total_slippage  # Buy at higher price
        else:  # sell
            execution_price = base_price - total_slippage  # Sell at lower price

        # Log significant slippage for analysis
        if participation_rate > 0.05:
            logger.debug(
                f"Market impact for {symbol}: {participation_rate:.1%} of ADV, "
                f"impact: {(execution_price/base_price - 1)*100:+.3f}%"
            )

        return execution_price

    def _simulate_partial_fill(self, quantity, symbol):
        """
        Simulate partial fills for large orders.

        Large orders may not fill completely, especially in backtest.
        This prevents unrealistic fills of huge positions.

        Args:
            quantity: Requested quantity
            symbol: Stock symbol

        Returns:
            Actual filled quantity (may be less than requested)
        """
        if not self.enable_partial_fills:
            return quantity

        # Use ACTUAL daily volume (not hardcoded 1M)
        current_date = self._current_date if self._current_date else datetime.now()
        avg_daily_volume = self._get_actual_daily_volume(symbol, current_date)

        # If order is >10% of daily volume, may not fill completely
        participation_rate = quantity / avg_daily_volume

        if participation_rate > 0.10:  # Order is >10% of daily volume
            # Fill rate decreases as participation rate increases
            # At 10% participation: ~95% fill
            # At 50% participation: ~70% fill
            # At 100%+ participation: ~50% fill
            if participation_rate >= 1.0:
                fill_rate = 0.5 + (np.random.rand() * 0.15)
            elif participation_rate >= 0.5:
                fill_rate = 0.65 + (np.random.rand() * 0.15)
            elif participation_rate >= 0.2:
                fill_rate = 0.75 + (np.random.rand() * 0.15)
            else:  # 10-20%
                fill_rate = 0.85 + (np.random.rand() * 0.10)

            filled_qty = int(quantity * fill_rate)
            logger.warning(
                f"Partial fill for {symbol}: {filled_qty}/{quantity} "
                f"({fill_rate:.1%}) - order is {participation_rate:.1%} of ADV"
            )
            return max(filled_qty, 1)  # Fill at least 1 share

        return quantity

    def place_order(self, symbol, quantity, side, price=None, order_type="market"):
        """
        Place an order with realistic slippage and partial fills.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            price: Limit price (optional, for limit orders)
            order_type: 'market' or 'limit'

        Returns:
            Order dict with execution details
        """
        current_date = self._current_date if self._current_date else datetime.now()

        # Get base price (mid price)
        base_price = price if price else self.get_price(symbol, current_date)

        # Apply slippage to get realistic execution price
        execution_price = self._calculate_slippage(symbol, quantity, side, base_price, order_type)

        # Simulate partial fills for large orders
        filled_quantity = self._simulate_partial_fill(quantity, symbol)

        order = {
            "id": len(self.orders) + 1,
            "symbol": symbol,
            "quantity": quantity,
            "filled_qty": filled_quantity,  # Actual filled amount
            "side": side,
            "price": base_price,  # Requested price
            "filled_avg_price": execution_price,  # Actual execution price (with slippage)
            "type": order_type,
            "status": "filled" if filled_quantity == quantity else "partially_filled",
            "created_at": current_date,
            "filled_at": current_date,
            "slippage_bps": abs((execution_price - base_price) / base_price) * 10000,
        }

        self.orders.append(order)

        # Update positions and cash using FILLED quantity and EXECUTION price
        cost = filled_quantity * execution_price

        if side == "buy":
            self.balance -= cost
            if symbol in self.positions:
                self.positions[symbol]["quantity"] += filled_quantity
                # Average the price
                total_qty = self.positions[symbol]["quantity"]
                prev_cost = self.positions[symbol]["entry_price"] * (total_qty - filled_quantity)
                new_cost = cost
                self.positions[symbol]["entry_price"] = (prev_cost + new_cost) / total_qty
            else:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": filled_quantity,
                    "entry_price": execution_price,
                }
        else:  # sell
            self.balance += cost
            if symbol in self.positions:
                self.positions[symbol]["quantity"] -= filled_quantity
                if self.positions[symbol]["quantity"] <= 0:
                    del self.positions[symbol]

        # Record the trade with actual execution details
        self.trades.append(
            {
                "id": len(self.trades) + 1,
                "symbol": symbol,
                "quantity": filled_quantity,
                "side": side,
                "price": execution_price,
                "slippage": execution_price - base_price,
                "timestamp": current_date,
            }
        )

        return order

    def get_position(self, symbol):
        """Get position for a symbol"""
        return self.positions.get(symbol, None)

    def get_positions(self):
        """Get all positions"""
        return list(self.positions.values())

    def get_balance(self):
        """Get current cash balance"""
        return self.balance

    def get_portfolio_value(self, date=None):
        """Get total portfolio value at given date"""
        value = self.balance
        current_date = date or datetime.now()

        for symbol, position in self.positions.items():
            price = self.get_price(symbol, current_date)
            value += position["quantity"] * price

        return value

    def get_orders(self, status=None):
        """Get orders with given status"""
        if status:
            return [order for order in self.orders if order["status"] == status]
        return self.orders

    def get_trades(self):
        """Get all trades"""
        return self.trades

    # =========================================================================
    # ASYNC WRAPPERS FOR STRATEGY COMPATIBILITY
    # =========================================================================

    async def get_account(self):
        """Async wrapper for getting account info (for strategy compatibility)."""

        class MockAccount:
            def __init__(self, balance, portfolio_value):
                self.equity = str(portfolio_value)
                self.cash = str(balance)
                self.buying_power = str(balance)

        portfolio_value = self.get_portfolio_value()
        return MockAccount(self.balance, portfolio_value)

    async def submit_order_advanced(self, order_request):
        """Async wrapper for submitting orders (for strategy compatibility)."""
        # Extract order details from the order request object
        symbol = getattr(order_request, "symbol", None)
        qty = getattr(order_request, "qty", None) or getattr(order_request, "quantity", None)
        side = getattr(order_request, "side", "buy")
        order_type = getattr(order_request, "type", "market") or getattr(
            order_request, "order_type", "market"
        )

        if symbol is None or qty is None:
            return None

        # Convert side to string if it's an enum
        if hasattr(side, "value"):
            side = side.value

        # Place the order
        result = self.place_order(symbol, int(qty), side, order_type=str(order_type))

        # Return a mock order response
        class MockOrder:
            def __init__(self, order_dict):
                self.id = order_dict["id"]
                self.symbol = order_dict["symbol"]
                self.qty = str(order_dict["quantity"])
                self.filled_qty = str(order_dict["filled_qty"])
                self.filled_avg_price = str(order_dict["filled_avg_price"])
                self.status = order_dict["status"]
                self.side = order_dict["side"]

        return MockOrder(result)

    async def get_latest_quote(self, symbol):
        """Async wrapper for getting latest quote (for strategy compatibility)."""
        current_date = self._current_date if self._current_date else datetime.now()
        price = self.get_price(symbol, current_date)

        class MockQuote:
            def __init__(self, p):
                self.ask_price = p
                self.bid_price = p * 0.999  # Small spread

        return MockQuote(price)

    async def get_bars(self, symbol, start=None, end=None, timeframe="1Day", limit=100):
        """Async wrapper for getting price bars (for strategy compatibility)."""
        if symbol not in self.price_data:
            return []

        df = self.price_data[symbol]

        class MockBar:
            def __init__(self, timestamp, o, h, l, c, v):
                self.timestamp = timestamp
                self.open = o
                self.high = h
                self.low = l
                self.close = c
                self.volume = v

        bars = []
        for idx, row in df.tail(limit).iterrows():
            bars.append(
                MockBar(idx, row["open"], row["high"], row["low"], row["close"], row["volume"])
            )

        return bars

    async def get_all_positions(self):
        """Async wrapper for getting all positions (for strategy compatibility)."""

        # Convert dict positions to mock position objects
        class MockPosition:
            def __init__(self, position_dict):
                self.symbol = position_dict["symbol"]
                self.qty = str(position_dict["quantity"])
                self.quantity = position_dict["quantity"]
                self.entry_price = position_dict["entry_price"]

        return [MockPosition(pos) for pos in self.positions.values()]

    # =========================================================================
    # GAP RISK MODELING - INSTITUTIONAL GRADE
    # =========================================================================

    def set_stop_order(self, symbol: str, stop_price: float, quantity: int, side: str = "sell") -> None:
        """
        Register a stop order for gap risk tracking.

        Args:
            symbol: Stock symbol
            stop_price: Stop trigger price
            quantity: Quantity to sell/cover
            side: 'sell' for long positions, 'buy' for short covers
        """
        self._stop_orders[symbol] = {
            "stop_price": stop_price,
            "quantity": quantity,
            "side": side,
            "created_at": self._current_date,
        }
        logger.debug(f"Stop order registered: {symbol} @ ${stop_price:.2f}")

    def clear_stop_order(self, symbol: str) -> None:
        """Clear stop order for a symbol (e.g., when position closed)."""
        if symbol in self._stop_orders:
            del self._stop_orders[symbol]

    def update_prev_day_close(self, symbol: str, close_price: float) -> None:
        """Update previous day's closing price for gap calculation."""
        self._prev_day_close[symbol] = close_price

    def simulate_overnight_gap(
        self,
        symbol: str,
        open_price: float,
        date: datetime,
    ) -> Optional[GapEvent]:
        """
        Simulate overnight gap impact on positions.

        This is the core gap risk modeling function. It:
        1. Calculates gap percentage from previous close
        2. Checks if any stop orders were gapped through
        3. If stop was gapped through, fills at OPEN price (not stop price)
        4. Records the gap event for analysis

        Args:
            symbol: Stock symbol
            open_price: Opening price for the day
            date: Current trading date

        Returns:
            GapEvent if a significant gap occurred, None otherwise
        """
        # Need previous close to calculate gap
        if symbol not in self._prev_day_close:
            return None

        prev_close = self._prev_day_close[symbol]
        gap_pct = (open_price - prev_close) / prev_close

        # Only track gaps > 0.5% (material gaps)
        if abs(gap_pct) < 0.005:
            return None

        # Check if we have a position in this symbol
        position = self.positions.get(symbol)
        if position is None:
            # No position, no risk impact (but still record for statistics)
            return None

        position_qty = position["quantity"]
        position_side = "long" if position_qty > 0 else "short"
        stop_order = self._stop_orders.get(symbol)
        stop_price = stop_order["stop_price"] if stop_order else None

        # Check if stop was gapped through
        stop_triggered = False
        slippage_from_stop = 0.0

        if stop_order:
            stop_price = stop_order["stop_price"]

            if position_side == "long":
                # Long position: stop triggers if open < stop_price (gap down)
                if open_price < stop_price:
                    stop_triggered = True
                    slippage_from_stop = stop_price - open_price
                    logger.warning(
                        f"GAP RISK: {symbol} stop GAPPED THROUGH! "
                        f"Stop: ${stop_price:.2f}, Open: ${open_price:.2f}, "
                        f"Slippage: ${slippage_from_stop:.2f} ({slippage_from_stop/stop_price*100:.2f}%)"
                    )
            else:
                # Short position: stop triggers if open > stop_price (gap up)
                if open_price > stop_price:
                    stop_triggered = True
                    slippage_from_stop = open_price - stop_price
                    logger.warning(
                        f"GAP RISK: {symbol} short cover GAPPED THROUGH! "
                        f"Stop: ${stop_price:.2f}, Open: ${open_price:.2f}, "
                        f"Slippage: ${slippage_from_stop:.2f}"
                    )

        # Record gap event
        gap_event = GapEvent(
            symbol=symbol,
            date=date,
            prev_close=prev_close,
            open_price=open_price,
            gap_pct=gap_pct,
            position_side=position_side,
            position_qty=abs(position_qty),
            stop_price=stop_price,
            stop_triggered=stop_triggered,
            slippage_from_stop=slippage_from_stop,
        )
        self._gap_events.append(gap_event)

        # If stop was gapped through, execute at OPEN price (not stop price)
        if stop_triggered:
            self._execute_gap_stop(symbol, open_price, stop_order)

        return gap_event

    def _execute_gap_stop(self, symbol: str, fill_price: float, stop_order: Dict) -> None:
        """
        Execute a stop order that was gapped through at the open price.

        This is institutional-grade behavior: stops that gap through fill at
        the market (open) price, not the stop price. This prevents the
        unrealistic assumption that stops always fill at their trigger price.

        Args:
            symbol: Stock symbol
            fill_price: Actual fill price (the open)
            stop_order: Stop order details
        """
        quantity = stop_order["quantity"]
        side = stop_order["side"]

        logger.info(
            f"Executing gapped stop for {symbol}: {side} {quantity} @ ${fill_price:.2f} "
            f"(stop was ${stop_order['stop_price']:.2f})"
        )

        # Place the order at the fill price (the open)
        self.place_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=fill_price,
            order_type="market",  # Gapped stops fill at market
        )

        # Clear the stop order
        self.clear_stop_order(symbol)

    def process_day_start_gaps(self, date: datetime) -> List[GapEvent]:
        """
        Process overnight gaps for all positions at the start of trading day.

        Call this method at the beginning of each trading day in backtest.

        Args:
            date: Current trading date

        Returns:
            List of GapEvent objects for positions that experienced gaps
        """
        gap_events = []

        for symbol in list(self.positions.keys()):
            if symbol not in self.price_data:
                continue

            # Get today's open price
            try:
                open_price = self._get_open_price(symbol, date)
                if open_price is None:
                    continue

                gap_event = self.simulate_overnight_gap(symbol, open_price, date)
                if gap_event:
                    gap_events.append(gap_event)

            except Exception as e:
                logger.debug(f"Error processing gap for {symbol}: {e}")
                continue

        return gap_events

    def _get_open_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get the opening price for a symbol on a given date."""
        if symbol not in self.price_data:
            return None

        df = self.price_data[symbol]

        try:
            import pytz

            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            try:
                # Try exact match
                if date in df.index:
                    return float(df.loc[date, "open"])
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, "tzinfo") else date
                if date_naive in df_naive.index:
                    return float(df_naive.loc[date_naive, "open"])

            # Get closest date using indexer
            try:
                idx = df.index.get_indexer([date], method="pad")[0]
                if idx >= 0 and idx < len(df):
                    return float(df.iloc[idx]["open"])
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, "tzinfo") else date
                idx = df_naive.index.get_indexer([date_naive], method="pad")[0]
                if idx >= 0 and idx < len(df):
                    return float(df_naive.iloc[idx]["open"])

        except Exception as e:
            logger.debug(f"Error getting open price for {symbol}: {e}")

        return None

    def update_prev_day_closes(self, date: datetime) -> None:
        """
        Update previous day closes for all held symbols at end of trading day.

        Call this method at the end of each trading day in backtest.

        Args:
            date: Current trading date (end of day)
        """
        for symbol in self.positions.keys():
            try:
                close_price = self.get_price(symbol, date)
                self._prev_day_close[symbol] = close_price
            except Exception as e:
                logger.debug(f"Error updating prev close for {symbol}: {e}")

    def get_gap_events(self) -> List[GapEvent]:
        """Get all recorded gap events."""
        return self._gap_events.copy()

    def get_gap_statistics(self) -> GapStatistics:
        """
        Calculate aggregate gap statistics for analysis.

        Returns:
            GapStatistics with summary metrics
        """
        if not self._gap_events:
            return GapStatistics(
                total_gaps=0,
                gaps_exceeding_2pct=0,
                stops_gapped_through=0,
                total_gap_slippage=0.0,
                largest_gap_pct=0.0,
                average_gap_pct=0.0,
            )

        gap_pcts = [abs(e.gap_pct) for e in self._gap_events]
        stops_gapped = [e for e in self._gap_events if e.stop_triggered]

        return GapStatistics(
            total_gaps=len(self._gap_events),
            gaps_exceeding_2pct=sum(1 for g in gap_pcts if g > 0.02),
            stops_gapped_through=len(stops_gapped),
            total_gap_slippage=sum(e.slippage_from_stop for e in stops_gapped),
            largest_gap_pct=max(gap_pcts) if gap_pcts else 0.0,
            average_gap_pct=sum(gap_pcts) / len(gap_pcts) if gap_pcts else 0.0,
        )

    def clear_gap_tracking(self) -> None:
        """Clear all gap tracking data (for new backtest run)."""
        self._gap_events.clear()
        self._stop_orders.clear()
        self._prev_day_close.clear()
