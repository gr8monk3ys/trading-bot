"""
Core state and price-data plumbing for :class:`BacktestBroker`.

Provides:
- :class:`ExecutionProfile` dataclass and ``EXECUTION_PROFILE_PRESETS``.
- :class:`BacktestBrokerCore` — initialization, execution-profile
  management, historical price-data storage, price retrieval, position
  and balance queries, and async wrappers used by strategies.

Slippage / partial-fill / order placement live in
:mod:`brokers.backtest.execution`; gap-risk modeling lives in
:mod:`brokers.backtest.gaps`. These mixins are combined into the
public :class:`BacktestBroker` in :mod:`brokers.backtest`.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from brokers.backtest.gaps import GapEvent

logger = logging.getLogger(__name__)


@dataclass
class ExecutionProfile:
    """Execution realism profile for paper/backtest fills."""

    name: str
    slippage_multiplier: float = 1.0
    partial_fill_multiplier: float = 1.0
    min_latency_ms: int = 15
    max_latency_ms: int = 120
    reject_probability: float = 0.0


EXECUTION_PROFILE_PRESETS: Dict[str, ExecutionProfile] = {
    "idealistic": ExecutionProfile(
        name="idealistic",
        slippage_multiplier=0.7,
        partial_fill_multiplier=1.1,
        min_latency_ms=2,
        max_latency_ms=20,
        reject_probability=0.0,
    ),
    "realistic": ExecutionProfile(
        name="realistic",
        slippage_multiplier=1.0,
        partial_fill_multiplier=1.0,
        min_latency_ms=15,
        max_latency_ms=120,
        reject_probability=0.0,
    ),
    "stressed": ExecutionProfile(
        name="stressed",
        slippage_multiplier=1.6,
        partial_fill_multiplier=0.75,
        min_latency_ms=80,
        max_latency_ms=500,
        reject_probability=0.02,
    ),
}


class BacktestBrokerCore:
    """Core state and price-data plumbing for :class:`BacktestBroker`."""

    def __init__(
        self,
        api_key=None,
        api_secret=None,
        paper=True,
        initial_balance=10000,
        slippage_bps=5.0,
        spread_bps=3.0,
        enable_partial_fills=True,
        execution_profile: str = "realistic",
        random_seed: Optional[int] = None,
        run_id: Optional[str] = None,
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
            execution_profile: Fill realism profile ('idealistic', 'realistic', 'stressed')
            random_seed: Optional seed for deterministic simulations
            run_id: Optional run identifier for observability correlation
        """
        # Note: api_key, api_secret, paper are accepted for API compatibility
        # but not stored (unused in backtesting, avoids accidental credential exposure)
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.trades: List[Dict] = []
        self.price_data: Dict[str, pd.DataFrame] = {}

        # Slippage parameters (basis points = 1/100th of 1%)
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.enable_partial_fills = enable_partial_fills
        self._rng = np.random.default_rng(random_seed)
        self.execution_profile = EXECUTION_PROFILE_PRESETS["realistic"]
        self.set_execution_profile(execution_profile)
        self.run_id = run_id

        # Current date for backtesting (set by BacktestEngine)
        self._current_date: Optional[datetime] = None

        # Gap risk tracking
        self._gap_events: List[GapEvent] = []
        self._stop_orders: Dict[str, Dict] = {}  # symbol -> stop order details
        self._prev_day_close: Dict[str, float] = {}  # symbol -> previous close

    def set_execution_profile(self, execution_profile) -> None:
        """Set execution realism profile."""
        if isinstance(execution_profile, ExecutionProfile):
            self.execution_profile = execution_profile
            return
        profile = EXECUTION_PROFILE_PRESETS.get(str(execution_profile).lower())
        if profile is None:
            raise ValueError(
                f"Unknown execution_profile '{execution_profile}'. "
                f"Expected one of: {', '.join(sorted(EXECUTION_PROFILE_PRESETS.keys()))}"
            )
        self.execution_profile = profile

    def get_execution_profile(self) -> ExecutionProfile:
        """Return active execution profile."""
        return self.execution_profile

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
            prices = price * np.cumprod(1 + daily_returns)

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

            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            # Get volume data up to (but not including) current date to avoid look-ahead
            try:
                historical_volume = df[df.index < date]["volume"]
            except TypeError:
                # Handle timezone issues
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, "tzinfo") else date
                historical_volume = df_naive[df_naive.index < date_naive]["volume"]

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

            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            try:
                historical = df[df.index < date]
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, "tzinfo") else date
                historical = df_naive[df_naive.index < date_naive]

            if len(historical) < 10:
                return base_spread_bps

            # Factor 1: Volume-based liquidity (lower volume = wider spread)
            avg_volume = historical["volume"].tail(20).mean()
            volume_factor = 1.0
            if avg_volume < 500000:
                volume_factor = 2.0  # Low liquidity
            elif avg_volume < 1000000:
                volume_factor = 1.5
            elif avg_volume > 10000000:
                volume_factor = 0.7  # Very liquid

            # Factor 2: Price-based (lower price = wider spread as % of price)
            price = historical["close"].iloc[-1]
            price_factor = 1.0
            if price < 10:
                price_factor = 2.0  # Penny stock territory
            elif price < 50:
                price_factor = 1.3
            elif price > 500:
                price_factor = 0.8  # High-price stocks often more liquid

            # Factor 3: Volatility (higher volatility = wider spread)
            returns = historical["close"].pct_change().tail(20)
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

            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            try:
                historical = df[df.index < date]
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, "tzinfo") else date
                historical = df_naive[df_naive.index < date_naive]

            if len(historical) < 20:
                return 0.30

            returns = historical["close"].pct_change().tail(20).dropna()
            return float(returns.std() * np.sqrt(252))

        except Exception:
            return 0.30

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
            def __init__(self, timestamp, o, h, low, c, v):
                self.timestamp = timestamp
                self.open = o
                self.high = h
                self.low = low
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
