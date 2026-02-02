"""
Tick Data Integration Layer

Provides institutional-grade microstructure data access:
1. Polygon.io Integration: Real-time and historical tick data
2. TAQ Data Support: NYSE Trade and Quote format parsing
3. Microstructure Features: Bid-ask spread, order flow, VWAP
4. Aggregation: Custom bar construction from ticks

Why tick data matters:
- Bar backtests miss 80%+ of price action
- Slippage modeling from bar data has 100-500 bps error
- Real execution happens at tick level
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator, Callable, AsyncIterator
import numpy as np

logger = logging.getLogger(__name__)


class TickType(Enum):
    """Type of tick data."""
    TRADE = "trade"
    QUOTE = "quote"
    NBBO = "nbbo"


class Exchange(Enum):
    """Trading venue identifiers."""
    NYSE = "N"
    NASDAQ = "Q"
    ARCA = "P"
    BATS = "Z"
    IEX = "V"
    EDGX = "K"
    EDGA = "J"
    BYX = "Y"
    BZX = "Z"
    MEMX = "U"
    DARK = "D"  # Dark pool (generic)

    @classmethod
    def from_code(cls, code: str) -> "Exchange":
        """Get exchange from single-character code."""
        for ex in cls:
            if ex.value == code:
                return ex
        return cls.DARK  # Unknown exchanges treated as dark


@dataclass
class Trade:
    """Single trade tick."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: Exchange
    conditions: List[str] = field(default_factory=list)

    # Trade condition flags
    is_odd_lot: bool = False
    is_intermarket_sweep: bool = False
    is_opening: bool = False
    is_closing: bool = False

    @property
    def notional(self) -> float:
        """Dollar value of trade."""
        return self.price * self.size

    def is_eligible_for_high_low(self) -> bool:
        """Whether trade should be used for high/low calculation."""
        # Exclude odd lots and certain condition codes
        return not self.is_odd_lot and "W" not in self.conditions


@dataclass
class Quote:
    """Single quote tick (NBBO or exchange-level)."""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    bid_exchange: Optional[Exchange] = None
    ask_exchange: Optional[Exchange] = None

    @property
    def mid_price(self) -> float:
        """Mid-point of bid-ask spread."""
        if self.bid_price <= 0 or self.ask_price <= 0:
            return 0.0
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Spread in basis points relative to mid."""
        mid = self.mid_price
        if mid <= 0:
            return 0.0
        return (self.spread / mid) * 10000

    @property
    def total_size(self) -> int:
        """Total liquidity at top of book."""
        return self.bid_size + self.ask_size

    def is_crossed(self) -> bool:
        """Check if bid >= ask (invalid market)."""
        return self.bid_price >= self.ask_price and self.bid_price > 0


@dataclass
class AggregatedBar:
    """Bar aggregated from tick data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int

    # Microstructure metrics
    avg_spread_bps: float = 0.0
    avg_trade_size: float = 0.0
    buy_volume: int = 0
    sell_volume: int = 0

    @property
    def dollar_volume(self) -> float:
        """Approximate dollar volume."""
        return self.vwap * self.volume

    @property
    def order_flow_imbalance(self) -> float:
        """Buy-sell imbalance (-1 to 1)."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total


@dataclass
class MicrostructureSnapshot:
    """Point-in-time microstructure state."""
    symbol: str
    timestamp: datetime

    # Price levels
    bid: float
    ask: float
    last_trade: float
    vwap: float

    # Liquidity
    bid_size: int
    ask_size: int
    spread_bps: float

    # Recent activity
    volume_1min: int
    trade_count_1min: int
    avg_trade_size_1min: float

    # Order flow
    buy_volume_1min: int
    sell_volume_1min: int
    order_flow_imbalance: float

    # Volatility
    realized_vol_1min: float
    high_1min: float
    low_1min: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "mid": (self.bid + self.ask) / 2,
            "last_trade": self.last_trade,
            "vwap": self.vwap,
            "spread_bps": self.spread_bps,
            "volume_1min": self.volume_1min,
            "order_flow_imbalance": self.order_flow_imbalance,
            "realized_vol_1min": self.realized_vol_1min,
        }


class TickDataProvider(ABC):
    """Abstract base class for tick data providers."""

    @abstractmethod
    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """Fetch historical trades."""
        pass

    @abstractmethod
    async def get_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None,
    ) -> List[Quote]:
        """Fetch historical quotes."""
        pass

    @abstractmethod
    async def stream_trades(
        self,
        symbols: List[str],
        callback: Callable[[Trade], None],
    ) -> None:
        """Stream real-time trades."""
        pass

    @abstractmethod
    async def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> None:
        """Stream real-time quotes."""
        pass


class PolygonTickProvider(TickDataProvider):
    """Polygon.io tick data provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polygon.io",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """Fetch historical trades from Polygon."""
        session = await self._get_session()

        trades = []
        start_ts = int(start.timestamp() * 1_000_000_000)  # nanoseconds

        url = f"{self.base_url}/v3/trades/{symbol}"
        params = {
            "timestamp.gte": start_ts,
            "timestamp.lt": int(end.timestamp() * 1_000_000_000),
            "limit": min(limit or 50000, 50000),
            "apiKey": self.api_key,
        }

        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Polygon API error: {response.status}")
                return trades

            data = await response.json()

            for result in data.get("results", []):
                trade = Trade(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(
                        result["sip_timestamp"] / 1_000_000_000
                    ),
                    price=result["price"],
                    size=result["size"],
                    exchange=Exchange.from_code(result.get("exchange", "D")),
                    conditions=result.get("conditions", []),
                )
                trades.append(trade)

        return trades

    async def get_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None,
    ) -> List[Quote]:
        """Fetch historical quotes from Polygon."""
        session = await self._get_session()

        quotes = []
        start_ts = int(start.timestamp() * 1_000_000_000)

        url = f"{self.base_url}/v3/quotes/{symbol}"
        params = {
            "timestamp.gte": start_ts,
            "timestamp.lt": int(end.timestamp() * 1_000_000_000),
            "limit": min(limit or 50000, 50000),
            "apiKey": self.api_key,
        }

        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Polygon API error: {response.status}")
                return quotes

            data = await response.json()

            for result in data.get("results", []):
                quote = Quote(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(
                        result["sip_timestamp"] / 1_000_000_000
                    ),
                    bid_price=result.get("bid_price", 0),
                    bid_size=result.get("bid_size", 0),
                    ask_price=result.get("ask_price", 0),
                    ask_size=result.get("ask_size", 0),
                    bid_exchange=Exchange.from_code(
                        result.get("bid_exchange", "D")
                    ),
                    ask_exchange=Exchange.from_code(
                        result.get("ask_exchange", "D")
                    ),
                )
                quotes.append(quote)

        return quotes

    async def stream_trades(
        self,
        symbols: List[str],
        callback: Callable[[Trade], None],
    ) -> None:
        """Stream real-time trades via WebSocket."""
        import websockets

        ws_url = f"wss://socket.polygon.io/stocks"

        async with websockets.connect(ws_url) as ws:
            # Authenticate
            await ws.send(f'{{"action":"auth","params":"{self.api_key}"}}')

            # Subscribe to trades
            symbols_str = ",".join([f"T.{s}" for s in symbols])
            await ws.send(f'{{"action":"subscribe","params":"{symbols_str}"}}')

            async for message in ws:
                import json
                data = json.loads(message)

                for item in data:
                    if item.get("ev") == "T":  # Trade event
                        trade = Trade(
                            symbol=item["sym"],
                            timestamp=datetime.fromtimestamp(item["t"] / 1000),
                            price=item["p"],
                            size=item["s"],
                            exchange=Exchange.from_code(item.get("x", "D")),
                            conditions=item.get("c", []),
                        )
                        callback(trade)

    async def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> None:
        """Stream real-time quotes via WebSocket."""
        import websockets

        ws_url = f"wss://socket.polygon.io/stocks"

        async with websockets.connect(ws_url) as ws:
            # Authenticate
            await ws.send(f'{{"action":"auth","params":"{self.api_key}"}}')

            # Subscribe to quotes
            symbols_str = ",".join([f"Q.{s}" for s in symbols])
            await ws.send(f'{{"action":"subscribe","params":"{symbols_str}"}}')

            async for message in ws:
                import json
                data = json.loads(message)

                for item in data:
                    if item.get("ev") == "Q":  # Quote event
                        quote = Quote(
                            symbol=item["sym"],
                            timestamp=datetime.fromtimestamp(item["t"] / 1000),
                            bid_price=item.get("bp", 0),
                            bid_size=item.get("bs", 0),
                            ask_price=item.get("ap", 0),
                            ask_size=item.get("as", 0),
                        )
                        callback(quote)


class TAQDataParser:
    """Parser for NYSE Trade and Quote (TAQ) data files."""

    # TAQ trade condition codes
    CONDITION_CODES = {
        "@": "regular",
        "A": "acquisition",
        "B": "bunched",
        "C": "cash",
        "D": "distribution",
        "E": "placeholder",
        "F": "intermarket_sweep",
        "G": "bunched_sold",
        "H": "price_variation",
        "I": "odd_lot",
        "K": "rule_155",
        "L": "sold_last",
        "M": "market_center_official_close",
        "N": "next_day",
        "O": "opening",
        "P": "prior_reference",
        "Q": "market_center_official_open",
        "R": "seller",
        "S": "split",
        "T": "form_t",
        "U": "extended_hours",
        "V": "contingent",
        "W": "average_price",
        "X": "cross",
        "Y": "sold_out_of_sequence",
        "Z": "sold",
    }

    def parse_trade_file(
        self,
        filepath: str,
        symbols: Optional[List[str]] = None,
    ) -> Iterator[Trade]:
        """
        Parse TAQ trade file.

        TAQ format (pipe-delimited):
        TIME|SYMBOL|SIZE|PRICE|EXCHANGE|SALE_CONDITION|...
        """
        with open(filepath, 'r') as f:
            # Skip header
            header = f.readline().strip().split('|')

            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 6:
                    continue

                symbol = parts[1].strip()
                if symbols and symbol not in symbols:
                    continue

                # Parse time (HHMMSS.sss format)
                time_str = parts[0]
                try:
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])
                    second = int(time_str[4:6])
                    microsecond = int(float("0" + time_str[6:]) * 1_000_000)

                    timestamp = datetime.now().replace(
                        hour=hour,
                        minute=minute,
                        second=second,
                        microsecond=microsecond,
                    )
                except (ValueError, IndexError):
                    continue

                try:
                    trade = Trade(
                        symbol=symbol,
                        timestamp=timestamp,
                        size=int(parts[2]),
                        price=float(parts[3]),
                        exchange=Exchange.from_code(parts[4]),
                        conditions=list(parts[5]) if len(parts) > 5 else [],
                    )

                    # Set condition flags
                    trade.is_odd_lot = "I" in trade.conditions
                    trade.is_intermarket_sweep = "F" in trade.conditions
                    trade.is_opening = "O" in trade.conditions
                    trade.is_closing = "M" in trade.conditions

                    yield trade
                except (ValueError, IndexError):
                    continue

    def parse_quote_file(
        self,
        filepath: str,
        symbols: Optional[List[str]] = None,
    ) -> Iterator[Quote]:
        """
        Parse TAQ quote file.

        TAQ format (pipe-delimited):
        TIME|SYMBOL|BID|BIDSIZ|ASK|ASKSIZ|BIDEX|ASKEX|...
        """
        with open(filepath, 'r') as f:
            # Skip header
            header = f.readline().strip().split('|')

            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 6:
                    continue

                symbol = parts[1].strip()
                if symbols and symbol not in symbols:
                    continue

                # Parse time
                time_str = parts[0]
                try:
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])
                    second = int(time_str[4:6])
                    microsecond = int(float("0" + time_str[6:]) * 1_000_000)

                    timestamp = datetime.now().replace(
                        hour=hour,
                        minute=minute,
                        second=second,
                        microsecond=microsecond,
                    )
                except (ValueError, IndexError):
                    continue

                try:
                    quote = Quote(
                        symbol=symbol,
                        timestamp=timestamp,
                        bid_price=float(parts[2]),
                        bid_size=int(parts[3]),
                        ask_price=float(parts[4]),
                        ask_size=int(parts[5]),
                        bid_exchange=Exchange.from_code(parts[6]) if len(parts) > 6 else None,
                        ask_exchange=Exchange.from_code(parts[7]) if len(parts) > 7 else None,
                    )
                    yield quote
                except (ValueError, IndexError):
                    continue


class TickAggregator:
    """Aggregates tick data into custom bars with microstructure metrics."""

    def __init__(self):
        self._trades: Dict[str, List[Trade]] = {}
        self._quotes: Dict[str, List[Quote]] = {}
        self._last_quote: Dict[str, Quote] = {}

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the aggregator."""
        if trade.symbol not in self._trades:
            self._trades[trade.symbol] = []
        self._trades[trade.symbol].append(trade)

    def add_quote(self, quote: Quote) -> None:
        """Add a quote to the aggregator."""
        if quote.symbol not in self._quotes:
            self._quotes[quote.symbol] = []
        self._quotes[quote.symbol].append(quote)
        self._last_quote[quote.symbol] = quote

    def classify_trade(
        self,
        trade: Trade,
        quote: Optional[Quote] = None,
    ) -> str:
        """
        Classify trade as buy or sell using tick rule.

        Returns:
            'buy', 'sell', or 'unknown'
        """
        if quote is None:
            quote = self._last_quote.get(trade.symbol)

        if quote is None:
            return "unknown"

        mid = quote.mid_price
        if trade.price > mid:
            return "buy"
        elif trade.price < mid:
            return "sell"
        else:
            return "unknown"

    def aggregate_to_bar(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Optional[AggregatedBar]:
        """Aggregate trades and quotes into a single bar."""
        trades = [
            t for t in self._trades.get(symbol, [])
            if start <= t.timestamp < end
        ]

        if not trades:
            return None

        quotes = [
            q for q in self._quotes.get(symbol, [])
            if start <= q.timestamp < end
        ]

        # OHLCV from trades
        prices = [t.price for t in trades]
        sizes = [t.size for t in trades]

        total_volume = sum(sizes)
        vwap = sum(t.price * t.size for t in trades) / total_volume if total_volume > 0 else 0

        # Buy/sell classification
        buy_volume = 0
        sell_volume = 0
        for trade in trades:
            classification = self.classify_trade(trade)
            if classification == "buy":
                buy_volume += trade.size
            elif classification == "sell":
                sell_volume += trade.size

        # Spread from quotes
        avg_spread_bps = 0.0
        if quotes:
            spreads = [q.spread_bps for q in quotes if q.spread_bps > 0]
            if spreads:
                avg_spread_bps = np.mean(spreads)

        return AggregatedBar(
            symbol=symbol,
            timestamp=start,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_volume,
            vwap=vwap,
            trade_count=len(trades),
            avg_spread_bps=avg_spread_bps,
            avg_trade_size=total_volume / len(trades) if trades else 0,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
        )

    def aggregate_time_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size_seconds: int = 60,
    ) -> List[AggregatedBar]:
        """Aggregate into fixed time bars."""
        bars = []
        current = start

        while current < end:
            bar_end = current + timedelta(seconds=bar_size_seconds)
            bar = self.aggregate_to_bar(symbol, current, bar_end)
            if bar:
                bars.append(bar)
            current = bar_end

        return bars

    def aggregate_volume_bars(
        self,
        symbol: str,
        volume_threshold: int,
    ) -> List[AggregatedBar]:
        """Aggregate into volume bars (fixed volume per bar)."""
        trades = sorted(
            self._trades.get(symbol, []),
            key=lambda t: t.timestamp
        )

        if not trades:
            return []

        bars = []
        current_trades = []
        current_volume = 0

        for trade in trades:
            current_trades.append(trade)
            current_volume += trade.size

            if current_volume >= volume_threshold:
                # Create bar
                prices = [t.price for t in current_trades]
                total_vol = sum(t.size for t in current_trades)
                vwap = sum(t.price * t.size for t in current_trades) / total_vol

                bar = AggregatedBar(
                    symbol=symbol,
                    timestamp=current_trades[0].timestamp,
                    open=prices[0],
                    high=max(prices),
                    low=min(prices),
                    close=prices[-1],
                    volume=total_vol,
                    vwap=vwap,
                    trade_count=len(current_trades),
                )
                bars.append(bar)

                current_trades = []
                current_volume = 0

        return bars

    def aggregate_dollar_bars(
        self,
        symbol: str,
        dollar_threshold: float,
    ) -> List[AggregatedBar]:
        """Aggregate into dollar bars (fixed notional per bar)."""
        trades = sorted(
            self._trades.get(symbol, []),
            key=lambda t: t.timestamp
        )

        if not trades:
            return []

        bars = []
        current_trades = []
        current_dollars = 0.0

        for trade in trades:
            current_trades.append(trade)
            current_dollars += trade.notional

            if current_dollars >= dollar_threshold:
                prices = [t.price for t in current_trades]
                total_vol = sum(t.size for t in current_trades)
                vwap = sum(t.price * t.size for t in current_trades) / total_vol

                bar = AggregatedBar(
                    symbol=symbol,
                    timestamp=current_trades[0].timestamp,
                    open=prices[0],
                    high=max(prices),
                    low=min(prices),
                    close=prices[-1],
                    volume=total_vol,
                    vwap=vwap,
                    trade_count=len(current_trades),
                )
                bars.append(bar)

                current_trades = []
                current_dollars = 0.0

        return bars

    def get_microstructure_snapshot(
        self,
        symbol: str,
        as_of: datetime,
        lookback_seconds: int = 60,
    ) -> Optional[MicrostructureSnapshot]:
        """Get point-in-time microstructure state."""
        lookback_start = as_of - timedelta(seconds=lookback_seconds)

        # Recent trades
        trades = [
            t for t in self._trades.get(symbol, [])
            if lookback_start <= t.timestamp <= as_of
        ]

        # Most recent quote
        quotes = [
            q for q in self._quotes.get(symbol, [])
            if q.timestamp <= as_of
        ]

        if not trades or not quotes:
            return None

        last_quote = max(quotes, key=lambda q: q.timestamp)

        # Calculate metrics
        prices = [t.price for t in trades]
        sizes = [t.size for t in trades]
        total_volume = sum(sizes)
        vwap = sum(t.price * t.size for t in trades) / total_volume if total_volume > 0 else 0

        # Buy/sell classification
        buy_volume = 0
        sell_volume = 0
        for trade in trades:
            classification = self.classify_trade(trade, last_quote)
            if classification == "buy":
                buy_volume += trade.size
            elif classification == "sell":
                sell_volume += trade.size

        # Realized volatility
        if len(prices) > 1:
            returns = np.diff(np.log(prices))
            realized_vol = np.std(returns) * np.sqrt(len(returns) * 60 / lookback_seconds)
        else:
            realized_vol = 0.0

        # Order flow imbalance
        ofi = 0.0
        if buy_volume + sell_volume > 0:
            ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume)

        return MicrostructureSnapshot(
            symbol=symbol,
            timestamp=as_of,
            bid=last_quote.bid_price,
            ask=last_quote.ask_price,
            last_trade=trades[-1].price,
            vwap=vwap,
            bid_size=last_quote.bid_size,
            ask_size=last_quote.ask_size,
            spread_bps=last_quote.spread_bps,
            volume_1min=total_volume,
            trade_count_1min=len(trades),
            avg_trade_size_1min=total_volume / len(trades) if trades else 0,
            buy_volume_1min=buy_volume,
            sell_volume_1min=sell_volume,
            order_flow_imbalance=ofi,
            realized_vol_1min=realized_vol,
            high_1min=max(prices),
            low_1min=min(prices),
        )

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear stored tick data."""
        if symbol:
            self._trades.pop(symbol, None)
            self._quotes.pop(symbol, None)
            self._last_quote.pop(symbol, None)
        else:
            self._trades.clear()
            self._quotes.clear()
            self._last_quote.clear()


class TickDataManager:
    """
    High-level manager for tick data operations.

    Provides unified interface for:
    - Historical tick data retrieval
    - Real-time streaming
    - Custom bar aggregation
    - Microstructure analytics
    """

    def __init__(
        self,
        provider: Optional[TickDataProvider] = None,
        cache_size_mb: int = 100,
    ):
        self.provider = provider
        self.aggregator = TickAggregator()
        self.cache_size_mb = cache_size_mb
        self._cache: Dict[str, List[Trade]] = {}
        self._streaming = False
        self._stream_callbacks: List[Callable] = []

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> List[Trade]:
        """Fetch historical trades."""
        if not self.provider:
            raise ValueError("No tick data provider configured")

        trades = await self.provider.get_trades(symbol, start, end)

        if use_cache:
            self._cache[f"{symbol}:trades:{start}:{end}"] = trades

        # Add to aggregator
        for trade in trades:
            self.aggregator.add_trade(trade)

        return trades

    async def get_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> List[Quote]:
        """Fetch historical quotes."""
        if not self.provider:
            raise ValueError("No tick data provider configured")

        quotes = await self.provider.get_quotes(symbol, start, end)

        if use_cache:
            self._cache[f"{symbol}:quotes:{start}:{end}"] = quotes

        # Add to aggregator
        for quote in quotes:
            self.aggregator.add_quote(quote)

        return quotes

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_type: str = "time",
        bar_size: int = 60,
    ) -> List[AggregatedBar]:
        """
        Get custom bars from tick data.

        Args:
            symbol: Symbol to fetch
            start: Start time
            end: End time
            bar_type: 'time', 'volume', or 'dollar'
            bar_size: Size of bar (seconds for time, shares for volume, dollars for dollar)
        """
        # Fetch underlying ticks
        await self.get_trades(symbol, start, end)
        await self.get_quotes(symbol, start, end)

        if bar_type == "time":
            return self.aggregator.aggregate_time_bars(symbol, start, end, bar_size)
        elif bar_type == "volume":
            return self.aggregator.aggregate_volume_bars(symbol, bar_size)
        elif bar_type == "dollar":
            return self.aggregator.aggregate_dollar_bars(symbol, float(bar_size))
        else:
            raise ValueError(f"Unknown bar type: {bar_type}")

    async def start_streaming(
        self,
        symbols: List[str],
        callback: Optional[Callable[[Trade], None]] = None,
    ) -> None:
        """Start real-time tick streaming."""
        if not self.provider:
            raise ValueError("No tick data provider configured")

        self._streaming = True

        def on_trade(trade: Trade):
            self.aggregator.add_trade(trade)
            if callback:
                callback(trade)
            for cb in self._stream_callbacks:
                cb(trade)

        await self.provider.stream_trades(symbols, on_trade)

    def stop_streaming(self) -> None:
        """Stop real-time streaming."""
        self._streaming = False

    def get_microstructure(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
    ) -> Optional[MicrostructureSnapshot]:
        """Get current microstructure state."""
        return self.aggregator.get_microstructure_snapshot(
            symbol,
            as_of or datetime.now(),
        )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self.aggregator.clear()


def create_tick_manager(
    provider_type: str = "polygon",
    api_key: Optional[str] = None,
    **kwargs,
) -> TickDataManager:
    """
    Factory function to create TickDataManager.

    Args:
        provider_type: 'polygon' or 'taq'
        api_key: API key for provider (if applicable)
        **kwargs: Additional provider configuration
    """
    provider = None

    if provider_type == "polygon" and api_key:
        provider = PolygonTickProvider(api_key=api_key, **kwargs)
    # TAQ doesn't need a provider (file-based)

    cache_size = kwargs.get("cache_size_mb", 100)
    return TickDataManager(provider=provider, cache_size_mb=cache_size)


def print_microstructure_report(snapshot: MicrostructureSnapshot) -> None:
    """Print formatted microstructure report."""
    print("\n" + "=" * 60)
    print(f"MICROSTRUCTURE SNAPSHOT: {snapshot.symbol}")
    print("=" * 60)

    print(f"\nTimestamp: {snapshot.timestamp}")
    print(f"\n{'Price Levels':-^40}")
    print(f"  Bid: ${snapshot.bid:.2f} x {snapshot.bid_size}")
    print(f"  Ask: ${snapshot.ask:.2f} x {snapshot.ask_size}")
    print(f"  Spread: {snapshot.spread_bps:.1f} bps")
    print(f"  Last Trade: ${snapshot.last_trade:.2f}")
    print(f"  VWAP: ${snapshot.vwap:.2f}")

    print(f"\n{'Activity (1 min)':-^40}")
    print(f"  Volume: {snapshot.volume_1min:,}")
    print(f"  Trades: {snapshot.trade_count_1min}")
    print(f"  Avg Trade Size: {snapshot.avg_trade_size_1min:.0f}")

    print(f"\n{'Order Flow':-^40}")
    print(f"  Buy Volume: {snapshot.buy_volume_1min:,}")
    print(f"  Sell Volume: {snapshot.sell_volume_1min:,}")
    ofi_pct = snapshot.order_flow_imbalance * 100
    ofi_label = "BUY" if ofi_pct > 0 else "SELL"
    print(f"  Imbalance: {abs(ofi_pct):.1f}% {ofi_label}")

    print(f"\n{'Volatility':-^40}")
    print(f"  Realized Vol: {snapshot.realized_vol_1min * 100:.2f}%")
    print(f"  Range: ${snapshot.low_1min:.2f} - ${snapshot.high_1min:.2f}")

    print("=" * 60 + "\n")
