"""
Order Flow Analyzer - Dark pool and unusual options activity detection.

This module analyzes order flow data to generate trading signals from:
- Dark pool prints (large block trades)
- Unusual options activity (sweeps, high volume vs OI)
- Put/call ratios
- Smart money flow indicators

Usage:
    from data.order_flow_analyzer import OrderFlowAnalyzer

    analyzer = OrderFlowAnalyzer(broker=alpaca_broker)
    await analyzer.initialize()
    signal = await analyzer.fetch_signal("AAPL")
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.alt_data_types import (
    AltDataSource,
    AlternativeSignal,
    OrderFlowSignal,
)
from data.alternative_data_provider import AlternativeDataProvider

logger = logging.getLogger(__name__)


@dataclass
class OptionsActivityData:
    """Options activity metrics for a symbol."""

    symbol: str
    timestamp: datetime

    # Volume metrics
    total_call_volume: int = 0
    total_put_volume: int = 0
    call_volume_vs_avg: float = 1.0  # Multiple of average
    put_volume_vs_avg: float = 1.0

    # Open interest
    call_oi: int = 0
    put_oi: int = 0

    # Sweeps (aggressive orders hitting multiple exchanges)
    call_sweep_count: int = 0
    put_sweep_count: int = 0
    call_sweep_premium: float = 0.0
    put_sweep_premium: float = 0.0

    # Large trades
    large_call_trades: int = 0  # > $100k premium
    large_put_trades: int = 0

    @property
    def put_call_ratio(self) -> float:
        """Put/call volume ratio."""
        if self.total_call_volume == 0:
            return 1.0
        return self.total_put_volume / self.total_call_volume

    @property
    def is_unusual_volume(self) -> bool:
        """Whether options volume is unusually high."""
        return self.call_volume_vs_avg > 2.0 or self.put_volume_vs_avg > 2.0

    @property
    def net_sweep_premium(self) -> float:
        """Net sweep premium (calls - puts)."""
        return self.call_sweep_premium - self.put_sweep_premium


@dataclass
class DarkPoolData:
    """Dark pool trading activity for a symbol."""

    symbol: str
    timestamp: datetime

    # Volume metrics
    dark_pool_volume: int = 0
    total_volume: int = 0
    dark_pool_pct: float = 0.0

    # Block trades (> 10,000 shares or $200k)
    block_trades: List[Dict[str, Any]] = None
    block_count: int = 0
    avg_block_size: float = 0.0

    # Price levels
    block_vwap: float = 0.0
    blocks_above_ask: int = 0
    blocks_below_bid: int = 0

    def __post_init__(self):
        if self.block_trades is None:
            self.block_trades = []

        if self.block_trades:
            self.block_count = len(self.block_trades)
            sizes = [b.get("size", 0) for b in self.block_trades]
            self.avg_block_size = sum(sizes) / len(sizes) if sizes else 0

    @property
    def block_trade_imbalance(self) -> float:
        """Imbalance of blocks above ask vs below bid (-1 to +1)."""
        total = self.blocks_above_ask + self.blocks_below_bid
        if total == 0:
            return 0.0
        return (self.blocks_above_ask - self.blocks_below_bid) / total


class OrderFlowAnalyzer(AlternativeDataProvider):
    """
    Analyzes order flow from dark pools and options markets.

    Generates signals based on:
    - Dark pool print analysis
    - Unusual options activity
    - Put/call ratios
    - Sweep detection
    """

    # Thresholds for signal generation
    UNUSUAL_VOLUME_THRESHOLD = 2.0  # 2x average
    HIGH_PUT_CALL_RATIO = 1.5  # Bearish threshold
    LOW_PUT_CALL_RATIO = 0.7  # Bullish threshold
    SWEEP_THRESHOLD = 5  # Minimum sweeps to be significant

    def __init__(
        self,
        broker=None,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize order flow analyzer.

        Args:
            broker: Broker instance for fetching options data (optional)
            cache_ttl_seconds: Cache TTL
        """
        super().__init__(
            source=AltDataSource.OPTIONS_FLOW,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        self._broker = broker

        # Historical averages for anomaly detection
        self._volume_history: Dict[str, List[Tuple[datetime, int, int]]] = {}  # call_vol, put_vol

    async def initialize(self) -> bool:
        """Initialize the order flow analyzer."""
        self._initialized = True
        logger.info("Order flow analyzer initialized")
        return True

    async def fetch_signal(self, symbol: str) -> Optional[OrderFlowSignal]:
        """
        Fetch order flow signal for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            OrderFlowSignal with options and dark pool data
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch options data
            options_data = await self._fetch_options_data(symbol)

            # Fetch dark pool data
            dark_pool_data = await self._fetch_dark_pool_data(symbol)

            # Generate combined signal
            return self._generate_signal(symbol, options_data, dark_pool_data)

        except Exception as e:
            logger.error(f"Error fetching order flow for {symbol}: {e}")
            self._error_count += 1
            return None

    async def _fetch_options_data(self, symbol: str) -> OptionsActivityData:
        """Fetch options activity data."""
        data = OptionsActivityData(symbol=symbol, timestamp=datetime.now())

        if self._broker is not None:
            try:
                # Try to fetch real options data from broker
                options_chain = await self._fetch_options_chain(symbol)
                if options_chain:
                    data = self._process_options_chain(symbol, options_chain)
            except Exception as e:
                logger.debug(f"Could not fetch real options data for {symbol}: {e}")

        # If no real data, generate synthetic data for development
        if data.total_call_volume == 0 and data.total_put_volume == 0:
            data = self._generate_synthetic_options_data(symbol)

        return data

    async def _fetch_options_chain(self, symbol: str) -> Optional[Dict]:
        """Fetch options chain from broker."""
        if self._broker is None:
            return None

        try:
            # Check if broker has options capabilities
            if hasattr(self._broker, "get_option_chain"):
                return await self._broker.get_option_chain(symbol)
        except Exception as e:
            logger.debug(f"Broker options fetch failed: {e}")

        return None

    def _process_options_chain(self, symbol: str, chain: Dict) -> OptionsActivityData:
        """Process raw options chain into activity data."""
        data = OptionsActivityData(symbol=symbol, timestamp=datetime.now())

        try:
            # Extract calls and puts
            calls = chain.get("calls", [])
            puts = chain.get("puts", [])

            # Sum volumes
            data.total_call_volume = sum(c.get("volume", 0) for c in calls)
            data.total_put_volume = sum(p.get("volume", 0) for p in puts)

            # Sum open interest
            data.call_oi = sum(c.get("open_interest", 0) for c in calls)
            data.put_oi = sum(p.get("open_interest", 0) for p in puts)

            # Calculate volume vs average
            avg_call, avg_put = self._get_average_volumes(symbol)
            if avg_call > 0:
                data.call_volume_vs_avg = data.total_call_volume / avg_call
            if avg_put > 0:
                data.put_volume_vs_avg = data.total_put_volume / avg_put

            # Update history
            self._update_volume_history(symbol, data.total_call_volume, data.total_put_volume)

        except Exception as e:
            logger.error(f"Error processing options chain: {e}")

        return data

    def _generate_synthetic_options_data(self, symbol: str) -> OptionsActivityData:
        """Generate synthetic options data for development/testing."""
        import random

        # Base volumes with some randomness
        base_call_vol = random.randint(5000, 50000)
        base_put_vol = random.randint(3000, 40000)

        # Occasionally generate unusual activity
        is_unusual = random.random() < 0.2

        if is_unusual:
            # Spike one side
            if random.random() < 0.5:
                base_call_vol *= random.uniform(2, 5)
            else:
                base_put_vol *= random.uniform(2, 5)

        data = OptionsActivityData(
            symbol=symbol,
            timestamp=datetime.now(),
            total_call_volume=int(base_call_vol),
            total_put_volume=int(base_put_vol),
            call_volume_vs_avg=1.0 + random.uniform(-0.3, 1.0),
            put_volume_vs_avg=1.0 + random.uniform(-0.3, 1.0),
            call_oi=int(base_call_vol * random.uniform(3, 10)),
            put_oi=int(base_put_vol * random.uniform(3, 10)),
            call_sweep_count=random.randint(0, 10) if is_unusual else random.randint(0, 3),
            put_sweep_count=random.randint(0, 10) if is_unusual else random.randint(0, 3),
            call_sweep_premium=random.uniform(0, 500000) if is_unusual else random.uniform(0, 100000),
            put_sweep_premium=random.uniform(0, 500000) if is_unusual else random.uniform(0, 100000),
            large_call_trades=random.randint(0, 5),
            large_put_trades=random.randint(0, 5),
        )

        return data

    async def _fetch_dark_pool_data(self, symbol: str) -> DarkPoolData:
        """Fetch dark pool activity data."""
        # Dark pool data typically requires paid data providers
        # (e.g., FINRA ADF data, FlowAlgo, etc.)
        # For now, generate synthetic data for development

        return self._generate_synthetic_dark_pool_data(symbol)

    def _generate_synthetic_dark_pool_data(self, symbol: str) -> DarkPoolData:
        """Generate synthetic dark pool data for development/testing."""
        import random

        total_volume = random.randint(1000000, 50000000)
        dark_pool_pct = random.uniform(0.3, 0.5)  # 30-50% is typical
        dark_pool_volume = int(total_volume * dark_pool_pct)

        # Generate some block trades
        block_count = random.randint(2, 15)
        blocks = []

        for _ in range(block_count):
            size = random.randint(10000, 100000)
            price = random.uniform(100, 500)
            side = "above_ask" if random.random() < 0.5 else "below_bid"

            blocks.append(
                {
                    "size": size,
                    "price": price,
                    "side": side,
                    "value": size * price,
                }
            )

        blocks_above = sum(1 for b in blocks if b["side"] == "above_ask")
        blocks_below = sum(1 for b in blocks if b["side"] == "below_bid")

        return DarkPoolData(
            symbol=symbol,
            timestamp=datetime.now(),
            dark_pool_volume=dark_pool_volume,
            total_volume=total_volume,
            dark_pool_pct=dark_pool_pct,
            block_trades=blocks,
            blocks_above_ask=blocks_above,
            blocks_below_bid=blocks_below,
        )

    def _generate_signal(
        self,
        symbol: str,
        options_data: OptionsActivityData,
        dark_pool_data: DarkPoolData,
    ) -> OrderFlowSignal:
        """Generate combined order flow signal."""
        # Calculate signal components

        # 1. Put/call ratio signal (-1 to +1)
        pc_ratio = options_data.put_call_ratio
        if pc_ratio > self.HIGH_PUT_CALL_RATIO:
            pc_signal = -min(1.0, (pc_ratio - self.HIGH_PUT_CALL_RATIO) / 0.5)
        elif pc_ratio < self.LOW_PUT_CALL_RATIO:
            pc_signal = min(1.0, (self.LOW_PUT_CALL_RATIO - pc_ratio) / 0.3)
        else:
            pc_signal = 0.0

        # 2. Sweep imbalance signal
        sweep_total = options_data.call_sweep_count + options_data.put_sweep_count
        if sweep_total >= self.SWEEP_THRESHOLD:
            sweep_imbalance = (
                options_data.call_sweep_premium - options_data.put_sweep_premium
            )
            max_premium = max(options_data.call_sweep_premium, options_data.put_sweep_premium, 1)
            sweep_signal = sweep_imbalance / max_premium
        else:
            sweep_signal = 0.0

        # 3. Dark pool block imbalance
        dp_signal = dark_pool_data.block_trade_imbalance

        # Combine signals (weighted average)
        combined_signal = 0.4 * pc_signal + 0.4 * sweep_signal + 0.2 * dp_signal
        combined_signal = max(-1.0, min(1.0, combined_signal))

        # Calculate confidence
        confidence = 0.3  # Base confidence

        # Increase confidence for unusual activity
        if options_data.is_unusual_volume:
            confidence += 0.2

        if sweep_total >= self.SWEEP_THRESHOLD:
            confidence += 0.2

        if dark_pool_data.block_count >= 5:
            confidence += 0.1

        confidence = min(0.9, confidence)

        return OrderFlowSignal(
            symbol=symbol,
            source=AltDataSource.OPTIONS_FLOW,
            timestamp=datetime.now(),
            signal_value=combined_signal,
            confidence=confidence,
            # Dark pool metrics
            dark_pool_volume=float(dark_pool_data.dark_pool_volume),
            dark_pool_pct_of_total=dark_pool_data.dark_pool_pct,
            block_trade_count=dark_pool_data.block_count,
            avg_block_size=dark_pool_data.avg_block_size,
            # Options metrics
            call_volume=options_data.total_call_volume,
            put_volume=options_data.total_put_volume,
            put_call_ratio=pc_ratio,
            unusual_options_activity=options_data.is_unusual_volume,
            sweep_count=sweep_total,
            raw_data={
                "call_sweep_premium": options_data.call_sweep_premium,
                "put_sweep_premium": options_data.put_sweep_premium,
                "call_volume_vs_avg": options_data.call_volume_vs_avg,
                "put_volume_vs_avg": options_data.put_volume_vs_avg,
            },
        )

    def _get_average_volumes(self, symbol: str) -> Tuple[float, float]:
        """Get average call and put volumes from history."""
        if symbol not in self._volume_history or not self._volume_history[symbol]:
            return 10000.0, 10000.0  # Default averages

        history = self._volume_history[symbol]
        avg_call = sum(c for _, c, _ in history) / len(history)
        avg_put = sum(p for _, _, p in history) / len(history)

        return avg_call, avg_put

    def _update_volume_history(self, symbol: str, call_vol: int, put_vol: int):
        """Update volume history for anomaly detection."""
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []

        now = datetime.now()
        self._volume_history[symbol].append((now, call_vol, put_vol))

        # Keep only last 20 data points
        if len(self._volume_history[symbol]) > 20:
            self._volume_history[symbol] = self._volume_history[symbol][-20:]


class DarkPoolProvider(AlternativeDataProvider):
    """
    Dedicated dark pool data provider.

    Focuses specifically on dark pool print analysis.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        super().__init__(
            source=AltDataSource.DARK_POOL,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def initialize(self) -> bool:
        """Initialize dark pool provider."""
        # Dark pool data typically requires:
        # - FINRA ADF (Alternative Display Facility) data
        # - Paid providers like FlowAlgo, Unusual Whales, etc.
        logger.info(
            "Dark pool provider initialized. "
            "Note: Premium data sources recommended for production use."
        )
        self._initialized = True
        return True

    async def fetch_signal(self, symbol: str) -> Optional[AlternativeSignal]:
        """Fetch dark pool signal."""
        if not self._initialized:
            await self.initialize()

        # Generate synthetic data for now
        # In production, integrate with FINRA ADF or premium providers
        import random

        dark_pool_pct = random.uniform(0.3, 0.5)
        block_imbalance = random.uniform(-0.3, 0.3)

        # Signal: High dark pool % with bullish block imbalance = bullish
        signal_value = block_imbalance * (1 + (dark_pool_pct - 0.4) * 2)
        signal_value = max(-1.0, min(1.0, signal_value))

        return AlternativeSignal(
            symbol=symbol,
            source=AltDataSource.DARK_POOL,
            timestamp=datetime.now(),
            signal_value=signal_value,
            confidence=0.4,  # Lower confidence for synthetic data
            metadata={
                "dark_pool_pct": dark_pool_pct,
                "block_imbalance": block_imbalance,
                "note": "Synthetic data - integrate premium provider for production",
            },
        )


# Factory function
def create_order_flow_provider(
    source: AltDataSource,
    broker=None,
    **kwargs,
) -> Optional[AlternativeDataProvider]:
    """
    Create an order flow data provider.

    Args:
        source: The order flow data source
        broker: Optional broker for real data
        **kwargs: Provider-specific configuration

    Returns:
        Configured provider instance
    """
    if source == AltDataSource.OPTIONS_FLOW:
        return OrderFlowAnalyzer(broker=broker, **kwargs)
    elif source == AltDataSource.DARK_POOL:
        return DarkPoolProvider(**kwargs)
    else:
        logger.warning(f"Unknown order flow source: {source}")
        return None
