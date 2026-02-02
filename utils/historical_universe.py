"""
Historical Universe - Survivorship Bias Correction.

CRITICAL FOR BACKTEST VALIDITY: Without this, backtests suffer from survivorship bias.

Survivorship bias occurs when backtests only include stocks that exist TODAY,
ignoring stocks that were delisted, went bankrupt, or were acquired. This creates
artificially inflated returns because:
1. Failed companies are excluded from historical analysis
2. Successful companies that still exist are overrepresented
3. Symbol changes (FB→META) can cause data gaps

This module provides:
1. IPO date tracking - Don't trade before a stock went public
2. Delisting tracking - Don't trade after a stock was delisted
3. Symbol change tracking - Handle corporate rebranding (FB→META, GOOG→GOOGL)
4. Universe filtering - Get tradeable symbols for any historical date

Usage:
    from utils.historical_universe import HistoricalUniverse

    universe = HistoricalUniverse()
    await universe.initialize()

    # Get symbols that were tradeable on a specific date
    tradeable = universe.get_tradeable_symbols(date(2023, 6, 15), candidate_symbols)

    # Check if a specific symbol was tradeable
    if universe.was_tradeable("AAPL", date(2023, 6, 15)):
        # Include in backtest
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a symbol's trading history."""

    symbol: str
    ipo_date: Optional[date] = None  # First tradeable date
    delisting_date: Optional[date] = None  # Last tradeable date (None = still active)
    previous_symbols: List[str] = field(default_factory=list)  # Previous tickers
    next_symbol: Optional[str] = None  # Symbol it was renamed to
    company_name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    is_active: bool = True

    def was_tradeable_on(self, check_date: date) -> bool:
        """Check if this symbol was tradeable on a specific date."""
        # Check IPO date
        if self.ipo_date and check_date < self.ipo_date:
            return False

        # Check delisting date
        if self.delisting_date and check_date > self.delisting_date:
            return False

        return True


# Known symbol changes (historical record)
# Format: (old_symbol, new_symbol, change_date)
KNOWN_SYMBOL_CHANGES: List[Tuple[str, str, date]] = [
    # Major tech rebrands
    ("FB", "META", date(2022, 6, 9)),
    ("GOOG", "GOOGL", date(2014, 4, 3)),  # Stock split created GOOGL
    ("TWTR", "X", date(2023, 7, 24)),  # Twitter to X (delisted)
    # Acquisitions/mergers
    ("ATVI", "MSFT", date(2023, 10, 13)),  # Activision acquired by Microsoft
    ("VMW", "AVGO", date(2023, 11, 22)),  # VMware acquired by Broadcom
    # Corporate actions
    ("GE", "GEV", date(2024, 4, 2)),  # GE Vernova spinoff
    ("AMZN", "AMZN", date(2022, 6, 6)),  # 20:1 stock split (same symbol)
    ("GOOGL", "GOOGL", date(2022, 7, 18)),  # 20:1 stock split (same symbol)
    ("TSLA", "TSLA", date(2022, 8, 25)),  # 3:1 stock split (same symbol)
]

# Known IPO dates for major stocks
KNOWN_IPO_DATES: Dict[str, date] = {
    # FAANG+
    "AAPL": date(1980, 12, 12),
    "AMZN": date(1997, 5, 15),
    "GOOGL": date(2004, 8, 19),
    "GOOG": date(2004, 8, 19),
    "META": date(2012, 5, 18),
    "FB": date(2012, 5, 18),
    "NFLX": date(2002, 5, 23),
    "MSFT": date(1986, 3, 13),
    "NVDA": date(1999, 1, 22),
    "TSLA": date(2010, 6, 29),
    # Other major tech
    "AMD": date(1972, 9, 27),
    "INTC": date(1971, 10, 13),
    "CRM": date(2004, 6, 23),
    "ADBE": date(1986, 8, 20),
    "PYPL": date(2015, 7, 20),
    "SQ": date(2015, 11, 19),  # Now BLOCK
    "SHOP": date(2015, 5, 21),
    "SNOW": date(2020, 9, 16),
    "PLTR": date(2020, 9, 30),
    "COIN": date(2021, 4, 14),
    "HOOD": date(2021, 7, 29),
    "RIVN": date(2021, 11, 10),
    "LCID": date(2021, 7, 26),
    # Recent IPOs (2023-2024)
    "ARM": date(2023, 9, 14),
    "BIRK": date(2023, 10, 11),
    "RDDT": date(2024, 3, 21),
    # Finance
    "JPM": date(1969, 3, 5),
    "BAC": date(1973, 1, 2),
    "GS": date(1999, 5, 4),
    "MS": date(1986, 3, 21),
    "V": date(2008, 3, 19),
    "MA": date(2006, 5, 25),
    # Healthcare
    "JNJ": date(1944, 9, 25),
    "UNH": date(1984, 10, 17),
    "PFE": date(1942, 6, 22),
    "MRNA": date(2018, 12, 7),
    "BNTX": date(2019, 10, 10),
    # Consumer
    "WMT": date(1972, 8, 25),
    "COST": date(1985, 12, 5),
    "HD": date(1981, 9, 22),
    "NKE": date(1980, 12, 2),
    "SBUX": date(1992, 6, 26),
    "MCD": date(1965, 4, 21),
    "DIS": date(1957, 11, 12),
    # Energy
    "XOM": date(1920, 1, 1),  # Approximate
    "CVX": date(1921, 1, 1),  # Approximate
    # ETFs (inception dates)
    "SPY": date(1993, 1, 22),
    "QQQ": date(1999, 3, 10),
    "IWM": date(2000, 5, 22),
    "DIA": date(1998, 1, 14),
    "VTI": date(2001, 5, 24),
    "VOO": date(2010, 9, 7),
}

# Known delisted stocks (for survivorship bias correction)
KNOWN_DELISTINGS: Dict[str, date] = {
    # Bankruptcies
    "WEWORK": date(2023, 11, 6),  # WeWork bankruptcy
    "BBBYQ": date(2023, 9, 29),  # Bed Bath & Beyond bankruptcy
    "SVB": date(2023, 3, 13),  # Silicon Valley Bank collapse
    "FRC": date(2023, 5, 1),  # First Republic Bank
    "SBNY": date(2023, 3, 12),  # Signature Bank
    # Acquisitions (stock no longer trades)
    "TWTR": date(2022, 10, 27),  # Twitter acquired by Elon Musk
    "ATVI": date(2023, 10, 13),  # Activision acquired by Microsoft
    "VMW": date(2023, 11, 22),  # VMware acquired by Broadcom
    "SAVE": date(2024, 1, 16),  # Spirit Airlines merger blocked, later delisted
}


class HistoricalUniverse:
    """
    Manages historical stock universe for survivorship-bias-free backtesting.

    This class tracks which symbols were tradeable on any given historical date,
    accounting for IPOs, delistings, and symbol changes.
    """

    def __init__(self, broker=None):
        """
        Initialize the historical universe.

        Args:
            broker: Optional broker instance for fetching symbol data
        """
        self.broker = broker
        self._symbols: Dict[str, SymbolInfo] = {}
        self._symbol_changes: List[Tuple[str, str, date]] = KNOWN_SYMBOL_CHANGES.copy()
        self._initialized = False

    async def initialize(self):
        """
        Initialize the universe with known data.

        Call this before using the universe for filtering.
        """
        # Load known IPO dates
        for symbol, ipo_date in KNOWN_IPO_DATES.items():
            self._symbols[symbol] = SymbolInfo(
                symbol=symbol,
                ipo_date=ipo_date,
                is_active=symbol not in KNOWN_DELISTINGS,
            )

        # Update with known delistings
        for symbol, delisting_date in KNOWN_DELISTINGS.items():
            if symbol in self._symbols:
                self._symbols[symbol].delisting_date = delisting_date
                self._symbols[symbol].is_active = False
            else:
                self._symbols[symbol] = SymbolInfo(
                    symbol=symbol,
                    delisting_date=delisting_date,
                    is_active=False,
                )

        # Process symbol changes
        for old_symbol, new_symbol, change_date in self._symbol_changes:
            if old_symbol in self._symbols:
                self._symbols[old_symbol].next_symbol = new_symbol
                if self._symbols[old_symbol].delisting_date is None:
                    self._symbols[old_symbol].delisting_date = change_date
            if new_symbol in self._symbols:
                if old_symbol not in self._symbols[new_symbol].previous_symbols:
                    self._symbols[new_symbol].previous_symbols.append(old_symbol)

        # Try to fetch additional data from broker if available
        if self.broker:
            await self._fetch_broker_data()

        self._initialized = True
        logger.info(
            f"HistoricalUniverse initialized: {len(self._symbols)} symbols tracked"
        )

    async def _fetch_broker_data(self):
        """Fetch additional symbol data from broker."""
        try:
            if hasattr(self.broker, "get_assets"):
                assets = await self.broker.get_assets()
                for asset in assets:
                    symbol = asset.symbol
                    if symbol not in self._symbols:
                        self._symbols[symbol] = SymbolInfo(
                            symbol=symbol,
                            exchange=getattr(asset, "exchange", None),
                            is_active=getattr(asset, "tradable", True),
                        )
        except Exception as e:
            logger.warning(f"Could not fetch broker data: {e}")

    def was_tradeable(self, symbol: str, check_date: date) -> bool:
        """
        Check if a symbol was tradeable on a specific date.

        Args:
            symbol: Stock symbol
            check_date: Date to check

        Returns:
            True if symbol was tradeable on that date
        """
        if symbol not in self._symbols:
            # Unknown symbol - assume it was tradeable (conservative)
            # Log warning for manual review
            logger.debug(
                f"Unknown symbol {symbol} - assuming tradeable on {check_date}"
            )
            return True

        return self._symbols[symbol].was_tradeable_on(check_date)

    def get_tradeable_symbols(
        self, check_date: date, candidates: List[str]
    ) -> List[str]:
        """
        Filter a list of symbols to only those tradeable on a specific date.

        Args:
            check_date: Date to check
            candidates: List of candidate symbols

        Returns:
            List of symbols that were tradeable on that date
        """
        tradeable = []
        for symbol in candidates:
            if self.was_tradeable(symbol, check_date):
                tradeable.append(symbol)
            else:
                logger.debug(
                    f"Excluding {symbol} from backtest on {check_date} "
                    f"(not tradeable)"
                )
        return tradeable

    def get_symbol_on_date(self, symbol: str, check_date: date) -> Optional[str]:
        """
        Get the correct symbol to use for a given date.

        Handles symbol changes (e.g., FB→META). If you're looking for META
        data on 2021-01-01, this will return "FB" since that was the symbol then.

        Args:
            symbol: Current/target symbol
            check_date: Historical date

        Returns:
            Symbol that was valid on that date, or None if not tradeable
        """
        if symbol not in self._symbols:
            return symbol if self.was_tradeable(symbol, check_date) else None

        info = self._symbols[symbol]

        # If the symbol was valid on that date, use it
        if info.was_tradeable_on(check_date):
            return symbol

        # Check if we should use a previous symbol
        for prev_symbol in info.previous_symbols:
            if prev_symbol in self._symbols:
                prev_info = self._symbols[prev_symbol]
                if prev_info.was_tradeable_on(check_date):
                    return prev_symbol

        # Check if we should use the next symbol
        if info.next_symbol and info.next_symbol in self._symbols:
            next_info = self._symbols[info.next_symbol]
            if next_info.was_tradeable_on(check_date):
                return info.next_symbol

        return None

    def get_ipo_date(self, symbol: str) -> Optional[date]:
        """Get the IPO date for a symbol."""
        if symbol in self._symbols:
            return self._symbols[symbol].ipo_date
        return None

    def get_delisting_date(self, symbol: str) -> Optional[date]:
        """Get the delisting date for a symbol (None if still active)."""
        if symbol in self._symbols:
            return self._symbols[symbol].delisting_date
        return None

    def is_active(self, symbol: str) -> bool:
        """Check if a symbol is currently active (not delisted)."""
        if symbol in self._symbols:
            return self._symbols[symbol].is_active
        return True  # Assume active if unknown

    def add_symbol_info(
        self,
        symbol: str,
        ipo_date: Optional[date] = None,
        delisting_date: Optional[date] = None,
        is_active: bool = True,
    ):
        """
        Add or update symbol information.

        Args:
            symbol: Stock symbol
            ipo_date: IPO/first trading date
            delisting_date: Delisting date (None if still active)
            is_active: Whether symbol is currently tradeable
        """
        if symbol in self._symbols:
            if ipo_date:
                self._symbols[symbol].ipo_date = ipo_date
            if delisting_date:
                self._symbols[symbol].delisting_date = delisting_date
            self._symbols[symbol].is_active = is_active
        else:
            self._symbols[symbol] = SymbolInfo(
                symbol=symbol,
                ipo_date=ipo_date,
                delisting_date=delisting_date,
                is_active=is_active,
            )

    def add_symbol_change(
        self, old_symbol: str, new_symbol: str, change_date: date
    ):
        """
        Record a symbol change (rebrand, merger, etc.).

        Args:
            old_symbol: Previous symbol
            new_symbol: New symbol
            change_date: Date of change
        """
        self._symbol_changes.append((old_symbol, new_symbol, change_date))

        # Update symbol info
        if old_symbol in self._symbols:
            self._symbols[old_symbol].next_symbol = new_symbol
            if self._symbols[old_symbol].delisting_date is None:
                self._symbols[old_symbol].delisting_date = change_date

        if new_symbol in self._symbols:
            if old_symbol not in self._symbols[new_symbol].previous_symbols:
                self._symbols[new_symbol].previous_symbols.append(old_symbol)

    def get_statistics(self) -> Dict:
        """Get universe statistics."""
        active_count = sum(1 for s in self._symbols.values() if s.is_active)
        with_ipo = sum(1 for s in self._symbols.values() if s.ipo_date)
        with_delisting = sum(1 for s in self._symbols.values() if s.delisting_date)

        return {
            "total_symbols": len(self._symbols),
            "active_symbols": active_count,
            "delisted_symbols": len(self._symbols) - active_count,
            "symbols_with_ipo_date": with_ipo,
            "symbols_with_delisting_date": with_delisting,
            "symbol_changes_tracked": len(self._symbol_changes),
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"HistoricalUniverse("
            f"symbols={stats['total_symbols']}, "
            f"active={stats['active_symbols']}, "
            f"delisted={stats['delisted_symbols']})"
        )


async def create_universe_for_backtest(
    broker=None,
    symbols: Optional[List[str]] = None,
) -> HistoricalUniverse:
    """
    Create and initialize a historical universe for backtesting.

    Convenience function that creates, initializes, and optionally
    pre-populates the universe with specific symbols.

    Args:
        broker: Optional broker instance
        symbols: Optional list of symbols to ensure are tracked

    Returns:
        Initialized HistoricalUniverse
    """
    universe = HistoricalUniverse(broker)
    await universe.initialize()

    # Add any additional symbols
    if symbols:
        for symbol in symbols:
            if symbol not in universe._symbols:
                universe.add_symbol_info(symbol)

    return universe
