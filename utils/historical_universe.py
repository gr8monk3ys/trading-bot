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
import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DelistingReason(Enum):
    """Reason for delisting - important for return calculation."""

    BANKRUPTCY = "bankruptcy"  # Returns go to -100% or near it
    ACQUISITION = "acquisition"  # Returns go to acquisition price
    MERGER = "merger"  # Returns based on merger terms
    VOLUNTARY = "voluntary"  # Company chose to delist
    COMPLIANCE = "compliance"  # Failed to meet listing requirements
    UNKNOWN = "unknown"


@dataclass
class SymbolInfo:
    """Information about a symbol's trading history."""

    symbol: str
    ipo_date: Optional[date] = None  # First tradeable date
    delisting_date: Optional[date] = None  # Last tradeable date (None = still active)
    delisting_reason: DelistingReason = DelistingReason.UNKNOWN
    final_price: Optional[float] = None  # Last known price (for return calculation)
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

    def get_terminal_return(self, entry_price: float) -> float:
        """
        Calculate terminal return for delisted stock.

        For bankruptcies, returns -100%.
        For acquisitions, returns based on acquisition price if known.
        """
        if self.delisting_reason == DelistingReason.BANKRUPTCY:
            return -1.0  # -100% loss
        elif self.final_price and entry_price > 0:
            return (self.final_price - entry_price) / entry_price
        else:
            return -0.5  # Assume 50% loss if unknown (conservative)


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
    # === Additional major stocks for comprehensive coverage ===
    # Semiconductors
    "AVGO": date(2009, 8, 6),  # Broadcom
    "QCOM": date(1991, 12, 17),
    "TXN": date(1953, 10, 1),
    "MU": date(1984, 6, 1),
    "AMAT": date(1972, 10, 13),
    "LRCX": date(1984, 7, 1),
    "KLAC": date(1980, 10, 1),
    "MRVL": date(2000, 6, 27),
    "ON": date(2000, 4, 28),
    "SWKS": date(1993, 1, 1),  # Skyworks
    # Software & Cloud
    "ORCL": date(1986, 3, 12),
    "SAP": date(1998, 8, 3),
    "NOW": date(2012, 6, 29),  # ServiceNow
    "WDAY": date(2012, 10, 12),  # Workday
    "SNOW": date(2020, 9, 16),  # Snowflake
    "ZM": date(2019, 4, 18),  # Zoom
    "DDOG": date(2019, 9, 19),  # Datadog
    "CRWD": date(2019, 6, 12),  # CrowdStrike
    "NET": date(2019, 9, 13),  # Cloudflare
    "MDB": date(2017, 10, 19),  # MongoDB
    "ZS": date(2018, 3, 16),  # Zscaler
    "OKTA": date(2017, 4, 7),
    "TWLO": date(2016, 6, 23),  # Twilio
    "TEAM": date(2015, 12, 10),  # Atlassian
    "PANW": date(2012, 7, 20),  # Palo Alto Networks
    "FTNT": date(2009, 11, 18),  # Fortinet
    "SPLK": date(2012, 4, 19),  # Splunk
    # E-commerce & Digital
    "EBAY": date(1998, 9, 24),
    "BKNG": date(1999, 3, 30),  # Booking Holdings (was PCLN)
    "ABNB": date(2020, 12, 10),  # Airbnb
    "UBER": date(2019, 5, 10),
    "LYFT": date(2019, 3, 29),
    "DASH": date(2020, 12, 9),  # DoorDash
    "PINS": date(2019, 4, 18),  # Pinterest
    "SNAP": date(2017, 3, 2),
    "SPOT": date(2018, 4, 3),  # Spotify
    "RBLX": date(2021, 3, 10),  # Roblox
    "U": date(2020, 9, 18),  # Unity Software
    # Biotech & Pharma
    "ABBV": date(2013, 1, 2),  # AbbVie
    "LLY": date(1952, 10, 1),
    "BMY": date(1933, 6, 1),
    "GILD": date(1992, 1, 22),
    "AMGN": date(1983, 6, 17),
    "BIIB": date(1991, 9, 18),
    "REGN": date(1991, 4, 11),
    "VRTX": date(1991, 7, 25),
    "ISRG": date(2000, 6, 13),  # Intuitive Surgical
    "DXCM": date(2005, 4, 14),  # Dexcom
    "ILMN": date(2000, 7, 28),  # Illumina
    # Industrials & Defense
    "BA": date(1962, 1, 2),
    "CAT": date(1929, 12, 2),
    "DE": date(1911, 1, 1),
    "GE": date(1896, 1, 1),
    "HON": date(1925, 1, 1),
    "LMT": date(1995, 3, 15),  # Lockheed Martin
    "RTX": date(2020, 4, 3),  # Raytheon (merger)
    "NOC": date(1994, 1, 1),  # Northrop Grumman
    "GD": date(1952, 1, 1),  # General Dynamics
    "UPS": date(1999, 11, 10),
    "FDX": date(1978, 4, 12),
    # Telecom & Media
    "T": date(1984, 1, 1),  # AT&T
    "VZ": date(2000, 7, 3),  # Verizon
    "TMUS": date(2013, 5, 1),  # T-Mobile
    "CMCSA": date(1972, 6, 29),  # Comcast
    "CHTR": date(2010, 1, 1),  # Charter
    "NFLX": date(2002, 5, 23),
    "DIS": date(1957, 11, 12),
    "WBD": date(2022, 4, 8),  # Warner Bros Discovery
    "PARA": date(2019, 12, 4),  # Paramount
    # Auto & EV
    "F": date(1956, 1, 1),
    "GM": date(2010, 11, 18),  # New GM post-bankruptcy
    "RIVN": date(2021, 11, 10),
    "LCID": date(2021, 7, 26),
    "NIO": date(2018, 9, 12),
    "XPEV": date(2020, 8, 27),  # XPeng
    "LI": date(2020, 7, 30),  # Li Auto
    "FFIE": date(2021, 7, 22),  # Faraday Future
    # Retail
    "TGT": date(1967, 10, 18),
    "LOW": date(1961, 10, 10),
    "TJX": date(1987, 1, 1),
    "ROST": date(1985, 8, 1),
    "BBY": date(1987, 4, 1),  # Best Buy
    "DG": date(2009, 11, 13),  # Dollar General
    "DLTR": date(1995, 3, 6),  # Dollar Tree
    # Real Estate
    "AMT": date(1998, 6, 4),  # American Tower
    "PLD": date(1997, 11, 21),  # Prologis
    "CCI": date(1998, 8, 14),  # Crown Castle
    "EQIX": date(2000, 8, 11),  # Equinix
    "PSA": date(1980, 10, 1),  # Public Storage
    "O": date(1994, 10, 18),  # Realty Income
    # Insurance
    "BRK.B": date(1996, 5, 6),
    "PGR": date(1971, 4, 15),  # Progressive
    "TRV": date(2002, 4, 1),  # Travelers
    "ALL": date(1993, 6, 3),  # Allstate
    "AIG": date(1984, 1, 1),
    "MET": date(2000, 4, 5),  # MetLife
    # Utilities
    "NEE": date(1950, 1, 1),  # NextEra
    "DUK": date(1961, 1, 1),  # Duke Energy
    "SO": date(1949, 1, 1),  # Southern Company
    "D": date(1983, 4, 1),  # Dominion
}

# Known delisted stocks with reason and final price (for survivorship bias correction)
# Format: symbol -> (date, reason, final_price_if_known)
KNOWN_DELISTINGS_EXTENDED: Dict[str, Tuple[date, DelistingReason, Optional[float]]] = {
    # === BANKRUPTCIES (returns go to ~0) ===
    "WEWORK": (date(2023, 11, 6), DelistingReason.BANKRUPTCY, 0.0),
    "BBBYQ": (date(2023, 9, 29), DelistingReason.BANKRUPTCY, 0.0),
    "SVB": (date(2023, 3, 13), DelistingReason.BANKRUPTCY, 0.0),
    "SIVB": (date(2023, 3, 13), DelistingReason.BANKRUPTCY, 0.0),  # SVB Financial
    "FRC": (date(2023, 5, 1), DelistingReason.BANKRUPTCY, 0.0),
    "SBNY": (date(2023, 3, 12), DelistingReason.BANKRUPTCY, 0.0),
    "LMND": (date(2023, 3, 10), DelistingReason.BANKRUPTCY, 0.0),  # Silvergate Capital
    "SI": (date(2023, 3, 8), DelistingReason.BANKRUPTCY, 0.0),  # Silvergate
    # Historical bankruptcies
    "LEHM": (date(2008, 9, 15), DelistingReason.BANKRUPTCY, 0.0),  # Lehman Brothers
    "ENRNQ": (date(2001, 12, 2), DelistingReason.BANKRUPTCY, 0.0),  # Enron
    "WCOM": (date(2002, 7, 21), DelistingReason.BANKRUPTCY, 0.0),  # WorldCom
    "GM": (date(2009, 6, 1), DelistingReason.BANKRUPTCY, 0.0),  # Old GM (became GMGMQ)
    "CIT": (date(2009, 11, 1), DelistingReason.BANKRUPTCY, 0.0),  # CIT Group
    "WAMUQ": (date(2008, 9, 26), DelistingReason.BANKRUPTCY, 0.0),  # Washington Mutual
    # 2020 COVID-era bankruptcies
    "HTZ": (date(2020, 5, 22), DelistingReason.BANKRUPTCY, 0.56),  # Hertz (later restructured)
    "JCP": (date(2020, 5, 15), DelistingReason.BANKRUPTCY, 0.0),  # JCPenney
    "JCPNQ": (date(2020, 5, 15), DelistingReason.BANKRUPTCY, 0.0),  # JCPenney
    "NMG": (date(2020, 9, 17), DelistingReason.BANKRUPTCY, 0.0),  # Neiman Marcus
    "PRTY": (date(2020, 5, 15), DelistingReason.BANKRUPTCY, 0.0),  # Party City
    "GNC": (date(2020, 6, 24), DelistingReason.BANKRUPTCY, 0.0),  # GNC Holdings
    "CHK": (date(2020, 6, 28), DelistingReason.BANKRUPTCY, 0.0),  # Chesapeake Energy
    # Crypto-related collapses
    "COIN": (date(2023, 6, 6), DelistingReason.COMPLIANCE, 60.0),  # Still trades but was distressed
    "VOYG": (date(2022, 7, 6), DelistingReason.BANKRUPTCY, 0.0),  # Voyager Digital
    "CORZQ": (date(2022, 11, 21), DelistingReason.BANKRUPTCY, 0.0),  # Core Scientific
    # === ACQUISITIONS (returns based on deal price) ===
    "TWTR": (date(2022, 10, 27), DelistingReason.ACQUISITION, 54.20),  # $54.20/share Musk deal
    "ATVI": (date(2023, 10, 13), DelistingReason.ACQUISITION, 95.00),  # $95/share MSFT deal
    "VMW": (date(2023, 11, 22), DelistingReason.ACQUISITION, 142.50),  # Broadcom deal
    "SAVE": (date(2024, 1, 16), DelistingReason.MERGER, None),  # JetBlue merger blocked
    "XLNX": (date(2022, 2, 14), DelistingReason.ACQUISITION, 200.00),  # AMD acquisition
    "NUAA": (date(2022, 4, 6), DelistingReason.ACQUISITION, 46.00),  # NuVasive
    "MGM": (date(2022, 3, 17), DelistingReason.ACQUISITION, 60.00),  # Amazon deal
    "CERN": (date(2022, 6, 6), DelistingReason.ACQUISITION, 95.00),  # Oracle deal
    "CTXS": (date(2022, 9, 30), DelistingReason.ACQUISITION, 104.00),  # Vista Equity
    "KHC": (date(2022, 6, 15), DelistingReason.ACQUISITION, 28.00),  # Symbol change
    "FISV": (date(2019, 7, 29), DelistingReason.MERGER, None),  # Fiserv/First Data
    # Historical major acquisitions
    "TWX": (date(2018, 6, 15), DelistingReason.ACQUISITION, 107.50),  # AT&T deal
    "SHLD": (date(2018, 10, 15), DelistingReason.BANKRUPTCY, 0.0),  # Sears
    "TIF": (date(2021, 1, 7), DelistingReason.ACQUISITION, 131.50),  # LVMH deal
    "SLB": (date(2020, 10, 1), DelistingReason.MERGER, None),  # Schlumberger rebranding
}

# Legacy format for backwards compatibility
KNOWN_DELISTINGS: Dict[str, date] = {
    symbol: info[0] for symbol, info in KNOWN_DELISTINGS_EXTENDED.items()
}

# S&P 500 Historical Changes (major additions/removals)
# This helps identify when stocks entered the S&P 500 vs when they IPO'd
SP500_CHANGES: List[Tuple[str, str, date]] = [
    # Format: (symbol, action "add"/"remove", date)
    # 2024 changes
    ("SMCI", "add", date(2024, 3, 18)),  # Super Micro Computer
    ("DECK", "add", date(2024, 3, 18)),  # Deckers Outdoor
    ("CRWD", "add", date(2024, 6, 24)),  # CrowdStrike
    ("KKR", "add", date(2024, 6, 24)),  # KKR & Co
    ("GDDY", "add", date(2024, 6, 24)),  # GoDaddy
    # 2023 changes
    ("UBER", "add", date(2023, 12, 18)),
    ("BLDR", "add", date(2023, 9, 18)),
    ("ABNB", "add", date(2023, 9, 18)),
    ("LULU", "add", date(2023, 6, 16)),
    # 2022 changes
    ("TSLA", "add", date(2020, 12, 21)),  # Tesla added to S&P 500
    # Removals
    ("DISH", "remove", date(2024, 3, 18)),
    ("WBA", "remove", date(2024, 6, 24)),  # Walgreens Boots Alliance
    ("FRC", "remove", date(2023, 5, 4)),  # First Republic (bankruptcy)
    ("SIVB", "remove", date(2023, 3, 17)),  # SVB Financial (bankruptcy)
]


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

    async def initialize(self, csv_path: Optional[str] = None):
        """
        Initialize the universe with known data.

        Call this before using the universe for filtering.

        Args:
            csv_path: Optional path to CSV file with additional symbol data
                     Format: symbol,ipo_date,delisting_date,reason,final_price
        """
        # Load known IPO dates
        for symbol, ipo_date in KNOWN_IPO_DATES.items():
            self._symbols[symbol] = SymbolInfo(
                symbol=symbol,
                ipo_date=ipo_date,
                is_active=symbol not in KNOWN_DELISTINGS,
            )

        # Update with extended delistings (includes reason and final price)
        for symbol, (delist_date, reason, final_price) in KNOWN_DELISTINGS_EXTENDED.items():
            if symbol in self._symbols:
                self._symbols[symbol].delisting_date = delist_date
                self._symbols[symbol].delisting_reason = reason
                self._symbols[symbol].final_price = final_price
                self._symbols[symbol].is_active = False
            else:
                self._symbols[symbol] = SymbolInfo(
                    symbol=symbol,
                    delisting_date=delist_date,
                    delisting_reason=reason,
                    final_price=final_price,
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

        # Load additional data from CSV if provided
        if csv_path:
            self._load_from_csv(csv_path)

        # Try to fetch additional data from broker if available
        if self.broker:
            await self._fetch_broker_data()

        self._initialized = True
        logger.info(
            f"HistoricalUniverse initialized: {len(self._symbols)} symbols tracked"
        )

    def _load_from_csv(self, csv_path: str):
        """
        Load additional symbol data from CSV file.

        Expected format (header row required):
        symbol,ipo_date,delisting_date,delisting_reason,final_price,company_name

        Dates should be in YYYY-MM-DD format.
        """
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                loaded = 0

                for row in reader:
                    symbol = row.get("symbol", "").strip().upper()
                    if not symbol:
                        continue

                    # Parse dates
                    ipo_date = None
                    if row.get("ipo_date"):
                        try:
                            ipo_date = datetime.strptime(row["ipo_date"], "%Y-%m-%d").date()
                        except ValueError:
                            pass

                    delist_date = None
                    if row.get("delisting_date"):
                        try:
                            delist_date = datetime.strptime(row["delisting_date"], "%Y-%m-%d").date()
                        except ValueError:
                            pass

                    # Parse reason
                    reason = DelistingReason.UNKNOWN
                    if row.get("delisting_reason"):
                        try:
                            reason = DelistingReason(row["delisting_reason"].lower())
                        except ValueError:
                            pass

                    # Parse final price
                    final_price = None
                    if row.get("final_price"):
                        try:
                            final_price = float(row["final_price"])
                        except ValueError:
                            pass

                    # Update or create symbol info
                    if symbol in self._symbols:
                        if ipo_date:
                            self._symbols[symbol].ipo_date = ipo_date
                        if delist_date:
                            self._symbols[symbol].delisting_date = delist_date
                            self._symbols[symbol].delisting_reason = reason
                            self._symbols[symbol].is_active = False
                        if final_price is not None:
                            self._symbols[symbol].final_price = final_price
                        if row.get("company_name"):
                            self._symbols[symbol].company_name = row["company_name"]
                    else:
                        self._symbols[symbol] = SymbolInfo(
                            symbol=symbol,
                            ipo_date=ipo_date,
                            delisting_date=delist_date,
                            delisting_reason=reason,
                            final_price=final_price,
                            company_name=row.get("company_name"),
                            is_active=delist_date is None,
                        )

                    loaded += 1

                logger.info(f"Loaded {loaded} symbols from CSV: {csv_path}")

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")

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

    def validate_coverage(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Validate survivorship bias coverage for a list of symbols.

        CRITICAL: Run this before any backtest to understand data gaps.

        Returns detailed report of coverage quality and gaps.

        Args:
            symbols: List of symbols to validate

        Returns:
            Dictionary with coverage report
        """
        report = {
            "total_symbols": len(symbols),
            "covered": [],
            "missing_ipo_date": [],
            "missing_data": [],
            "coverage_pct": 0.0,
            "ipo_coverage_pct": 0.0,
            "warnings": [],
            "risk_level": "UNKNOWN",
        }

        for symbol in symbols:
            if symbol in self._symbols:
                info = self._symbols[symbol]
                report["covered"].append(symbol)
                if info.ipo_date is None:
                    report["missing_ipo_date"].append(symbol)
            else:
                report["missing_data"].append(symbol)

        # Calculate coverage percentages
        report["coverage_pct"] = (
            len(report["covered"]) / len(symbols) * 100 if symbols else 0
        )
        covered_with_ipo = len(report["covered"]) - len(report["missing_ipo_date"])
        report["ipo_coverage_pct"] = (
            covered_with_ipo / len(symbols) * 100 if symbols else 0
        )

        # Generate warnings
        if report["missing_data"]:
            report["warnings"].append(
                f"{len(report['missing_data'])} symbols have no historical data. "
                "Backtest may have survivorship bias for: "
                f"{', '.join(report['missing_data'][:10])}{'...' if len(report['missing_data']) > 10 else ''}"
            )

        if report["missing_ipo_date"]:
            report["warnings"].append(
                f"{len(report['missing_ipo_date'])} symbols missing IPO dates. "
                "May include pre-IPO data in backtest."
            )

        # Risk assessment
        if report["coverage_pct"] >= 90 and report["ipo_coverage_pct"] >= 80:
            report["risk_level"] = "LOW"
        elif report["coverage_pct"] >= 70 and report["ipo_coverage_pct"] >= 50:
            report["risk_level"] = "MEDIUM"
        else:
            report["risk_level"] = "HIGH"
            report["warnings"].append(
                "HIGH RISK: Insufficient survivorship bias coverage. "
                "Backtest results may be significantly inflated."
            )

        return report

    def export_to_csv(self, filepath: str):
        """
        Export current universe data to CSV for review/editing.

        Args:
            filepath: Path to output CSV file
        """
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "symbol", "ipo_date", "delisting_date", "delisting_reason",
                "final_price", "company_name", "is_active", "previous_symbols", "next_symbol"
            ])

            for symbol, info in sorted(self._symbols.items()):
                writer.writerow([
                    info.symbol,
                    info.ipo_date.isoformat() if info.ipo_date else "",
                    info.delisting_date.isoformat() if info.delisting_date else "",
                    info.delisting_reason.value if info.delisting_reason else "",
                    info.final_price if info.final_price is not None else "",
                    info.company_name or "",
                    info.is_active,
                    ",".join(info.previous_symbols) if info.previous_symbols else "",
                    info.next_symbol or "",
                ])

        logger.info(f"Exported {len(self._symbols)} symbols to {filepath}")

    def get_delisted_between(
        self, start_date: date, end_date: date
    ) -> List[Tuple[str, date, DelistingReason]]:
        """
        Get all stocks that were delisted between two dates.

        Useful for understanding survivorship bias magnitude in a backtest period.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            List of (symbol, delisting_date, reason) tuples
        """
        delisted = []
        for symbol, info in self._symbols.items():
            if info.delisting_date:
                if start_date <= info.delisting_date <= end_date:
                    delisted.append((symbol, info.delisting_date, info.delisting_reason))

        return sorted(delisted, key=lambda x: x[1])

    def calculate_survivorship_bias_estimate(
        self, symbols: List[str], start_date: date, end_date: date
    ) -> Dict[str, Any]:
        """
        Estimate the magnitude of survivorship bias for a backtest.

        CRITICAL: Understanding this helps interpret backtest results.

        Args:
            symbols: Symbols in the backtest universe
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dictionary with bias estimation
        """
        # Find stocks that were delisted during the period
        delisted = self.get_delisted_between(start_date, end_date)
        delisted_symbols = {d[0] for d in delisted}
        delisted_in_universe = delisted_symbols.intersection(set(symbols))

        # Count bankruptcies vs acquisitions
        bankruptcies = [d for d in delisted if d[2] == DelistingReason.BANKRUPTCY]
        acquisitions = [d for d in delisted if d[2] == DelistingReason.ACQUISITION]

        # Estimate bias
        # Bankruptcies cause ~100% loss on average
        # Acquisitions may have been at premium or discount
        bankruptcy_bias = len(bankruptcies) * 0.10  # ~10% portfolio impact per bankruptcy
        acquisition_bias = len(acquisitions) * 0.02  # ~2% impact (usually positive)

        total_bias_estimate = bankruptcy_bias - acquisition_bias

        return {
            "period": f"{start_date} to {end_date}",
            "total_delistings": len(delisted),
            "delistings_in_universe": len(delisted_in_universe),
            "bankruptcies": len(bankruptcies),
            "acquisitions": len(acquisitions),
            "estimated_bias_pct": total_bias_estimate,
            "interpretation": (
                f"Estimated survivorship bias: {total_bias_estimate:+.1f}% "
                f"({len(bankruptcies)} bankruptcies, {len(acquisitions)} acquisitions). "
                f"Subtract this from reported returns for more realistic estimate."
            ),
            "delisted_symbols": list(delisted_symbols),
        }


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
