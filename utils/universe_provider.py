"""
Universe Provider - Provides tradeable stock universes

Fetches and caches stock universe data from various sources:
- S&P 500 constituents
- NASDAQ-100 constituents
- Sector-specific stocks
- Custom universe files

Usage:
    from utils.universe_provider import UniverseProvider

    provider = UniverseProvider()
    sp500 = await provider.get_sp500_constituents()
    nasdaq100 = await provider.get_nasdaq100_constituents()
    tech_stocks = await provider.get_sector_stocks("Technology")
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import aiohttp

logger = logging.getLogger(__name__)

# Cache directory for universe data
CACHE_DIR = Path(__file__).parent.parent / ".cache" / "universes"


class UniverseProvider:
    """
    Provides tradeable stock universes with caching.

    Fetches universe data from Wikipedia or other sources,
    caches locally to avoid repeated API calls.
    """

    # Wikipedia URLs for fetching constituents
    UNIVERSE_URLS = {
        "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    }

    # Hardcoded fallback for S&P 500 (major constituents, updated periodically)
    SP500_FALLBACK = [
        # Top 50 by market cap (as of 2024)
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK.B",
        "UNH",
        "LLY",
        "JPM",
        "XOM",
        "JNJ",
        "V",
        "PG",
        "MA",
        "AVGO",
        "HD",
        "CVX",
        "MRK",
        "ABBV",
        "COST",
        "PEP",
        "KO",
        "ADBE",
        "WMT",
        "MCD",
        "CSCO",
        "CRM",
        "BAC",
        "ACN",
        "TMO",
        "LIN",
        "ABT",
        "NFLX",
        "DHR",
        "ORCL",
        "AMD",
        "DIS",
        "PFE",
        "CMCSA",
        "VZ",
        "INTC",
        "PM",
        "WFC",
        "TXN",
        "COP",
        "NEE",
        "QCOM",
        # 51-100
        "INTU",
        "RTX",
        "AMAT",
        "BMY",
        "UPS",
        "HON",
        "CAT",
        "UNP",
        "LOW",
        "SPGI",
        "IBM",
        "T",
        "GE",
        "MS",
        "ELV",
        "BA",
        "SBUX",
        "DE",
        "GS",
        "BLK",
        "PLD",
        "NOW",
        "ISRG",
        "MDLZ",
        "GILD",
        "ADP",
        "AMT",
        "ADI",
        "BKNG",
        "SYK",
        "LMT",
        "TJX",
        "VRTX",
        "CI",
        "CVS",
        "REGN",
        "MMC",
        "CB",
        "C",
        "MO",
        "ZTS",
        "SO",
        "LRCX",
        "SCHW",
        "BSX",
        "PGR",
        "BDX",
        "FI",
        "TMUS",
        "DUK",
        # 101-150
        "SLB",
        "EOG",
        "ETN",
        "ICE",
        "PYPL",
        "CME",
        "AON",
        "MU",
        "SNPS",
        "NOC",
        "FCX",
        "KLAC",
        "EQIX",
        "APD",
        "SHW",
        "HUM",
        "ITW",
        "CL",
        "ORLY",
        "WM",
        "MCK",
        "CMG",
        "CDNS",
        "CSX",
        "PNC",
        "USB",
        "TGT",
        "EMR",
        "PSA",
        "NSC",
        "GD",
        "MAR",
        "MSI",
        "AJG",
        "APH",
        "MCO",
        "EW",
        "PH",
        "ATVI",
        "ECL",
        "TT",
        "ROP",
        "DXCM",
        "AZO",
        "OXY",
        "CTAS",
        "CARR",
        "MET",
        "TDG",
        "PCAR",
        # 151-200
        "WELL",
        "HLT",
        "AIG",
        "AFL",
        "HES",
        "COF",
        "STZ",
        "TRV",
        "PSX",
        "SRE",
        "ROST",
        "MCHP",
        "NEM",
        "AMP",
        "D",
        "PAYX",
        "KMB",
        "MSCI",
        "FTNT",
        "DHI",
        "IDXX",
        "TEL",
        "ODFL",
        "JCI",
        "FAST",
        "MNST",
        "CTSH",
        "SPG",
        "AEP",
        "O",
        "ALL",
        "CMI",
        "CPRT",
        "EA",
        "KHC",
        "YUM",
        "BIIB",
        "KEYS",
        "BK",
        "PRU",
        "AME",
        "VRSK",
        "HCA",
        "HSY",
        "A",
        "OTIS",
        "SYY",
        "DD",
        "PPG",
        "GWW",
        # 201-250
        "DOW",
        "FANG",
        "GEHC",
        "EXC",
        "CNC",
        "ILMN",
        "EL",
        "LHX",
        "XEL",
        "WMB",
        "GIS",
        "CBRE",
        "KDP",
        "MTD",
        "HIG",
        "IQV",
        "VLO",
        "NDAQ",
        "NUE",
        "KR",
        "RMD",
        "EIX",
        "ADM",
        "VICI",
        "WEC",
        "CTVA",
        "FTV",
        "ROK",
        "IT",
        "AWK",
        "ACGL",
        "ED",
        "OKE",
        "DG",
        "HAL",
        "PCG",
        "ANSS",
        "WST",
        "ON",
        "DLTR",
        "DLR",
        "VMC",
        "BKR",
        "DVN",
        "GLW",
        "CDW",
        "HPQ",
        "PWR",
        "EFX",
        "APTV",
    ]

    # NASDAQ-100 fallback
    NASDAQ100_FALLBACK = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AVGO",
        "COST",
        "ADBE",
        "PEP",
        "CSCO",
        "NFLX",
        "AMD",
        "CMCSA",
        "INTC",
        "TMUS",
        "TXN",
        "QCOM",
        "INTU",
        "AMAT",
        "HON",
        "SBUX",
        "BKNG",
        "ISRG",
        "MDLZ",
        "GILD",
        "ADP",
        "ADI",
        "VRTX",
        "REGN",
        "LRCX",
        "PYPL",
        "MU",
        "SNPS",
        "KLAC",
        "CDNS",
        "CSX",
        "ORLY",
        "PANW",
        "MAR",
        "MELI",
        "FTNT",
        "MNST",
        "AZN",
        "DXCM",
        "CTAS",
        "PCAR",
        "PDD",
        "CPRT",
        "MCHP",
        "KHC",
        "ODFL",
        "EXC",
        "KDP",
        "PAYX",
        "CRWD",
        "AEP",
        "WDAY",
        "IDXX",
        "CEG",
        "ROST",
        "FAST",
        "MRNA",
        "ABNB",
        "ON",
        "CTSH",
        "EA",
        "BIIB",
        "ILMN",
        "BKR",
        "VRSK",
        "XEL",
        "CSGP",
        "GFS",
        "GEHC",
        "FANG",
        "ANSS",
        "DLTR",
        "WBD",
        "ZS",
        "TTD",
        "DDOG",
        "TEAM",
        "ALGN",
        "ENPH",
        "SIRI",
        "WBA",
        "LCID",
        "RIVN",
        "ZM",
        "OKTA",
        "SPLK",
        "DOCU",
        "NXPI",
        "EBAY",
        "JD",
        "ASML",
        "LULU",
    ]

    # DOW 30 fallback
    DOW30_FALLBACK = [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CRM",
        "CSCO",
        "CVX",
        "DIS",
        "DOW",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PG",
        "TRV",
        "UNH",
        "V",
        "VZ",
        "WBA",
        "WMT",
    ]

    # Sector ETF mappings
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Materials": "XLB",
        "Industrials": "XLI",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication Services": "XLC",
    }

    def __init__(self, cache_hours: int = 24):
        """
        Initialize universe provider.

        Args:
            cache_hours: Hours to cache universe data (default 24)
        """
        self.cache_hours = cache_hours
        self._memory_cache: Dict[str, tuple] = {}  # universe -> (data, timestamp)

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def get_sp500_constituents(self, use_fallback: bool = True) -> List[str]:
        """
        Get current S&P 500 constituents.

        Args:
            use_fallback: Use hardcoded fallback if fetch fails

        Returns:
            List of S&P 500 ticker symbols
        """
        return await self._get_universe("sp500", self.SP500_FALLBACK, use_fallback)

    async def get_nasdaq100_constituents(self, use_fallback: bool = True) -> List[str]:
        """
        Get current NASDAQ-100 constituents.

        Returns:
            List of NASDAQ-100 ticker symbols
        """
        return await self._get_universe("nasdaq100", self.NASDAQ100_FALLBACK, use_fallback)

    async def get_dow30_constituents(self, use_fallback: bool = True) -> List[str]:
        """
        Get current Dow Jones Industrial Average constituents.

        Returns:
            List of DOW 30 ticker symbols
        """
        return await self._get_universe("dow30", self.DOW30_FALLBACK, use_fallback)

    async def get_sector_stocks(self, sector: str) -> List[str]:
        """
        Get stocks in a specific sector.

        Uses yfinance to identify sector membership.

        Args:
            sector: Sector name (e.g., "Technology", "Healthcare")

        Returns:
            List of ticker symbols in the sector
        """
        # Get full S&P 500 universe
        sp500 = await self.get_sp500_constituents()

        # Get sector for each stock
        sector_stocks = []

        try:
            import yfinance as yf

            for symbol in sp500:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    stock_sector = info.get("sector", "")

                    if stock_sector.lower() == sector.lower():
                        sector_stocks.append(symbol)
                except Exception:
                    continue

        except ImportError:
            logger.warning("yfinance not available for sector filtering")
            return []

        logger.info(f"Found {len(sector_stocks)} stocks in {sector} sector")
        return sector_stocks

    async def get_sector_map(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get sector mapping for a list of symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict of symbol -> sector
        """
        sector_map = {}

        try:
            import yfinance as yf

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    sector = info.get("sector", "Unknown")
                    sector_map[symbol] = sector
                except Exception:
                    sector_map[symbol] = "Unknown"

        except ImportError:
            logger.warning("yfinance not available for sector mapping")
            return dict.fromkeys(symbols, "Unknown")

        return sector_map

    async def get_custom_universe(self, file_path: str) -> List[str]:
        """
        Load custom universe from a file.

        File should contain one ticker symbol per line.

        Args:
            file_path: Path to universe file

        Returns:
            List of ticker symbols
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Universe file not found: {file_path}")
            return []

        with open(path, "r") as f:
            symbols = [line.strip().upper() for line in f if line.strip()]

        # Filter out comments and empty lines
        symbols = [s for s in symbols if s and not s.startswith("#")]

        logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
        return symbols

    async def _get_universe(
        self,
        universe_name: str,
        fallback: List[str],
        use_fallback: bool = True,
    ) -> List[str]:
        """
        Internal method to get universe data with caching.
        """
        # Check memory cache first
        if universe_name in self._memory_cache:
            data, timestamp = self._memory_cache[universe_name]
            if (datetime.now() - timestamp).total_seconds() < self.cache_hours * 3600:
                logger.debug(f"Using memory-cached {universe_name} ({len(data)} symbols)")
                return data

        # Check file cache
        cache_file = CACHE_DIR / f"{universe_name}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                    cached_time = datetime.fromisoformat(cached["timestamp"])

                    if (datetime.now() - cached_time).total_seconds() < self.cache_hours * 3600:
                        symbols = cached["symbols"]
                        self._memory_cache[universe_name] = (symbols, cached_time)
                        logger.debug(f"Using file-cached {universe_name} ({len(symbols)} symbols)")
                        return symbols
            except Exception as e:
                logger.debug(f"Cache read error: {e}")

        # Try to fetch from Wikipedia
        symbols = await self._fetch_from_wikipedia(universe_name)

        if symbols:
            # Cache the result
            self._cache_universe(universe_name, symbols)
            return symbols

        # Use fallback if fetch failed
        if use_fallback:
            logger.warning(f"Using fallback data for {universe_name} ({len(fallback)} symbols)")
            self._memory_cache[universe_name] = (fallback, datetime.now())
            return fallback

        return []

    async def _fetch_from_wikipedia(self, universe_name: str) -> List[str]:
        """
        Fetch universe constituents from Wikipedia.
        """
        url = self.UNIVERSE_URLS.get(universe_name)
        if not url:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.warning(f"Wikipedia fetch failed: HTTP {response.status}")
                        return []

                    html = await response.text()

                    # Parse ticker symbols from HTML
                    symbols = self._parse_wikipedia_table(html, universe_name)

                    if symbols:
                        logger.info(f"Fetched {len(symbols)} symbols for {universe_name}")
                        return symbols

        except asyncio.TimeoutError:
            logger.warning(f"Wikipedia fetch timed out for {universe_name}")
        except Exception as e:
            logger.warning(f"Wikipedia fetch error: {e}")

        return []

    def _parse_wikipedia_table(self, html: str, universe_name: str) -> List[str]:
        """
        Parse ticker symbols from Wikipedia HTML.

        Different universes have different table structures.
        """
        symbols = []

        # Simple regex-based parsing (works for most Wikipedia tables)
        # Look for ticker symbols in table cells

        if universe_name == "sp500":
            # S&P 500 table has symbols in first column with stock-symbol class
            pattern = r'class="external text"[^>]*>([A-Z.]+)</a>'
            matches = re.findall(pattern, html)
            symbols = list(set(matches))

            # Also try alternate pattern
            if len(symbols) < 100:
                pattern = r"<td[^>]*>([A-Z]{1,5})</td>"
                matches = re.findall(pattern, html)
                symbols = [m for m in matches if len(m) <= 5 and m.isalpha()]
                symbols = list(set(symbols))

        elif universe_name == "nasdaq100":
            pattern = r">([A-Z]{1,5})</a>"
            matches = re.findall(pattern, html)
            symbols = [m for m in matches if len(m) <= 5]
            symbols = list(set(symbols))

        elif universe_name == "dow30":
            pattern = r">([A-Z]{1,5})</a>"
            matches = re.findall(pattern, html)
            symbols = [m for m in matches if len(m) <= 5]
            symbols = list(set(symbols))

        # Filter to valid-looking ticker symbols
        symbols = [s for s in symbols if self._is_valid_ticker(s)]

        return symbols

    def _is_valid_ticker(self, symbol: str) -> bool:
        """Check if a string looks like a valid ticker symbol."""
        if not symbol:
            return False
        if len(symbol) > 5:
            return False
        if not symbol.replace(".", "").isalpha():
            return False
        # Exclude common false positives
        exclude = {"NYSE", "NASDAQ", "SEC", "USA", "USD", "ETF", "CEO", "CFO", "IPO"}
        if symbol.upper() in exclude:
            return False
        return True

    def _cache_universe(self, universe_name: str, symbols: List[str]):
        """Save universe data to cache."""
        # Memory cache
        self._memory_cache[universe_name] = (symbols, datetime.now())

        # File cache
        try:
            cache_file = CACHE_DIR / f"{universe_name}.json"
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "universe": universe_name,
                        "symbols": symbols,
                        "timestamp": datetime.now().isoformat(),
                        "count": len(symbols),
                    },
                    f,
                    indent=2,
                )
            logger.debug(f"Cached {universe_name} to {cache_file}")
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def clear_cache(self):
        """Clear all cached universe data."""
        self._memory_cache.clear()

        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass

        logger.info("Universe cache cleared")


# Convenience functions
async def get_sp500() -> List[str]:
    """Quick access to S&P 500 constituents."""
    provider = UniverseProvider()
    return await provider.get_sp500_constituents()


async def get_nasdaq100() -> List[str]:
    """Quick access to NASDAQ-100 constituents."""
    provider = UniverseProvider()
    return await provider.get_nasdaq100_constituents()
