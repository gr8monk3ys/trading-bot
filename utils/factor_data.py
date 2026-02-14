"""
Factor Data Pipeline

Fetches and processes fundamental data for factor calculations.
Supports multiple data sources with fallback mechanisms.

Data Sources:
1. Alpaca API (market data, limited fundamentals)
2. External APIs (Alpha Vantage, Financial Modeling Prep, etc.)
3. CSV files (for backtesting with historical data)

Data Processing:
- Point-in-time data handling (no look-ahead bias)
- Proper winsorization and outlier handling
- Sector classification for sector-neutral factors
- Caching for performance
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Fundamental data for a single symbol at a point in time."""

    symbol: str
    as_of_date: datetime
    # Data provenance / source tracking. Not included in `to_dict()` to avoid
    # polluting numeric factor inputs.
    source: str = "unknown"

    # Value metrics
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None

    # Quality metrics
    roe: Optional[float] = None  # Return on equity
    roa: Optional[float] = None  # Return on assets
    roic: Optional[float] = None  # Return on invested capital
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    earnings_variability: Optional[float] = None  # StdDev of EPS

    # Size metrics
    market_cap: Optional[float] = None

    # Sector classification
    sector: Optional[str] = None
    industry: Optional[str] = None

    def to_dict(self) -> Dict[str, float | str | None]:
        """Convert to dictionary for factor calculations."""
        return {
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "ps_ratio": self.ps_ratio,
            "ev_ebitda": self.ev_ebitda,
            "roe": self.roe,
            "roa": self.roa,
            "roic": self.roic,
            "debt_to_equity": self.debt_to_equity,
            "current_ratio": self.current_ratio,
            "earnings_variability": self.earnings_variability,
            "market_cap": self.market_cap,
            "sector": self.sector,
            "industry": self.industry,
        }


class FactorDataProvider:
    """
    Provides fundamental data for factor calculations.

    Features:
    - Multiple data source support
    - Caching with configurable TTL
    - Point-in-time data for backtesting
    - Batch fetching for efficiency
    """

    # Sector classifications (GICS-like)
    SECTORS = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Consumer Discretionary",
        "META": "Technology",
        "NVDA": "Technology",
        "TSLA": "Consumer Discretionary",
        "JPM": "Financials",
        "V": "Financials",
        "JNJ": "Healthcare",
        "UNH": "Healthcare",
        "PG": "Consumer Staples",
        "HD": "Consumer Discretionary",
        "MA": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
        "PFE": "Healthcare",
        "ABBV": "Healthcare",
        "KO": "Consumer Staples",
        "PEP": "Consumer Staples",
        "WMT": "Consumer Staples",
        "COST": "Consumer Staples",
        "DIS": "Communication Services",
        "NFLX": "Communication Services",
        "INTC": "Technology",
        "AMD": "Technology",
        "CRM": "Technology",
        "ORCL": "Technology",
        "ADBE": "Technology",
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: int = 24,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        allow_synthetic_fallback: bool = True,
    ):
        """
        Initialize the factor data provider.

        Args:
            cache_dir: Directory for caching data
            cache_ttl_hours: Cache time-to-live in hours
            alpaca_api_key: Alpaca API key (optional)
            alpaca_secret_key: Alpaca secret key (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".factor_cache")
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.alpaca_api_key = alpaca_api_key or os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.allow_synthetic_fallback = allow_synthetic_fallback

        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, FundamentalData] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def _get_cache_key(self, symbol: str, as_of_date: Optional[datetime] = None) -> str:
        """Generate cache key."""
        date_str = (as_of_date or datetime.now()).strftime("%Y-%m-%d")
        return f"{symbol}_{date_str}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        age = datetime.now() - self._cache_timestamps[cache_key]
        return age < self.cache_ttl

    async def get_fundamental_data(
        self,
        symbol: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[FundamentalData]:
        """
        Get fundamental data for a single symbol.

        Args:
            symbol: Stock symbol
            as_of_date: Point-in-time date (default: now)

        Returns:
            FundamentalData or None if unavailable
        """
        as_of_date = as_of_date or datetime.now()
        cache_key = self._get_cache_key(symbol, as_of_date)

        # Check cache
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Fetch from sources (with fallback)
        data = await self._fetch_from_sources(symbol, as_of_date)

        if data:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()

        return data

    async def get_batch_fundamental_data(
        self,
        symbols: List[str],
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, FundamentalData]:
        """
        Get fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols
            as_of_date: Point-in-time date

        Returns:
            Dictionary mapping symbols to FundamentalData
        """
        tasks = [self.get_fundamental_data(s, as_of_date) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results, strict=False):
            if isinstance(result, FundamentalData):
                data[symbol] = result
            elif isinstance(result, Exception):
                logger.warning(f"Failed to fetch data for {symbol}: {result}")

        return data

    async def _fetch_from_sources(
        self,
        symbol: str,
        as_of_date: datetime,
    ) -> Optional[FundamentalData]:
        """
        Fetch fundamental data from available sources.

        Tries sources in order:
        1. Local CSV files (for backtesting)
        2. Alpaca API
        3. Synthetic data (fallback for testing)
        """
        # Try local CSV first
        csv_data = await self._fetch_from_csv(symbol, as_of_date)
        if csv_data:
            return csv_data

        # Try Alpaca (limited fundamental data)
        if self.alpaca_api_key:
            alpaca_data = await self._fetch_from_alpaca(symbol)
            if alpaca_data:
                return alpaca_data

        # Fallback: synthetic data for testing
        if not self.allow_synthetic_fallback:
            return None

        return self._generate_synthetic_data(symbol, as_of_date)

    async def _fetch_from_csv(
        self,
        symbol: str,
        as_of_date: datetime,
    ) -> Optional[FundamentalData]:
        """Load fundamental data from CSV files."""
        csv_path = self.cache_dir / "fundamentals.csv"
        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])

            # Point-in-time: get most recent data before as_of_date
            symbol_data = df[(df["symbol"] == symbol) & (df["date"] <= as_of_date)].sort_values(
                "date", ascending=False
            )

            if symbol_data.empty:
                return None

            row = symbol_data.iloc[0]
            return FundamentalData(
                symbol=symbol,
                as_of_date=row["date"].to_pydatetime(),
                source="csv",
                pe_ratio=row.get("pe_ratio"),
                pb_ratio=row.get("pb_ratio"),
                ps_ratio=row.get("ps_ratio"),
                ev_ebitda=row.get("ev_ebitda"),
                roe=row.get("roe"),
                roa=row.get("roa"),
                roic=row.get("roic"),
                debt_to_equity=row.get("debt_to_equity"),
                current_ratio=row.get("current_ratio"),
                earnings_variability=row.get("earnings_variability"),
                market_cap=row.get("market_cap"),
                sector=row.get("sector"),
                industry=row.get("industry"),
            )
        except Exception as e:
            logger.warning(f"Failed to load CSV data for {symbol}: {e}")
            return None

    async def _fetch_from_alpaca(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch from Alpaca API (limited fundamental data available)."""
        # Alpaca has limited fundamental data - mainly through news/assets
        # This is a placeholder for when they add more fundamental data
        return None

    def _generate_synthetic_data(
        self,
        symbol: str,
        as_of_date: datetime,
    ) -> FundamentalData:
        """
        Generate synthetic fundamental data for testing.

        IMPORTANT: This should only be used for testing/development.
        Production systems should use real data sources.
        """
        # Use symbol hash for consistent random values
        np.random.seed(hash(symbol) % (2**32))

        sector = self.SECTORS.get(symbol, "Other")

        # Generate realistic ranges based on sector
        sector_adjustments = {
            "Technology": {"pe": 1.5, "growth": 1.3},
            "Financials": {"pe": 0.7, "growth": 0.9},
            "Healthcare": {"pe": 1.2, "growth": 1.1},
            "Consumer Discretionary": {"pe": 1.0, "growth": 1.0},
            "Consumer Staples": {"pe": 0.9, "growth": 0.8},
            "Energy": {"pe": 0.6, "growth": 0.7},
            "Communication Services": {"pe": 1.1, "growth": 1.0},
            "Other": {"pe": 1.0, "growth": 1.0},
        }

        adj = sector_adjustments.get(sector, {"pe": 1.0, "growth": 1.0})

        return FundamentalData(
            symbol=symbol,
            as_of_date=as_of_date,
            source="synthetic",
            pe_ratio=max(5, np.random.normal(20 * adj["pe"], 8)),
            pb_ratio=max(0.5, np.random.normal(3.0, 1.5)),
            ps_ratio=max(0.5, np.random.normal(4.0 * adj["pe"], 2.0)),
            ev_ebitda=max(3, np.random.normal(12 * adj["pe"], 5)),
            roe=max(-0.1, np.random.normal(0.15 * adj["growth"], 0.08)),
            roa=max(-0.05, np.random.normal(0.08 * adj["growth"], 0.04)),
            roic=max(-0.05, np.random.normal(0.12 * adj["growth"], 0.06)),
            debt_to_equity=max(0, np.random.normal(0.8, 0.5)),
            current_ratio=max(0.5, np.random.normal(1.5, 0.5)),
            earnings_variability=max(0.05, np.random.normal(0.2, 0.1)),
            market_cap=10 ** np.random.uniform(9, 12),  # $1B to $1T
            sector=sector,
            industry=None,
        )

    def save_to_csv(
        self,
        data: Dict[str, FundamentalData],
        filename: str = "fundamentals.csv",
    ):
        """
        Save fundamental data to CSV for backtesting.

        Args:
            data: Dictionary of symbol -> FundamentalData
            filename: Output filename
        """
        rows = []
        for symbol, fd in data.items():
            row = fd.to_dict()
            row["symbol"] = symbol
            row["date"] = fd.as_of_date.strftime("%Y-%m-%d")
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.cache_dir / filename

        # Append if exists, otherwise create
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            existing_records = existing.dropna(how="all").to_dict(orient="records")
            new_records = df.dropna(how="all").to_dict(orient="records")

            merged_by_key = {}
            for record in existing_records + new_records:
                key = (record.get("symbol"), record.get("date"))
                merged_by_key[key] = record

            if merged_by_key:
                all_columns = sorted(
                    {column for record in merged_by_key.values() for column in record.keys()}
                )
                df = pd.DataFrame(merged_by_key.values()).reindex(columns=all_columns)
            else:
                df = pd.DataFrame(columns=existing.columns if not existing.empty else df.columns)

        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(rows)} records to {csv_path}")

    async def build_factor_inputs(
        self,
        symbols: List[str],
        price_data: pd.DataFrame,
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Build all inputs needed for factor calculations.

        Args:
            symbols: List of symbols
            price_data: Price data DataFrame
            as_of_date: Point-in-time date

        Returns:
            Dictionary with:
            - fundamental_data: {symbol: {metric: value}}
            - fundamental_data_real: {symbol: {metric: value}} (non-synthetic sources)
            - market_caps: {symbol: market_cap}
            - market_caps_real: {symbol: market_cap} (non-synthetic sources)
            - sectors: {symbol: sector}
            - data_provenance: summary of data sources/coverage
        """
        as_of_date = as_of_date or datetime.now()

        # Fetch fundamental data
        fundamental_batch = await self.get_batch_fundamental_data(symbols, as_of_date)

        # Build output dictionaries
        fundamental_data = {}
        fundamental_data_real = {}
        market_caps = {}
        market_caps_real = {}
        sectors = {}
        sources: Dict[str, str] = {}
        counts_by_source: Dict[str, int] = {}

        real_sources = {"csv", "alpaca"}

        for symbol, fd in fundamental_batch.items():
            source = (getattr(fd, "source", None) or "unknown").strip().lower()
            sources[symbol] = source
            counts_by_source[source] = counts_by_source.get(source, 0) + 1

            fundamental_data[symbol] = fd.to_dict()
            if source in real_sources:
                fundamental_data_real[symbol] = fd.to_dict()
            if fd.market_cap:
                market_caps[symbol] = fd.market_cap
                if source in real_sources:
                    market_caps_real[symbol] = fd.market_cap
            if fd.sector:
                sectors[symbol] = fd.sector

        total_symbols = len(symbols)
        symbols_with_data = len(fundamental_batch)
        missing_symbols = [s for s in symbols if s not in fundamental_batch]
        synthetic_count = counts_by_source.get("synthetic", 0)
        real_count = sum(counts_by_source.get(s, 0) for s in real_sources)
        coverage_ratio = (symbols_with_data / total_symbols) if total_symbols else 0.0
        synthetic_ratio = (synthetic_count / total_symbols) if total_symbols else 0.0
        real_ratio = (real_count / total_symbols) if total_symbols else 0.0
        missing_ratio = (
            ((total_symbols - symbols_with_data) / total_symbols) if total_symbols else 0.0
        )

        return {
            "fundamental_data": fundamental_data,
            "fundamental_data_real": fundamental_data_real,
            "market_caps": market_caps,
            "market_caps_real": market_caps_real,
            "sectors": sectors,
            "data_provenance": {
                "as_of_date": as_of_date.isoformat(),
                "total_symbols": total_symbols,
                "symbols_with_data": symbols_with_data,
                "missing_symbols": missing_symbols,
                "real_sources": sorted(real_sources),
                "sources": sources,
                "counts_by_source": counts_by_source,
                "ratios": {
                    "coverage_ratio": coverage_ratio,
                    "real_ratio": real_ratio,
                    "synthetic_ratio": synthetic_ratio,
                    "missing_ratio": missing_ratio,
                },
            },
        }


class PointInTimeDataManager:
    """
    Manages point-in-time data for backtesting to avoid look-ahead bias.

    CRITICAL for institutional-grade backtesting:
    - Only use data that would have been available at each point in time
    - Account for data publication delays (earnings announced days after quarter end)
    - Handle data revisions properly
    """

    # Typical data availability delays (in days)
    PUBLICATION_DELAYS = {
        "earnings": 45,  # Quarterly earnings usually available ~45 days after quarter end
        "balance_sheet": 45,
        "market_cap": 1,  # Next day
        "price": 0,  # Same day
    }

    def __init__(self, data_provider: FactorDataProvider):
        self.data_provider = data_provider
        self._historical_data: Dict[str, pd.DataFrame] = {}

    def get_available_date(
        self,
        data_type: str,
        as_of_date: datetime,
    ) -> datetime:
        """
        Get the date of most recent available data.

        Args:
            data_type: Type of data ('earnings', 'balance_sheet', etc.)
            as_of_date: Current simulation date

        Returns:
            Date of most recent available data
        """
        delay = self.PUBLICATION_DELAYS.get(data_type, 0)
        return as_of_date - timedelta(days=delay)

    async def get_point_in_time_fundamentals(
        self,
        symbols: List[str],
        as_of_date: datetime,
    ) -> Dict[str, Dict[str, float | str | None]]:
        """
        Get fundamental data that would have been available at as_of_date.

        This accounts for publication delays to prevent look-ahead bias.
        """
        # For earnings-based metrics, use data from ~45 days prior
        available_date = self.get_available_date("earnings", as_of_date)

        batch = await self.data_provider.get_batch_fundamental_data(symbols, available_date)

        return {symbol: fd.to_dict() for symbol, fd in batch.items()}


def create_sample_fundamentals_csv(output_dir: str = ".factor_cache"):
    """
    Create sample fundamentals CSV for testing.

    This generates realistic-looking fundamental data for common stocks.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "V",
        "JNJ",
        "UNH",
        "PG",
        "HD",
        "MA",
        "BAC",
        "XOM",
        "CVX",
        "PFE",
        "ABBV",
        "KO",
        "PEP",
        "WMT",
        "COST",
    ]

    provider = FactorDataProvider(cache_dir=output_dir)

    # Generate data for multiple dates
    rows = []
    dates = pd.date_range(end=datetime.now(), periods=12, freq="ME")

    for date in dates:
        for symbol in symbols:
            fd = provider._generate_synthetic_data(symbol, date.to_pydatetime())
            row = fd.to_dict()
            row["symbol"] = symbol
            row["date"] = date.strftime("%Y-%m-%d")
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_path / "fundamentals.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created sample fundamentals at {csv_path}")

    return csv_path


if __name__ == "__main__":
    # Generate sample data when run directly
    create_sample_fundamentals_csv()
