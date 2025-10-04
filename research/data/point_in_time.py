"""
Point-in-Time Database - Survivorship-Bias-Free Data Access

Provides data exactly as it was known at any historical point:
- Avoids look-ahead bias by respecting announcement dates
- Handles restatements and corrections properly
- Tracks company events (M&A, delistings, bankruptcies)
- Supports multiple data sources (Compustat, FactSet, Bloomberg)

Why Point-in-Time Matters:
- Regular databases show current data, not what was known historically
- Earnings announced Feb 15 shouldn't appear in Jan 31 backtests
- Restated financials shouldn't retroactively change old signals
- Delisted/bankrupt companies must be included (survivorship bias)

Usage:
    pit = PointInTimeDB(storage_path="./pit_data")

    # Get fundamental data as it was known on a specific date
    eps = await pit.get_fundamental(
        symbol="AAPL",
        field="eps_diluted",
        as_of_date=datetime(2024, 1, 15),  # What was known on Jan 15
    )

    # Get universe that existed at a point in time
    sp500_2020 = await pit.get_universe(
        universe="SP500",
        as_of_date=datetime(2020, 1, 1),
    )

Data Sources (requires subscription):
- Compustat: Point-in-time fundamentals ($50K+/year)
- FactSet: Alternative PIT data
- Bloomberg: Terminal or B-PIPE access
- Sharadar/Quandl: Budget option (~$500/month)
"""

import asyncio
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DataField(Enum):
    """Standard fundamental data fields."""
    # Income Statement
    REVENUE = "revenue"
    GROSS_PROFIT = "gross_profit"
    OPERATING_INCOME = "operating_income"
    NET_INCOME = "net_income"
    EPS_BASIC = "eps_basic"
    EPS_DILUTED = "eps_diluted"

    # Balance Sheet
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    TOTAL_EQUITY = "total_equity"
    CASH = "cash"
    TOTAL_DEBT = "total_debt"
    BOOK_VALUE_PER_SHARE = "book_value_per_share"

    # Cash Flow
    OPERATING_CASH_FLOW = "operating_cash_flow"
    CAPEX = "capex"
    FREE_CASH_FLOW = "free_cash_flow"
    DIVIDENDS_PAID = "dividends_paid"

    # Ratios
    ROE = "roe"
    ROA = "roa"
    ROIC = "roic"
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    NET_MARGIN = "net_margin"
    DEBT_TO_EQUITY = "debt_to_equity"
    CURRENT_RATIO = "current_ratio"

    # Valuation (requires price)
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    PS_RATIO = "ps_ratio"
    EV_EBITDA = "ev_ebitda"
    DIVIDEND_YIELD = "dividend_yield"


class CorporateEvent(Enum):
    """Corporate events that affect data interpretation."""
    EARNINGS_RELEASE = "earnings_release"
    EARNINGS_RESTATEMENT = "earnings_restatement"
    STOCK_SPLIT = "stock_split"
    DIVIDEND_ANNOUNCEMENT = "dividend_announcement"
    MERGER_ANNOUNCED = "merger_announced"
    MERGER_COMPLETED = "merger_completed"
    SPINOFF = "spinoff"
    DELISTING = "delisting"
    BANKRUPTCY = "bankruptcy"
    IPO = "ipo"
    TICKER_CHANGE = "ticker_change"


@dataclass
class PITDataPoint:
    """A single point-in-time data observation."""
    symbol: str
    field: str
    value: float
    period_end: datetime  # Fiscal period end date
    announced_date: datetime  # When the data became public
    source: str  # Data provider
    version: int = 1  # For restatements (higher = newer)
    is_preliminary: bool = False
    is_restated: bool = False
    original_value: Optional[float] = None  # Before restatement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "field": self.field,
            "value": self.value,
            "period_end": self.period_end.isoformat(),
            "announced_date": self.announced_date.isoformat(),
            "source": self.source,
            "version": self.version,
            "is_preliminary": self.is_preliminary,
            "is_restated": self.is_restated,
            "original_value": self.original_value,
        }


@dataclass
class UniverseMembership:
    """Universe membership record."""
    symbol: str
    universe: str  # e.g., "SP500", "RUSSELL2000"
    added_date: datetime
    removed_date: Optional[datetime] = None
    removal_reason: Optional[str] = None  # e.g., "delisted", "dropped", "merged"


@dataclass
class PITQueryResult:
    """Result of a point-in-time query."""
    symbol: str
    field: str
    as_of_date: datetime
    value: Optional[float]
    data_point: Optional[PITDataPoint]
    data_age_days: Optional[int]  # Days since announcement
    next_release: Optional[datetime]  # Expected next data point


class DataSourceAdapter(ABC):
    """Abstract adapter for data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Source name."""
        pass

    @abstractmethod
    async def fetch_fundamental(
        self,
        symbol: str,
        field: DataField,
        start_date: datetime,
        end_date: datetime,
    ) -> List[PITDataPoint]:
        """Fetch fundamental data points."""
        pass

    @abstractmethod
    async def fetch_universe(
        self,
        universe: str,
        as_of_date: datetime,
    ) -> List[str]:
        """Fetch universe constituents."""
        pass

    @abstractmethod
    async def fetch_corporate_events(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, CorporateEvent, Dict[str, Any]]]:
        """Fetch corporate events."""
        pass


class CompustatAdapter(DataSourceAdapter):
    """
    Adapter for Compustat Point-in-Time data.

    Requires: WRDS subscription with Compustat access
    """

    def __init__(
        self,
        wrds_username: Optional[str] = None,
        wrds_password: Optional[str] = None,
    ):
        self._username = wrds_username or os.getenv("WRDS_USERNAME")
        self._password = wrds_password or os.getenv("WRDS_PASSWORD")
        self._connection = None

    @property
    def name(self) -> str:
        return "compustat"

    async def connect(self):
        """Connect to WRDS."""
        try:
            import wrds
            self._connection = wrds.Connection(
                wrds_username=self._username,
                wrds_password=self._password,
            )
            logger.info("Connected to WRDS/Compustat")
        except ImportError:
            logger.warning("wrds package not installed. Install with: pip install wrds")
        except Exception as e:
            logger.error(f"Failed to connect to WRDS: {e}")

    async def fetch_fundamental(
        self,
        symbol: str,
        field: DataField,
        start_date: datetime,
        end_date: datetime,
    ) -> List[PITDataPoint]:
        """Fetch from Compustat Point-in-Time."""
        if not self._connection:
            await self.connect()

        if not self._connection:
            return []

        # Map our field names to Compustat variables
        field_mapping = {
            DataField.REVENUE: "revtq",
            DataField.NET_INCOME: "niq",
            DataField.EPS_DILUTED: "epspxq",
            DataField.TOTAL_ASSETS: "atq",
            DataField.TOTAL_EQUITY: "seqq",
            DataField.BOOK_VALUE_PER_SHARE: "bkvlps",
            DataField.ROE: "roeq",
        }

        compustat_var = field_mapping.get(field)
        if not compustat_var:
            logger.warning(f"Field {field} not mapped to Compustat")
            return []

        # Query Compustat Point-in-Time
        # Note: This is simplified; real query needs proper date handling
        query = f"""
        SELECT tic, datadate, rdq, {compustat_var}
        FROM comp.fundq
        WHERE tic = '{symbol}'
        AND datadate BETWEEN '{start_date.strftime('%Y-%m-%d')}'
        AND '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY datadate
        """

        try:
            df = self._connection.raw_sql(query)

            data_points = []
            for _, row in df.iterrows():
                if row[compustat_var] is not None:
                    data_points.append(PITDataPoint(
                        symbol=symbol,
                        field=field.value,
                        value=float(row[compustat_var]),
                        period_end=row["datadate"],
                        announced_date=row["rdq"],  # Report Date of Quarterly
                        source=self.name,
                    ))

            return data_points

        except Exception as e:
            logger.error(f"Compustat query failed: {e}")
            return []

    async def fetch_universe(
        self,
        universe: str,
        as_of_date: datetime,
    ) -> List[str]:
        """Fetch S&P 500 constituents from Compustat."""
        if not self._connection:
            await self.connect()

        if not self._connection:
            return []

        # Query S&P 500 constituents
        query = f"""
        SELECT gvkey, tic, from_, thru
        FROM comp.idxcst_his
        WHERE gvkeyx = '000003'  -- S&P 500 index
        AND from_ <= '{as_of_date.strftime('%Y-%m-%d')}'
        AND (thru IS NULL OR thru >= '{as_of_date.strftime('%Y-%m-%d')}')
        """

        try:
            df = self._connection.raw_sql(query)
            return df["tic"].tolist()
        except Exception as e:
            logger.error(f"Compustat universe query failed: {e}")
            return []

    async def fetch_corporate_events(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, CorporateEvent, Dict[str, Any]]]:
        """Fetch corporate events."""
        # Simplified - real implementation would query multiple tables
        return []


class SharadarAdapter(DataSourceAdapter):
    """
    Adapter for Sharadar/Nasdaq Data Link (budget option).

    Requires: Nasdaq Data Link subscription (~$500/month)
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("NASDAQ_DATA_LINK_API_KEY")

    @property
    def name(self) -> str:
        return "sharadar"

    async def fetch_fundamental(
        self,
        symbol: str,
        field: DataField,
        start_date: datetime,
        end_date: datetime,
    ) -> List[PITDataPoint]:
        """Fetch from Sharadar SF1 table."""
        try:
            import nasdaqdatalink
            nasdaqdatalink.ApiConfig.api_key = self._api_key

            # Map fields to Sharadar columns
            field_mapping = {
                DataField.REVENUE: "revenue",
                DataField.NET_INCOME: "netinc",
                DataField.EPS_DILUTED: "epsdil",
                DataField.TOTAL_ASSETS: "assets",
                DataField.TOTAL_EQUITY: "equity",
                DataField.ROE: "roe",
                DataField.FREE_CASH_FLOW: "fcf",
            }

            sharadar_col = field_mapping.get(field)
            if not sharadar_col:
                return []

            # Fetch data
            df = nasdaqdatalink.get_table(
                "SHARADAR/SF1",
                ticker=symbol,
                calendardate={"gte": start_date.strftime("%Y-%m-%d")},
            )

            data_points = []
            for _, row in df.iterrows():
                if sharadar_col in row and row[sharadar_col] is not None:
                    # Sharadar uses datekey as announcement date
                    data_points.append(PITDataPoint(
                        symbol=symbol,
                        field=field.value,
                        value=float(row[sharadar_col]),
                        period_end=row["calendardate"],
                        announced_date=row.get("datekey", row["calendardate"]),
                        source=self.name,
                    ))

            return data_points

        except ImportError:
            logger.warning("nasdaqdatalink not installed")
            return []
        except Exception as e:
            logger.error(f"Sharadar fetch failed: {e}")
            return []

    async def fetch_universe(
        self,
        universe: str,
        as_of_date: datetime,
    ) -> List[str]:
        """Fetch universe from Sharadar."""
        # Sharadar provides S&P 500 constituents in their tickers table
        return []

    async def fetch_corporate_events(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, CorporateEvent, Dict[str, Any]]]:
        """Fetch corporate events from Sharadar."""
        return []


class PointInTimeDB:
    """
    Point-in-Time database for survivorship-bias-free backtesting.

    Features:
    - Stores data with announcement dates
    - Handles restatements properly
    - Tracks universe membership over time
    - Supports multiple data sources
    """

    def __init__(
        self,
        storage_path: str = "./pit_data",
        adapters: Optional[List[DataSourceAdapter]] = None,
    ):
        """
        Initialize Point-in-Time database.

        Args:
            storage_path: Path for local cache
            adapters: Data source adapters (Compustat, Sharadar, etc.)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.adapters = adapters or []
        self._db_path = self.storage_path / "pit_cache.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Data points table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                field TEXT NOT NULL,
                value REAL NOT NULL,
                period_end TEXT NOT NULL,
                announced_date TEXT NOT NULL,
                source TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_preliminary INTEGER DEFAULT 0,
                is_restated INTEGER DEFAULT 0,
                original_value REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, field, period_end, source, version)
            )
        """)

        # Universe membership table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universe_membership (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                universe TEXT NOT NULL,
                added_date TEXT NOT NULL,
                removed_date TEXT,
                removal_reason TEXT,
                UNIQUE(symbol, universe, added_date)
            )
        """)

        # Corporate events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corporate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                UNIQUE(symbol, event_date, event_type)
            )
        """)

        # Indices for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_symbol_field
            ON data_points(symbol, field, announced_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_universe_date
            ON universe_membership(universe, added_date, removed_date)
        """)

        conn.commit()
        conn.close()

    async def get_fundamental(
        self,
        symbol: str,
        field: Union[str, DataField],
        as_of_date: datetime,
        allow_stale_days: int = 120,  # Max staleness for quarterly data
    ) -> PITQueryResult:
        """
        Get fundamental data as it was known on a specific date.

        Args:
            symbol: Stock symbol
            field: Data field to retrieve
            as_of_date: Point in time to query
            allow_stale_days: Maximum age of data in days

        Returns:
            PITQueryResult with value and metadata
        """
        if isinstance(field, DataField):
            field = field.value

        # Query local cache first
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Get most recent data point announced before as_of_date
        cursor.execute("""
            SELECT symbol, field, value, period_end, announced_date,
                   source, version, is_preliminary, is_restated, original_value
            FROM data_points
            WHERE symbol = ? AND field = ?
            AND announced_date <= ?
            ORDER BY announced_date DESC, version DESC
            LIMIT 1
        """, (symbol, field, as_of_date.isoformat()))

        row = cursor.fetchone()
        conn.close()

        if row:
            data_point = PITDataPoint(
                symbol=row[0],
                field=row[1],
                value=row[2],
                period_end=datetime.fromisoformat(row[3]),
                announced_date=datetime.fromisoformat(row[4]),
                source=row[5],
                version=row[6],
                is_preliminary=bool(row[7]),
                is_restated=bool(row[8]),
                original_value=row[9],
            )

            data_age = (as_of_date - data_point.announced_date).days

            if data_age > allow_stale_days:
                logger.warning(
                    f"Stale data for {symbol}.{field}: {data_age} days old"
                )

            return PITQueryResult(
                symbol=symbol,
                field=field,
                as_of_date=as_of_date,
                value=data_point.value,
                data_point=data_point,
                data_age_days=data_age,
                next_release=None,  # Would need earnings calendar
            )

        # Data not in cache - try to fetch from adapters
        for adapter in self.adapters:
            try:
                data_points = await adapter.fetch_fundamental(
                    symbol=symbol,
                    field=DataField(field),
                    start_date=as_of_date - timedelta(days=allow_stale_days),
                    end_date=as_of_date,
                )

                if data_points:
                    # Cache the data
                    await self._cache_data_points(data_points)

                    # Find the right data point
                    valid_points = [
                        p for p in data_points
                        if p.announced_date <= as_of_date
                    ]
                    if valid_points:
                        dp = max(valid_points, key=lambda x: x.announced_date)
                        return PITQueryResult(
                            symbol=symbol,
                            field=field,
                            as_of_date=as_of_date,
                            value=dp.value,
                            data_point=dp,
                            data_age_days=(as_of_date - dp.announced_date).days,
                            next_release=None,
                        )

            except Exception as e:
                logger.error(f"Adapter {adapter.name} failed: {e}")

        # No data found
        return PITQueryResult(
            symbol=symbol,
            field=field,
            as_of_date=as_of_date,
            value=None,
            data_point=None,
            data_age_days=None,
            next_release=None,
        )

    async def get_fundamentals_batch(
        self,
        symbols: List[str],
        fields: List[Union[str, DataField]],
        as_of_date: datetime,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Get multiple fundamentals for multiple symbols.

        Returns:
            Nested dict: {symbol: {field: value}}
        """
        results = {}

        # Run queries in parallel
        tasks = []
        for symbol in symbols:
            for field in fields:
                tasks.append(self.get_fundamental(symbol, field, as_of_date))

        query_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results
        idx = 0
        for symbol in symbols:
            results[symbol] = {}
            for field in fields:
                field_name = field.value if isinstance(field, DataField) else field
                result = query_results[idx]
                if isinstance(result, Exception):
                    results[symbol][field_name] = None
                else:
                    results[symbol][field_name] = result.value
                idx += 1

        return results

    async def get_universe(
        self,
        universe: str,
        as_of_date: datetime,
    ) -> List[str]:
        """
        Get universe constituents as of a specific date.

        Args:
            universe: Universe name (e.g., "SP500", "RUSSELL2000")
            as_of_date: Point in time

        Returns:
            List of symbols that were in the universe
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT symbol FROM universe_membership
            WHERE universe = ?
            AND added_date <= ?
            AND (removed_date IS NULL OR removed_date > ?)
        """, (universe, as_of_date.isoformat(), as_of_date.isoformat()))

        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not symbols:
            # Try to fetch from adapters
            for adapter in self.adapters:
                try:
                    symbols = await adapter.fetch_universe(universe, as_of_date)
                    if symbols:
                        break
                except Exception as e:
                    logger.error(f"Universe fetch failed: {e}")

        return symbols

    async def get_corporate_events(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[CorporateEvent]] = None,
    ) -> List[Tuple[datetime, CorporateEvent, Dict[str, Any]]]:
        """Get corporate events for a symbol in date range."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        query = """
            SELECT event_date, event_type, details
            FROM corporate_events
            WHERE symbol = ?
            AND event_date BETWEEN ? AND ?
        """
        params = [symbol, start_date.isoformat(), end_date.isoformat()]

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend([e.value for e in event_types])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append((
                datetime.fromisoformat(row[0]),
                CorporateEvent(row[1]),
                json.loads(row[2]) if row[2] else {},
            ))

        return events

    async def _cache_data_points(self, data_points: List[PITDataPoint]):
        """Cache data points to local database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        for dp in data_points:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO data_points
                    (symbol, field, value, period_end, announced_date,
                     source, version, is_preliminary, is_restated, original_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dp.symbol, dp.field, dp.value,
                    dp.period_end.isoformat(), dp.announced_date.isoformat(),
                    dp.source, dp.version, dp.is_preliminary, dp.is_restated,
                    dp.original_value,
                ))
            except Exception as e:
                logger.error(f"Failed to cache data point: {e}")

        conn.commit()
        conn.close()

    async def add_universe_membership(
        self,
        symbol: str,
        universe: str,
        added_date: datetime,
        removed_date: Optional[datetime] = None,
        removal_reason: Optional[str] = None,
    ):
        """Add or update universe membership record."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO universe_membership
            (symbol, universe, added_date, removed_date, removal_reason)
            VALUES (?, ?, ?, ?, ?)
        """, (
            symbol, universe, added_date.isoformat(),
            removed_date.isoformat() if removed_date else None,
            removal_reason,
        ))

        conn.commit()
        conn.close()

    async def import_from_csv(
        self,
        filepath: str,
        field: DataField,
        source: str = "csv_import",
    ):
        """
        Import historical data from CSV file.

        Expected columns: symbol, period_end, announced_date, value
        """
        import csv

        data_points = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data_points.append(PITDataPoint(
                        symbol=row["symbol"],
                        field=field.value,
                        value=float(row["value"]),
                        period_end=datetime.fromisoformat(row["period_end"]),
                        announced_date=datetime.fromisoformat(row["announced_date"]),
                        source=source,
                    ))
                except Exception as e:
                    logger.warning(f"Skipping row: {e}")

        await self._cache_data_points(data_points)
        logger.info(f"Imported {len(data_points)} data points from {filepath}")

    def get_data_coverage(self, symbol: str) -> Dict[str, Any]:
        """Get data coverage statistics for a symbol."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT field, COUNT(*) as count,
                   MIN(period_end) as earliest,
                   MAX(period_end) as latest
            FROM data_points
            WHERE symbol = ?
            GROUP BY field
        """, (symbol,))

        coverage = {}
        for row in cursor.fetchall():
            coverage[row[0]] = {
                "count": row[1],
                "earliest": row[2],
                "latest": row[3],
            }

        conn.close()
        return coverage

    def get_supported_sources(self) -> List[str]:
        """Get list of configured data sources."""
        sources = []
        for adapter in self.adapters:
            sources.append(adapter.__class__.__name__.replace("Adapter", ""))
        # Always include local cache
        sources.append("local_cache")
        return sources


def create_pit_db(
    storage_path: str = "./pit_data",
    use_compustat: bool = False,
    use_sharadar: bool = False,
    sources: Optional[List[str]] = None,
) -> PointInTimeDB:
    """
    Factory function to create Point-in-Time database with adapters.

    Args:
        storage_path: Path for local cache
        use_compustat: Enable Compustat adapter (requires WRDS subscription)
        use_sharadar: Enable Sharadar adapter (requires Nasdaq Data Link subscription)

    Returns:
        Configured PointInTimeDB instance
    """
    adapters = []

    if use_compustat:
        adapters.append(CompustatAdapter())

    if use_sharadar:
        adapters.append(SharadarAdapter())

    return PointInTimeDB(storage_path=storage_path, adapters=adapters)
