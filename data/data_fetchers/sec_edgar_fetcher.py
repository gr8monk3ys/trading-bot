"""
SEC EDGAR Filing Fetcher

Fetches SEC filings from EDGAR:
- 10-K Annual Reports (Risk Factors, MD&A)
- 10-Q Quarterly Reports
- 8-K Current Reports (Material Events)
"""

import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """SEC filing data."""

    symbol: str
    cik: str  # Central Index Key
    filing_type: str  # 10-K, 10-Q, 8-K
    filing_date: datetime
    accession_number: str

    # Content
    content: str = ""

    # Parsed sections (for 10-K/10-Q)
    risk_factors: str = ""
    mda_section: str = ""  # Management Discussion & Analysis
    business_section: str = ""

    # For 8-K
    items_reported: List[str] = field(default_factory=list)

    # Metadata
    company_name: str = ""
    fiscal_year_end: str = ""
    word_count: int = 0
    url: str = ""
    source: str = ""
    cached: bool = False

    def __post_init__(self):
        """Calculate word count."""
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())


class SECEdgarFetcher:
    """
    Fetches SEC filings from EDGAR.

    Primary: SEC EDGAR API (data.sec.gov)
    Cache: 30 days
    """

    # SEC EDGAR API base URL
    BASE_URL = "https://data.sec.gov"

    # User agent required by SEC
    USER_AGENT = "TradingBot/1.0 (contact@example.com)"

    # Cache TTL: 30 days
    CACHE_TTL_DAYS = 30

    # Rate limit: 10 requests per second per SEC guidelines
    REQUEST_DELAY = 0.1

    # 8-K Item codes and descriptions
    ITEM_8K_CODES = {
        "1.01": "Entry into Material Agreement",
        "1.02": "Termination of Material Agreement",
        "1.03": "Bankruptcy or Receivership",
        "2.01": "Completion of Acquisition/Disposition",
        "2.02": "Results of Operations and Financial Condition",
        "2.03": "Creation of Direct Financial Obligation",
        "2.04": "Triggering Events That Accelerate Obligations",
        "2.05": "Costs Associated with Exit Activities",
        "2.06": "Material Impairments",
        "3.01": "Notice of Delisting",
        "3.02": "Unregistered Sales of Equity Securities",
        "3.03": "Material Modification to Rights of Security Holders",
        "4.01": "Changes in Registrant's Certifying Accountant",
        "4.02": "Non-Reliance on Previously Issued Financials",
        "5.01": "Changes in Control",
        "5.02": "Departure/Appointment of Directors or Officers",
        "5.03": "Amendments to Articles of Incorporation or Bylaws",
        "5.04": "Temporary Suspension of Trading",
        "5.05": "Amendment to Code of Ethics",
        "5.06": "Change in Shell Company Status",
        "5.07": "Submission of Matters to Vote of Security Holders",
        "5.08": "Shareholder Nominations",
        "6.01": "ABS Informational and Computational Material",
        "7.01": "Regulation FD Disclosure",
        "8.01": "Other Events",
        "9.01": "Financial Statements and Exhibits",
    }

    def __init__(self, cache_dir: str = ".cache/sec_filings"):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory for caching filings
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._cache_dir / "filings.db"
        self._init_db()

        # CIK lookup cache
        self._cik_cache: Dict[str, str] = {}

    def _init_db(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS filings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    cik TEXT NOT NULL,
                    filing_type TEXT NOT NULL,
                    filing_date TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    content TEXT,
                    risk_factors TEXT,
                    mda_section TEXT,
                    business_section TEXT,
                    items_reported TEXT,
                    company_name TEXT,
                    fiscal_year_end TEXT,
                    word_count INTEGER,
                    url TEXT,
                    source TEXT,
                    fetched_at TEXT NOT NULL,
                    UNIQUE(symbol, filing_type, accession_number)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_filings_symbol
                ON filings(symbol, filing_type, filing_date)
            """)
            conn.commit()

    async def fetch_latest(
        self,
        symbol: str,
        filing_type: str = "10-K",
    ) -> Optional[SECFiling]:
        """
        Fetch the most recent filing of a given type.

        Args:
            symbol: Stock symbol
            filing_type: 10-K, 10-Q, or 8-K

        Returns:
            SECFiling or None if not available
        """
        # Check cache first
        cached = self._get_cached_latest(symbol, filing_type)
        if cached:
            logger.debug(f"Cache hit for {symbol} {filing_type}")
            return cached

        # Get CIK for symbol
        cik = await self._get_cik(symbol)
        if not cik:
            logger.warning(f"Could not find CIK for {symbol}")
            return self._generate_mock_filing(symbol, filing_type)

        # Fetch filing list
        filings = await self._fetch_filing_list(cik, filing_type)
        if not filings:
            return self._generate_mock_filing(symbol, filing_type)

        # Get most recent filing
        latest = filings[0]

        # Fetch full filing content
        filing = await self._fetch_filing_content(symbol, cik, latest, filing_type)
        if filing:
            self._cache_filing(filing)
            return filing

        return self._generate_mock_filing(symbol, filing_type)

    async def fetch_by_date(
        self,
        symbol: str,
        filing_type: str,
        after_date: datetime,
    ) -> List[SECFiling]:
        """
        Fetch filings after a specific date.

        Args:
            symbol: Stock symbol
            filing_type: 10-K, 10-Q, or 8-K
            after_date: Only return filings after this date

        Returns:
            List of SECFiling objects
        """
        cik = await self._get_cik(symbol)
        if not cik:
            return []

        filings = await self._fetch_filing_list(cik, filing_type)
        results = []

        for filing_info in filings:
            filing_date = datetime.fromisoformat(filing_info.get("filedAt", "")[:10])
            if filing_date > after_date:
                filing = await self._fetch_filing_content(
                    symbol, cik, filing_info, filing_type
                )
                if filing:
                    results.append(filing)
                    self._cache_filing(filing)

            # Rate limiting
            await asyncio.sleep(self.REQUEST_DELAY)

        return results

    async def fetch_8k_material_events(
        self,
        symbol: str,
        days_back: int = 30,
    ) -> List[SECFiling]:
        """
        Fetch recent 8-K filings with material events.

        Args:
            symbol: Stock symbol
            days_back: Days to look back

        Returns:
            List of 8-K filings with material events
        """
        after_date = datetime.now() - timedelta(days=days_back)
        filings = await self.fetch_by_date(symbol, "8-K", after_date)

        # Filter for material events
        material_items = {"1.01", "1.02", "1.03", "2.01", "2.05", "2.06", "5.02"}
        material_filings = [
            f for f in filings
            if any(item in material_items for item in f.items_reported)
        ]

        return material_filings

    async def _get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK for a stock symbol."""
        if symbol in self._cik_cache:
            return self._cik_cache[symbol]

        try:
            url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "company": symbol,
                "type": "10-K",
                "dateb": "",
                "owner": "include",
                "count": "1",
                "output": "atom",
            }

            headers = {"User-Agent": self.USER_AGENT}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, headers=headers, timeout=30
                ) as response:
                    if response.status != 200:
                        return None

                    content = await response.text()

                    # Extract CIK from response
                    match = re.search(r"CIK=(\d+)", content)
                    if match:
                        cik = match.group(1).zfill(10)
                        self._cik_cache[symbol] = cik
                        return cik

        except Exception as e:
            logger.error(f"CIK lookup error for {symbol}: {e}")

        # Try alternative lookup via company tickers JSON
        return await self._lookup_cik_from_tickers(symbol)

    async def _lookup_cik_from_tickers(self, symbol: str) -> Optional[str]:
        """Lookup CIK from SEC company tickers JSON."""
        try:
            url = f"{self.BASE_URL}/files/company_tickers.json"
            headers = {"User-Agent": self.USER_AGENT}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()

                    for entry in data.values():
                        if entry.get("ticker", "").upper() == symbol.upper():
                            cik = str(entry.get("cik_str", "")).zfill(10)
                            self._cik_cache[symbol] = cik
                            return cik

        except Exception as e:
            logger.error(f"Ticker lookup error for {symbol}: {e}")

        return None

    async def _fetch_filing_list(
        self,
        cik: str,
        filing_type: str,
    ) -> List[Dict[str, Any]]:
        """Fetch list of filings for a CIK."""
        try:
            url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": cik,
                "type": filing_type,
                "dateb": "",
                "owner": "include",
                "count": "10",
                "output": "atom",
            }

            headers = {"User-Agent": self.USER_AGENT}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, headers=headers, timeout=30
                ) as response:
                    if response.status != 200:
                        return []

                    content = await response.text()
                    return self._parse_filing_list(content)

        except Exception as e:
            logger.error(f"Filing list fetch error: {e}")
            return []

    def _parse_filing_list(self, content: str) -> List[Dict[str, Any]]:
        """Parse filing list from EDGAR response."""
        filings = []

        # Extract entries from Atom feed
        # Simple regex parsing - in production use proper XML parsing
        entries = re.findall(
            r"<entry>(.*?)</entry>",
            content,
            re.DOTALL
        )

        for entry in entries:
            filing_info = {}

            # Extract accession number
            match = re.search(r"accession-number>(\d+-\d+-\d+)<", entry)
            if match:
                filing_info["accessionNumber"] = match.group(1)

            # Extract filing date
            match = re.search(r"filing-date>(\d{4}-\d{2}-\d{2})<", entry)
            if match:
                filing_info["filedAt"] = match.group(1)

            # Extract filing type
            match = re.search(r"filing-type>([^<]+)<", entry)
            if match:
                filing_info["filingType"] = match.group(1)

            # Extract link
            match = re.search(r'href="([^"]+)"', entry)
            if match:
                filing_info["link"] = match.group(1)

            if filing_info.get("accessionNumber"):
                filings.append(filing_info)

        return filings

    async def _fetch_filing_content(
        self,
        symbol: str,
        cik: str,
        filing_info: Dict[str, Any],
        filing_type: str,
    ) -> Optional[SECFiling]:
        """Fetch full filing content."""
        try:
            accession = filing_info.get("accessionNumber", "").replace("-", "")
            filing_date = filing_info.get("filedAt", "")

            # Construct filing URL
            url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession}"

            headers = {"User-Agent": self.USER_AGENT}

            async with aiohttp.ClientSession() as session:
                # Get filing index
                async with session.get(
                    f"{url}/index.json", headers=headers, timeout=30
                ) as response:
                    if response.status != 200:
                        # Try to get the primary document directly
                        return await self._fetch_primary_document(
                            session, url, symbol, cik, filing_info, filing_type, headers
                        )

                    index_data = await response.json()

                # Find primary document
                primary_doc = None
                for item in index_data.get("directory", {}).get("item", []):
                    name = item.get("name", "")
                    if filing_type.lower() in name.lower() and name.endswith(".htm"):
                        primary_doc = name
                        break

                if not primary_doc:
                    # Look for any .htm file
                    for item in index_data.get("directory", {}).get("item", []):
                        if item.get("name", "").endswith(".htm"):
                            primary_doc = item["name"]
                            break

                if not primary_doc:
                    return None

                # Fetch primary document
                async with session.get(
                    f"{url}/{primary_doc}", headers=headers, timeout=60
                ) as response:
                    if response.status != 200:
                        return None

                    html_content = await response.text()

                    # Parse content
                    content = self._extract_text_from_html(html_content)

                    # Extract sections based on filing type
                    risk_factors = ""
                    mda_section = ""
                    items_reported = []

                    if filing_type in ("10-K", "10-Q"):
                        risk_factors = self._extract_risk_factors(content)
                        mda_section = self._extract_mda(content)
                    elif filing_type == "8-K":
                        items_reported = self._extract_8k_items(content)

                    return SECFiling(
                        symbol=symbol,
                        cik=cik,
                        filing_type=filing_type,
                        filing_date=datetime.fromisoformat(filing_date),
                        accession_number=filing_info.get("accessionNumber", ""),
                        content=content[:100000],  # Limit to 100k chars
                        risk_factors=risk_factors,
                        mda_section=mda_section,
                        items_reported=items_reported,
                        url=f"{url}/{primary_doc}",
                        source="sec_edgar",
                    )

        except Exception as e:
            logger.error(f"Filing content fetch error: {e}")
            return None

    async def _fetch_primary_document(
        self,
        session: aiohttp.ClientSession,
        url: str,
        symbol: str,
        cik: str,
        filing_info: Dict[str, Any],
        filing_type: str,
        headers: Dict[str, str],
    ) -> Optional[SECFiling]:
        """Try to fetch primary document directly."""
        # Try common document names
        doc_names = [
            f"{filing_type.lower().replace('-', '')}.htm",
            "0001.htm",
            "primary_doc.htm",
        ]

        for doc_name in doc_names:
            try:
                async with session.get(
                    f"{url}/{doc_name}", headers=headers, timeout=60
                ) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        content = self._extract_text_from_html(html_content)

                        return SECFiling(
                            symbol=symbol,
                            cik=cik,
                            filing_type=filing_type,
                            filing_date=datetime.fromisoformat(
                                filing_info.get("filedAt", datetime.now().isoformat()[:10])
                            ),
                            accession_number=filing_info.get("accessionNumber", ""),
                            content=content[:100000],
                            url=f"{url}/{doc_name}",
                            source="sec_edgar",
                        )
            except Exception:
                continue

        return None

    def _extract_text_from_html(self, html: str) -> str:
        """Extract text from HTML content."""
        # Remove scripts and styles
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)

        # Decode HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')

        return text.strip()

    def _extract_risk_factors(self, content: str) -> str:
        """Extract Risk Factors section from filing."""
        patterns = [
            r"Item\s*1A[\.\s]+Risk\s*Factors(.*?)Item\s*1B",
            r"RISK\s*FACTORS(.*?)(?:Item\s*1B|UNRESOLVED\s*STAFF\s*COMMENTS)",
            r"Risk\s*Factors(.*?)(?:Unresolved|Properties)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                risk_text = match.group(1).strip()
                # Limit to 30k chars
                return risk_text[:30000]

        return ""

    def _extract_mda(self, content: str) -> str:
        """Extract Management Discussion & Analysis section."""
        patterns = [
            r"Item\s*7[\.\s]+Management.s\s*Discussion(.*?)Item\s*7A",
            r"MANAGEMENT.S\s*DISCUSSION\s*AND\s*ANALYSIS(.*?)(?:Item\s*7A|QUANTITATIVE)",
            r"Management.s\s*Discussion(.*?)(?:Quantitative|Market\s*Risk)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                mda_text = match.group(1).strip()
                return mda_text[:30000]

        return ""

    def _extract_8k_items(self, content: str) -> List[str]:
        """Extract reported items from 8-K filing."""
        items = []

        for item_code in self.ITEM_8K_CODES.keys():
            pattern = rf"Item\s*{re.escape(item_code)}"
            if re.search(pattern, content, re.IGNORECASE):
                items.append(item_code)

        return items

    def _get_cached_latest(
        self,
        symbol: str,
        filing_type: str,
    ) -> Optional[SECFiling]:
        """Get most recent cached filing."""
        cutoff = datetime.now() - timedelta(days=self.CACHE_TTL_DAYS)

        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute(
                """
                SELECT symbol, cik, filing_type, filing_date, accession_number,
                       content, risk_factors, mda_section, business_section,
                       items_reported, company_name, fiscal_year_end, word_count,
                       url, source
                FROM filings
                WHERE symbol = ? AND filing_type = ? AND fetched_at > ?
                ORDER BY filing_date DESC
                LIMIT 1
                """,
                (symbol, filing_type, cutoff.isoformat())
            ).fetchone()

            if result:
                return self._row_to_filing(result)

        return None

    def _row_to_filing(self, row: tuple) -> SECFiling:
        """Convert database row to SECFiling."""
        items = json.loads(row[9]) if row[9] else []

        return SECFiling(
            symbol=row[0],
            cik=row[1],
            filing_type=row[2],
            filing_date=datetime.fromisoformat(row[3]),
            accession_number=row[4],
            content=row[5] or "",
            risk_factors=row[6] or "",
            mda_section=row[7] or "",
            business_section=row[8] or "",
            items_reported=items,
            company_name=row[10] or "",
            fiscal_year_end=row[11] or "",
            word_count=row[12] or 0,
            url=row[13] or "",
            source=row[14] or "",
            cached=True,
        )

    def _cache_filing(self, filing: SECFiling) -> None:
        """Cache filing to database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO filings
                (symbol, cik, filing_type, filing_date, accession_number,
                 content, risk_factors, mda_section, business_section,
                 items_reported, company_name, fiscal_year_end, word_count,
                 url, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filing.symbol,
                    filing.cik,
                    filing.filing_type,
                    filing.filing_date.isoformat(),
                    filing.accession_number,
                    filing.content,
                    filing.risk_factors,
                    filing.mda_section,
                    filing.business_section,
                    json.dumps(filing.items_reported),
                    filing.company_name,
                    filing.fiscal_year_end,
                    filing.word_count,
                    filing.url,
                    filing.source,
                    datetime.now().isoformat(),
                )
            )
            conn.commit()

    def _generate_mock_filing(
        self,
        symbol: str,
        filing_type: str,
    ) -> Optional[SECFiling]:
        """Generate mock filing for testing."""
        now = datetime.now()

        if filing_type == "10-K":
            content = f"""
{symbol} Annual Report (Form 10-K)
Fiscal Year Ended December 31, {now.year - 1}

PART I

Item 1. Business

{symbol} is a leading company in its industry. The company has shown consistent growth
and continues to invest in innovation and expansion.

Item 1A. Risk Factors

The following risk factors should be carefully considered:

1. Competition Risk: The company faces significant competition from established players
and new entrants in the market.

2. Regulatory Risk: Changes in government regulations could adversely affect operations.

3. Economic Risk: Economic downturns could reduce consumer spending and impact revenue.

4. Technology Risk: Rapid technological change requires continuous investment in R&D.

5. Cybersecurity Risk: Potential security breaches could harm reputation and operations.

PART II

Item 7. Management's Discussion and Analysis

Results of Operations:
Revenue increased by X% compared to the prior year. Gross margins improved due to
operational efficiencies. Operating expenses were well-controlled.

Liquidity and Capital Resources:
The company maintains a strong balance sheet with adequate liquidity. Cash flows from
operations remain healthy and support ongoing investment needs.

Outlook:
Management remains optimistic about future growth opportunities while monitoring
potential headwinds from macroeconomic conditions.
            """.strip()

            return SECFiling(
                symbol=symbol,
                cik="0000000000",
                filing_type="10-K",
                filing_date=now - timedelta(days=60),
                accession_number="0000000000-00-000001",
                content=content,
                risk_factors=content[content.find("Risk Factors"):content.find("PART II")],
                mda_section=content[content.find("Management's Discussion"):],
                source="mock",
            )

        elif filing_type == "8-K":
            content = f"""
{symbol} Current Report (Form 8-K)
Date of Report: {now.strftime('%B %d, %Y')}

Item 2.02 Results of Operations and Financial Condition

On {now.strftime('%B %d, %Y')}, {symbol} issued a press release announcing its financial
results for the quarter ended {now.strftime('%B %d, %Y')}.

The company reported revenue of $X billion, exceeding analyst expectations.
Earnings per share came in at $X.XX, compared to $X.XX in the prior year quarter.

Management provided guidance for the upcoming quarter and full year.

Item 9.01 Financial Statements and Exhibits

The press release is furnished as Exhibit 99.1.
            """.strip()

            return SECFiling(
                symbol=symbol,
                cik="0000000000",
                filing_type="8-K",
                filing_date=now - timedelta(days=5),
                accession_number="0000000000-00-000002",
                content=content,
                items_reported=["2.02", "9.01"],
                source="mock",
            )

        return None
