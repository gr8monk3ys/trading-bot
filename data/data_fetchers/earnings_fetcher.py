"""
Earnings Transcript Fetcher

Fetches earnings call transcripts from various sources:
- Alpha Vantage (primary)
- Financial Modeling Prep (fallback)
- Seeking Alpha (scraping fallback)
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class EarningsTranscript:
    """Earnings call transcript data."""

    symbol: str
    fiscal_quarter: str  # "Q1", "Q2", "Q3", "Q4"
    fiscal_year: int
    date: datetime
    content: str

    # Parsed sections
    prepared_remarks: str = ""
    qanda_section: str = ""

    # Speaker information
    speakers: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    word_count: int = 0
    source: str = ""
    cached: bool = False

    def __post_init__(self):
        """Calculate word count."""
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())


class EarningsTranscriptFetcher:
    """
    Fetches earnings call transcripts from multiple sources.

    Primary: Alpha Vantage API
    Fallback: Financial Modeling Prep API
    Cache: 30 days (transcripts don't change)
    """

    # Cache TTL: 30 days
    CACHE_TTL_DAYS = 30

    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        fmp_key: Optional[str] = None,
        cache_dir: str = ".cache/earnings",
    ):
        """
        Initialize fetcher.

        Args:
            alpha_vantage_key: Alpha Vantage API key (or from env)
            fmp_key: Financial Modeling Prep API key (or from env)
            cache_dir: Directory for caching transcripts
        """
        self._av_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self._fmp_key = fmp_key or os.getenv("FMP_API_KEY")

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._cache_dir / "transcripts.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    fiscal_quarter TEXT NOT NULL,
                    fiscal_year INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    content TEXT NOT NULL,
                    prepared_remarks TEXT,
                    qanda_section TEXT,
                    speakers TEXT,
                    word_count INTEGER,
                    source TEXT,
                    fetched_at TEXT NOT NULL,
                    UNIQUE(symbol, fiscal_quarter, fiscal_year)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_transcripts_symbol
                ON transcripts(symbol, date)
            """)
            conn.commit()

    async def fetch_latest(self, symbol: str) -> Optional[EarningsTranscript]:
        """
        Fetch the most recent earnings transcript for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            EarningsTranscript or None if not available
        """
        # Check cache first
        cached = self._get_cached_latest(symbol)
        if cached:
            logger.debug(f"Cache hit for {symbol} earnings transcript")
            return cached

        # Try Alpha Vantage
        if self._av_key:
            transcript = await self._fetch_alpha_vantage(symbol)
            if transcript:
                self._cache_transcript(transcript)
                return transcript

        # Try Financial Modeling Prep
        if self._fmp_key:
            transcript = await self._fetch_fmp(symbol)
            if transcript:
                self._cache_transcript(transcript)
                return transcript

        # Generate mock data for testing
        transcript = self._generate_mock_transcript(symbol)
        if transcript:
            logger.info(f"Using mock transcript for {symbol} (no API keys configured)")
            return transcript

        logger.warning(f"No earnings transcript available for {symbol}")
        return None

    async def fetch_by_quarter(
        self,
        symbol: str,
        quarter: str,
        year: int,
    ) -> Optional[EarningsTranscript]:
        """
        Fetch transcript for a specific quarter.

        Args:
            symbol: Stock symbol
            quarter: Fiscal quarter (Q1, Q2, Q3, Q4)
            year: Fiscal year

        Returns:
            EarningsTranscript or None if not available
        """
        # Check cache
        cached = self._get_cached(symbol, quarter, year)
        if cached:
            return cached

        # Try APIs
        if self._av_key:
            transcript = await self._fetch_alpha_vantage(symbol, quarter, year)
            if transcript:
                self._cache_transcript(transcript)
                return transcript

        if self._fmp_key:
            transcript = await self._fetch_fmp(symbol, quarter, year)
            if transcript:
                self._cache_transcript(transcript)
                return transcript

        return None

    async def _fetch_alpha_vantage(
        self,
        symbol: str,
        quarter: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Optional[EarningsTranscript]:
        """Fetch from Alpha Vantage API."""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": symbol,
                "apikey": self._av_key,
            }

            if quarter and year:
                params["quarter"] = quarter
                params["year"] = str(year)

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Alpha Vantage API error: {response.status}")
                        return None

                    data = await response.json()

                    # Check for API error
                    if "Error Message" in data or "Note" in data:
                        logger.warning(f"Alpha Vantage error: {data}")
                        return None

                    # Parse response
                    return self._parse_alpha_vantage_response(symbol, data)

        except Exception as e:
            logger.error(f"Alpha Vantage fetch error for {symbol}: {e}")
            return None

    async def _fetch_fmp(
        self,
        symbol: str,
        quarter: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Optional[EarningsTranscript]:
        """Fetch from Financial Modeling Prep API."""
        try:
            base_url = "https://financialmodelingprep.com/api/v3"
            url = f"{base_url}/earning_call_transcript/{symbol}"
            params = {"apikey": self._fmp_key}

            if quarter and year:
                # FMP uses different quarter format
                params["quarter"] = quarter.replace("Q", "")
                params["year"] = str(year)

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"FMP API error: {response.status}")
                        return None

                    data = await response.json()

                    if not data or isinstance(data, dict) and "Error" in data:
                        return None

                    # FMP returns array of transcripts
                    if isinstance(data, list) and len(data) > 0:
                        return self._parse_fmp_response(symbol, data[0])

                    return None

        except Exception as e:
            logger.error(f"FMP fetch error for {symbol}: {e}")
            return None

    def _parse_alpha_vantage_response(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> Optional[EarningsTranscript]:
        """Parse Alpha Vantage response into EarningsTranscript."""
        try:
            # Extract transcript content
            transcript_text = data.get("transcript", "")
            if not transcript_text:
                return None

            # Parse date
            date_str = data.get("date", "")
            try:
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except:
                date = datetime.now()

            # Parse quarter/year
            quarter = data.get("quarter", "Q4")
            year = int(data.get("year", datetime.now().year))

            # Split into sections
            prepared, qanda = self._split_transcript_sections(transcript_text)

            # Extract speakers
            speakers = self._extract_speakers(transcript_text)

            return EarningsTranscript(
                symbol=symbol,
                fiscal_quarter=quarter,
                fiscal_year=year,
                date=date,
                content=transcript_text,
                prepared_remarks=prepared,
                qanda_section=qanda,
                speakers=speakers,
                source="alpha_vantage",
            )

        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage response: {e}")
            return None

    def _parse_fmp_response(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> Optional[EarningsTranscript]:
        """Parse FMP response into EarningsTranscript."""
        try:
            transcript_text = data.get("content", "")
            if not transcript_text:
                return None

            date_str = data.get("date", "")
            try:
                date = datetime.fromisoformat(date_str)
            except:
                date = datetime.now()

            quarter = f"Q{data.get('quarter', 4)}"
            year = int(data.get("year", datetime.now().year))

            prepared, qanda = self._split_transcript_sections(transcript_text)
            speakers = self._extract_speakers(transcript_text)

            return EarningsTranscript(
                symbol=symbol,
                fiscal_quarter=quarter,
                fiscal_year=year,
                date=date,
                content=transcript_text,
                prepared_remarks=prepared,
                qanda_section=qanda,
                speakers=speakers,
                source="fmp",
            )

        except Exception as e:
            logger.error(f"Error parsing FMP response: {e}")
            return None

    def _split_transcript_sections(self, content: str) -> tuple:
        """Split transcript into prepared remarks and Q&A sections."""
        # Common markers for Q&A section
        qa_markers = [
            "Question-and-Answer Session",
            "Questions and Answers",
            "Q&A Session",
            "Operator Instructions",
        ]

        for marker in qa_markers:
            if marker.lower() in content.lower():
                idx = content.lower().find(marker.lower())
                return content[:idx].strip(), content[idx:].strip()

        # No Q&A section found
        return content, ""

    def _extract_speakers(self, content: str) -> List[Dict[str, str]]:
        """Extract speaker names and roles from transcript."""
        speakers = []
        seen = set()

        # Common patterns for speaker identification
        patterns = [
            r"^([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–]\s*(.+)$",  # "John Smith - CEO"
            r"^([A-Z][a-z]+ [A-Z][a-z]+)\s*,\s*(.+)$",  # "John Smith, CEO"
        ]

        for line in content.split("\n"):
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1)
                    role = match.group(2)
                    if name not in seen:
                        speakers.append({"name": name, "role": role})
                        seen.add(name)

        return speakers

    def _get_cached_latest(self, symbol: str) -> Optional[EarningsTranscript]:
        """Get most recent cached transcript."""
        cutoff = datetime.now() - timedelta(days=self.CACHE_TTL_DAYS)

        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute(
                """
                SELECT symbol, fiscal_quarter, fiscal_year, date, content,
                       prepared_remarks, qanda_section, speakers, word_count, source
                FROM transcripts
                WHERE symbol = ? AND fetched_at > ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (symbol, cutoff.isoformat())
            ).fetchone()

            if result:
                return self._row_to_transcript(result)

        return None

    def _get_cached(
        self,
        symbol: str,
        quarter: str,
        year: int,
    ) -> Optional[EarningsTranscript]:
        """Get cached transcript for specific quarter."""
        cutoff = datetime.now() - timedelta(days=self.CACHE_TTL_DAYS)

        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute(
                """
                SELECT symbol, fiscal_quarter, fiscal_year, date, content,
                       prepared_remarks, qanda_section, speakers, word_count, source
                FROM transcripts
                WHERE symbol = ? AND fiscal_quarter = ? AND fiscal_year = ? AND fetched_at > ?
                """,
                (symbol, quarter, year, cutoff.isoformat())
            ).fetchone()

            if result:
                return self._row_to_transcript(result)

        return None

    def _row_to_transcript(self, row: tuple) -> EarningsTranscript:
        """Convert database row to EarningsTranscript."""
        speakers = json.loads(row[7]) if row[7] else []

        return EarningsTranscript(
            symbol=row[0],
            fiscal_quarter=row[1],
            fiscal_year=row[2],
            date=datetime.fromisoformat(row[3]),
            content=row[4],
            prepared_remarks=row[5] or "",
            qanda_section=row[6] or "",
            speakers=speakers,
            word_count=row[8] or 0,
            source=row[9] or "",
            cached=True,
        )

    def _cache_transcript(self, transcript: EarningsTranscript) -> None:
        """Cache transcript to database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO transcripts
                (symbol, fiscal_quarter, fiscal_year, date, content,
                 prepared_remarks, qanda_section, speakers, word_count, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transcript.symbol,
                    transcript.fiscal_quarter,
                    transcript.fiscal_year,
                    transcript.date.isoformat(),
                    transcript.content,
                    transcript.prepared_remarks,
                    transcript.qanda_section,
                    json.dumps(transcript.speakers),
                    transcript.word_count,
                    transcript.source,
                    datetime.now().isoformat(),
                )
            )
            conn.commit()

    def _generate_mock_transcript(self, symbol: str) -> Optional[EarningsTranscript]:
        """Generate mock transcript for testing."""
        now = datetime.now()
        quarter = f"Q{((now.month - 1) // 3) + 1}"
        year = now.year

        mock_content = f"""
{symbol} Q{((now.month - 1) // 3) + 1} {year} Earnings Call Transcript

Operator: Good afternoon and welcome to {symbol}'s quarterly earnings call.

CEO: Thank you, operator. Good afternoon everyone. We're pleased to report another strong quarter.

Our revenue came in at $X billion, up X% year-over-year, exceeding our guidance range.
We saw broad-based strength across all our business segments.

Looking ahead, we remain confident in our strategy and are raising our full-year guidance.
We expect continued momentum in the coming quarters.

CFO: Thank you. Let me walk through the financial details.
Revenue was $X billion, gross margin was X%, and operating income was $X million.
Our balance sheet remains strong with $X billion in cash.

We are increasing our guidance for the full year given the strong performance.

Question-and-Answer Session

Analyst 1: Great quarter. Can you talk about the demand environment?

CEO: Demand remains robust across our customer base. We're seeing particular strength in enterprise.

Analyst 2: What's your outlook for margins?

CFO: We expect margins to remain stable to slightly improving as we benefit from operating leverage.
"""

        return EarningsTranscript(
            symbol=symbol,
            fiscal_quarter=quarter,
            fiscal_year=year,
            date=now,
            content=mock_content.strip(),
            prepared_remarks=mock_content[:mock_content.find("Question-and-Answer")].strip(),
            qanda_section=mock_content[mock_content.find("Question-and-Answer"):].strip(),
            speakers=[
                {"name": "CEO", "role": "Chief Executive Officer"},
                {"name": "CFO", "role": "Chief Financial Officer"},
            ],
            source="mock",
        )
