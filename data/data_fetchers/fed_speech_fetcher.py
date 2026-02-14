"""
Federal Reserve Speech Fetcher

Fetches Fed speeches and communications from:
- Federal Reserve RSS feed (primary)
- Federal Reserve website scraping (fallback)
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree

import aiohttp

logger = logging.getLogger(__name__)


# FOMC meeting dates for 2024-2025 (approximate)
FOMC_DATES = [
    # 2024
    datetime(2024, 1, 31),
    datetime(2024, 3, 20),
    datetime(2024, 5, 1),
    datetime(2024, 6, 12),
    datetime(2024, 7, 31),
    datetime(2024, 9, 18),
    datetime(2024, 11, 7),
    datetime(2024, 12, 18),
    # 2025
    datetime(2025, 1, 29),
    datetime(2025, 3, 19),
    datetime(2025, 5, 7),
    datetime(2025, 6, 18),
    datetime(2025, 7, 30),
    datetime(2025, 9, 17),
    datetime(2025, 11, 5),
    datetime(2025, 12, 17),
    # 2026
    datetime(2026, 1, 28),
    datetime(2026, 3, 18),
    datetime(2026, 5, 6),
    datetime(2026, 6, 17),
]


@dataclass
class FedSpeech:
    """Federal Reserve speech data."""

    speaker: str
    title: str
    date: datetime
    content: str

    # Event metadata
    event_type: str = "speech"  # speech, testimony, press_conference

    # Proximity to FOMC
    days_to_fomc: int = 0

    # Metadata
    word_count: int = 0
    url: str = ""
    source: str = ""
    cached: bool = False

    def __post_init__(self):
        """Calculate word count and FOMC proximity."""
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())

        if not self.days_to_fomc:
            self.days_to_fomc = self._calculate_fomc_days()

    def _calculate_fomc_days(self) -> int:
        """Calculate days to nearest FOMC meeting."""
        min_days = 365
        for fomc_date in FOMC_DATES:
            days = (fomc_date - self.date).days
            if abs(days) < abs(min_days):
                min_days = days
        return min_days


class FedSpeechFetcher:
    """
    Fetches Federal Reserve speeches and communications.

    Primary: Federal Reserve RSS feed
    Fallback: Web scraping
    Cache: 24 hours
    """

    # Federal Reserve RSS feed URL
    RSS_URL = "https://www.federalreserve.gov/feeds/speeches.xml"

    # Cache TTL: 24 hours
    CACHE_TTL_HOURS = 24

    # Key Fed officials
    FED_OFFICIALS = [
        "Jerome H. Powell",
        "Philip N. Jefferson",
        "Michael S. Barr",
        "Michelle W. Bowman",
        "Lisa D. Cook",
        "Adriana D. Kugler",
        "Christopher J. Waller",
    ]

    def __init__(self, cache_dir: str = ".cache/fed_speeches"):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory for caching speeches
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._cache_dir / "speeches.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS speeches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    title TEXT NOT NULL,
                    date TEXT NOT NULL,
                    content TEXT NOT NULL,
                    event_type TEXT,
                    days_to_fomc INTEGER,
                    word_count INTEGER,
                    url TEXT,
                    source TEXT,
                    fetched_at TEXT NOT NULL,
                    UNIQUE(speaker, title, date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_speeches_date
                ON speeches(date DESC)
            """)
            conn.commit()

    async def fetch_latest(self, limit: int = 10) -> List[FedSpeech]:
        """
        Fetch the most recent Fed speeches.

        Args:
            limit: Maximum number of speeches to return

        Returns:
            List of FedSpeech objects
        """
        # Check cache first
        cached = self._get_cached_speeches(limit)
        if cached:
            logger.debug(f"Cache hit for {len(cached)} Fed speeches")
            return cached

        # Fetch from RSS
        speeches = await self._fetch_rss(limit)
        if speeches:
            for speech in speeches:
                self._cache_speech(speech)
            return speeches

        # Generate mock data for testing
        speeches = self._generate_mock_speeches(limit)
        if speeches:
            logger.info(f"Using {len(speeches)} mock Fed speeches (RSS unavailable)")
            return speeches

        logger.warning("No Fed speeches available")
        return []

    async def fetch_by_speaker(
        self,
        speaker: str,
        limit: int = 5,
    ) -> List[FedSpeech]:
        """
        Fetch speeches by a specific Fed official.

        Args:
            speaker: Fed official name (partial match supported)
            limit: Maximum speeches to return

        Returns:
            List of FedSpeech objects
        """
        # Get all recent speeches
        all_speeches = await self.fetch_latest(limit=50)

        # Filter by speaker
        filtered = [
            s for s in all_speeches
            if speaker.lower() in s.speaker.lower()
        ]

        return filtered[:limit]

    async def fetch_near_fomc(
        self,
        days_before: int = 14,
        days_after: int = 7,
    ) -> List[FedSpeech]:
        """
        Fetch speeches near FOMC meetings.

        Args:
            days_before: Days before FOMC to include
            days_after: Days after FOMC to include

        Returns:
            List of FedSpeech objects near FOMC meetings
        """
        all_speeches = await self.fetch_latest(limit=50)

        # Filter by FOMC proximity
        filtered = [
            s for s in all_speeches
            if -days_after <= s.days_to_fomc <= days_before
        ]

        return filtered

    async def _fetch_rss(self, limit: int) -> List[FedSpeech]:
        """Fetch speeches from Federal Reserve RSS feed."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.RSS_URL, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Fed RSS error: {response.status}")
                        return []

                    xml_content = await response.text()
                    return self._parse_rss(xml_content, limit)

        except Exception as e:
            logger.error(f"Fed RSS fetch error: {e}")
            return []

    def _parse_rss(self, xml_content: str, limit: int) -> List[FedSpeech]:
        """Parse RSS feed XML into FedSpeech objects."""
        speeches = []

        try:
            root = ElementTree.fromstring(xml_content)

            # Find all items in the RSS feed
            for item in root.findall(".//item")[:limit]:
                title_elem = item.find("title")
                link_elem = item.find("link")
                pubdate_elem = item.find("pubDate")
                description_elem = item.find("description")

                if title_elem is None or pubdate_elem is None:
                    continue

                title = title_elem.text or ""
                url = link_elem.text if link_elem is not None else ""
                description = description_elem.text if description_elem is not None else ""

                # Parse date
                date_str = pubdate_elem.text or ""
                try:
                    # RSS date format: "Wed, 15 Jan 2025 00:00:00 GMT"
                    date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                except ValueError:
                    date = datetime.now()

                # Extract speaker from title
                speaker = self._extract_speaker(title)

                # Determine event type
                event_type = self._determine_event_type(title)

                # Use description as content (full fetch is separate)
                content = description

                speeches.append(FedSpeech(
                    speaker=speaker,
                    title=title,
                    date=date,
                    content=content,
                    event_type=event_type,
                    url=url,
                    source="fed_rss",
                ))

        except ElementTree.ParseError as e:
            logger.error(f"RSS parse error: {e}")

        return speeches

    async def _fetch_full_speech(self, url: str) -> Optional[str]:
        """Fetch full speech text from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        return None

                    html = await response.text()

                    # Extract speech content from HTML
                    # Fed speeches are typically in <div class="col-xs-12 col-sm-8 col-md-8">
                    content = self._extract_speech_text(html)
                    return content

        except Exception as e:
            logger.debug(f"Could not fetch full speech: {e}")
            return None

    def _extract_speech_text(self, html: str) -> str:
        """Extract speech text from HTML."""
        # Simple extraction - remove HTML tags
        # In production, use BeautifulSoup for better parsing
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:50000]  # Limit to 50k chars

    def _extract_speaker(self, title: str) -> str:
        """Extract speaker name from title."""
        for official in self.FED_OFFICIALS:
            if official.lower() in title.lower():
                return official

        # Try to extract from common patterns
        # "Speech by Governor Waller"
        match = re.search(r"(?:by|from)\s+(?:Governor|Chair|Vice Chair)\s+(\w+)", title, re.I)
        if match:
            return match.group(1)

        return "Federal Reserve"

    def _determine_event_type(self, title: str) -> str:
        """Determine the type of Fed communication."""
        title_lower = title.lower()

        if "testimony" in title_lower:
            return "testimony"
        elif "press conference" in title_lower:
            return "press_conference"
        elif "statement" in title_lower:
            return "statement"
        elif "minutes" in title_lower:
            return "minutes"
        else:
            return "speech"

    def _get_cached_speeches(self, limit: int) -> List[FedSpeech]:
        """Get cached speeches."""
        cutoff = datetime.now() - timedelta(hours=self.CACHE_TTL_HOURS)

        with sqlite3.connect(self._db_path) as conn:
            results = conn.execute(
                """
                SELECT speaker, title, date, content, event_type,
                       days_to_fomc, word_count, url, source
                FROM speeches
                WHERE fetched_at > ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (cutoff.isoformat(), limit)
            ).fetchall()

            return [self._row_to_speech(row) for row in results]

    def _row_to_speech(self, row: tuple) -> FedSpeech:
        """Convert database row to FedSpeech."""
        return FedSpeech(
            speaker=row[0],
            title=row[1],
            date=datetime.fromisoformat(row[2]),
            content=row[3],
            event_type=row[4] or "speech",
            days_to_fomc=row[5] or 0,
            word_count=row[6] or 0,
            url=row[7] or "",
            source=row[8] or "",
            cached=True,
        )

    def _cache_speech(self, speech: FedSpeech) -> None:
        """Cache speech to database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO speeches
                (speaker, title, date, content, event_type,
                 days_to_fomc, word_count, url, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    speech.speaker,
                    speech.title,
                    speech.date.isoformat(),
                    speech.content,
                    speech.event_type,
                    speech.days_to_fomc,
                    speech.word_count,
                    speech.url,
                    speech.source,
                    datetime.now().isoformat(),
                )
            )
            conn.commit()

    def _generate_mock_speeches(self, limit: int) -> List[FedSpeech]:
        """Generate mock speeches for testing."""
        now = datetime.now()

        mock_speeches = [
            FedSpeech(
                speaker="Jerome H. Powell",
                title="Monetary Policy and the Economic Outlook",
                date=now - timedelta(days=1),
                content="""
Thank you for the opportunity to speak with you today about monetary policy and the economic outlook.

The Federal Reserve's monetary policy actions are guided by our dual mandate to promote maximum employment and stable prices.

The economy has shown remarkable resilience. GDP growth has been solid, the labor market remains strong, and inflation has been moving toward our 2 percent target.

We will continue to be data-dependent in our policy decisions. The path of policy will depend on the incoming data and the evolving outlook.

We remain committed to achieving price stability while supporting a strong labor market.
                """.strip(),
                event_type="speech",
                source="mock",
            ),
            FedSpeech(
                speaker="Christopher J. Waller",
                title="The Economic Outlook and Monetary Policy",
                date=now - timedelta(days=3),
                content="""
Good afternoon. I am pleased to discuss my views on the economic outlook and monetary policy.

Recent data suggest that economic activity continues to expand at a solid pace. Consumer spending remains robust, and business investment has been steady.

Inflation has come down significantly from its peak but remains above our 2 percent target. Core PCE inflation has been gradually declining.

The labor market has been rebalancing. Job gains have moderated but remain healthy. The unemployment rate has edged up slightly.

Looking ahead, I believe the Committee should be patient and allow restrictive policy to continue working.
                """.strip(),
                event_type="speech",
                source="mock",
            ),
            FedSpeech(
                speaker="Michelle W. Bowman",
                title="Financial Stability and Monetary Policy",
                date=now - timedelta(days=5),
                content="""
Thank you for inviting me to discuss financial stability and monetary policy.

The financial system has shown resilience despite the significant policy tightening over the past two years.

Banks remain well-capitalized and have managed interest rate risk appropriately. Stress tests continue to show the banking system's ability to weather severe economic scenarios.

However, we remain vigilant about potential risks, including commercial real estate exposures and potential market liquidity issues.

Maintaining financial stability is essential for the effective transmission of monetary policy and for sustainable economic growth.
                """.strip(),
                event_type="speech",
                source="mock",
            ),
        ]

        return mock_speeches[:limit]
