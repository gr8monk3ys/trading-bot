"""
AlpacaBroker — thin facade composing the sub-client mixins.

The implementation lives in `brokers/alpaca/`:

    - brokers/alpaca/account.py       — connection / auth / account / positions
    - brokers/alpaca/orders.py        — submission, cancel, replace, query, impact
    - brokers/alpaca/market_data.py   — stock bars / quotes / news
    - brokers/alpaca/crypto.py        — 24/7 crypto bars / quotes / orders
    - brokers/alpaca/streaming.py     — websocket handlers + subscriptions
    - brokers/alpaca/portfolio.py     — portfolio history / equity curve

The exception hierarchy and `retry_with_backoff` decorator live in
`brokers/alpaca/_retry.py` and are re-exported here so external callers can
continue to do `from brokers.alpaca_broker import AlpacaBroker, OrderError`.
"""

import asyncio
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.trading.client import TradingClient

# NOTE: Removed lumibot imports - they crash at import time due to
# lumibot.credentials.py trying to instantiate Alpaca broker before config is ready
# We don't actually need lumibot's Broker class - we built our own implementation
from config import ALPACA_CREDS, SYMBOLS
from utils.audit_log import AuditLog
from utils.order_lifecycle import OrderLifecycleTracker

from brokers.alpaca import (  # noqa: F401 - re-exported for callers
    AlpacaAccountMixin,
    AlpacaCryptoMixin,
    AlpacaMarketDataMixin,
    AlpacaOrdersMixin,
    AlpacaPortfolioMixin,
    AlpacaStreamingMixin,
    BrokerConnectionError,
    BrokerError,
    GatewayBypassError,
    OrderError,
    retry_with_backoff,
)
from brokers.alpaca._retry import DEBUG_MODE

logger = logging.getLogger(__name__)


__all__ = [
    "AlpacaBroker",
    "BrokerError",
    "BrokerConnectionError",
    "OrderError",
    "GatewayBypassError",
    "retry_with_backoff",
]


class AlpacaBroker(
    AlpacaAccountMixin,
    AlpacaOrdersMixin,
    AlpacaMarketDataMixin,
    AlpacaCryptoMixin,
    AlpacaStreamingMixin,
    AlpacaPortfolioMixin,
):
    """
    Alpaca broker implementation.

    Direct implementation without lumibot dependency to avoid import-time crashes.
    Provides all necessary broker functionality for live and paper trading.
    Supports both stocks and 24/7 cryptocurrency trading.

    Composed from six mixins (see module docstring). The public API is
    unchanged from the pre-split monolith — `broker.get_account()`,
    `broker.submit_order_advanced()`, `broker.get_bars()`, etc. all work
    exactly as before.
    """

    NAME = "alpaca"
    IS_BACKTESTING_BROKER = False

    # CRYPTO_PAIRS is now imported from utils.crypto_utils for consistency

    # Default timeout for API calls (in seconds)
    DEFAULT_API_TIMEOUT = 30.0
    # Timeout for data-heavy operations (bars, portfolio history)
    DATA_API_TIMEOUT = 60.0
    # Timeout for order operations (more critical, shorter timeout)
    ORDER_API_TIMEOUT = 15.0

    def __init__(self, paper=True, audit_log: Optional[AuditLog] = None):
        """Initialize the AlpacaBroker.

        Sets up the shared state (credentials, TradingClient, StockDataStream,
        crypto clients, partial-fill tracker) that the mixin methods rely on.
        """
        try:
            # Ensure paper is a boolean
            if isinstance(paper, str):
                paper = paper.lower() == "true"
            self.paper = bool(paper)
            logger.info(
                f"AlpacaBroker initialized with paper={self.paper} (type: {type(self.paper)})"
            )

            # Initialize position tracking
            self._filled_positions: List[Any] = []
            self._subscribers: set[object] = set()
            self._ws_lock = asyncio.Lock()
            self._ws_task: Optional[asyncio.Task] = None
            self._connected = False
            self._reconnect_attempts = 0
            self._reconnect_delay = 1  # Initial reconnect delay in seconds
            self._max_reconnect_delay = 60  # Max reconnect delay in seconds

            # P0 FIX: Use local variables for credentials instead of storing as attributes
            # This prevents accidental exposure through logging, serialization, or debugging
            _api_key = ALPACA_CREDS["API_KEY"]
            _api_secret = ALPACA_CREDS["API_SECRET"]

            if not _api_key or not _api_secret:
                raise ValueError(
                    "Alpaca API credentials not found. Please set them in your environment variables."
                )

            # Store credentials for crypto client initialization (lazy loaded)
            self._api_key = _api_key
            self._api_secret = _api_secret

            # Initialize the trading client
            self.trading_client = TradingClient(
                api_key=_api_key,
                secret_key=_api_secret,
                paper=self.paper,
                url_override="https://paper-api.alpaca.markets" if self.paper else None,
            )

            # Initialize the data client for stocks
            self.data_client = StockHistoricalDataClient(api_key=_api_key, secret_key=_api_secret)

            # Crypto clients (lazy initialized to avoid unnecessary connections)
            self._crypto_data_client: Optional[CryptoHistoricalDataClient] = None
            self._crypto_stream: Optional[CryptoDataStream] = None

            # Initialize the stream for WebSockets (stocks)
            self.stream = StockDataStream(
                api_key=_api_key,
                secret_key=_api_secret,
            )
            self._active_stream: Any = self.stream
            self._ws_asset_class: str = "stock"  # "stock" | "crypto"
            self._ws_symbols: List[str] = list(SYMBOLS)

            self._subscribed_symbols: set[str] = set()  # Keep track of subscribed symbols

            # Performance optimization: TTL-based price cache to reduce API calls
            self._price_cache: Dict[str, tuple[float, Any]] = {}  # {symbol: (price, ts)}
            self._price_cache_ttl = timedelta(seconds=5)  # Cache prices for 5 seconds

            # INSTITUTIONAL SAFETY: Gateway enforcement flag
            # When True, direct calls to submit_order_advanced() will raise GatewayBypassError
            # All orders must route through OrderGateway for safety checks
            self._gateway_required: bool = False  # Set to True after OrderGateway is initialized
            self._gateway_caller_token: Optional[str] = None  # Token for authorized gateway calls

            # INSTITUTIONAL SAFETY: Partial fill tracking
            # Tracks order fills and handles unfilled quantities
            from utils.partial_fill_tracker import PartialFillPolicy, PartialFillTracker

            self._partial_fill_tracker = PartialFillTracker(
                policy=PartialFillPolicy.ALERT_ONLY,  # Default to alerting
            )

            # Audit log (optional)
            self._audit_log = audit_log
            self._order_metadata: Dict[str, Dict[str, Any]] = {}
            self._lifecycle_tracker: Optional[OrderLifecycleTracker] = None
            self._position_manager = None

            # News client is lazy-loaded by get_news()
            self._news_client = None

            # P0 FIX: Removed unused config dict that stored credentials in memory

        except Exception as e:
            logger.error(f"Error initializing AlpacaBroker: {e}", exc_info=DEBUG_MODE)
            raise
