"""
AlpacaBroker sub-package.

The original `brokers/alpaca_broker.py` (~2,977 LOC) was split into focused
mixin modules that the top-level `AlpacaBroker` class composes via multiple
inheritance:

    - account.py       — connection/auth helpers, account/positions/asset queries
    - orders.py        — order submission, modification, query, market impact
    - market_data.py   — stock bars/quotes/news
    - crypto.py        — 24/7 crypto bars/quotes/orders/positions
    - streaming.py     — websocket handlers + WebSocketManager-backed streaming
    - portfolio.py     — portfolio history / equity curve / performance summary

Shared utilities live in `_retry.py` (retry decorator, exception hierarchy).

External callers continue to import `AlpacaBroker` and the exception classes
from `brokers.alpaca_broker` — this sub-package is an internal implementation
detail.
"""

from brokers.alpaca._retry import (
    BrokerConnectionError,
    BrokerError,
    GatewayBypassError,
    OrderError,
    retry_with_backoff,
)
from brokers.alpaca.account import AlpacaAccountMixin
from brokers.alpaca.crypto import AlpacaCryptoMixin
from brokers.alpaca.market_data import AlpacaMarketDataMixin
from brokers.alpaca.orders import AlpacaOrdersMixin
from brokers.alpaca.portfolio import AlpacaPortfolioMixin
from brokers.alpaca.streaming import AlpacaStreamingMixin

__all__ = [
    "AlpacaAccountMixin",
    "AlpacaCryptoMixin",
    "AlpacaMarketDataMixin",
    "AlpacaOrdersMixin",
    "AlpacaPortfolioMixin",
    "AlpacaStreamingMixin",
    "BrokerError",
    "BrokerConnectionError",
    "OrderError",
    "GatewayBypassError",
    "retry_with_backoff",
]
