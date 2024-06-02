"""
Broker Integration Package

Provides broker integrations for trading:
- AlpacaBroker: Integration with Alpaca trading platform for stocks and crypto

Exception Classes:
- BrokerError: Base exception for all broker errors
- BrokerConnectionError: Connection failures
- OrderError: Order submission/modification failures
"""

from brokers.alpaca_broker import (
    AlpacaBroker,
    BrokerConnectionError,
    BrokerError,
    OrderError,
)

__all__ = [
    # Stock/Crypto Broker
    "AlpacaBroker",
    # Exceptions
    "BrokerError",
    "BrokerConnectionError",
    "OrderError",
]
