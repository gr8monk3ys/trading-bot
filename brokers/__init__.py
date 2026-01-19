"""
Broker Integration Package

Provides broker integrations for trading:
- Alpaca: Integration with Alpaca trading platform

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
    "AlpacaBroker",
    "BrokerError",
    "BrokerConnectionError",
    "OrderError",
]
