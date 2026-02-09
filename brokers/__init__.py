"""
Broker Integration Package

Provides broker integrations for trading:
- AlpacaBroker: Integration with Alpaca trading platform for stocks and crypto
- OptionsBroker: Options trading support via Alpaca (requires options approval)

Exception Classes:
- BrokerError: Base exception for all broker errors
- BrokerConnectionError: Connection failures
- OrderError: Order submission/modification failures
- OptionsError: Options-specific errors
- InvalidContractError: Invalid option contract specification
"""

from brokers.alpaca_broker import (
    AlpacaBroker,
    BrokerConnectionError,
    BrokerError,
    OrderError,
)
from brokers.options_broker import (
    InvalidContractError,
    OptionChain,
    OptionContract,
    OptionsBroker,
    OptionsError,
    OptionType,
)

__all__ = [
    # Stock/Crypto Broker
    "AlpacaBroker",
    # Options Broker
    "OptionsBroker",
    "OptionContract",
    "OptionChain",
    "OptionType",
    # Exceptions
    "BrokerError",
    "BrokerConnectionError",
    "OrderError",
    "OptionsError",
    "InvalidContractError",
]
