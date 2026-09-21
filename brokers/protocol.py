"""The broker seam: the one surface both brokers present to strategies and engines.

Two adapters satisfy it, ``AlpacaBroker`` (paper) and ``BacktestBroker``
(simulator). Everything a strategy, gateway or engine needs from "the broker"
is here; anything else on either class is that adapter's internal seam.
See docs/adr/0003-broker-seam-is-a-stated-protocol-with-two-adapters.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Position:
    """A signed quantity of one symbol held at the broker: positive long, negative short."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0

    @property
    def side(self) -> str:
        return "short" if self.qty < 0 else "long"

    @property
    def quantity(self) -> float:
        """Alias kept for callers that predate the protocol; prefer ``qty``."""
        return self.qty


@runtime_checkable
class Broker(Protocol):
    """What both brokers guarantee. All calls are async; positions are ``Position``."""

    async def get_positions(self) -> List[Position]: ...

    async def get_position(self, symbol: str) -> Optional[Position]: ...

    async def get_account(self) -> Any: ...

    async def submit_order_advanced(self, order_request: Any) -> Any: ...
