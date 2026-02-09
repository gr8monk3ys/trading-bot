"""
Smart Order Router (SOR)

Provides institutional-grade order routing across multiple venues:
1. Multi-Venue Access: NYSE, Nasdaq, IEX, BATS, EDGX, dark pools
2. Liquidity Analysis: Top-of-book and depth analysis
3. Fee Optimization: Consider maker/taker fees and rebates
4. Best Execution: Route to minimize market impact

Why SOR matters:
- Institutional traders check 25+ venues before routing
- Missing 50-200 bps on large orders without proper routing
- Citadel's SOR checks 500+ signals before each order
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Venue(Enum):
    """Trading venue identifiers."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    ARCA = "ARCA"
    BATS_BZX = "BATS_BZX"
    BATS_BYX = "BATS_BYX"
    EDGX = "EDGX"
    EDGA = "EDGA"
    IEX = "IEX"
    MEMX = "MEMX"

    # Dark pools
    SIGMA_X = "SIGMA_X"          # Goldman Sachs
    CROSSFINDER = "CROSSFINDER"  # Credit Suisse
    LEVEL_ATS = "LEVEL_ATS"      # Citadel
    MS_POOL = "MS_POOL"          # Morgan Stanley
    UBS_ATS = "UBS_ATS"          # UBS
    VIRTU_MATCHIT = "VIRTU_MATCHIT"  # Virtu

    # Retail/wholesale
    CITADEL_CONNECT = "CITADEL_CONNECT"
    VIRTU_AMERICAS = "VIRTU_AMERICAS"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class RoutingStrategy(Enum):
    """Order routing strategy."""
    BEST_PRICE = "best_price"           # Lowest ask / highest bid
    BEST_SIZE = "best_size"             # Most liquidity at NBBO
    LEAST_COST = "least_cost"           # Minimize total cost (fees + impact)
    PRICE_IMPROVEMENT = "price_improvement"  # Maximize price improvement
    MINIMIZE_INFORMATION = "minimize_information"  # Minimize information leakage
    PASSIVE = "passive"                 # Prioritize rebate capture
    AGGRESSIVE = "aggressive"           # Prioritize fill rate


@dataclass
class VenueQuote:
    """Quote from a single venue."""
    venue: Venue
    symbol: str
    timestamp: datetime

    # Prices
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int

    # Depth (if available)
    bid_depth: Optional[List[Tuple[float, int]]] = None  # [(price, size), ...]
    ask_depth: Optional[List[Tuple[float, int]]] = None

    # Venue characteristics
    is_lit: bool = True
    is_dark: bool = False
    supports_midpoint: bool = False

    @property
    def mid_price(self) -> float:
        """Mid-point price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid_price <= 0:
            return 0
        return (self.spread / self.mid_price) * 10000

    def available_size(self, side: OrderSide) -> int:
        """Size available at top of book."""
        return self.bid_size if side == OrderSide.SELL else self.ask_size

    def price_for_side(self, side: OrderSide) -> float:
        """Get relevant price for order side."""
        return self.ask_price if side == OrderSide.BUY else self.bid_price


@dataclass
class VenueFees:
    """Fee structure for a venue."""
    venue: Venue

    # Per-share fees (in dollars)
    maker_fee: float = 0.0      # Negative = rebate
    taker_fee: float = 0.0

    # Access fees
    access_fee: float = 0.0     # Fixed per order
    routing_fee: float = 0.0    # Fee for routing away

    # Minimum charges
    min_per_order: float = 0.0

    @classmethod
    def from_venue(cls, venue: Venue) -> "VenueFees":
        """Get standard fee schedule for venue."""
        # Simplified fee schedules (actual fees vary by tier)
        fee_schedules = {
            Venue.NYSE: cls(
                venue=venue,
                maker_fee=-0.0020,  # $0.20 rebate per 100 shares
                taker_fee=0.0030,   # $0.30 per 100 shares
            ),
            Venue.NASDAQ: cls(
                venue=venue,
                maker_fee=-0.0020,
                taker_fee=0.0030,
            ),
            Venue.IEX: cls(
                venue=venue,
                maker_fee=0.0,      # No rebate
                taker_fee=0.0009,   # Very low taker fee
            ),
            Venue.BATS_BZX: cls(
                venue=venue,
                maker_fee=-0.0025,  # Higher rebate
                taker_fee=0.0030,
            ),
            Venue.BATS_BYX: cls(
                venue=venue,
                maker_fee=0.0,      # Inverted pricing
                taker_fee=-0.0002,  # Taker rebate
            ),
            Venue.EDGX: cls(
                venue=venue,
                maker_fee=-0.0025,
                taker_fee=0.0030,
            ),
            Venue.EDGA: cls(
                venue=venue,
                maker_fee=0.0,
                taker_fee=0.0004,   # Low taker fee
            ),
        }
        return fee_schedules.get(venue, cls(venue=venue))

    def calculate_cost(
        self,
        shares: int,
        is_maker: bool,
    ) -> float:
        """Calculate total fee for order (negative = rebate)."""
        per_share = self.maker_fee if is_maker else self.taker_fee
        cost = per_share * shares + self.access_fee

        # Apply minimum only for positive costs (not rebates)
        if cost > 0:
            return max(cost, self.min_per_order)
        return cost


@dataclass
class NBBO:
    """National Best Bid and Offer."""
    symbol: str
    timestamp: datetime

    # Best prices
    best_bid: float
    best_ask: float
    best_bid_size: int
    best_ask_size: int

    # Which venues are at NBBO
    bid_venues: List[Venue] = field(default_factory=list)
    ask_venues: List[Venue] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        if self.mid_price <= 0:
            return 0
        return (self.spread / self.mid_price) * 10000


@dataclass
class RoutingDecision:
    """Decision for how to route an order."""
    venue: Venue
    quantity: int
    price: Optional[float]
    is_marketable: bool
    expected_fill_probability: float
    expected_cost: float  # Fees + estimated impact
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue": self.venue.value,
            "quantity": self.quantity,
            "price": self.price,
            "is_marketable": self.is_marketable,
            "expected_fill_probability": self.expected_fill_probability,
            "expected_cost": self.expected_cost,
            "reasoning": self.reasoning,
        }


@dataclass
class RouteResult:
    """Result of order routing."""
    success: bool
    decisions: List[RoutingDecision]
    total_quantity: int
    estimated_total_cost: float
    estimated_avg_price: float
    nbbo_at_decision: NBBO
    decision_time_us: int  # Microseconds

    # Analytics
    venues_considered: int
    venues_rejected: int
    price_improvement_expected: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "decisions": [d.to_dict() for d in self.decisions],
            "total_quantity": self.total_quantity,
            "estimated_total_cost": self.estimated_total_cost,
            "estimated_avg_price": self.estimated_avg_price,
            "decision_time_us": self.decision_time_us,
            "venues_considered": self.venues_considered,
            "price_improvement_expected": self.price_improvement_expected,
        }


class VenueConnector(ABC):
    """Abstract interface for venue connectivity."""

    @property
    @abstractmethod
    def venue(self) -> Venue:
        """Get venue identifier."""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[VenueQuote]:
        """Get current quote from venue."""
        pass

    @abstractmethod
    async def send_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Optional[float],
        order_type: str = "limit",
    ) -> Dict[str, Any]:
        """Send order to venue."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass


class MockVenueConnector(VenueConnector):
    """Mock venue connector for testing."""

    def __init__(
        self,
        venue: Venue,
        base_bid: float = 100.0,
        base_ask: float = 100.05,
        base_size: int = 500,
    ):
        self._venue = venue
        self.base_bid = base_bid
        self.base_ask = base_ask
        self.base_size = base_size
        self._orders: Dict[str, Dict] = {}

    @property
    def venue(self) -> Venue:
        return self._venue

    async def get_quote(self, symbol: str) -> Optional[VenueQuote]:
        """Generate mock quote with some randomness."""
        # Add slight venue-specific variation
        venue_offset = hash(self._venue.value) % 10 / 10000

        return VenueQuote(
            venue=self._venue,
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=self.base_bid + venue_offset,
            ask_price=self.base_ask + venue_offset,
            bid_size=self.base_size + (hash(self._venue.value) % 200),
            ask_size=self.base_size + (hash(self._venue.value) % 200),
            is_lit=self._venue not in [
                Venue.SIGMA_X, Venue.CROSSFINDER, Venue.LEVEL_ATS
            ],
            is_dark=self._venue in [
                Venue.SIGMA_X, Venue.CROSSFINDER, Venue.LEVEL_ATS
            ],
            supports_midpoint=self._venue in [
                Venue.IEX, Venue.SIGMA_X, Venue.CROSSFINDER
            ],
        )

    async def send_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Optional[float],
        order_type: str = "limit",
    ) -> Dict[str, Any]:
        """Mock order submission."""
        order_id = f"{self._venue.value}_{symbol}_{datetime.now().timestamp()}"

        self._orders[order_id] = {
            "order_id": order_id,
            "venue": self._venue.value,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "status": "accepted",
            "timestamp": datetime.now().isoformat(),
        }

        return self._orders[order_id]

    async def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation."""
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            return True
        return False


class SmartOrderRouter:
    """
    Smart Order Router for institutional-grade execution.

    Features:
    - Multi-venue quote aggregation
    - NBBO calculation
    - Routing strategy optimization
    - Fee-aware routing
    - Market impact estimation
    """

    def __init__(
        self,
        connectors: Optional[Dict[Venue, VenueConnector]] = None,
        default_strategy: RoutingStrategy = RoutingStrategy.LEAST_COST,
        max_venues_per_order: int = 5,
        min_fill_probability: float = 0.5,
    ):
        self.connectors = connectors or {}
        self.default_strategy = default_strategy
        self.max_venues_per_order = max_venues_per_order
        self.min_fill_probability = min_fill_probability

        # Fee schedules
        self._fees: Dict[Venue, VenueFees] = {}
        for venue in Venue:
            self._fees[venue] = VenueFees.from_venue(venue)

        # Historical data for venue quality
        self._venue_fill_rates: Dict[Venue, float] = {}
        self._venue_latencies: Dict[Venue, float] = {}

    def add_connector(self, connector: VenueConnector) -> None:
        """Add a venue connector."""
        self.connectors[connector.venue] = connector
        logger.info(f"Added connector for {connector.venue.value}")

    async def get_consolidated_quote(
        self,
        symbol: str,
        venues: Optional[List[Venue]] = None,
    ) -> Tuple[NBBO, Dict[Venue, VenueQuote]]:
        """Get consolidated quote from all venues."""
        target_venues = venues or list(self.connectors.keys())

        # Fetch quotes in parallel
        tasks = []
        for venue in target_venues:
            if venue in self.connectors:
                tasks.append(self._get_quote_with_venue(venue, symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        quotes: Dict[Venue, VenueQuote] = {}
        best_bid = 0.0
        best_ask = float('inf')
        best_bid_size = 0
        best_ask_size = 0
        bid_venues: List[Venue] = []
        ask_venues: List[Venue] = []

        for result in results:
            if isinstance(result, Exception):
                continue

            venue, quote = result
            if quote is None:
                continue

            quotes[venue] = quote

            # Update NBBO
            if quote.bid_price > best_bid:
                best_bid = quote.bid_price
                best_bid_size = quote.bid_size
                bid_venues = [venue]
            elif quote.bid_price == best_bid:
                best_bid_size += quote.bid_size
                bid_venues.append(venue)

            if quote.ask_price < best_ask:
                best_ask = quote.ask_price
                best_ask_size = quote.ask_size
                ask_venues = [venue]
            elif quote.ask_price == best_ask:
                best_ask_size += quote.ask_size
                ask_venues.append(venue)

        nbbo = NBBO(
            symbol=symbol,
            timestamp=datetime.now(),
            best_bid=best_bid,
            best_ask=best_ask if best_ask != float('inf') else 0,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
            bid_venues=bid_venues,
            ask_venues=ask_venues,
        )

        return nbbo, quotes

    async def _get_quote_with_venue(
        self,
        venue: Venue,
        symbol: str,
    ) -> Tuple[Venue, Optional[VenueQuote]]:
        """Helper to get quote and return with venue identifier."""
        try:
            quote = await self.connectors[venue].get_quote(symbol)
            return venue, quote
        except Exception as e:
            logger.warning(f"Failed to get quote from {venue.value}: {e}")
            return venue, None

    def _estimate_fill_probability(
        self,
        quote: VenueQuote,
        side: OrderSide,
        quantity: int,
        price: Optional[float],
    ) -> float:
        """Estimate probability of fill at venue."""
        available = quote.available_size(side)
        venue_price = quote.price_for_side(side)

        # Size-based probability
        size_prob = min(1.0, available / quantity) if quantity > 0 else 0

        # Price-based probability
        if price is None:
            price_prob = 1.0  # Market order
        elif side == OrderSide.BUY:
            price_prob = 1.0 if price >= venue_price else 0.0
        else:
            price_prob = 1.0 if price <= venue_price else 0.0

        # Venue historical fill rate
        venue_rate = self._venue_fill_rates.get(quote.venue, 0.85)

        # Dark pool discount
        dark_factor = 0.6 if quote.is_dark else 1.0

        return size_prob * price_prob * venue_rate * dark_factor

    def _estimate_market_impact(
        self,
        quote: VenueQuote,
        side: OrderSide,
        quantity: int,
    ) -> float:
        """Estimate market impact in bps."""
        available = quote.available_size(side)

        if available == 0:
            return 50.0  # High impact if no liquidity

        # Simple square-root model
        participation = quantity / available
        base_impact = 5.0  # 5 bps base

        return base_impact * np.sqrt(participation) * 100

    def _calculate_routing_cost(
        self,
        venue: Venue,
        quote: VenueQuote,
        side: OrderSide,
        quantity: int,
        is_marketable: bool,
    ) -> float:
        """Calculate total cost of routing to venue (fees + impact)."""
        fees = self._fees.get(venue, VenueFees(venue=venue))

        # Fee cost
        fee_cost = fees.calculate_cost(quantity, is_maker=not is_marketable)

        # Market impact cost
        impact_bps = self._estimate_market_impact(quote, side, quantity)
        impact_cost = (impact_bps / 10000) * quote.mid_price * quantity

        return fee_cost + impact_cost

    def _select_venues_best_price(
        self,
        quotes: Dict[Venue, VenueQuote],
        side: OrderSide,
        quantity: int,
    ) -> List[RoutingDecision]:
        """Select venues by best price."""
        decisions = []
        remaining = quantity

        # Sort by price
        sorted_venues = sorted(
            quotes.items(),
            key=lambda x: x[1].price_for_side(side),
            reverse=(side == OrderSide.SELL),
        )

        for venue, quote in sorted_venues[:self.max_venues_per_order]:
            if remaining <= 0:
                break

            available = quote.available_size(side)
            fill_qty = min(remaining, available)

            if fill_qty > 0:
                decisions.append(RoutingDecision(
                    venue=venue,
                    quantity=fill_qty,
                    price=quote.price_for_side(side),
                    is_marketable=True,
                    expected_fill_probability=self._estimate_fill_probability(
                        quote, side, fill_qty, None
                    ),
                    expected_cost=self._calculate_routing_cost(
                        venue, quote, side, fill_qty, True
                    ),
                    reasoning=f"Best price at {venue.value}",
                ))
                remaining -= fill_qty

        return decisions

    def _select_venues_least_cost(
        self,
        quotes: Dict[Venue, VenueQuote],
        side: OrderSide,
        quantity: int,
    ) -> List[RoutingDecision]:
        """Select venues by minimum total cost."""
        decisions = []
        remaining = quantity

        # Calculate cost for each venue
        venue_costs = []
        for venue, quote in quotes.items():
            available = quote.available_size(side)
            if available > 0:
                cost = self._calculate_routing_cost(
                    venue, quote, side, min(remaining, available), True
                )
                venue_costs.append((venue, quote, cost, available))

        # Sort by cost
        sorted_venues = sorted(venue_costs, key=lambda x: x[2])

        for venue, quote, cost, available in sorted_venues[:self.max_venues_per_order]:
            if remaining <= 0:
                break

            fill_qty = min(remaining, available)

            if fill_qty > 0:
                decisions.append(RoutingDecision(
                    venue=venue,
                    quantity=fill_qty,
                    price=quote.price_for_side(side),
                    is_marketable=True,
                    expected_fill_probability=self._estimate_fill_probability(
                        quote, side, fill_qty, None
                    ),
                    expected_cost=cost,
                    reasoning=f"Lowest cost at {venue.value} (${cost:.4f})",
                ))
                remaining -= fill_qty

        return decisions

    def _select_venues_passive(
        self,
        quotes: Dict[Venue, VenueQuote],
        nbbo: NBBO,
        side: OrderSide,
        quantity: int,
    ) -> List[RoutingDecision]:
        """Select venues for passive (rebate-capturing) execution."""
        decisions = []
        remaining = quantity

        # Look for venues with maker rebates
        rebate_venues = []
        for venue, quote in quotes.items():
            fees = self._fees.get(venue, VenueFees(venue=venue))
            if fees.maker_fee < 0:  # Negative fee = rebate
                rebate_venues.append((venue, quote, -fees.maker_fee))

        # Sort by rebate amount
        sorted_venues = sorted(rebate_venues, key=lambda x: x[2], reverse=True)

        # Post at NBBO on rebate venues
        passive_price = nbbo.best_bid if side == OrderSide.BUY else nbbo.best_ask

        for venue, _quote, rebate in sorted_venues[:self.max_venues_per_order]:
            if remaining <= 0:
                break

            fill_qty = min(remaining, quantity // len(sorted_venues))

            if fill_qty > 0:
                decisions.append(RoutingDecision(
                    venue=venue,
                    quantity=fill_qty,
                    price=passive_price,
                    is_marketable=False,
                    expected_fill_probability=0.3,  # Passive orders have lower fill rate
                    expected_cost=-rebate * fill_qty,  # Negative cost = rebate
                    reasoning=f"Passive at {venue.value} for ${rebate:.4f}/share rebate",
                ))
                remaining -= fill_qty

        return decisions

    def _select_venues_price_improvement(
        self,
        quotes: Dict[Venue, VenueQuote],
        nbbo: NBBO,
        side: OrderSide,
        quantity: int,
    ) -> List[RoutingDecision]:
        """Select venues that may offer price improvement."""
        decisions = []
        remaining = quantity

        # Look for dark pools and midpoint venues
        midpoint_venues = [
            (venue, quote) for venue, quote in quotes.items()
            if quote.supports_midpoint or quote.is_dark
        ]

        midpoint = nbbo.mid_price

        for venue, quote in midpoint_venues[:self.max_venues_per_order]:
            if remaining <= 0:
                break

            fill_qty = min(remaining, quantity // max(1, len(midpoint_venues)))

            if fill_qty > 0:
                # Estimate improvement vs NBBO
                improvement = abs(
                    midpoint - quote.price_for_side(side)
                ) if side == OrderSide.BUY else abs(
                    quote.price_for_side(side) - midpoint
                )

                decisions.append(RoutingDecision(
                    venue=venue,
                    quantity=fill_qty,
                    price=midpoint,  # Midpoint pegged
                    is_marketable=False,
                    expected_fill_probability=0.4,  # Dark pool uncertainty
                    expected_cost=0,  # Midpoint execution
                    reasoning=f"Midpoint at {venue.value} for ~{improvement*10000:.1f}bps improvement",
                ))
                remaining -= fill_qty

        return decisions

    async def route_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        strategy: Optional[RoutingStrategy] = None,
        price: Optional[float] = None,
        venues: Optional[List[Venue]] = None,
    ) -> RouteResult:
        """
        Route an order across venues.

        Args:
            symbol: Symbol to trade
            side: Buy or sell
            quantity: Number of shares
            strategy: Routing strategy (uses default if not specified)
            price: Limit price (None for market order)
            venues: Specific venues to consider (None for all)

        Returns:
            RouteResult with routing decisions
        """
        start_time = datetime.now()
        strategy = strategy or self.default_strategy

        # Get consolidated quotes
        nbbo, quotes = await self.get_consolidated_quote(symbol, venues)

        if not quotes:
            return RouteResult(
                success=False,
                decisions=[],
                total_quantity=0,
                estimated_total_cost=0,
                estimated_avg_price=0,
                nbbo_at_decision=nbbo,
                decision_time_us=0,
                venues_considered=0,
                venues_rejected=0,
                price_improvement_expected=0,
            )

        # Select venues based on strategy
        if strategy == RoutingStrategy.BEST_PRICE:
            decisions = self._select_venues_best_price(quotes, side, quantity)
        elif strategy == RoutingStrategy.LEAST_COST:
            decisions = self._select_venues_least_cost(quotes, side, quantity)
        elif strategy == RoutingStrategy.PASSIVE:
            decisions = self._select_venues_passive(quotes, nbbo, side, quantity)
        elif strategy == RoutingStrategy.PRICE_IMPROVEMENT:
            decisions = self._select_venues_price_improvement(
                quotes, nbbo, side, quantity
            )
        else:
            # Default to least cost
            decisions = self._select_venues_least_cost(quotes, side, quantity)

        # Calculate aggregates
        total_qty = sum(d.quantity for d in decisions)
        total_cost = sum(d.expected_cost for d in decisions)

        # Estimate average price
        if total_qty > 0:
            weighted_price = sum(
                (d.price or nbbo.mid_price) * d.quantity for d in decisions
            )
            avg_price = weighted_price / total_qty
        else:
            avg_price = nbbo.mid_price

        # Calculate expected price improvement
        nbbo_price = nbbo.best_ask if side == OrderSide.BUY else nbbo.best_bid
        improvement = (nbbo_price - avg_price) if side == OrderSide.BUY else (avg_price - nbbo_price)

        # Decision time
        decision_time = (datetime.now() - start_time).microseconds

        return RouteResult(
            success=len(decisions) > 0,
            decisions=decisions,
            total_quantity=total_qty,
            estimated_total_cost=total_cost,
            estimated_avg_price=avg_price,
            nbbo_at_decision=nbbo,
            decision_time_us=decision_time,
            venues_considered=len(quotes),
            venues_rejected=len(quotes) - len(decisions),
            price_improvement_expected=improvement,
        )

    async def execute_route(
        self,
        route_result: RouteResult,
        symbol: str,
        side: OrderSide,
    ) -> List[Dict[str, Any]]:
        """Execute the routing decisions."""
        if not route_result.success:
            return []

        # Send orders in parallel
        tasks = []
        for decision in route_result.decisions:
            if decision.venue in self.connectors:
                tasks.append(
                    self.connectors[decision.venue].send_order(
                        symbol=symbol,
                        side=side,
                        quantity=decision.quantity,
                        price=decision.price,
                        order_type="limit" if decision.price else "market",
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        orders = [r for r in results if not isinstance(r, Exception)]
        return orders

    def get_venue_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each venue."""
        stats = {}
        for venue in self.connectors:
            stats[venue.value] = {
                "fill_rate": self._venue_fill_rates.get(venue, 0.85),
                "avg_latency_us": self._venue_latencies.get(venue, 1000),
                "fees": {
                    "maker": self._fees[venue].maker_fee,
                    "taker": self._fees[venue].taker_fee,
                },
            }
        return stats

    def update_venue_stats(
        self,
        venue: Venue,
        filled: bool,
        latency_us: int,
    ) -> None:
        """Update venue statistics from execution feedback."""
        # Update fill rate with exponential moving average
        current_rate = self._venue_fill_rates.get(venue, 0.85)
        fill_value = 1.0 if filled else 0.0
        self._venue_fill_rates[venue] = 0.95 * current_rate + 0.05 * fill_value

        # Update latency
        current_latency = self._venue_latencies.get(venue, 1000)
        self._venue_latencies[venue] = 0.95 * current_latency + 0.05 * latency_us


def create_smart_router(
    venues: Optional[List[Venue]] = None,
    strategy: RoutingStrategy = RoutingStrategy.LEAST_COST,
) -> SmartOrderRouter:
    """
    Factory function to create a SmartOrderRouter with mock connectors.

    Args:
        venues: List of venues to include (defaults to major lit exchanges)
        strategy: Default routing strategy
    """
    if venues is None:
        venues = [
            Venue.NYSE,
            Venue.NASDAQ,
            Venue.ARCA,
            Venue.IEX,
            Venue.BATS_BZX,
            Venue.EDGX,
        ]

    router = SmartOrderRouter(default_strategy=strategy)

    for venue in venues:
        connector = MockVenueConnector(venue)
        router.add_connector(connector)

    return router


def print_routing_report(result: RouteResult) -> None:
    """Print formatted routing report."""
    print("\n" + "=" * 60)
    print("SMART ORDER ROUTING REPORT")
    print("=" * 60)

    print(f"\n{'NBBO at Decision':-^40}")
    print(f"  Bid: ${result.nbbo_at_decision.best_bid:.4f} x {result.nbbo_at_decision.best_bid_size}")
    print(f"  Ask: ${result.nbbo_at_decision.best_ask:.4f} x {result.nbbo_at_decision.best_ask_size}")
    print(f"  Spread: {result.nbbo_at_decision.spread_bps:.1f} bps")

    print(f"\n{'Routing Decisions':-^40}")
    for i, decision in enumerate(result.decisions, 1):
        print(f"\n  [{i}] {decision.venue.value}")
        print(f"      Quantity: {decision.quantity:,}")
        print(f"      Price: ${decision.price:.4f}" if decision.price else "      Price: Market")
        print(f"      Fill Prob: {decision.expected_fill_probability:.1%}")
        print(f"      Est Cost: ${decision.expected_cost:.4f}")
        print(f"      Reason: {decision.reasoning}")

    print(f"\n{'Summary':-^40}")
    print(f"  Total Quantity: {result.total_quantity:,}")
    print(f"  Est. Avg Price: ${result.estimated_avg_price:.4f}")
    print(f"  Total Cost: ${result.estimated_total_cost:.4f}")
    print(f"  Price Improvement: ${result.price_improvement_expected:.4f}")
    print(f"  Venues Considered: {result.venues_considered}")
    print(f"  Decision Time: {result.decision_time_us} us")

    print("=" * 60 + "\n")
