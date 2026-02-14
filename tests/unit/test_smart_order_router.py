"""
Tests for Smart Order Router

Tests:
- Venue quote aggregation
- NBBO calculation
- Routing strategy selection
- Fee optimization
- Market impact estimation
"""

from datetime import datetime

import pytest

from execution.smart_order_router import (
    NBBO,
    MockVenueConnector,
    OrderSide,
    RouteResult,
    RoutingDecision,
    RoutingStrategy,
    Venue,
    VenueFees,
    VenueQuote,
    create_smart_router,
)


class TestVenue:
    """Tests for Venue enum."""

    def test_lit_exchanges_exist(self):
        """Test lit exchange venues exist."""
        expected = ["NYSE", "NASDAQ", "ARCA", "IEX", "BATS_BZX", "EDGX"]
        for name in expected:
            assert hasattr(Venue, name)

    def test_dark_pools_exist(self):
        """Test dark pool venues exist."""
        expected = ["SIGMA_X", "CROSSFINDER", "LEVEL_ATS"]
        for name in expected:
            assert hasattr(Venue, name)


class TestVenueQuote:
    """Tests for VenueQuote dataclass."""

    def test_create_quote(self):
        """Test creating a venue quote."""
        quote = VenueQuote(
            venue=Venue.NYSE,
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.05,
            ask_size=300,
        )

        assert quote.venue == Venue.NYSE
        assert quote.bid_price == 150.00

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = VenueQuote(
            venue=Venue.NYSE,
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.10,
            ask_size=300,
        )

        assert quote.mid_price == 150.05

    def test_spread_bps(self):
        """Test spread in basis points."""
        quote = VenueQuote(
            venue=Venue.NYSE,
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.15,
            ask_size=300,
        )

        # 0.15 / 150.075 * 10000 = ~10 bps
        assert abs(quote.spread_bps - 10.0) < 0.5

    def test_available_size_buy(self):
        """Test available size for buy."""
        quote = VenueQuote(
            venue=Venue.NYSE,
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.05,
            ask_size=300,
        )

        assert quote.available_size(OrderSide.BUY) == 300

    def test_available_size_sell(self):
        """Test available size for sell."""
        quote = VenueQuote(
            venue=Venue.NYSE,
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.05,
            ask_size=300,
        )

        assert quote.available_size(OrderSide.SELL) == 500


class TestVenueFees:
    """Tests for VenueFees dataclass."""

    def test_create_fees(self):
        """Test creating fee structure."""
        fees = VenueFees(
            venue=Venue.NYSE,
            maker_fee=-0.0020,
            taker_fee=0.0030,
        )

        assert fees.maker_fee == -0.0020
        assert fees.taker_fee == 0.0030

    def test_from_venue_nyse(self):
        """Test getting NYSE fees."""
        fees = VenueFees.from_venue(Venue.NYSE)
        assert fees.maker_fee < 0  # Rebate
        assert fees.taker_fee > 0

    def test_from_venue_iex(self):
        """Test getting IEX fees."""
        fees = VenueFees.from_venue(Venue.IEX)
        assert fees.maker_fee == 0  # No rebate
        assert fees.taker_fee > 0

    def test_calculate_cost_taker(self):
        """Test calculating taker cost."""
        fees = VenueFees(
            venue=Venue.NYSE,
            maker_fee=-0.0020,
            taker_fee=0.0030,
        )

        # 1000 shares at $0.003/share = $3
        cost = fees.calculate_cost(1000, is_maker=False)
        assert cost == 3.0

    def test_calculate_cost_maker(self):
        """Test calculating maker rebate."""
        fees = VenueFees(
            venue=Venue.NYSE,
            maker_fee=-0.0020,
            taker_fee=0.0030,
        )

        # 1000 shares at -$0.002/share = -$2 (rebate)
        cost = fees.calculate_cost(1000, is_maker=True)
        assert cost == -2.0


class TestNBBO:
    """Tests for NBBO dataclass."""

    def test_create_nbbo(self):
        """Test creating NBBO."""
        nbbo = NBBO(
            symbol="AAPL",
            timestamp=datetime.now(),
            best_bid=150.00,
            best_ask=150.05,
            best_bid_size=1000,
            best_ask_size=800,
            bid_venues=[Venue.NYSE, Venue.NASDAQ],
            ask_venues=[Venue.ARCA],
        )

        assert nbbo.best_bid == 150.00
        assert len(nbbo.bid_venues) == 2

    def test_mid_price(self):
        """Test mid price calculation."""
        nbbo = NBBO(
            symbol="AAPL",
            timestamp=datetime.now(),
            best_bid=150.00,
            best_ask=150.10,
            best_bid_size=1000,
            best_ask_size=800,
        )

        assert nbbo.mid_price == 150.05

    def test_spread_bps(self):
        """Test spread in basis points."""
        nbbo = NBBO(
            symbol="AAPL",
            timestamp=datetime.now(),
            best_bid=150.00,
            best_ask=150.15,
            best_bid_size=1000,
            best_ask_size=800,
        )

        # ~10 bps
        assert abs(nbbo.spread_bps - 10.0) < 0.5


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_create_decision(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            venue=Venue.NYSE,
            quantity=500,
            price=150.05,
            is_marketable=True,
            expected_fill_probability=0.95,
            expected_cost=1.50,
            reasoning="Best price at NYSE",
        )

        assert decision.venue == Venue.NYSE
        assert decision.quantity == 500

    def test_to_dict(self):
        """Test serialization."""
        decision = RoutingDecision(
            venue=Venue.NYSE,
            quantity=500,
            price=150.05,
            is_marketable=True,
            expected_fill_probability=0.95,
            expected_cost=1.50,
            reasoning="Best price",
        )

        d = decision.to_dict()
        assert d["venue"] == "NYSE"
        assert d["quantity"] == 500


class TestMockVenueConnector:
    """Tests for MockVenueConnector class."""

    @pytest.fixture
    def connector(self):
        """Create a mock connector."""
        return MockVenueConnector(Venue.NYSE)

    @pytest.mark.asyncio
    async def test_get_quote(self, connector):
        """Test getting a quote."""
        quote = await connector.get_quote("AAPL")
        assert quote is not None
        assert quote.venue == Venue.NYSE
        assert quote.bid_price > 0

    @pytest.mark.asyncio
    async def test_send_order(self, connector):
        """Test sending an order."""
        result = await connector.send_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
        )

        assert result["status"] == "accepted"
        assert result["venue"] == "NYSE"

    @pytest.mark.asyncio
    async def test_cancel_order(self, connector):
        """Test canceling an order."""
        # First send an order
        order = await connector.send_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
        )

        # Then cancel it
        result = await connector.cancel_order(order["order_id"])
        assert result is True


class TestSmartOrderRouter:
    """Tests for SmartOrderRouter class."""

    @pytest.fixture
    def router(self):
        """Create a router with mock connectors."""
        return create_smart_router()

    @pytest.mark.asyncio
    async def test_get_consolidated_quote(self, router):
        """Test getting consolidated quote."""
        nbbo, quotes = await router.get_consolidated_quote("AAPL")

        assert nbbo is not None
        assert nbbo.best_bid > 0
        assert len(quotes) > 0

    @pytest.mark.asyncio
    async def test_route_order_best_price(self, router):
        """Test routing with best price strategy."""
        result = await router.route_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000,
            strategy=RoutingStrategy.BEST_PRICE,
        )

        assert result.success
        assert len(result.decisions) > 0
        assert result.total_quantity <= 1000

    @pytest.mark.asyncio
    async def test_route_order_least_cost(self, router):
        """Test routing with least cost strategy."""
        result = await router.route_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=500,
            strategy=RoutingStrategy.LEAST_COST,
        )

        assert result.success
        assert result.estimated_total_cost is not None

    @pytest.mark.asyncio
    async def test_route_order_passive(self, router):
        """Test routing with passive strategy."""
        result = await router.route_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
            strategy=RoutingStrategy.PASSIVE,
        )

        assert result is not None
        # Passive may have fewer fills
        for decision in result.decisions:
            assert not decision.is_marketable

    @pytest.mark.asyncio
    async def test_execute_route(self, router):
        """Test executing routing decisions."""
        route_result = await router.route_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
        )

        orders = await router.execute_route(
            route_result,
            symbol="AAPL",
            side=OrderSide.BUY,
        )

        assert len(orders) > 0

    def test_get_venue_stats(self, router):
        """Test getting venue statistics."""
        stats = router.get_venue_stats()
        assert len(stats) > 0
        assert "NYSE" in stats

    def test_update_venue_stats(self, router):
        """Test updating venue statistics."""
        router.update_venue_stats(Venue.NYSE, filled=True, latency_us=500)

        stats = router.get_venue_stats()
        # Stats should be updated
        assert "NYSE" in stats


class TestRoutingStrategy:
    """Tests for RoutingStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all routing strategies exist."""
        expected = [
            "BEST_PRICE",
            "BEST_SIZE",
            "LEAST_COST",
            "PRICE_IMPROVEMENT",
            "PASSIVE",
            "AGGRESSIVE",
        ]
        for name in expected:
            assert hasattr(RoutingStrategy, name)


class TestRouteResult:
    """Tests for RouteResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = RouteResult(
            success=True,
            decisions=[],
            total_quantity=1000,
            estimated_total_cost=3.0,
            estimated_avg_price=150.05,
            nbbo_at_decision=NBBO(
                symbol="AAPL",
                timestamp=datetime.now(),
                best_bid=150.0,
                best_ask=150.1,
                best_bid_size=1000,
                best_ask_size=800,
            ),
            decision_time_us=100,
            venues_considered=6,
            venues_rejected=1,
            price_improvement_expected=0.01,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["total_quantity"] == 1000


class TestCreateSmartRouter:
    """Tests for create_smart_router factory."""

    def test_create_with_defaults(self):
        """Test creating router with defaults."""
        router = create_smart_router()
        assert router is not None
        assert len(router.connectors) > 0

    def test_create_with_specific_venues(self):
        """Test creating router with specific venues."""
        router = create_smart_router(venues=[Venue.NYSE, Venue.NASDAQ, Venue.IEX])
        assert len(router.connectors) == 3

    def test_create_with_strategy(self):
        """Test creating router with specific strategy."""
        router = create_smart_router(strategy=RoutingStrategy.PASSIVE)
        assert router.default_strategy == RoutingStrategy.PASSIVE
