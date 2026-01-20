"""
Unit tests for PortfolioRebalancer.

Tests the portfolio rebalancing system including:
- Initialization with target allocations
- Equal weight allocation
- Drift calculation
- Rebalancing schedule check
- Order generation
- Order execution
- Report generation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestPortfolioRebalancerInit:
    """Test PortfolioRebalancer initialization."""

    def test_init_with_target_allocations(self):
        """Test initialization with explicit target allocations."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.50},
            rebalance_threshold=0.05,
            rebalance_frequency="weekly"
        )

        assert rebalancer.target_allocations == {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.50}
        assert rebalancer.rebalance_threshold == 0.05
        assert rebalancer.rebalance_frequency == "weekly"
        assert rebalancer.last_rebalance is None
        assert rebalancer.rebalance_history == []

    def test_init_with_equal_weight(self):
        """Test initialization with equal weight symbols."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            equal_weight_symbols=["AAPL", "MSFT", "GOOGL", "AMZN"]
        )

        assert len(rebalancer.target_allocations) == 4
        assert rebalancer.target_allocations["AAPL"] == 0.25
        assert rebalancer.target_allocations["MSFT"] == 0.25
        assert rebalancer.target_allocations["GOOGL"] == 0.25
        assert rebalancer.target_allocations["AMZN"] == 0.25

    def test_init_invalid_allocations_sum(self):
        """Test initialization fails when allocations don't sum to 1.0."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()

        with pytest.raises(ValueError) as excinfo:
            PortfolioRebalancer(
                broker=mock_broker,
                target_allocations={"AAPL": 0.25, "MSFT": 0.25}  # Sums to 0.50
            )

        assert "must sum to 1.0" in str(excinfo.value)

    def test_init_no_allocations(self):
        """Test initialization fails when no allocations provided."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()

        with pytest.raises(ValueError) as excinfo:
            PortfolioRebalancer(broker=mock_broker)

        assert "Must provide" in str(excinfo.value)

    def test_init_with_custom_min_trade_size(self):
        """Test initialization with custom minimum trade size."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            min_trade_size=500.0
        )

        assert rebalancer.min_trade_size == 500.0

    def test_init_dry_run_mode(self):
        """Test initialization with dry run mode."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            dry_run=True
        )

        assert rebalancer.dry_run is True

    def test_init_allocations_approximately_one(self):
        """Test initialization accepts allocations that sum very close to 1.0."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        # Sum = 0.999 which is within tolerance
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.333, "MSFT": 0.333, "GOOGL": 0.333}
        )

        assert len(rebalancer.target_allocations) == 3


class TestGetCurrentAllocations:
    """Test getting current portfolio allocations."""

    @pytest.mark.asyncio
    async def test_get_allocations_with_positions(self):
        """Test getting allocations when positions exist."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        mock_positions = [
            MagicMock(symbol="AAPL", market_value="25000"),
            MagicMock(symbol="MSFT", market_value="25000"),
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.50}
        )

        allocations = await rebalancer.get_current_allocations()

        assert allocations["AAPL"] == 0.25
        assert allocations["MSFT"] == 0.25
        assert allocations["GOOGL"] == 0.0

    @pytest.mark.asyncio
    async def test_get_allocations_no_positions(self):
        """Test getting allocations when no positions exist."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.get_positions = AsyncMock(return_value=[])

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.50, "MSFT": 0.50}
        )

        allocations = await rebalancer.get_current_allocations()

        assert allocations["AAPL"] == 0.0
        assert allocations["MSFT"] == 0.0

    @pytest.mark.asyncio
    async def test_get_allocations_zero_equity(self):
        """Test getting allocations when equity is zero."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "0"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0}
        )

        allocations = await rebalancer.get_current_allocations()

        assert allocations == {}

    @pytest.mark.asyncio
    async def test_get_allocations_exception(self):
        """Test getting allocations handles exceptions."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_broker.get_account = AsyncMock(side_effect=Exception("API Error"))

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0}
        )

        allocations = await rebalancer.get_current_allocations()

        assert allocations == {}


class TestCalculateDrift:
    """Test drift calculation."""

    @pytest.mark.asyncio
    async def test_calculate_drift_underweight(self):
        """Test drift calculation when underweight."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        # AAPL is 20%, target is 30% => underweight by 10%
        mock_positions = [
            MagicMock(symbol="AAPL", market_value="20000"),
            MagicMock(symbol="MSFT", market_value="80000"),
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.30, "MSFT": 0.70}
        )

        drift = await rebalancer.calculate_drift()

        assert abs(drift["AAPL"] - (-0.10)) < 0.001  # Underweight
        assert abs(drift["MSFT"] - 0.10) < 0.001    # Overweight

    @pytest.mark.asyncio
    async def test_calculate_drift_no_positions(self):
        """Test drift calculation with no positions."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.get_positions = AsyncMock(return_value=[])

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.50, "MSFT": 0.50}
        )

        drift = await rebalancer.calculate_drift()

        # All underweight
        assert drift["AAPL"] == -0.50
        assert drift["MSFT"] == -0.50


class TestNeedsRebalancing:
    """Test rebalancing need detection."""

    @pytest.mark.asyncio
    async def test_needs_rebalancing_exceeds_threshold(self):
        """Test that rebalancing is needed when drift exceeds threshold."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        # 10% drift exceeds 5% threshold
        mock_positions = [
            MagicMock(symbol="AAPL", market_value="35000"),  # 35% vs 25% target
            MagicMock(symbol="MSFT", market_value="65000"),  # 65% vs 75% target
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.25, "MSFT": 0.75},
            rebalance_threshold=0.05
        )

        assert await rebalancer.needs_rebalancing() is True

    @pytest.mark.asyncio
    async def test_no_rebalancing_within_threshold(self):
        """Test that rebalancing is not needed when drift is within threshold."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        # 2% drift is within 5% threshold
        mock_positions = [
            MagicMock(symbol="AAPL", market_value="27000"),  # 27% vs 25% target
            MagicMock(symbol="MSFT", market_value="73000"),  # 73% vs 75% target
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.25, "MSFT": 0.75},
            rebalance_threshold=0.05
        )

        assert await rebalancer.needs_rebalancing() is False


class TestShouldRebalanceBySchedule:
    """Test schedule-based rebalancing check."""

    @pytest.mark.asyncio
    async def test_should_rebalance_never_rebalanced(self):
        """Test schedule check when never rebalanced before."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="weekly"
        )

        assert await rebalancer._should_rebalance_by_schedule() is True

    @pytest.mark.asyncio
    async def test_should_rebalance_daily_not_due(self):
        """Test daily schedule when not enough time has passed."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="daily"
        )
        rebalancer.last_rebalance = datetime.now() - timedelta(hours=12)

        assert await rebalancer._should_rebalance_by_schedule() is False

    @pytest.mark.asyncio
    async def test_should_rebalance_daily_due(self):
        """Test daily schedule when time has passed."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="daily"
        )
        rebalancer.last_rebalance = datetime.now() - timedelta(days=2)

        assert await rebalancer._should_rebalance_by_schedule() is True

    @pytest.mark.asyncio
    async def test_should_rebalance_weekly_not_due(self):
        """Test weekly schedule when not enough time has passed."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="weekly"
        )
        rebalancer.last_rebalance = datetime.now() - timedelta(days=3)

        assert await rebalancer._should_rebalance_by_schedule() is False

    @pytest.mark.asyncio
    async def test_should_rebalance_weekly_due(self):
        """Test weekly schedule when time has passed."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="weekly"
        )
        rebalancer.last_rebalance = datetime.now() - timedelta(weeks=2)

        assert await rebalancer._should_rebalance_by_schedule() is True

    @pytest.mark.asyncio
    async def test_should_rebalance_monthly_not_due(self):
        """Test monthly schedule when not enough time has passed."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="monthly"
        )
        rebalancer.last_rebalance = datetime.now() - timedelta(days=15)

        assert await rebalancer._should_rebalance_by_schedule() is False

    @pytest.mark.asyncio
    async def test_should_rebalance_monthly_due(self):
        """Test monthly schedule when time has passed."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="monthly"
        )
        rebalancer.last_rebalance = datetime.now() - timedelta(days=45)

        assert await rebalancer._should_rebalance_by_schedule() is True

    @pytest.mark.asyncio
    async def test_should_rebalance_unknown_frequency(self):
        """Test unknown frequency defaults to True."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            rebalance_frequency="unknown"
        )
        rebalancer.last_rebalance = datetime.now()

        assert await rebalancer._should_rebalance_by_schedule() is True


class TestGenerateRebalanceOrders:
    """Test rebalance order generation."""

    @pytest.mark.asyncio
    async def test_generate_orders_buy_and_sell(self):
        """Test generating buy and sell orders."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        mock_positions = [
            MagicMock(symbol="AAPL", market_value="40000", current_price="150"),  # 40% vs 30%
            MagicMock(symbol="MSFT", market_value="60000", current_price="300"),  # 60% vs 70%
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.30, "MSFT": 0.70},
            min_trade_size=100.0
        )

        orders = await rebalancer.generate_rebalance_orders()

        assert len(orders) == 2

        # Find the AAPL order (should be sell)
        aapl_order = next(o for o in orders if o["symbol"] == "AAPL")
        assert aapl_order["side"] == "sell"
        assert "overweight" in aapl_order["reason"]

        # Find the MSFT order (should be buy)
        msft_order = next(o for o in orders if o["symbol"] == "MSFT")
        assert msft_order["side"] == "buy"
        assert "underweight" in msft_order["reason"]

    @pytest.mark.asyncio
    async def test_generate_orders_skip_small_adjustments(self):
        """Test that small adjustments are skipped."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        # Only 1% drift = $1000 adjustment, but min_trade_size is $2000
        mock_positions = [
            MagicMock(symbol="AAPL", market_value="51000", current_price="150"),  # 51% vs 50%
            MagicMock(symbol="MSFT", market_value="49000", current_price="300"),  # 49% vs 50%
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.50, "MSFT": 0.50},
            min_trade_size=2000.0
        )

        orders = await rebalancer.generate_rebalance_orders()

        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_generate_orders_no_position(self):
        """Test generating orders when no position exists."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        mock_positions = []
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        # Mock getting quote for symbols without positions
        mock_quote = MagicMock(ask="150.0")
        mock_broker.get_quote = AsyncMock(return_value=mock_quote)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            min_trade_size=100.0
        )

        orders = await rebalancer.generate_rebalance_orders()

        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["side"] == "buy"

    @pytest.mark.asyncio
    async def test_generate_orders_no_price(self):
        """Test generating orders when price is unavailable."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.get_positions = AsyncMock(return_value=[])
        mock_broker.get_quote = AsyncMock(side_effect=Exception("No quote"))

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            min_trade_size=100.0
        )

        orders = await rebalancer.generate_rebalance_orders()

        # Order skipped due to no price
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_generate_orders_exception(self):
        """Test order generation handles exceptions."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_broker.get_account = AsyncMock(side_effect=Exception("API Error"))

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0}
        )

        orders = await rebalancer.generate_rebalance_orders()

        assert orders == []


class TestExecuteRebalancing:
    """Test rebalancing execution."""

    @pytest.mark.asyncio
    async def test_execute_no_orders(self):
        """Test execution with no orders."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0}
        )

        result = await rebalancer.execute_rebalancing([])

        assert result["status"] == "no_action"
        assert result["orders_executed"] == 0

    @pytest.mark.asyncio
    async def test_execute_dry_run(self):
        """Test execution in dry run mode."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            dry_run=True
        )

        orders = [{
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.0,
            "value": 1500.0,
            "current_weight": 0.0,
            "target_weight": 1.0,
            "drift": -1.0,
            "reason": "underweight"
        }]

        result = await rebalancer.execute_rebalancing(orders)

        assert result["status"] == "dry_run"
        assert result["orders_generated"] == 1

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful order execution."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_broker.submit_order_advanced = AsyncMock(return_value=MagicMock())

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            dry_run=False
        )

        orders = [{
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.0,
            "value": 1500.0,
            "current_weight": 0.0,
            "target_weight": 1.0,
            "drift": -1.0,
            "reason": "underweight"
        }]

        result = await rebalancer.execute_rebalancing(orders)

        assert result["status"] == "success"
        assert result["orders_executed"] == 1
        assert result["orders_failed"] == 0
        assert rebalancer.last_rebalance is not None
        assert len(rebalancer.rebalance_history) == 1

    @pytest.mark.asyncio
    async def test_execute_order_failure(self):
        """Test handling of order execution failure."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_broker.submit_order_advanced = AsyncMock(return_value=None)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            dry_run=False
        )

        orders = [{
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.0,
            "value": 1500.0,
            "current_weight": 0.0,
            "target_weight": 1.0,
            "drift": -1.0,
            "reason": "underweight"
        }]

        result = await rebalancer.execute_rebalancing(orders)

        assert result["status"] == "success"
        assert result["orders_executed"] == 0
        assert result["orders_failed"] == 1

    @pytest.mark.asyncio
    async def test_execute_exception_during_order(self):
        """Test handling of exception during order execution."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_broker.submit_order_advanced = AsyncMock(side_effect=Exception("Order failed"))

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0},
            dry_run=False
        )

        orders = [{
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.0,
            "value": 1500.0,
            "current_weight": 0.0,
            "target_weight": 1.0,
            "drift": -1.0,
            "reason": "underweight"
        }]

        result = await rebalancer.execute_rebalancing(orders)

        assert result["orders_failed"] == 1

    @pytest.mark.asyncio
    async def test_execute_multiple_orders(self):
        """Test executing multiple orders."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_broker.submit_order_advanced = AsyncMock(return_value=MagicMock())

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.5, "MSFT": 0.5},
            dry_run=False
        )

        orders = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "price": 150.0,
                "value": 1500.0,
                "current_weight": 0.3,
                "target_weight": 0.5,
                "drift": -0.2,
                "reason": "underweight"
            },
            {
                "symbol": "MSFT",
                "side": "sell",
                "quantity": 5,
                "price": 300.0,
                "value": 1500.0,
                "current_weight": 0.7,
                "target_weight": 0.5,
                "drift": 0.2,
                "reason": "overweight"
            }
        ]

        result = await rebalancer.execute_rebalancing(orders)

        assert result["orders_executed"] == 2


class TestGetRebalanceReport:
    """Test report generation."""

    @pytest.mark.asyncio
    async def test_report_generation(self):
        """Test generating rebalance report."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        mock_positions = [
            MagicMock(symbol="AAPL", market_value="50000"),
            MagicMock(symbol="MSFT", market_value="50000"),
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 0.50, "MSFT": 0.50},
            rebalance_threshold=0.05
        )

        report = await rebalancer.get_rebalance_report()

        assert "PORTFOLIO REBALANCING REPORT" in report
        assert "AAPL" in report
        assert "MSFT" in report
        assert "Target Allocations" in report

    @pytest.mark.asyncio
    async def test_report_shows_last_rebalance(self):
        """Test that report shows last rebalance time."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.get_positions = AsyncMock(return_value=[])

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0}
        )
        rebalancer.last_rebalance = datetime(2024, 1, 15, 10, 30, 0)

        report = await rebalancer.get_rebalance_report()

        assert "2024-01-15" in report
        assert "10:30:00" in report

    @pytest.mark.asyncio
    async def test_report_never_rebalanced(self):
        """Test report when never rebalanced."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_broker.get_positions = AsyncMock(return_value=[])

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0}
        )

        report = await rebalancer.get_rebalance_report()

        assert "Never" in report


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_single_symbol_portfolio(self):
        """Test with single symbol portfolio."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        mock_positions = [
            MagicMock(symbol="AAPL", market_value="100000"),
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            target_allocations={"AAPL": 1.0}
        )

        allocations = await rebalancer.get_current_allocations()
        drift = await rebalancer.calculate_drift()

        assert allocations["AAPL"] == 1.0
        assert drift["AAPL"] == 0.0
        assert await rebalancer.needs_rebalancing() is False

    @pytest.mark.asyncio
    async def test_many_symbols_portfolio(self):
        """Test with many symbols."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = AsyncMock()
        mock_account = MagicMock()
        mock_account.equity = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        # 10 equal weight symbols
        symbols = [f"SYM{i}" for i in range(10)]
        mock_positions = [
            MagicMock(symbol=sym, market_value="10000")
            for sym in symbols
        ]
        mock_broker.get_positions = AsyncMock(return_value=mock_positions)

        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            equal_weight_symbols=symbols
        )

        allocations = await rebalancer.get_current_allocations()
        drift = await rebalancer.calculate_drift()

        for sym in symbols:
            assert abs(allocations[sym] - 0.1) < 0.001
            assert abs(drift[sym]) < 0.001

    def test_equal_weight_two_symbols(self):
        """Test equal weight with just two symbols."""
        from utils.portfolio_rebalancer import PortfolioRebalancer

        mock_broker = MagicMock()
        rebalancer = PortfolioRebalancer(
            broker=mock_broker,
            equal_weight_symbols=["AAPL", "MSFT"]
        )

        assert rebalancer.target_allocations["AAPL"] == 0.5
        assert rebalancer.target_allocations["MSFT"] == 0.5
