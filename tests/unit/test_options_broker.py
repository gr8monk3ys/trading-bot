#!/usr/bin/env python3
"""
Comprehensive unit tests for OptionsBroker module.

Tests cover:
1. OCC symbol building and parsing
2. OptionContract and OptionChain dataclasses
3. Option chain filtering
4. Order submission (mocked)
5. Position management (mocked)
6. Strategy helpers
7. Utility functions
"""

import os
import sys
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brokers.options_broker import (
    InvalidContractError,
    OptionChain,
    OptionContract,
    OptionsBroker,
    OptionType,
    calculate_contract_value,
    get_monthly_expiration,
    get_weekly_expiration,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def options_broker():
    """Create an OptionsBroker instance for testing."""
    with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
        return OptionsBroker(api_key="test_api_key", secret_key="test_secret_key", paper=True)


@pytest.fixture
def sample_expiration():
    """Get a sample expiration date (30 days out)."""
    return date.today() + timedelta(days=30)


@pytest.fixture
def sample_call_contract(sample_expiration):
    """Create a sample call option contract."""
    return OptionContract(
        symbol="AAPL  250120C00150000",
        underlying="AAPL",
        expiration=sample_expiration,
        strike=150.0,
        option_type=OptionType.CALL,
        bid=5.50,
        ask=5.80,
        last=5.65,
        volume=1000,
        open_interest=5000,
        delta=0.55,
        gamma=0.05,
        theta=-0.10,
        vega=0.20,
        implied_volatility=0.30,
    )


@pytest.fixture
def sample_put_contract(sample_expiration):
    """Create a sample put option contract."""
    return OptionContract(
        symbol="AAPL  250120P00140000",
        underlying="AAPL",
        expiration=sample_expiration,
        strike=140.0,
        option_type=OptionType.PUT,
        bid=3.20,
        ask=3.50,
        last=3.35,
        volume=800,
        open_interest=3000,
        delta=-0.35,
        gamma=0.04,
        theta=-0.08,
        vega=0.15,
        implied_volatility=0.28,
    )


@pytest.fixture
def sample_option_chain(sample_expiration, sample_call_contract, sample_put_contract):
    """Create a sample option chain."""
    # Create additional contracts at different strikes
    calls = [
        OptionContract(
            symbol="AAPL  250120C00140000",
            underlying="AAPL",
            expiration=sample_expiration,
            strike=140.0,
            option_type=OptionType.CALL,
            bid=12.00,
            ask=12.50,
            delta=0.75,
        ),
        sample_call_contract,
        OptionContract(
            symbol="AAPL  250120C00160000",
            underlying="AAPL",
            expiration=sample_expiration,
            strike=160.0,
            option_type=OptionType.CALL,
            bid=2.00,
            ask=2.30,
            delta=0.30,
        ),
    ]
    puts = [
        OptionContract(
            symbol="AAPL  250120P00130000",
            underlying="AAPL",
            expiration=sample_expiration,
            strike=130.0,
            option_type=OptionType.PUT,
            bid=1.50,
            ask=1.80,
            delta=-0.20,
        ),
        sample_put_contract,
        OptionContract(
            symbol="AAPL  250120P00150000",
            underlying="AAPL",
            expiration=sample_expiration,
            strike=150.0,
            option_type=OptionType.PUT,
            bid=5.00,
            ask=5.30,
            delta=-0.45,
        ),
    ]
    return OptionChain(underlying="AAPL", expiration=sample_expiration, calls=calls, puts=puts)


# =============================================================================
# OCC SYMBOL TESTS
# =============================================================================


class TestOCCSymbolBuilding:
    """Test OCC symbol building functionality."""

    def test_build_occ_symbol_basic_call(self):
        """Test building a basic call OCC symbol."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="AAPL",
            expiration=date(2025, 1, 20),
            option_type=OptionType.CALL,
            strike=150.0,
        )
        assert symbol == "AAPL  250120C00150000"

    def test_build_occ_symbol_basic_put(self):
        """Test building a basic put OCC symbol."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="AAPL",
            expiration=date(2025, 1, 20),
            option_type=OptionType.PUT,
            strike=140.0,
        )
        assert symbol == "AAPL  250120P00140000"

    def test_build_occ_symbol_fractional_strike(self):
        """Test building symbol with fractional strike."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="SPY",
            expiration=date(2025, 3, 21),
            option_type=OptionType.CALL,
            strike=475.50,
        )
        assert symbol == "SPY   250321C00475500"

    def test_build_occ_symbol_short_underlying(self):
        """Test that short underlying is properly padded."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="F", expiration=date(2025, 6, 20), option_type=OptionType.PUT, strike=12.0
        )
        # F padded to 6 chars: "F     "
        assert symbol.startswith("F     ")
        assert "P" in symbol

    def test_build_occ_symbol_low_strike(self):
        """Test building symbol with low strike price."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="SIRI", expiration=date(2025, 2, 21), option_type=OptionType.CALL, strike=5.0
        )
        assert "C00005000" in symbol

    def test_build_occ_symbol_high_strike(self):
        """Test building symbol with high strike price."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="AMZN",
            expiration=date(2025, 1, 17),
            option_type=OptionType.CALL,
            strike=3500.0,
        )
        assert "C03500000" in symbol

    def test_build_occ_symbol_empty_underlying_raises(self):
        """Test that empty underlying raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            OptionsBroker.build_occ_symbol(
                underlying="",
                expiration=date(2025, 1, 20),
                option_type=OptionType.CALL,
                strike=150.0,
            )

    def test_build_occ_symbol_long_underlying_raises(self):
        """Test that underlying > 6 chars raises error."""
        with pytest.raises(ValueError, match="too long"):
            OptionsBroker.build_occ_symbol(
                underlying="TOOLONG",
                expiration=date(2025, 1, 20),
                option_type=OptionType.CALL,
                strike=150.0,
            )


class TestOCCSymbolParsing:
    """Test OCC symbol parsing functionality."""

    def test_parse_occ_symbol_call(self):
        """Test parsing a call OCC symbol."""
        result = OptionsBroker.parse_occ_symbol("AAPL  250120C00150000")

        assert result["underlying"] == "AAPL"
        assert result["expiration"] == date(2025, 1, 20)
        assert result["option_type"] == OptionType.CALL
        assert result["strike"] == 150.0

    def test_parse_occ_symbol_put(self):
        """Test parsing a put OCC symbol."""
        result = OptionsBroker.parse_occ_symbol("MSFT  250221P00400000")

        assert result["underlying"] == "MSFT"
        assert result["expiration"] == date(2025, 2, 21)
        assert result["option_type"] == OptionType.PUT
        assert result["strike"] == 400.0

    def test_parse_occ_symbol_fractional_strike(self):
        """Test parsing symbol with fractional strike."""
        result = OptionsBroker.parse_occ_symbol("SPY   250321C00475500")

        assert result["underlying"] == "SPY"
        assert result["strike"] == 475.5

    def test_parse_occ_symbol_short_underlying(self):
        """Test parsing symbol with short underlying."""
        result = OptionsBroker.parse_occ_symbol("F     250620P00012000")

        assert result["underlying"] == "F"
        assert result["strike"] == 12.0

    def test_parse_occ_symbol_invalid_format_raises(self):
        """Test that invalid format raises error."""
        with pytest.raises(InvalidContractError, match="Invalid OCC symbol"):
            OptionsBroker.parse_occ_symbol("AAPL")

    def test_parse_occ_symbol_invalid_option_type_raises(self):
        """Test that invalid option type raises error."""
        with pytest.raises(InvalidContractError, match="Invalid option type"):
            OptionsBroker.parse_occ_symbol("AAPL  250120X00150000")

    def test_parse_occ_symbol_roundtrip(self):
        """Test that build and parse are inverse operations."""
        original = {
            "underlying": "GOOGL",
            "expiration": date(2025, 6, 20),
            "option_type": OptionType.CALL,
            "strike": 175.0,
        }

        symbol = OptionsBroker.build_occ_symbol(**original)
        parsed = OptionsBroker.parse_occ_symbol(symbol)

        assert parsed["underlying"] == original["underlying"]
        assert parsed["expiration"] == original["expiration"]
        assert parsed["option_type"] == original["option_type"]
        assert parsed["strike"] == original["strike"]


class TestIsOptionSymbol:
    """Test option symbol detection."""

    def test_is_option_symbol_valid_call(self):
        """Test detecting valid call symbol."""
        assert OptionsBroker.is_option_symbol("AAPL  250120C00150000") is True

    def test_is_option_symbol_valid_put(self):
        """Test detecting valid put symbol."""
        assert OptionsBroker.is_option_symbol("MSFT  250221P00400000") is True

    def test_is_option_symbol_stock_symbol(self):
        """Test that stock symbol returns False."""
        assert OptionsBroker.is_option_symbol("AAPL") is False

    def test_is_option_symbol_empty(self):
        """Test that empty string returns False."""
        assert OptionsBroker.is_option_symbol("") is False

    def test_is_option_symbol_none(self):
        """Test that None returns False."""
        assert OptionsBroker.is_option_symbol(None) is False


# =============================================================================
# OPTION CONTRACT TESTS
# =============================================================================


class TestOptionContract:
    """Test OptionContract dataclass."""

    def test_mid_price_calculation(self, sample_call_contract):
        """Test mid price calculation."""
        # bid=5.50, ask=5.80 -> mid=5.65
        assert sample_call_contract.mid_price == 5.65

    def test_mid_price_fallback_to_last(self):
        """Test mid price falls back to last when bid/ask missing."""
        contract = OptionContract(
            symbol="AAPL  250120C00150000",
            underlying="AAPL",
            expiration=date.today() + timedelta(days=30),
            strike=150.0,
            option_type=OptionType.CALL,
            last=5.50,
        )
        assert contract.mid_price == 5.50

    def test_spread_calculation(self, sample_call_contract):
        """Test spread calculation."""
        # bid=5.50, ask=5.80 -> spread=0.30
        assert sample_call_contract.spread == 0.30

    def test_spread_pct_calculation(self, sample_call_contract):
        """Test spread percentage calculation."""
        # spread=0.30, mid=5.65 -> spread_pct=5.31%
        assert sample_call_contract.spread_pct == pytest.approx(5.31, rel=0.01)

    def test_days_to_expiration(self, sample_expiration):
        """Test DTE calculation."""
        contract = OptionContract(
            symbol="TEST",
            underlying="TEST",
            expiration=sample_expiration,
            strike=100.0,
            option_type=OptionType.CALL,
        )
        expected_dte = (sample_expiration - date.today()).days
        assert contract.days_to_expiration == expected_dte

    def test_repr_output(self, sample_call_contract):
        """Test string representation."""
        repr_str = repr(sample_call_contract)
        assert "AAPL" in repr_str
        assert "150" in repr_str
        assert "CALL" in repr_str


# =============================================================================
# OPTION CHAIN TESTS
# =============================================================================


class TestOptionChain:
    """Test OptionChain dataclass."""

    def test_num_strikes(self, sample_option_chain):
        """Test counting unique strikes."""
        # Strikes: 130, 140, 150, 160 = 4 unique
        assert sample_option_chain.num_strikes == 4

    def test_get_call_at_strike(self, sample_option_chain):
        """Test getting call at specific strike."""
        call = sample_option_chain.get_call_at_strike(150.0)
        assert call is not None
        assert call.strike == 150.0
        assert call.option_type == OptionType.CALL

    def test_get_call_at_strike_not_found(self, sample_option_chain):
        """Test getting call at non-existent strike."""
        call = sample_option_chain.get_call_at_strike(999.0)
        assert call is None

    def test_get_put_at_strike(self, sample_option_chain):
        """Test getting put at specific strike."""
        put = sample_option_chain.get_put_at_strike(140.0)
        assert put is not None
        assert put.strike == 140.0
        assert put.option_type == OptionType.PUT

    def test_get_atm_strike(self, sample_option_chain):
        """Test finding ATM strike."""
        # With strikes 130, 140, 150, 160 and underlying at 147
        atm = sample_option_chain.get_atm_strike(147.0)
        assert atm == 150.0  # Closest to 147

    def test_filter_by_delta(self, sample_option_chain):
        """Test filtering by delta range."""
        filtered = sample_option_chain.filter_by_delta(min_delta=0.25, max_delta=0.50)

        # Should include contracts with abs(delta) between 0.25 and 0.50
        for call in filtered.calls:
            if call.delta is not None:
                assert 0.25 <= abs(call.delta) <= 0.50

    def test_repr_output(self, sample_option_chain):
        """Test string representation."""
        repr_str = repr(sample_option_chain)
        assert "AAPL" in repr_str
        assert "calls" in repr_str
        assert "puts" in repr_str


# =============================================================================
# OPTIONS BROKER INITIALIZATION TESTS
# =============================================================================


class TestOptionsBrokerInit:
    """Test OptionsBroker initialization."""

    def test_init_with_paper_true(self):
        """Test initialization with paper=True."""
        broker = OptionsBroker(api_key="test_key", secret_key="test_secret", paper=True)
        assert broker.paper is True

    def test_init_with_paper_false(self):
        """Test initialization with paper=False."""
        broker = OptionsBroker(api_key="test_key", secret_key="test_secret", paper=False)
        assert broker.paper is False

    def test_init_with_paper_string(self):
        """Test initialization with paper as string."""
        broker = OptionsBroker(api_key="test_key", secret_key="test_secret", paper="true")
        assert broker.paper is True

    def test_lazy_client_initialization(self, options_broker):
        """Test that clients are not initialized until needed."""
        assert options_broker._trading_client is None
        assert options_broker._options_client is None


# =============================================================================
# ORDER SUBMISSION TESTS (MOCKED)
# =============================================================================


class TestOptionOrderSubmission:
    """Test option order submission with mocks."""

    @pytest.mark.asyncio
    async def test_submit_option_order_limit(self, options_broker, sample_expiration):
        """Test submitting a limit option order."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "test-order-123"
        mock_order.symbol = "AAPL  250120C00150000"
        mock_order.side.value = "buy"
        mock_order.qty = "5"
        mock_order.type.value = "limit"
        mock_order.status.value = "pending_new"
        mock_order.created_at = "2025-01-01T10:00:00Z"
        mock_order.limit_price = "5.50"

        mock_client.submit_order.return_value = mock_order

        with patch.object(options_broker, "_get_trading_client", return_value=mock_client):
            result = await options_broker.submit_option_order(
                occ_symbol="AAPL  250120C00150000",
                side="buy",
                qty=5,
                order_type="limit",
                limit_price=5.50,
            )

        assert result is not None
        assert result["id"] == "test-order-123"
        assert result["side"] == "buy"
        assert result["qty"] == "5"

    @pytest.mark.asyncio
    async def test_submit_option_order_invalid_symbol_raises(self, options_broker):
        """Test that invalid symbol raises error."""
        with pytest.raises(InvalidContractError, match="Invalid OCC symbol"):
            await options_broker.submit_option_order(
                occ_symbol="AAPL", side="buy", qty=1  # Stock symbol, not option
            )

    @pytest.mark.asyncio
    async def test_submit_option_order_invalid_side_raises(self, options_broker):
        """Test that invalid side raises error."""
        with pytest.raises(ValueError, match="must be 'buy' or 'sell'"):
            await options_broker.submit_option_order(
                occ_symbol="AAPL  250120C00150000", side="hold", qty=1
            )

    @pytest.mark.asyncio
    async def test_submit_option_order_invalid_qty_raises(self, options_broker):
        """Test that invalid quantity raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            await options_broker.submit_option_order(
                occ_symbol="AAPL  250120C00150000", side="buy", qty=0
            )

    @pytest.mark.asyncio
    async def test_submit_option_order_limit_without_price_raises(self, options_broker):
        """Test that limit order without price raises error."""
        with pytest.raises(ValueError, match="limit_price required"):
            await options_broker.submit_option_order(
                occ_symbol="AAPL  250120C00150000", side="buy", qty=1, order_type="limit"
            )


# =============================================================================
# STRATEGY HELPER TESTS
# =============================================================================


class TestStrategyHelpers:
    """Test strategy helper methods."""

    @pytest.mark.asyncio
    async def test_buy_call(self, options_broker, sample_expiration):
        """Test buy_call helper."""
        mock_submit = AsyncMock(return_value={"id": "order-123"})

        with patch.object(options_broker, "submit_option_order", mock_submit):
            await options_broker.buy_call(
                underlying="AAPL",
                expiration=sample_expiration,
                strike=150.0,
                qty=1,
                limit_price=5.50,
            )

        mock_submit.assert_called_once()
        call_args = mock_submit.call_args
        assert call_args[1]["side"] == "buy"
        assert call_args[1]["order_type"] == "limit"

    @pytest.mark.asyncio
    async def test_buy_put(self, options_broker, sample_expiration):
        """Test buy_put helper."""
        mock_submit = AsyncMock(return_value={"id": "order-123"})

        with patch.object(options_broker, "submit_option_order", mock_submit):
            await options_broker.buy_put(
                underlying="AAPL",
                expiration=sample_expiration,
                strike=140.0,
                qty=1,
                limit_price=3.50,
            )

        mock_submit.assert_called_once()
        call_args = mock_submit.call_args
        assert "P" in call_args[1]["occ_symbol"]  # Put symbol

    @pytest.mark.asyncio
    async def test_sell_covered_call(self, options_broker, sample_expiration):
        """Test sell_covered_call helper."""
        mock_submit = AsyncMock(return_value={"id": "order-123"})

        with patch.object(options_broker, "submit_option_order", mock_submit):
            await options_broker.sell_covered_call(
                underlying="AAPL",
                expiration=sample_expiration,
                strike=160.0,
                qty=1,
                limit_price=2.00,
            )

        mock_submit.assert_called_once()
        call_args = mock_submit.call_args
        assert call_args[1]["side"] == "sell"
        assert "C" in call_args[1]["occ_symbol"]

    @pytest.mark.asyncio
    async def test_sell_cash_secured_put(self, options_broker, sample_expiration):
        """Test sell_cash_secured_put helper."""
        mock_submit = AsyncMock(return_value={"id": "order-123"})

        with patch.object(options_broker, "submit_option_order", mock_submit):
            await options_broker.sell_cash_secured_put(
                underlying="AAPL",
                expiration=sample_expiration,
                strike=140.0,
                qty=1,
                limit_price=3.00,
            )

        mock_submit.assert_called_once()
        call_args = mock_submit.call_args
        assert call_args[1]["side"] == "sell"
        assert "P" in call_args[1]["occ_symbol"]


# =============================================================================
# UTILITY METHOD TESTS
# =============================================================================


class TestUtilityMethods:
    """Test utility methods."""

    def test_calculate_max_loss_long_call(self, options_broker):
        """Test max loss for long call."""
        max_loss = options_broker.calculate_max_loss(
            option_type=OptionType.CALL, side="buy", strike=150.0, premium=5.00, qty=1
        )
        # Max loss = premium * 100 * qty
        assert max_loss == 500.0

    def test_calculate_max_loss_long_put(self, options_broker):
        """Test max loss for long put."""
        max_loss = options_broker.calculate_max_loss(
            option_type=OptionType.PUT, side="buy", strike=140.0, premium=3.00, qty=2
        )
        assert max_loss == 600.0  # 3.00 * 100 * 2

    def test_calculate_max_loss_short_call_unlimited(self, options_broker):
        """Test max loss for short call is unlimited."""
        max_loss = options_broker.calculate_max_loss(
            option_type=OptionType.CALL, side="sell", strike=150.0, premium=5.00, qty=1
        )
        assert max_loss == float("inf")

    def test_calculate_max_loss_short_put(self, options_broker):
        """Test max loss for short put."""
        max_loss = options_broker.calculate_max_loss(
            option_type=OptionType.PUT, side="sell", strike=140.0, premium=3.00, qty=1
        )
        # Max loss = (strike - premium) * 100
        assert max_loss == 13700.0  # (140 - 3) * 100

    def test_calculate_breakeven_long_call(self, options_broker):
        """Test breakeven for long call."""
        breakeven = options_broker.calculate_breakeven(
            option_type=OptionType.CALL, side="buy", strike=150.0, premium=5.00
        )
        assert breakeven == 155.0

    def test_calculate_breakeven_long_put(self, options_broker):
        """Test breakeven for long put."""
        breakeven = options_broker.calculate_breakeven(
            option_type=OptionType.PUT, side="buy", strike=140.0, premium=3.00
        )
        assert breakeven == 137.0

    def test_clear_cache(self, options_broker):
        """Test cache clearing."""
        # Add something to cache
        options_broker._chain_cache["test"] = ("data", "time")
        assert len(options_broker._chain_cache) == 1

        options_broker.clear_cache()
        assert len(options_broker._chain_cache) == 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_calculate_contract_value(self):
        """Test contract value calculation."""
        result = calculate_contract_value(strike=150.0, premium=5.00, qty=2)

        assert result["total_premium"] == 1000.0  # 5 * 100 * 2
        assert result["notional_value"] == 30000.0  # 150 * 100 * 2
        assert result["shares_controlled"] == 200  # 100 * 2
        assert result["cost_to_buy"] == 1000.0
        assert result["cash_for_csp"] == 30000.0

    def test_get_monthly_expiration(self):
        """Test getting monthly expiration."""
        exp = get_monthly_expiration(months_out=1)

        # Should be a Friday
        assert exp.weekday() == 4  # Friday is 4

        # Should be in the future
        assert exp > date.today()

        # Should be at least 2 weeks out (3rd Friday is always >= 15th)
        assert exp.day >= 15

    def test_get_monthly_expiration_multiple_months(self):
        """Test getting monthly expiration multiple months out."""
        exp1 = get_monthly_expiration(months_out=1)
        exp2 = get_monthly_expiration(months_out=2)
        exp3 = get_monthly_expiration(months_out=3)

        # Each should be further out
        assert exp2 > exp1
        assert exp3 > exp2

    def test_get_weekly_expiration(self):
        """Test getting weekly expiration."""
        exp = get_weekly_expiration(weeks_out=1)

        # Should be a Friday
        assert exp.weekday() == 4

        # Should be in the future
        assert exp > date.today()

    def test_get_weekly_expiration_multiple_weeks(self):
        """Test getting weekly expiration multiple weeks out."""
        exp1 = get_weekly_expiration(weeks_out=1)
        exp2 = get_weekly_expiration(weeks_out=2)

        # Should be exactly 7 days apart
        assert (exp2 - exp1).days == 7


# =============================================================================
# POSITION MANAGEMENT TESTS (MOCKED)
# =============================================================================


class TestPositionManagement:
    """Test position management with mocks."""

    @pytest.mark.asyncio
    async def test_get_option_positions(self, options_broker):
        """Test getting option positions."""
        mock_client = MagicMock()

        # Create mock positions (mix of stocks and options)
        mock_stock_pos = MagicMock()
        mock_stock_pos.symbol = "AAPL"

        mock_option_pos = MagicMock()
        mock_option_pos.symbol = "AAPL  250120C00150000"
        mock_option_pos.qty = "5"
        mock_option_pos.avg_entry_price = "5.50"
        mock_option_pos.market_value = "2900.00"
        mock_option_pos.unrealized_pl = "150.00"
        mock_option_pos.unrealized_plpc = "0.05"
        mock_option_pos.cost_basis = "2750.00"

        mock_client.get_all_positions.return_value = [mock_stock_pos, mock_option_pos]

        with patch.object(options_broker, "_get_trading_client", return_value=mock_client):
            positions = await options_broker.get_option_positions()

        # Should only return option positions
        assert len(positions) == 1
        assert positions[0]["underlying"] == "AAPL"
        assert positions[0]["strike"] == 150.0
        assert positions[0]["option_type"] == "call"

    @pytest.mark.asyncio
    async def test_cancel_option_order(self, options_broker):
        """Test canceling an option order."""
        mock_client = MagicMock()

        with patch.object(options_broker, "_get_trading_client", return_value=mock_client):
            result = await options_broker.cancel_option_order("order-123")

        assert result is True
        mock_client.cancel_order_by_id.assert_called_once_with("order-123")


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_build_symbol_with_spaces_in_underlying(self):
        """Test that underlying with spaces is handled."""
        # Should strip spaces
        symbol = OptionsBroker.build_occ_symbol(
            underlying=" AAPL ",
            expiration=date(2025, 1, 20),
            option_type=OptionType.CALL,
            strike=150.0,
        )
        assert symbol.startswith("AAPL")

    def test_parse_symbol_lowercase(self):
        """Test parsing lowercase symbol."""
        # OCC symbols should handle case insensitivity
        result = OptionsBroker.parse_occ_symbol("aapl  250120c00150000")
        assert result["underlying"] == "aapl"  # Preserves case from input
        assert result["option_type"] == OptionType.CALL

    def test_very_low_strike_symbol(self):
        """Test building symbol with very low strike."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="SNDL",
            expiration=date(2025, 1, 17),
            option_type=OptionType.CALL,
            strike=0.50,
        )
        assert "C00000500" in symbol

    def test_high_precision_strike_rounding(self):
        """Test that strike is properly rounded."""
        symbol = OptionsBroker.build_occ_symbol(
            underlying="SPY",
            expiration=date(2025, 1, 17),
            option_type=OptionType.CALL,
            strike=475.555,  # Will round to 475.56 -> 475560
        )
        # Should round to nearest cent
        parsed = OptionsBroker.parse_occ_symbol(symbol)
        assert parsed["strike"] == pytest.approx(475.555, abs=0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
