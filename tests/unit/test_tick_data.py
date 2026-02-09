"""
Tests for Tick Data Integration

Tests:
- Trade and Quote dataclasses
- Tick aggregation to bars
- Microstructure calculations
- Volume/dollar bar construction
- Order flow imbalance
"""

from datetime import datetime, timedelta

import pytest

from data.tick_data import (
    AggregatedBar,
    Exchange,
    MicrostructureSnapshot,
    Quote,
    TAQDataParser,
    TickAggregator,
    Trade,
    create_tick_manager,
)


class TestTrade:
    """Tests for Trade dataclass."""

    def test_create_trade(self):
        """Test creating a trade."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.25,
            size=100,
            exchange=Exchange.NYSE,
        )

        assert trade.symbol == "AAPL"
        assert trade.price == 150.25
        assert trade.size == 100

    def test_trade_notional(self):
        """Test notional value calculation."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.0,
            size=100,
            exchange=Exchange.NYSE,
        )

        assert trade.notional == 15000.0

    def test_trade_with_conditions(self):
        """Test trade with condition flags."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.0,
            size=50,
            exchange=Exchange.NYSE,
            conditions=["I", "F"],
            is_odd_lot=True,
            is_intermarket_sweep=True,
        )

        assert trade.is_odd_lot is True
        assert trade.is_intermarket_sweep is True
        assert not trade.is_eligible_for_high_low()


class TestQuote:
    """Tests for Quote dataclass."""

    def test_create_quote(self):
        """Test creating a quote."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.05,
            ask_size=300,
        )

        assert quote.bid_price == 150.00
        assert quote.ask_price == 150.05

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.10,
            ask_size=300,
        )

        assert quote.mid_price == 150.05

    def test_spread_calculation(self):
        """Test spread calculation."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.00,
            bid_size=500,
            ask_price=150.10,
            ask_size=300,
        )

        # Use approximate comparison for floating point
        assert abs(quote.spread - 0.10) < 0.0001
        # spread_bps = (0.10 / 150.05) * 10000 ≈ 6.66 bps
        assert 6.0 < quote.spread_bps < 7.0

    def test_crossed_market_detection(self):
        """Test crossed market detection."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.10,
            bid_size=500,
            ask_price=150.00,
            ask_size=300,
        )

        assert quote.is_crossed() is True


class TestAggregatedBar:
    """Tests for AggregatedBar dataclass."""

    def test_create_bar(self):
        """Test creating an aggregated bar."""
        bar = AggregatedBar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=10000,
            vwap=150.3,
            trade_count=50,
        )

        assert bar.open == 150.0
        assert bar.high == 151.0
        assert bar.trade_count == 50

    def test_dollar_volume(self):
        """Test dollar volume calculation."""
        bar = AggregatedBar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=10000,
            vwap=150.0,
            trade_count=50,
        )

        assert bar.dollar_volume == 1500000.0

    def test_order_flow_imbalance(self):
        """Test order flow imbalance calculation."""
        bar = AggregatedBar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=10000,
            vwap=150.0,
            trade_count=50,
            buy_volume=6000,
            sell_volume=4000,
        )

        # (6000 - 4000) / (6000 + 4000) = 0.2
        assert bar.order_flow_imbalance == 0.2


class TestTickAggregator:
    """Tests for TickAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create a tick aggregator."""
        return TickAggregator()

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        trades = []

        for i in range(100):
            trades.append(Trade(
                symbol="AAPL",
                timestamp=base_time + timedelta(seconds=i),
                price=150.0 + (i % 10) * 0.01,
                size=100 + (i % 5) * 10,
                exchange=Exchange.NYSE,
            ))

        return trades

    @pytest.fixture
    def sample_quotes(self):
        """Create sample quotes."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        quotes = []

        for i in range(100):
            quotes.append(Quote(
                symbol="AAPL",
                timestamp=base_time + timedelta(seconds=i),
                bid_price=149.95 + (i % 10) * 0.01,
                bid_size=500,
                ask_price=150.00 + (i % 10) * 0.01,
                ask_size=300,
            ))

        return quotes

    def test_add_trade(self, aggregator):
        """Test adding trades."""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.0,
            size=100,
            exchange=Exchange.NYSE,
        )

        aggregator.add_trade(trade)
        assert len(aggregator._trades.get("AAPL", [])) == 1

    def test_add_quote(self, aggregator):
        """Test adding quotes."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.0,
            bid_size=500,
            ask_price=150.05,
            ask_size=300,
        )

        aggregator.add_quote(quote)
        assert len(aggregator._quotes.get("AAPL", [])) == 1

    def test_classify_trade_buy(self, aggregator):
        """Test trade classification as buy."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.0,
            bid_size=500,
            ask_price=150.10,
            ask_size=300,
        )
        aggregator.add_quote(quote)

        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.08,  # Above mid (150.05)
            size=100,
            exchange=Exchange.NYSE,
        )

        assert aggregator.classify_trade(trade, quote) == "buy"

    def test_classify_trade_sell(self, aggregator):
        """Test trade classification as sell."""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=150.0,
            bid_size=500,
            ask_price=150.10,
            ask_size=300,
        )

        trade = Trade(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.02,  # Below mid (150.05)
            size=100,
            exchange=Exchange.NYSE,
        )

        assert aggregator.classify_trade(trade, quote) == "sell"

    def test_aggregate_to_bar(self, aggregator, sample_trades, sample_quotes):
        """Test aggregating to a single bar."""
        for trade in sample_trades:
            aggregator.add_trade(trade)
        for quote in sample_quotes:
            aggregator.add_quote(quote)

        bar = aggregator.aggregate_to_bar(
            symbol="AAPL",
            start=datetime(2024, 1, 15, 10, 0, 0),
            end=datetime(2024, 1, 15, 10, 1, 0),
        )

        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.trade_count > 0
        assert bar.volume > 0

    def test_aggregate_time_bars(self, aggregator, sample_trades, sample_quotes):
        """Test time bar aggregation."""
        for trade in sample_trades:
            aggregator.add_trade(trade)
        for quote in sample_quotes:
            aggregator.add_quote(quote)

        bars = aggregator.aggregate_time_bars(
            symbol="AAPL",
            start=datetime(2024, 1, 15, 10, 0, 0),
            end=datetime(2024, 1, 15, 10, 2, 0),
            bar_size_seconds=30,
        )

        assert len(bars) >= 2  # At least 2 bars in 2 minutes with 30s bars

    def test_aggregate_volume_bars(self, aggregator, sample_trades):
        """Test volume bar aggregation."""
        for trade in sample_trades:
            aggregator.add_trade(trade)

        bars = aggregator.aggregate_volume_bars(
            symbol="AAPL",
            volume_threshold=1000,
        )

        # Each bar should have ~1000 shares
        for bar in bars:
            assert bar.volume >= 1000 or bar == bars[-1]

    def test_aggregate_dollar_bars(self, aggregator, sample_trades):
        """Test dollar bar aggregation."""
        for trade in sample_trades:
            aggregator.add_trade(trade)

        bars = aggregator.aggregate_dollar_bars(
            symbol="AAPL",
            dollar_threshold=150000.0,
        )

        assert len(bars) >= 1

    def test_microstructure_snapshot(self, aggregator, sample_trades, sample_quotes):
        """Test microstructure snapshot generation."""
        for trade in sample_trades[:50]:
            aggregator.add_trade(trade)
        for quote in sample_quotes[:50]:
            aggregator.add_quote(quote)

        snapshot = aggregator.get_microstructure_snapshot(
            symbol="AAPL",
            as_of=datetime(2024, 1, 15, 10, 0, 45),
            lookback_seconds=60,
        )

        assert snapshot is not None
        assert snapshot.symbol == "AAPL"
        assert snapshot.volume_1min > 0

    def test_clear_data(self, aggregator, sample_trades):
        """Test clearing data."""
        for trade in sample_trades:
            aggregator.add_trade(trade)

        aggregator.clear("AAPL")
        assert len(aggregator._trades.get("AAPL", [])) == 0


class TestMicrostructureSnapshot:
    """Tests for MicrostructureSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        snapshot = MicrostructureSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.0,
            ask=150.05,
            last_trade=150.02,
            vwap=150.01,
            bid_size=500,
            ask_size=300,
            spread_bps=3.33,
            volume_1min=5000,
            trade_count_1min=25,
            avg_trade_size_1min=200,
            buy_volume_1min=3000,
            sell_volume_1min=2000,
            order_flow_imbalance=0.2,
            realized_vol_1min=0.001,
            high_1min=150.10,
            low_1min=149.95,
        )

        assert snapshot.symbol == "AAPL"
        assert snapshot.order_flow_imbalance == 0.2

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        snapshot = MicrostructureSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid=150.0,
            ask=150.05,
            last_trade=150.02,
            vwap=150.01,
            bid_size=500,
            ask_size=300,
            spread_bps=3.33,
            volume_1min=5000,
            trade_count_1min=25,
            avg_trade_size_1min=200,
            buy_volume_1min=3000,
            sell_volume_1min=2000,
            order_flow_imbalance=0.2,
            realized_vol_1min=0.001,
            high_1min=150.10,
            low_1min=149.95,
        )

        d = snapshot.to_dict()
        assert "symbol" in d
        assert "mid" in d
        assert d["order_flow_imbalance"] == 0.2


class TestExchange:
    """Tests for Exchange enum."""

    def test_all_exchanges_exist(self):
        """Test all expected exchanges exist."""
        expected = ["NYSE", "NASDAQ", "ARCA", "BATS", "IEX", "EDGX"]
        for name in expected:
            assert hasattr(Exchange, name)

    def test_from_code(self):
        """Test getting exchange from code."""
        assert Exchange.from_code("N") == Exchange.NYSE
        assert Exchange.from_code("Q") == Exchange.NASDAQ


class TestTickDataManager:
    """Tests for TickDataManager class."""

    def test_create_manager(self):
        """Test creating a manager."""
        manager = create_tick_manager()
        assert manager is not None

    def test_aggregator_access(self):
        """Test accessing aggregator."""
        manager = create_tick_manager()
        assert manager.aggregator is not None

    def test_clear_cache(self):
        """Test clearing cache."""
        manager = create_tick_manager()
        manager.clear_cache()  # Should not raise


class TestTAQDataParser:
    """Tests for TAQDataParser class."""

    def test_condition_codes_defined(self):
        """Test that condition codes are defined."""
        parser = TAQDataParser()
        assert "@" in parser.CONDITION_CODES
        assert parser.CONDITION_CODES["@"] == "regular"
        assert parser.CONDITION_CODES["I"] == "odd_lot"
