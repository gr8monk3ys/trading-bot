from engine.backtest_engine import BacktestEngine


def test_calculate_trade_pnl_simple_roundtrip():
    engine = BacktestEngine()

    trades = [
        {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 100.0},
        {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 110.0},
    ]

    records = engine._calculate_trade_pnl(trades)

    assert records
    assert records[0]["symbol"] == "AAPL"
    assert records[0]["pnl"] == 0
    assert records[1]["pnl"] == 100.0


def test_calculate_trade_pnl_partial_sell():
    engine = BacktestEngine()

    trades = [
        {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 100.0},
        {"symbol": "AAPL", "side": "sell", "quantity": 5, "price": 105.0},
        {"symbol": "AAPL", "side": "sell", "quantity": 5, "price": 95.0},
    ]

    records = engine._calculate_trade_pnl(trades)

    assert len(records) == 3
    assert records[0]["side"] == "buy"
    assert sum(r["pnl"] for r in records[1:]) == 0.0


def test_calculate_trade_pnl_sell_without_position():
    engine = BacktestEngine()

    trades = [
        {"symbol": "AAPL", "side": "sell", "quantity": 5, "price": 100.0},
    ]

    records = engine._calculate_trade_pnl(trades)

    assert len(records) == 1
    assert records[0]["pnl"] == 0


def test_calculate_trade_pnl_multiple_symbols():
    engine = BacktestEngine()

    trades = [
        {"symbol": "AAPL", "side": "buy", "quantity": 5, "price": 100.0},
        {"symbol": "MSFT", "side": "buy", "quantity": 5, "price": 200.0},
        {"symbol": "AAPL", "side": "sell", "quantity": 5, "price": 110.0},
        {"symbol": "MSFT", "side": "sell", "quantity": 5, "price": 190.0},
    ]

    records = engine._calculate_trade_pnl(trades)

    assert len(records) == 4
    assert records[2]["pnl"] == 50.0
    assert records[3]["pnl"] == -50.0


def test_calculate_trade_pnl_multiple_buys_avg_price():
    engine = BacktestEngine()

    trades = [
        {"symbol": "AAPL", "side": "buy", "quantity": 5, "price": 100.0},
        {"symbol": "AAPL", "side": "buy", "quantity": 5, "price": 120.0},
        {"symbol": "AAPL", "side": "sell", "quantity": 10, "price": 130.0},
    ]

    records = engine._calculate_trade_pnl(trades)

    # Avg entry price = 110, pnl = (130-110)*10 = 200
    assert records[-1]["pnl"] == 200.0
