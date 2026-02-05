from engine.backtest_engine import BacktestEngine


def test_calculate_sharpe_from_equity_insufficient_data():
    engine = BacktestEngine()
    assert engine._calculate_sharpe_from_equity([100]) == 0.0


def test_calculate_sharpe_from_equity_zero_variance():
    engine = BacktestEngine()
    assert engine._calculate_sharpe_from_equity([100, 100, 100]) == 0.0


def test_calculate_sharpe_from_equity_positive():
    engine = BacktestEngine()
    sharpe = engine._calculate_sharpe_from_equity([100, 102, 101, 104])

    assert sharpe > 0


def test_calculate_sharpe_from_equity_negative():
    engine = BacktestEngine()
    sharpe = engine._calculate_sharpe_from_equity([100, 98, 97, 95])

    assert sharpe < 0


def test_calculate_sharpe_from_equity_handles_small_variance():
    engine = BacktestEngine()
    sharpe = engine._calculate_sharpe_from_equity([100, 100.01, 100.02, 100.01])

    assert sharpe != 0
