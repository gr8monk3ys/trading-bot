from engine.performance_metrics import Verdict, verdict


def _metrics(**kw):
    base = {"sharpe_ratio": 0.5, "win_rate": 0.5, "max_drawdown": -0.2, "total_return": 0.1}
    base.update(kw)
    return base


def test_quotable_when_data_complete_and_enough_trades():
    v = verdict(_metrics(), 60, {"symbols_requested": 4, "symbols_loaded": 4})
    assert v == Verdict("REPORTED") and v.quotable


def test_too_few_trades_is_inconclusive_with_the_reason():
    v = verdict(_metrics(), 26, {"symbols_requested": 4, "symbols_loaded": 4})
    assert v.status == "INCONCLUSIVE" and not v.quotable
    assert v.reasons == ("26 trades, below the 50-trade significance bar",)


def test_missing_symbols_dominate_everything_else():
    v = verdict(_metrics(), 500, {"symbols_requested": 4, "symbols_loaded": 3})
    assert v.status == "DATA_UNAVAILABLE"


def test_suspicious_statistics_are_named():
    v = verdict(_metrics(sharpe_ratio=4.0, win_rate=0.9, max_drawdown=0.0, total_return=0.5), 80)
    assert v.status == "INCONCLUSIVE" and len(v.reasons) == 3
