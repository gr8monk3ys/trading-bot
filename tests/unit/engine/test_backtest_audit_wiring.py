"""The engine's trade log must stay readable by `backtest-audit`.

`backtest-audit` reads a trade log and reports the defects that silently
inflate a backtest — the same ones this repo shipped for months (#74, #84).
Its input schema is already what `BacktestEngine.run_backtest` records per
fill, so nothing is reshaped here: the artifact these tests write is the
engine's own `trades`/`equity_curve`/`exposure` output, serialised.

These tests prove the wiring, not the strategy. A run may legitimately
audit as NOT TRUSTWORTHY — two committed artifacts do, on purpose — so
nothing here asserts a verdict. What is asserted is that the audit reads
the engine's output, judges every check it can, and returns a report.

The real verdicts on the committed artifacts come from
`uv run python scripts/audit_results.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
from backtest_audit import audit_file, load

from engine.backtest_engine import BacktestEngine
from tests.unit.engine.test_backtest_end_liquidation import (
    _bars_by_symbol,
    _BuyAndHoldStrategy,
)

VERDICTS = {
    "TRUSTWORTHY",
    "TRUSTWORTHY (some checks unjudged)",
    "TRUSTWORTHY WITH CAVEATS",
    "NOT TRUSTWORTHY",
}


async def _run_and_export(path):
    """Run a short deterministic backtest and write its audit artifact."""
    symbols = ["AAA", "BBB"]
    start = datetime(2024, 1, 2)
    n_days = 10
    get_bars, _ = _bars_by_symbol(symbols, start, n=n_days)

    data_broker = MagicMock()
    data_broker.get_bars = AsyncMock(side_effect=get_bars)

    engine = BacktestEngine(broker=data_broker)
    result = await engine.run_backtest(
        strategy_class=_BuyAndHoldStrategy,
        symbols=symbols,
        start_date=start,
        end_date=start + pd.Timedelta(days=n_days + 1),
        initial_capital=100_000,
        execution_profile="realistic",
    )

    # Same field set `scripts/run_etf_baseline.py` writes. The per-fill
    # records go in untouched — that they are already the audit's schema
    # is the thing under test.
    artifact = {
        "trades": result["trades"],
        "equity_curve": [float(v) for v in result["equity_curve"]],
        "exposure": {
            "avg_gross_exposure": result["avg_gross_exposure"],
            "peak_gross_exposure": result["peak_gross_exposure"],
        },
        "metrics": {"total_return": result["total_return"]},
    }
    path.write_text(json.dumps(artifact, indent=2, default=str))
    return result


async def test_engine_trade_log_loads_unreshaped(tmp_path):
    """Every fill the engine recorded survives the round trip into the audit."""
    artifact = tmp_path / "backtest.json"
    result = await _run_and_export(artifact)

    assert result["trades"], "the fixture backtest recorded no trades"

    loaded = load(artifact)
    assert len(loaded.trades) == len(result["trades"])
    assert {t.symbol for t in loaded.trades} == {t["symbol"] for t in result["trades"]}
    assert {t.side for t in loaded.trades} <= {"buy", "sell"}
    assert loaded.equity_curve[0] == 100_000
    assert loaded.exposure["avg_gross_exposure"] == result["avg_gross_exposure"]


async def test_audit_runs_over_engine_output(tmp_path):
    """The audit judges the artifact and returns a verdict — any verdict."""
    artifact = tmp_path / "backtest.json"
    await _run_and_export(artifact)

    report = audit_file(artifact)

    assert report.verdict in VERDICTS
    assert report.findings, "the audit produced no findings at all"
    for finding in report.findings:
        assert finding.check_id and finding.title and finding.summary

    # Exporting the full equity curve (not just the summary) is what makes
    # the reconciliation judgeable; an unjudged check is not a passed one.
    pnl = next(f for f in report.findings if f.check_id == "pnl-reconciliation")
    assert pnl.severity.value != "skipped", pnl.summary


async def test_btaudit_cli_parses_engine_output(tmp_path):
    """`btaudit --json` reads the same artifact, for CI use."""
    artifact = tmp_path / "backtest.json"
    await _run_and_export(artifact)

    proc = subprocess.run(
        [sys.executable, "-m", "backtest_audit.cli", "--json", str(artifact)],
        capture_output=True,
        text=True,
        check=False,
    )

    # 0 = clean, 1 = blocking findings. 2 is "cannot audit" — the failure
    # this test exists to catch.
    assert proc.returncode in (0, 1), proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["verdict"] in VERDICTS
    assert payload["findings"]
