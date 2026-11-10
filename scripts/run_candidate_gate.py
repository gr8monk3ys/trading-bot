"""Run a strategy candidate through the pass bar, on the dev window or the locked holdout.

Usage:
    uv run python scripts/run_candidate_gate.py --candidate trend --window dev
    uv run python scripts/run_candidate_gate.py --candidate trend --window holdout   # once

The holdout (2020-01-01 .. 2026-10-30) is the shared test set. The first
holdout run writes results/holdout_lock.json with the candidate, commit and
parameters. Any later holdout run, for any candidate, refuses unless
``--reopen "<reason>"`` is given; the reason is appended so every extra look
stays visible in git (commit the lock file with ``git add -f``). Only open
the holdout for a candidate that passed dev (ADR 0011).

Pass bar (ADR 0011), on the window only, after the engine's costs
(40 bps slippage + 10 bps spread per fill, config.BASELINE), with dividends
credited to positions held into each ex-date:
    1. Beats SPY total return (dividends reinvested) held at the strategy's
       average gross exposure.
    2. At least 100 closed round trips (fills that realise P&L).
    3. Beats that same-exposure SPY in both halves of the window.
Alpha and its t-stat are reported, not gated: a few years cannot prove a
modest Sharpe either way (years needed ~ (1.645 / Sharpe)^2).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import BASELINE  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# name -> (module, class). Each class declares its frozen Params (ADR 0010).
CANDIDATES = {
    "trend": ("strategies.trend_following_strategy", "TrendFollowingStrategy"),
    "volmanaged": ("strategies.vol_managed_strategy", "VolManagedStrategy"),
}
WINDOWS = {
    "dev": ("2008-01-01", "2019-12-31"),
    "holdout": ("2020-01-01", "2026-10-30"),
}
WARMUP_DAYS = 400  # calendar days loaded before the window so 12-month signals are ready
MIN_ROUND_TRIPS = 100
RESULTS = REPO_ROOT / "results"
LOCK = RESULTS / "holdout_lock.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _strategy_class(candidate: str):
    module, name = CANDIDATES[candidate]
    return getattr(importlib.import_module(module), name)


def _check_lock(candidate: str, reopen: str | None) -> None:
    cls = _strategy_class(candidate)
    entry = {
        "at": _now(),
        "candidate": candidate,
        "commit": _git_sha(),
        "params": cls.Params.defaults(),
    }
    if LOCK.exists():
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        if not reopen:
            first = lock["openings"][0]
            raise SystemExit(
                f"Holdout already opened on {first['at']} for {first['candidate']} at "
                f"{first['commit'][:7]}. Another look spends the test set; "
                'pass --reopen "<reason>" to record why.'
            )
        entry["reason"] = reopen
        lock["openings"].append(entry)
    else:
        lock = {"window": WINDOWS["holdout"], "openings": [entry]}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def _prices(symbol: str, start: str, end: str, adjusted: bool) -> pd.Series:
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=adjusted, progress=False)
    s = df["Close"].squeeze().dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def _stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    sd = r.std()
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "annual_return": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0,
        "sharpe_rf0": float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }


async def _run(candidate: str, window: str) -> dict:
    from engine.backtest_engine import BacktestEngine
    from engine.historical_bars import load_dividends, resolve_bars_source

    cls = _strategy_class(candidate)
    params = cls.Params.defaults()
    start, end = WINDOWS[window]
    load_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=WARMUP_DAYS)).strftime(
        "%Y-%m-%d"
    )
    symbols = list(params["symbols"])
    data_broker, source = await resolve_bars_source(preferred=BASELINE["data_source"])
    dividends = await load_dividends(symbols)
    engine = BacktestEngine(broker=data_broker)
    result = await engine.run_backtest(
        strategy_class=cls,
        symbols=symbols,
        start_date=datetime.strptime(load_start, "%Y-%m-%d"),
        end_date=datetime.strptime(end, "%Y-%m-%d"),
        initial_capital=BASELINE["initial_capital"],
        execution_profile="realistic",
        dividends=dividends,
    )

    # Session dates from SPY's bars over the same load range (the engine's own
    # session list). The curve is stamped one session late: value i+1 is the
    # close of session i.
    end_excl = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    spy_tr = _prices("SPY", load_start, end_excl, adjusted=True)
    sessions = spy_tr.index
    curve = list(result["equity_curve"])
    n = len(sessions)
    if len(curve) not in (n, n + 1):
        raise SystemExit(f"curve length {len(curve)} does not match {n} sessions")
    vals = curve[1 : n + 1] if len(curve) == n + 1 else curve[1:] + [curve[-1]]
    eq = pd.Series(vals, index=sessions)
    expo_vals = list(result["exposure_curve"])[:n]
    expo = pd.Series(expo_vals, index=sessions[: len(expo_vals)])

    first = int(np.argmax(sessions >= pd.Timestamp(start)))
    eq_w = eq.iloc[first - 1 :]  # include the prior close so day one has a return
    strat_r = eq_w.pct_change().dropna()
    spy_r = spy_tr.pct_change().reindex(strat_r.index).fillna(0)
    gross = float(expo[expo.index >= pd.Timestamp(start)].mean())
    same_r = gross * spy_r
    spy_total = float((1 + spy_r).prod() - 1)

    half = len(strat_r) // 2
    halves = []
    for part in (slice(0, half), slice(half, None)):
        s = float((1 + strat_r.iloc[part]).prod() - 1)
        b = float((1 + same_r.iloc[part]).prod() - 1)
        idx = strat_r.index[part]
        halves.append(
            {
                "from": str(idx[0].date()),
                "to": str(idx[-1].date()),
                "strategy": s,
                "spy_same_exposure": b,
                "beats": s > b,
            }
        )

    x = spy_r.to_numpy(dtype=float)
    y = strat_r.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    cov = resid.var(ddof=2) * np.linalg.inv(X.T @ X)

    trades = result.get("trades", [])
    win_start = pd.Timestamp(start)
    closed = [
        t
        for t in trades
        if abs(float(t.get("pnl") or 0)) > 1e-9
        and pd.Timestamp(str(t.get("timestamp"))[:10]) >= win_start
    ]

    strat = _stats(strat_r)
    same = _stats(same_r)
    checks = {
        "beats_spy_same_exposure": strat["total_return"] > same["total_return"],
        f"at_least_{MIN_ROUND_TRIPS}_round_trips": len(closed) >= MIN_ROUND_TRIPS,
        "beats_in_both_halves": all(h["beats"] for h in halves),
    }
    return {
        "generated_at": _now(),
        "candidate": candidate,
        "strategy_class": cls.__name__,
        "window": window,
        "start": start,
        "end": end,
        "data_source": source,
        "commit": _git_sha(),
        "params": params,
        "costs_bps": {"slippage": BASELINE["slippage_bps"], "spread": BASELINE["spread_bps"]},
        "dividends_credited": result.get("dividends_credited", 0.0),
        "avg_gross_exposure": gross,
        "strategy": strat,
        "spy_same_exposure": same,
        "spy_buy_hold_total_return": spy_total,
        "alpha_annual": float(coef[0] * 252),
        "alpha_t": float(coef[0] / math.sqrt(cov[0, 0])),
        "beta": float(coef[1]),
        "round_trips": len(closed),
        "fills": len(trades),
        "halves": halves,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "equity": {str(d.date()): round(float(v), 2) for d, v in eq_w.items()},
        "trades": [
            {
                "date": str(t.get("timestamp"))[:10],
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "qty": t.get("quantity"),
                "pnl": round(float(t.get("pnl") or 0), 2),
            }
            for t in trades
        ],
    }


def _markdown(a: dict) -> str:
    s, b = a["strategy"], a["spy_same_exposure"]
    rows = "\n".join(
        f"| {h['from']} .. {h['to']} | {h['strategy']:+.2%} | {h['spy_same_exposure']:+.2%} | "
        f"{'yes' if h['beats'] else 'no'} |"
        for h in a["halves"]
    )
    checks = "\n".join(f"- [{'x' if v else ' '}] {k}" for k, v in a["checks"].items())
    return f"""# Candidate `{a['candidate']}` ({a['strategy_class']}), {a['window']} window ({a['start']} .. {a['end']})

**Verdict: {a['verdict']}** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

{checks}

| | Strategy | SPY total return at same exposure ({a['avg_gross_exposure']:.0%}) |
|---|---|---|
| Total return | {s['total_return']:+.2%} | {b['total_return']:+.2%} |
| Annual return | {s['annual_return']:+.2%} | {b['annual_return']:+.2%} |
| Sharpe (rf=0) | {s['sharpe_rf0']:.2f} | {b['sharpe_rf0']:.2f} |
| Max drawdown | {s['max_drawdown']:.2%} | {b['max_drawdown']:.2%} |

SPY buy-and-hold, 100% invested, dividends reinvested: {a['spy_buy_hold_total_return']:+.2%}.
Alpha {a['alpha_annual']:+.2%}/yr (t = {a['alpha_t']:.2f}), beta {a['beta']:.2f}.
Closed round trips: {a['round_trips']} ({a['fills']} fills incl. warm-up and final liquidation).
Dividends credited: ${a['dividends_credited']:,.0f}.
Costs: {a['costs_bps']['slippage']} bps slippage + {a['costs_bps']['spread']} bps spread per fill.
Data: `{a['data_source']}`, commit `{a['commit'][:7]}`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
{rows}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    ap.add_argument("--window", choices=sorted(WINDOWS), required=True)
    ap.add_argument("--reopen", help="reason for another look at the holdout")
    args = ap.parse_args()
    if args.window == "holdout":
        _check_lock(args.candidate, args.reopen)
    artifact = asyncio.run(_run(args.candidate, args.window))
    stem = RESULTS / f"gate_{args.candidate}_{args.window}"
    stem.with_suffix(".json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(_markdown(artifact), encoding="utf-8")
    print(_markdown(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
