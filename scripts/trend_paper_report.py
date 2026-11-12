"""Summarise the trend candidate's paper fills against the backtest's cost assumption.

Usage:
    uv run python scripts/trend_paper_report.py            # reads logs/trend_paper.csv

``scripts/run_trend_paper.py`` appends one row per order. Each filled row
carries the last trade price read just before submission (``ref_price``) and
the fill price. Slippage is signed so that a positive number is a cost:

    buy:  (fill - ref) / ref * 10_000
    sell: (ref - fill) / ref * 10_000

The gate charges 50 bps per fill (ADR 0013). If paper fills cost well under
that, the gate was conservative; if they cost more, every gate verdict was
too kind. Paper fills are simulated by Alpaca, so this checks the plumbing
and gives a floor, not a measurement of live market impact.
"""

from __future__ import annotations

import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "logs" / "trend_paper.csv"
GATE_COST_BPS = 50.0

COLUMNS = [
    "run_at",
    "symbol",
    "side",
    "qty_requested",
    "qty_filled",
    "fill_price",
    "status",
    "reason",
    "ref_price",
    "slippage_bps",
]


def slippage_bps(side: str, ref_price: float | None, fill_price: float | None) -> float | None:
    """Signed cost of one fill in basis points; None when either price is missing."""
    if not ref_price or not fill_price or ref_price <= 0:
        return None
    if side.lower() == "buy":
        return (fill_price - ref_price) / ref_price * 10_000
    if side.lower() == "sell":
        return (ref_price - fill_price) / ref_price * 10_000
    return None


@dataclass(frozen=True)
class Summary:
    runs: int
    orders: int
    fills_with_ref: int
    mean_bps: float | None
    median_bps: float | None
    worst_bps: float | None


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "", "None") else None
    except ValueError:
        return None


def summarize(rows: list[dict[str, str]]) -> Summary:
    costs = [c for c in (_float(r.get("slippage_bps")) for r in rows) if c is not None]
    return Summary(
        runs=len({r["run_at"] for r in rows}),
        orders=len(rows),
        fills_with_ref=len(costs),
        mean_bps=statistics.fmean(costs) if costs else None,
        median_bps=statistics.median(costs) if costs else None,
        worst_bps=max(costs) if costs else None,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read the log. Rows written before ref_price existed have 8 fields; they count
    as orders but carry no slippage."""
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for fields in csv.reader(f):
            if not fields or fields[0] == "run_at":
                continue
            rows.append(dict(zip(COLUMNS, fields, strict=False)))
    return rows


def render(s: Summary) -> str:
    if s.orders == 0:
        return "No paper orders logged yet."

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:+.1f} bps"

    lines = [
        f"Rebalances: {s.runs}   orders: {s.orders}   fills with a reference price: "
        f"{s.fills_with_ref}",
        f"Slippage per fill: mean {fmt(s.mean_bps)}, median {fmt(s.median_bps)}, "
        f"worst {fmt(s.worst_bps)}",
        f"Gate assumption: {GATE_COST_BPS:.0f} bps per fill.",
    ]
    if s.mean_bps is not None:
        verdict = "under" if s.mean_bps <= GATE_COST_BPS else "OVER"
        lines.append(f"Mean paper cost is {verdict} the gate assumption.")
    if s.fills_with_ref < 30:
        lines.append(f"Only {s.fills_with_ref} priced fills; under 30 this is anecdote.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    path = Path(argv[0]) if argv else DEFAULT_CSV
    if not path.exists():
        print(f"No log at {path}. Run scripts/run_trend_paper.py first.")
        return 1
    print(render(summarize(read_rows(path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
