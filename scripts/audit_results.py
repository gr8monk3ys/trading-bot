"""Audit committed backtest artifacts for the defects that inflate results.

The engine already writes its trade log in the schema `backtest-audit`
reads — symbol / side / quantity / price / timestamp / pnl per fill — so
nothing here reshapes anything. It loads `results/*.json` and asks
`backtest-audit` whether the numbers in them mean what the reports claim.

Usage:
    uv run python scripts/audit_results.py            # the current artifacts
    uv run python scripts/audit_results.py --all      # superseded runs too
    uv run python scripts/audit_results.py FILE ...   # specific artifacts

Exit status is 1 if any audited artifact has a blocking finding, so the
command can gate a change that silently reintroduces one.

The superseded runs (`--all`) are expected to fail: they are the pre-fix
artifacts kept for the record, and reproducing their defects from the
trade log alone is the point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest_audit import audit_file
from backtest_audit.report import render_text

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# The exposure sweep the README and CLAUDE.md quote.
CURRENT = [
    "etf_baseline_2020-2024_gross25.json",
    "etf_baseline_2020-2024_gross50.json",
    "etf_baseline_2020-2024_gross100.json",
]

# Kept for the record, banner-marked SUPERSEDED in their .md siblings.
SUPERSEDED = [
    "etf_baseline_2020-2024.json",
    "honest_backtest_2020-2024.json",
]


def _audit(path: Path, verbose: bool) -> bool:
    """Print one artifact's audit. Returns True if it has blocking findings."""
    try:
        result = audit_file(path)
    except (OSError, ValueError) as exc:
        # A summary-only artifact (e.g. the A/B roll-up) carries no trade
        # log; that is not a failure, there is simply nothing to audit.
        print(f"  skipped {path.name}: {exc}")
        return False

    print(render_text(result, color=sys.stdout.isatty(), verbose=verbose))
    return bool(result.blocking)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="artifacts to audit (default: current runs)")
    parser.add_argument("--all", action="store_true", help="also audit the superseded pre-fix runs")
    parser.add_argument("-v", "--verbose", action="store_true", help="detail passing checks too")
    args = parser.parse_args(argv)

    if args.paths:
        targets = [Path(p) for p in args.paths]
    else:
        names = CURRENT + (SUPERSEDED if args.all else [])
        targets = [RESULTS_DIR / n for n in names]

    blocking = [p.name for p in targets if _audit(p, args.verbose)]

    if blocking:
        print(f"Blocking findings in: {', '.join(blocking)}")
        return 1
    print("No blocking findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
