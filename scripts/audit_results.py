"""Audit committed backtest artifacts for the defects that inflate results.

The engine already writes its trade log in the schema `backtest-audit`
reads — symbol / side / quantity / price / timestamp / pnl per fill — so
nothing here reshapes anything. It loads `results/*.json` and asks
`backtest-audit` whether the numbers in them mean what the reports claim.

Usage:
    uv run python scripts/audit_results.py            # the current artifacts
    uv run python scripts/audit_results.py --all      # superseded runs too
    uv run python scripts/audit_results.py FILE ...   # specific artifacts

Exit status: 0 clean, 1 a blocking finding, 2 an artifact could not be read
at all. The last one is deliberately not a skip — a typo in a filename must
not drain the gate and still report success.

The superseded runs (`--all`) are expected to fail: they are the pre-fix
artifacts kept for the record, and their defects are still visible in the
trade log.
"""

from __future__ import annotations

import argparse
import json
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


# _audit outcomes, in order of how loudly they should be reported.
CLEAN, SKIPPED, BLOCKING, ERROR = "clean", "skipped", "blocking", "error"


def _audit(path: Path, verbose: bool) -> str:
    """Print one artifact's audit. Returns CLEAN / SKIPPED / BLOCKING / ERROR."""
    try:
        result = audit_file(path)
    except OSError as exc:
        # Missing or unreadable. Never a skip: an artifact this command was
        # told to audit and could not open is a failure of the gate itself,
        # and a silent skip here would report success having audited nothing.
        print(f"  ERROR   {path}: {exc}")
        return ERROR
    except json.JSONDecodeError as exc:
        # Corrupt for the same reason: a truncated artifact is not an
        # artifact with nothing to say.
        print(f"  ERROR   {path}: not valid JSON: {exc}")
        return ERROR
    except ValueError as exc:
        # Readable, but carries no trade log — the A/B roll-up is a summary
        # of two runs, not a run. Nothing to audit, and that is not a
        # failure. Counted and reported so it is never silent.
        print(f"  skipped {path.name}: {exc}")
        return SKIPPED

    print(render_text(result, color=sys.stdout.isatty(), verbose=verbose))
    return BLOCKING if result.blocking else CLEAN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="artifacts to audit (default: current runs)")
    parser.add_argument("--all", action="store_true", help="also audit the superseded pre-fix runs")
    parser.add_argument("-v", "--verbose", action="store_true", help="detail passing checks too")
    args = parser.parse_args(argv)

    if args.paths:
        if args.all:
            parser.error("--all selects the default artifact set; drop it or drop the paths")
        targets = [Path(p) for p in args.paths]
    else:
        names = CURRENT + (SUPERSEDED if args.all else [])
        targets = [RESULTS_DIR / n for n in names]

    outcomes = [(p, _audit(p, args.verbose)) for p in targets]
    errors = [p.name for p, o in outcomes if o is ERROR]
    blocking = [p.name for p, o in outcomes if o is BLOCKING]
    skipped = [p.name for p, o in outcomes if o is SKIPPED]

    print(
        f"{len(outcomes) - len(errors) - len(skipped)} audited, "
        f"{len(skipped)} skipped, {len(errors)} unreadable."
    )
    if errors:
        print(f"Could not audit: {', '.join(errors)}")
        return 2
    if blocking:
        print(f"Blocking findings in: {', '.join(blocking)}")
        return 1
    print("No blocking findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
