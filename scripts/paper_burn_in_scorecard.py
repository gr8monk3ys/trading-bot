#!/usr/bin/env python3
"""
Generate long-horizon paper trading burn-in scorecards and sign-off gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.research_registry import ResearchRegistry
from utils.paper_burn_in import (
    BurnInCriteria,
    build_paper_burn_in_scorecard,
    format_paper_burn_in_markdown,
)


def _parse_args() -> argparse.Namespace:
    defaults = BurnInCriteria()
    parser = argparse.ArgumentParser(description="Paper burn-in scorecard")
    parser.add_argument(
        "--paper-results-json",
        default=None,
        help="Path to paper results JSON payload",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Research experiment ID to read paper results from registry",
    )
    parser.add_argument(
        "--registry-path",
        default=".research/experiments",
        help="Experiment registry path (for --experiment-id)",
    )
    parser.add_argument(
        "--production-path",
        default=".research/production",
        help="Production registry path (for --experiment-id)",
    )
    parser.add_argument(
        "--parameter-registry-path",
        default=".research/parameters",
        help="Parameter registry path (for --experiment-id)",
    )
    parser.add_argument(
        "--artifacts-path",
        default=".research/artifacts",
        help="Artifacts path (for --experiment-id)",
    )
    parser.add_argument("--min-trading-days", type=int, default=defaults.min_trading_days)
    parser.add_argument("--min-trades", type=int, default=defaults.min_trades)
    parser.add_argument("--max-drawdown", type=float, default=defaults.max_drawdown)
    parser.add_argument(
        "--min-reconciliation-pass-rate",
        type=float,
        default=defaults.min_reconciliation_pass_rate,
    )
    parser.add_argument(
        "--max-operational-error-rate",
        type=float,
        default=defaults.max_operational_error_rate,
    )
    parser.add_argument(
        "--min-execution-quality-score",
        type=float,
        default=defaults.min_execution_quality_score,
    )
    parser.add_argument("--max-shadow-drift", type=float, default=defaults.max_shadow_drift)
    parser.add_argument(
        "--max-critical-slo-breaches",
        type=int,
        default=defaults.max_critical_slo_breaches,
    )
    parser.add_argument(
        "--require-manual-signoff",
        action="store_true",
        help="Require manual sign-off approval in paper results",
    )
    parser.add_argument(
        "--fail-on",
        choices=["none", "not_ready"],
        default="not_ready",
        help="Exit non-zero when scorecard is not ready",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output file path",
    )
    parser.add_argument(
        "--md-output",
        default=None,
        help="Optional markdown output file path",
    )
    return parser.parse_args()


def _load_paper_results(args: argparse.Namespace) -> Dict[str, Any]:
    if args.paper_results_json:
        payload = json.loads(Path(args.paper_results_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("paper_results"), dict):
            return payload["paper_results"]
        if isinstance(payload, dict):
            return payload
        raise ValueError("Paper results JSON must be an object payload")

    if args.experiment_id:
        registry = ResearchRegistry(
            registry_path=args.registry_path,
            production_path=args.production_path,
            parameter_registry_path=args.parameter_registry_path,
            artifacts_path=args.artifacts_path,
        )
        experiment = registry.experiments.get(args.experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment not found: {args.experiment_id}")
        if not isinstance(experiment.paper_results, dict):
            raise ValueError(f"Experiment has no paper_results: {args.experiment_id}")
        return experiment.paper_results

    raise ValueError("Provide either --paper-results-json or --experiment-id")


def _build_criteria(args: argparse.Namespace) -> BurnInCriteria:
    return BurnInCriteria(
        min_trading_days=max(0, int(args.min_trading_days)),
        min_trades=max(0, int(args.min_trades)),
        max_drawdown=max(0.0, float(args.max_drawdown)),
        min_reconciliation_pass_rate=max(0.0, float(args.min_reconciliation_pass_rate)),
        max_operational_error_rate=max(0.0, float(args.max_operational_error_rate)),
        min_execution_quality_score=max(0.0, float(args.min_execution_quality_score)),
        max_shadow_drift=max(0.0, float(args.max_shadow_drift)),
        max_critical_slo_breaches=max(0, int(args.max_critical_slo_breaches)),
        require_manual_signoff=bool(args.require_manual_signoff),
    )


def main() -> None:
    args = _parse_args()

    try:
        paper_results = _load_paper_results(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from e

    scorecard = build_paper_burn_in_scorecard(
        paper_results,
        criteria=_build_criteria(args),
    )
    markdown = format_paper_burn_in_markdown(scorecard)
    print(markdown)

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    if args.md_output:
        out = Path(args.md_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")

    should_fail = args.fail_on == "not_ready" and not bool(scorecard.get("ready_for_signoff"))
    raise SystemExit(1 if should_fail else 0)


if __name__ == "__main__":
    main()
