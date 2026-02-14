#!/usr/bin/env python3
"""
Strategy promotion checklist + CI gate.

This script evaluates strict promotion readiness for a research experiment.
It exits non-zero when required criteria are not met.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path for local execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.research_registry import ResearchRegistry
from utils.paper_burn_in import build_paper_burn_in_scorecard
from utils.shadow_drift_dashboard import build_shadow_drift_dashboard


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy promotion gate")
    parser.add_argument("--experiment-id", required=True, help="Research experiment ID")
    parser.add_argument(
        "--registry-path",
        default=".research/experiments",
        help="Experiment registry path",
    )
    parser.add_argument(
        "--production-path",
        default=".research/production",
        help="Production registry path",
    )
    parser.add_argument(
        "--parameter-registry-path",
        default=".research/parameters",
        help="Parameter registry path",
    )
    parser.add_argument(
        "--artifacts-path",
        default=".research/artifacts",
        help="Artifacts path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict readiness checks",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for checklist",
    )
    return parser.parse_args()


def run_gate(args: argparse.Namespace) -> int:
    registry = ResearchRegistry(
        registry_path=args.registry_path,
        production_path=args.production_path,
        parameter_registry_path=args.parameter_registry_path,
        artifacts_path=args.artifacts_path,
    )

    checklist = registry.generate_promotion_checklist(args.experiment_id)
    ready = registry.is_promotion_ready(args.experiment_id, strict=args.strict)
    blockers = registry.get_promotion_blockers(args.experiment_id, strict=args.strict)
    experiment = registry.experiments.get(args.experiment_id)
    shadow_dashboard = None
    burn_in_scorecard = None
    if experiment and isinstance(experiment.paper_results, dict):
        shadow_dashboard = build_shadow_drift_dashboard(
            experiment.paper_results,
            critical_threshold=registry.DEFAULT_GATES["paper_live_shadow_drift"]["threshold"],
        )
        burn_in_scorecard = build_paper_burn_in_scorecard(experiment.paper_results)

    payload = {
        "experiment_id": args.experiment_id,
        "strict": args.strict,
        "ready": ready,
        "blockers": blockers,
        "checklist": checklist,
        "shadow_drift_dashboard": shadow_dashboard,
        "burn_in_scorecard": burn_in_scorecard,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

    print("=" * 72)
    print(f"STRATEGY PROMOTION GATE | experiment={args.experiment_id} | strict={args.strict}")
    print("=" * 72)
    print(f"Ready: {'YES' if ready else 'NO'}")

    if blockers:
        print("Blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    else:
        print("Blockers: none")
    if shadow_dashboard:
        print("Shadow Drift Status: " f"{str(shadow_dashboard.get('status', 'unknown')).upper()}")
    if burn_in_scorecard:
        print(
            "Burn-In Signoff: "
            f"{'READY' if burn_in_scorecard.get('ready_for_signoff') else 'NOT READY'} "
            f"(score={float(burn_in_scorecard.get('score', 0.0)):.2%})"
        )

    return 0 if ready else 1


def main() -> None:
    args = _parse_args()
    raise SystemExit(run_gate(args))


if __name__ == "__main__":
    main()
