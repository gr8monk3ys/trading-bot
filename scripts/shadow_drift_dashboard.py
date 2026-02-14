#!/usr/bin/env python3
"""
Generate paper/live shadow drift dashboard and enforce alert thresholds.
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
from utils.shadow_drift_dashboard import (
    build_shadow_drift_dashboard,
    format_shadow_drift_dashboard_markdown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow drift dashboard")
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
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=None,
        help="Warning threshold (decimal drift)",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=0.15,
        help="Critical threshold (decimal drift)",
    )
    parser.add_argument(
        "--fail-on",
        choices=["none", "warning", "critical"],
        default="critical",
        help="Exit non-zero when status reaches this level",
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


def _should_fail(status: str, fail_on: str) -> bool:
    rank = {"none": 0, "unknown": 0, "ok": 0, "warning": 1, "critical": 2}
    return rank.get(status, 0) >= rank.get(fail_on, 0) and fail_on != "none"


def main() -> None:
    args = _parse_args()
    try:
        paper_results = _load_paper_results(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from e

    dashboard = build_shadow_drift_dashboard(
        paper_results,
        critical_threshold=args.critical_threshold,
        warning_threshold=args.warning_threshold,
    )
    markdown = format_shadow_drift_dashboard_markdown(dashboard)
    print(markdown)

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    if args.md_output:
        out = Path(args.md_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")

    status = str(dashboard.get("status", "unknown"))
    raise SystemExit(1 if _should_fail(status, args.fail_on) else 0)


if __name__ == "__main__":
    main()
