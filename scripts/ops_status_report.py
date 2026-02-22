#!/usr/bin/env python3
"""Generate a compact operational status report from run artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_artifacts import read_jsonl, write_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate operational status report")
    parser.add_argument(
        "--run-dir", required=True, help="Run artifact directory (results/runs/<run_id>)"
    )
    parser.add_argument("--json-output", default=None, help="Optional JSON output path")
    parser.add_argument("--md-output", default=None, help="Optional markdown output path")
    return parser.parse_args()


def _count_slo(rows: list[dict]) -> dict:
    critical = 0
    warning = 0
    for row in rows:
        if row.get("event_type") != "slo_breach":
            continue
        severity = str(row.get("severity", "")).strip().lower()
        if severity == "critical":
            critical += 1
        elif severity == "warning":
            warning += 1
    return {"critical": critical, "warning": warning, "total": critical + warning}


def _build_report(run_dir: Path) -> dict:
    incident_rows = read_jsonl(run_dir / "incident_events.jsonl")
    slo_rows = read_jsonl(run_dir / "ops_slo_events.jsonl")
    position_rows = read_jsonl(run_dir / "position_reconciliation_events.jsonl")
    order_rows = read_jsonl(run_dir / "order_reconciliation_events.jsonl")

    incident_types = [str(r.get("event_type", "")) for r in incident_rows]
    opened = incident_types.count("incident_open")
    acknowledged = incident_types.count("incident_ack")
    sla_breaches = incident_types.count("incident_sla_breach")

    pos_failed = sum(1 for r in position_rows if bool(r.get("has_mismatch")))
    order_failed = sum(1 for r in order_rows if int(r.get("mismatch_count", 0) or 0) > 0)

    return {
        "run_dir": str(run_dir),
        "incidents": {
            "opened": opened,
            "acknowledged": acknowledged,
            "unacknowledged_estimate": max(0, opened - acknowledged),
            "sla_breaches": sla_breaches,
        },
        "slo": _count_slo(slo_rows),
        "reconciliation": {
            "position_runs": len(position_rows),
            "position_failures": pos_failed,
            "order_runs": len(order_rows),
            "order_mismatch_runs": order_failed,
        },
    }


def _to_markdown(report: dict) -> str:
    incidents = report.get("incidents", {})
    slo = report.get("slo", {})
    recon = report.get("reconciliation", {})
    return "\n".join(
        [
            "# Ops Status Report",
            "",
            f"- Run Dir: `{report.get('run_dir')}`",
            f"- Incidents Opened: `{incidents.get('opened', 0)}`",
            f"- Incidents Acknowledged: `{incidents.get('acknowledged', 0)}`",
            f"- Unacknowledged (est): `{incidents.get('unacknowledged_estimate', 0)}`",
            f"- Incident SLA Breaches: `{incidents.get('sla_breaches', 0)}`",
            f"- SLO Critical Breaches: `{slo.get('critical', 0)}`",
            f"- SLO Warning Breaches: `{slo.get('warning', 0)}`",
            f"- Position Recon Failures: `{recon.get('position_failures', 0)}` of `{recon.get('position_runs', 0)}` runs",
            f"- Order Recon Mismatch Runs: `{recon.get('order_mismatch_runs', 0)}` of `{recon.get('order_runs', 0)}` runs",
        ]
    )


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    report = _build_report(run_dir)
    markdown = _to_markdown(report)
    print(markdown)

    if args.json_output:
        write_json(args.json_output, report)
    if args.md_output:
        md_path = Path(args.md_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
