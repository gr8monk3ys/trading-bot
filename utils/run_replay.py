"""
Replay helpers for run-scoped observability artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from utils.run_artifacts import read_json, read_jsonl


def resolve_run_directory(run_id: str, artifacts_dir: str = "results/runs") -> Path:
    """Resolve run directory for the given run ID."""
    run_dir = Path(artifacts_dir) / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def load_run_artifacts(run_id: str, artifacts_dir: str = "results/runs") -> dict[str, Any]:
    """Load summary, decision events, and trade events for a run."""
    run_dir = resolve_run_directory(run_id, artifacts_dir=artifacts_dir)
    summary = read_json(run_dir / "summary.json", default={}) or {}
    decisions = read_jsonl(run_dir / "decision_events.jsonl")
    trades = read_jsonl(run_dir / "trades.jsonl")
    order_reconciliation = read_jsonl(run_dir / "order_reconciliation_events.jsonl")
    position_reconciliation = read_jsonl(run_dir / "position_reconciliation_events.jsonl")
    slo_events = read_jsonl(run_dir / "ops_slo_events.jsonl")
    incident_events = read_jsonl(run_dir / "incident_events.jsonl")
    data_quality_events = read_jsonl(run_dir / "data_quality_events.jsonl")
    manifest = read_json(run_dir / "manifest.json", default={}) or {}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "decisions": decisions,
        "trades": trades,
        "order_reconciliation": order_reconciliation,
        "position_reconciliation": position_reconciliation,
        "slo_events": slo_events,
        "incident_events": incident_events,
        "data_quality_events": data_quality_events,
        "manifest": manifest,
    }


def filter_events(
    events: list[dict[str, Any]],
    symbol: Optional[str] = None,
    date_prefix: Optional[str] = None,
    errors_only: bool = False,
    event_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Filter event records for replay display."""
    filtered: list[dict[str, Any]] = []
    for event in events:
        if symbol and str(event.get("symbol", "")).upper() != symbol.upper():
            continue

        if event_type and str(event.get("event_type")) != event_type:
            continue

        if errors_only and not event.get("error"):
            continue

        if date_prefix:
            event_date = str(event.get("date", ""))
            if not event_date.startswith(date_prefix):
                continue

        filtered.append(event)

    if limit is not None and limit >= 0:
        return filtered[:limit]
    return filtered


def format_replay_report(
    summary: dict[str, Any],
    decisions: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    order_reconciliation: Optional[list[dict[str, Any]]] = None,
    position_reconciliation: Optional[list[dict[str, Any]]] = None,
    slo_events: Optional[list[dict[str, Any]]] = None,
    incident_events: Optional[list[dict[str, Any]]] = None,
    data_quality_events: Optional[list[dict[str, Any]]] = None,
    limit: int = 20,
) -> str:
    """Format replay output for terminal display."""
    lines: list[str] = []
    run_id = summary.get("run_id", "unknown")
    strategy = summary.get("strategy", "unknown")
    total_return = summary.get("total_return")
    final_equity = summary.get("final_equity")

    lines.append("=" * 72)
    lines.append(f"REPLAY REPORT | run_id={run_id} | strategy={strategy}")
    lines.append("=" * 72)
    if final_equity is not None and total_return is not None:
        lines.append(
            f"Final Equity: ${float(final_equity):,.2f} | Total Return: {float(total_return):+.2%}"
        )
    lines.append(f"Decision events: {len(decisions)} | Trade events: {len(trades)}")
    lines.append(
        "Reconciliation snapshots: "
        f"order={len(order_reconciliation or [])}, "
        f"position={len(position_reconciliation or [])} | "
        f"SLO events: {len(slo_events or [])} | "
        f"Incident events: {len(incident_events or [])} | "
        f"Data-quality events: {len(data_quality_events or [])}"
    )

    lines.append("-" * 72)
    lines.append("Decision Timeline")
    for event in decisions[:limit]:
        date = event.get("date", "?")
        symbol = event.get("symbol", "-")
        action = event.get("action", "-")
        err = event.get("error")
        suffix = f" | ERROR: {err}" if err else ""
        lines.append(f"{date} | {symbol:>8} | {action:<10}{suffix}")

    lines.append("-" * 72)
    lines.append("Operations Timeline")
    latest_order = (order_reconciliation or [])[-1] if order_reconciliation else None
    latest_position = (position_reconciliation or [])[-1] if position_reconciliation else None
    latest_slo = (slo_events or [])[-1] if slo_events else None
    latest_incident = (incident_events or [])[-1] if incident_events else None
    latest_quality = (data_quality_events or [])[-1] if data_quality_events else None

    if latest_order:
        lines.append(
            "Order Recon: "
            f"status={latest_order.get('status')} "
            f"consecutive_mismatch_runs={latest_order.get('consecutive_mismatch_runs')}"
        )
    if latest_position:
        lines.append(
            "Position Recon: "
            f"positions_match={latest_position.get('positions_match')} "
            f"mismatch_count={latest_position.get('mismatch_count')}"
        )
    if latest_quality:
        lines.append(
            "Data Quality: "
            f"errors={latest_quality.get('total_errors')} "
            f"stale_warnings={latest_quality.get('stale_warnings')}"
        )
    if latest_slo:
        lines.append(
            "Latest SLO: "
            f"{latest_slo.get('severity', '?').upper()} "
            f"{latest_slo.get('name', '?')} - {latest_slo.get('message', '')}"
        )
    if latest_incident:
        lines.append(
            "Latest Incident: "
            f"{latest_incident.get('event_type', '?')} "
            f"id={latest_incident.get('incident_id', '?')}"
        )

    lines.append("-" * 72)
    lines.append("Trade Timeline")
    for event in trades[:limit]:
        date = event.get("date", event.get("timestamp", "?"))
        symbol = event.get("symbol", "-")
        side = event.get("side", "-")
        qty = event.get("quantity", event.get("filled_qty", "-"))
        price = event.get("price", event.get("filled_avg_price", "-"))
        lines.append(f"{date} | {symbol:>8} | {side:<4} | qty={qty} | price={price}")

    return "\n".join(lines)
