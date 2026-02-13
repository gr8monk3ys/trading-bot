"""
Paper/live shadow drift dashboard and threshold evaluation helpers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from utils.execution_quality_gate import (
    extract_execution_quality_metrics,
    extract_paper_live_shadow_drift,
)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_drift_series(paper_results: Mapping[str, Any]) -> list[Dict[str, Any]]:
    series: list[Dict[str, Any]] = []
    raw_series = paper_results.get("shadow_drift_series")
    if isinstance(raw_series, list):
        for row in raw_series:
            if not isinstance(row, Mapping):
                continue
            drift = _as_float(row.get("paper_live_shadow_drift"))
            if drift is None:
                drift = _as_float(row.get("shadow_drift"))
            if drift is None:
                drift = _as_float(row.get("drift"))
            if drift is None:
                continue
            series.append(
                {
                    "date": str(
                        row.get("date")
                        or row.get("timestamp")
                        or row.get("t")
                        or "unknown"
                    ),
                    "drift": abs(drift),
                }
            )

    if series:
        return series

    state = paper_results.get("state")
    daily_stats = state.get("daily_stats") if isinstance(state, Mapping) else None
    if not isinstance(daily_stats, list):
        return []

    for day in daily_stats:
        if not isinstance(day, Mapping):
            continue
        drift = extract_paper_live_shadow_drift(day)
        if drift is None:
            continue
        series.append(
            {
                "date": str(day.get("date") or day.get("timestamp") or "unknown"),
                "drift": abs(drift),
            }
        )
    return series


def build_shadow_drift_dashboard(
    paper_results: Mapping[str, Any],
    *,
    critical_threshold: float = 0.15,
    warning_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a machine-readable dashboard payload for paper/live shadow drift.
    """
    critical = max(0.0, float(critical_threshold))
    warning = (
        max(0.0, float(warning_threshold))
        if warning_threshold is not None
        else max(0.0, critical * 0.8)
    )
    if warning > critical:
        warning = critical

    execution_metrics = extract_execution_quality_metrics(paper_results)
    drift = extract_paper_live_shadow_drift(paper_results)
    series = _extract_drift_series(paper_results)
    max_series_drift = max((p["drift"] for p in series), default=None)
    avg_series_drift = (
        sum(p["drift"] for p in series) / len(series)
        if series
        else None
    )

    status = "unknown"
    if drift is not None:
        if drift >= critical:
            status = "critical"
        elif drift >= warning:
            status = "warning"
        else:
            status = "ok"

    actions: list[str] = []
    if status == "critical":
        actions.extend(
            [
                "Block live promotion and keep strategy in paper mode.",
                "Investigate broker fill/slippage path divergence.",
                "Run broker/API fault-injection matrix before retry.",
            ]
        )
    elif status == "warning":
        actions.extend(
            [
                "Increase shadow-monitoring cadence and review fill quality.",
                "Recheck routing, quote staleness, and slippage assumptions.",
            ]
        )
    elif status == "ok":
        actions.append("Drift is within thresholds; continue burn-in monitoring.")
    else:
        actions.append("Shadow drift metric missing; cannot evaluate promotion safety.")

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "status": status,
        "alert": status in {"warning", "critical"},
        "thresholds": {
            "warning": warning,
            "critical": critical,
        },
        "metrics": {
            "paper_live_shadow_drift": drift,
            "execution_quality_score": execution_metrics.get("execution_quality_score"),
            "avg_actual_slippage_bps": execution_metrics.get("avg_actual_slippage_bps"),
            "fill_rate": execution_metrics.get("fill_rate"),
            "series_points": len(series),
            "series_max_drift": max_series_drift,
            "series_avg_drift": avg_series_drift,
        },
        "drift_series_tail": series[-30:],
        "actions": actions,
    }


def format_shadow_drift_dashboard_markdown(dashboard: Mapping[str, Any]) -> str:
    """
    Render a concise markdown summary suitable for reports/runbooks.
    """
    status = str(dashboard.get("status", "unknown")).upper()
    metrics = dashboard.get("metrics", {}) if isinstance(dashboard.get("metrics"), Mapping) else {}
    thresholds = (
        dashboard.get("thresholds", {})
        if isinstance(dashboard.get("thresholds"), Mapping)
        else {}
    )
    lines = [
        "# Shadow Drift Dashboard",
        "",
        f"- Status: **{status}**",
        (
            f"- Drift: `{metrics.get('paper_live_shadow_drift')}` "
            f"(warning>={thresholds.get('warning')}, critical>={thresholds.get('critical')})"
        ),
        f"- Execution quality score: `{metrics.get('execution_quality_score')}`",
        f"- Avg slippage (bps): `{metrics.get('avg_actual_slippage_bps')}`",
        f"- Fill rate: `{metrics.get('fill_rate')}`",
        f"- Drift series points: `{metrics.get('series_points')}`",
        "",
        "## Actions",
    ]
    for action in dashboard.get("actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines)
