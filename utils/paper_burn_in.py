"""
Long-horizon paper trading burn-in scorecard and sign-off criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from utils.execution_quality_gate import (
    extract_execution_quality_metrics,
    extract_paper_live_shadow_drift,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class BurnInCriteria:
    min_trading_days: int = 60
    min_trades: int = 100
    max_drawdown: float = 0.15
    min_reconciliation_pass_rate: float = 0.995
    max_operational_error_rate: float = 0.02
    min_execution_quality_score: float = 70.0
    max_shadow_drift: float = 0.15
    max_critical_slo_breaches: int = 0
    require_manual_signoff: bool = False


def _extract_metrics(paper_results: Mapping[str, Any]) -> Dict[str, Any]:
    execution = extract_execution_quality_metrics(paper_results)
    trading_days = int(_as_float(paper_results.get("trading_days"), 0))

    trades = int(
        _as_float(
            paper_results.get("total_trades")
            or paper_results.get("trade_count")
            or paper_results.get("trades")
            or paper_results.get("signals_executed"),
            0,
        )
    )

    raw_drawdown = _as_float(
        paper_results.get("max_drawdown") or paper_results.get("max_drawdown_pct"),
        0.0,
    )
    max_drawdown = abs(raw_drawdown)

    recon_rate = paper_results.get("reconciliation_pass_rate")
    if recon_rate is None:
        runs = _as_float(paper_results.get("reconciliation_runs"), 0.0)
        mismatches = _as_float(paper_results.get("reconciliation_mismatch_count"), 0.0)
        recon_rate = 1.0 if runs <= 0 else max(0.0, 1.0 - (mismatches / runs))
    recon_rate = _as_float(recon_rate, 0.0)

    error_rate = paper_results.get("operational_error_rate")
    if error_rate is None:
        decisions = _as_float(paper_results.get("decision_events"), 0.0)
        errors = _as_float(paper_results.get("decision_errors"), 0.0)
        error_rate = 0.0 if decisions <= 0 else max(0.0, errors / decisions)
    error_rate = _as_float(error_rate, 0.0)

    critical_slo = paper_results.get("critical_slo_breaches")
    if critical_slo is None:
        events = paper_results.get("ops_slo_events")
        if isinstance(events, list):
            critical_slo = sum(
                1
                for event in events
                if isinstance(event, Mapping)
                and str(event.get("severity", "")).lower() == "critical"
            )
        else:
            critical_slo = 0
    critical_slo = int(_as_float(critical_slo, 0))

    return {
        "trading_days": trading_days,
        "total_trades": trades,
        "max_drawdown": max_drawdown,
        "reconciliation_pass_rate": recon_rate,
        "operational_error_rate": error_rate,
        "execution_quality_score": execution.get("execution_quality_score"),
        "avg_actual_slippage_bps": execution.get("avg_actual_slippage_bps"),
        "fill_rate": execution.get("fill_rate"),
        "paper_live_shadow_drift": extract_paper_live_shadow_drift(paper_results),
        "critical_slo_breaches": critical_slo,
        "manual_signoff_approved": bool(
            paper_results.get("manual_signoff_approved") or paper_results.get("signoff_approved")
        ),
    }


def build_paper_burn_in_scorecard(
    paper_results: Mapping[str, Any],
    *,
    criteria: Optional[BurnInCriteria] = None,
) -> Dict[str, Any]:
    """
    Build long-horizon burn-in scorecard with promotion sign-off readiness.
    """
    c = criteria or BurnInCriteria()
    metrics = _extract_metrics(paper_results)

    checks = [
        {
            "name": "burn_in_trading_days",
            "required": True,
            "passed": metrics["trading_days"] >= c.min_trading_days,
            "details": f"trading_days={metrics['trading_days']} required>={c.min_trading_days}",
        },
        {
            "name": "burn_in_trade_count",
            "required": True,
            "passed": metrics["total_trades"] >= c.min_trades,
            "details": f"total_trades={metrics['total_trades']} required>={c.min_trades}",
        },
        {
            "name": "burn_in_max_drawdown",
            "required": True,
            "passed": metrics["max_drawdown"] <= c.max_drawdown,
            "details": f"max_drawdown={metrics['max_drawdown']:.3f} required<={c.max_drawdown:.3f}",
        },
        {
            "name": "burn_in_reconciliation_rate",
            "required": True,
            "passed": metrics["reconciliation_pass_rate"] >= c.min_reconciliation_pass_rate,
            "details": (
                "reconciliation_pass_rate="
                f"{metrics['reconciliation_pass_rate']:.3f} required>={c.min_reconciliation_pass_rate:.3f}"
            ),
        },
        {
            "name": "burn_in_operational_error_rate",
            "required": True,
            "passed": metrics["operational_error_rate"] <= c.max_operational_error_rate,
            "details": (
                "operational_error_rate="
                f"{metrics['operational_error_rate']:.4f} required<={c.max_operational_error_rate:.4f}"
            ),
        },
        {
            "name": "burn_in_execution_quality_score",
            "required": True,
            "passed": (
                metrics["execution_quality_score"] is not None
                and metrics["execution_quality_score"] >= c.min_execution_quality_score
            ),
            "details": (
                "execution_quality_score="
                f"{metrics['execution_quality_score']} required>={c.min_execution_quality_score:.1f}"
            ),
        },
        {
            "name": "burn_in_shadow_drift",
            "required": True,
            "passed": (
                metrics["paper_live_shadow_drift"] is not None
                and metrics["paper_live_shadow_drift"] <= c.max_shadow_drift
            ),
            "details": (
                "paper_live_shadow_drift="
                f"{metrics['paper_live_shadow_drift']} required<={c.max_shadow_drift:.3f}"
            ),
        },
        {
            "name": "burn_in_critical_slo_breaches",
            "required": True,
            "passed": metrics["critical_slo_breaches"] <= c.max_critical_slo_breaches,
            "details": (
                "critical_slo_breaches="
                f"{metrics['critical_slo_breaches']} required<={c.max_critical_slo_breaches}"
            ),
        },
        {
            "name": "burn_in_manual_signoff",
            "required": bool(c.require_manual_signoff),
            "passed": metrics["manual_signoff_approved"] is True,
            "details": ("manual_signoff_approved=" f"{metrics['manual_signoff_approved']}"),
        },
    ]

    required_checks = [check for check in checks if check["required"]]
    passed_required = sum(1 for check in required_checks if check["passed"])
    blockers = [
        f"{check['name']}: {check['details']}" for check in required_checks if not check["passed"]
    ]
    score = (passed_required / len(required_checks)) if required_checks else 0.0
    ready = len(blockers) == 0

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "ready_for_signoff": ready,
        "score": score,
        "criteria": {
            "min_trading_days": c.min_trading_days,
            "min_trades": c.min_trades,
            "max_drawdown": c.max_drawdown,
            "min_reconciliation_pass_rate": c.min_reconciliation_pass_rate,
            "max_operational_error_rate": c.max_operational_error_rate,
            "min_execution_quality_score": c.min_execution_quality_score,
            "max_shadow_drift": c.max_shadow_drift,
            "max_critical_slo_breaches": c.max_critical_slo_breaches,
            "require_manual_signoff": c.require_manual_signoff,
        },
        "metrics": metrics,
        "checks": checks,
        "blockers": blockers,
    }


def format_paper_burn_in_markdown(scorecard: Mapping[str, Any]) -> str:
    """
    Render burn-in scorecard as concise markdown.
    """
    status = "READY" if scorecard.get("ready_for_signoff") else "NOT READY"
    lines = [
        "# Paper Burn-In Scorecard",
        "",
        f"- Status: **{status}**",
        f"- Score: `{float(scorecard.get('score', 0.0)):.2%}`",
        "",
        "## Checks",
    ]
    for check in scorecard.get("checks", []):
        if not isinstance(check, Mapping):
            continue
        label = "PASS" if check.get("passed") else "FAIL"
        req = "required" if check.get("required") else "advisory"
        lines.append(f"- [{label}] `{check.get('name')}` ({req}) - {check.get('details')}")
    blockers = scorecard.get("blockers", [])
    lines.append("")
    lines.append("## Blockers")
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    return "\n".join(lines)
