"""
Execution quality metric extraction and gating helpers.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_execution_quality_metrics(paper_results: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """
    Normalize execution-quality metrics from paper trading result payloads.
    """
    execution_quality = paper_results.get("execution_quality")
    quality = execution_quality if isinstance(execution_quality, Mapping) else {}

    score = _as_float(paper_results.get("execution_quality_score"))
    if score is None:
        score = _as_float(paper_results.get("execution_score"))
    if score is None:
        score = _as_float(quality.get("execution_score"))

    slippage_bps = _as_float(paper_results.get("avg_actual_slippage_bps"))
    if slippage_bps is None:
        slippage_bps = _as_float(quality.get("avg_actual_slippage_bps"))
    if slippage_bps is None:
        slippage_bps = _as_float(quality.get("avg_slippage_bps"))

    slippage_pct = _as_float(paper_results.get("avg_actual_slippage"))
    if slippage_pct is None:
        slippage_pct = _as_float(paper_results.get("avg_slippage"))
    if slippage_pct is None:
        slippage_pct = _as_float(quality.get("avg_actual_slippage"))
    if slippage_pct is None:
        slippage_pct = _as_float(quality.get("avg_slippage"))
    if slippage_bps is None and slippage_pct is not None:
        slippage_bps = slippage_pct * 10000.0

    fill_rate = _as_float(paper_results.get("fill_rate"))
    if fill_rate is None:
        fill_rate = _as_float(quality.get("fill_rate"))
    if fill_rate is None:
        signals_generated = _as_float(paper_results.get("signals_generated"))
        signals_executed = _as_float(paper_results.get("signals_executed"))
        if signals_generated is not None and signals_generated > 0 and signals_executed is not None:
            fill_rate = max(0.0, min(1.0, signals_executed / signals_generated))
    if fill_rate is None:
        state = paper_results.get("state")
        if isinstance(state, Mapping):
            daily_stats = state.get("daily_stats")
            if isinstance(daily_stats, list):
                rates = [
                    _as_float(day.get("fill_rate"))
                    for day in daily_stats
                    if isinstance(day, Mapping) and _as_float(day.get("fill_rate")) is not None
                ]
                if rates:
                    fill_rate = max(0.0, min(1.0, sum(rates) / len(rates)))

    if score is None and (slippage_bps is not None or fill_rate is not None):
        penalty = 0.0
        if slippage_bps is not None:
            # 5 bps baseline before penalties.
            penalty += max(0.0, slippage_bps - 5.0) * 1.5
        if fill_rate is not None and fill_rate < 1.0:
            penalty += (1.0 - fill_rate) * 200.0
        score = max(0.0, min(100.0, 100.0 - penalty))

    return {
        "execution_quality_score": score,
        "avg_actual_slippage_bps": slippage_bps,
        "fill_rate": fill_rate,
    }


def extract_paper_live_shadow_drift(paper_results: Mapping[str, Any]) -> Optional[float]:
    """
    Normalize paper/live shadow drift from result payloads.

    Returned value is an absolute decimal drift (e.g. 0.12 = 12%).
    """
    direct_keys = (
        "paper_live_shadow_drift",
        "shadow_drift",
        "paper_live_shadow_drift_pct",
    )
    for key in direct_keys:
        value = _as_float(paper_results.get(key))
        if value is not None:
            return abs(value)

    shadow = paper_results.get("shadow")
    if isinstance(shadow, Mapping):
        for key in ("paper_live_shadow_drift", "shadow_drift", "drift"):
            value = _as_float(shadow.get(key))
            if value is not None:
                return abs(value)

    paper_return = _as_float(paper_results.get("net_return"))
    if paper_return is None:
        paper_return = _as_float(paper_results.get("paper_return"))
    if paper_return is None:
        paper_return = _as_float(paper_results.get("paper_net_return"))

    live_shadow_return = _as_float(paper_results.get("live_shadow_return"))
    if live_shadow_return is None and isinstance(shadow, Mapping):
        live_shadow_return = _as_float(shadow.get("live_return"))

    if paper_return is None or live_shadow_return is None:
        return None

    baseline = max(abs(paper_return), 1e-6)
    return abs(live_shadow_return - paper_return) / baseline
