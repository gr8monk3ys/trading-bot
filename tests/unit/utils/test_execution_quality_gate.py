#!/usr/bin/env python3
"""
Unit tests for execution quality metric normalization helpers.
"""

import pytest

from utils.execution_quality_gate import (
    extract_execution_quality_metrics,
    extract_paper_live_shadow_drift,
)


def test_extract_execution_quality_metrics_prefers_explicit_bps_values():
    metrics = extract_execution_quality_metrics(
        {
            "execution_quality_score": 82.0,
            "avg_actual_slippage_bps": 14.5,
            "fill_rate": 0.97,
        }
    )

    assert metrics["execution_quality_score"] == 82.0
    assert metrics["avg_actual_slippage_bps"] == 14.5
    assert metrics["fill_rate"] == 0.97


def test_extract_execution_quality_metrics_converts_pct_to_bps():
    metrics = extract_execution_quality_metrics(
        {
            "execution_quality": {
                "execution_score": 75,
                "avg_actual_slippage": 0.0012,
                "fill_rate": 0.95,
            }
        }
    )

    assert metrics["execution_quality_score"] == 75.0
    assert metrics["avg_actual_slippage_bps"] == pytest.approx(12.0)
    assert metrics["fill_rate"] == 0.95


def test_extract_execution_quality_metrics_handles_missing_values():
    metrics = extract_execution_quality_metrics({"trading_days": 20})
    assert metrics["execution_quality_score"] is None
    assert metrics["avg_actual_slippage_bps"] is None
    assert metrics["fill_rate"] is None


def test_extract_paper_live_shadow_drift_direct_metric():
    drift = extract_paper_live_shadow_drift({"paper_live_shadow_drift": 0.12})
    assert drift == pytest.approx(0.12)


def test_extract_paper_live_shadow_drift_from_returns():
    drift = extract_paper_live_shadow_drift({"net_return": 0.10, "live_shadow_return": 0.08})
    assert drift == pytest.approx(0.20)
