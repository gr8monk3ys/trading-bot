#!/usr/bin/env python3
"""
Unit tests for shadow drift dashboard utilities.
"""

from utils.shadow_drift_dashboard import (
    build_shadow_drift_dashboard,
    format_shadow_drift_dashboard_markdown,
)


def test_build_shadow_drift_dashboard_marks_critical():
    dashboard = build_shadow_drift_dashboard(
        {"paper_live_shadow_drift": 0.2, "execution_quality_score": 72},
        critical_threshold=0.15,
        warning_threshold=0.10,
    )
    assert dashboard["status"] == "critical"
    assert dashboard["alert"] is True
    assert dashboard["metrics"]["paper_live_shadow_drift"] == 0.2


def test_build_shadow_drift_dashboard_marks_warning():
    dashboard = build_shadow_drift_dashboard(
        {
            "paper_return": 0.10,
            "live_shadow_return": 0.085,
            "state": {
                "daily_stats": [
                    {"date": "2026-01-01", "paper_return": 0.01, "live_shadow_return": 0.009},
                    {"date": "2026-01-02", "paper_return": 0.02, "live_shadow_return": 0.018},
                ]
            },
        },
        critical_threshold=0.20,
        warning_threshold=0.10,
    )
    assert dashboard["status"] == "warning"
    assert dashboard["metrics"]["series_points"] == 2
    assert len(dashboard["drift_series_tail"]) == 2


def test_build_shadow_drift_dashboard_unknown_when_metric_missing():
    dashboard = build_shadow_drift_dashboard({}, critical_threshold=0.15)
    assert dashboard["status"] == "unknown"
    assert dashboard["metrics"]["paper_live_shadow_drift"] is None


def test_format_shadow_drift_dashboard_markdown_contains_sections():
    dashboard = build_shadow_drift_dashboard({"paper_live_shadow_drift": 0.05})
    text = format_shadow_drift_dashboard_markdown(dashboard)
    assert "# Shadow Drift Dashboard" in text
    assert "## Actions" in text
