#!/usr/bin/env python3
"""
Unit tests for paper burn-in scorecard helpers.
"""

from utils.paper_burn_in import (
    BurnInCriteria,
    build_paper_burn_in_scorecard,
    format_paper_burn_in_markdown,
)


def _strong_paper_results() -> dict:
    return {
        "trading_days": 75,
        "total_trades": 140,
        "max_drawdown": -0.11,
        "reconciliation_pass_rate": 0.998,
        "operational_error_rate": 0.005,
        "execution_quality_score": 82.0,
        "avg_actual_slippage_bps": 12.0,
        "fill_rate": 0.97,
        "paper_live_shadow_drift": 0.06,
        "critical_slo_breaches": 0,
    }


def test_build_paper_burn_in_scorecard_ready_with_strong_metrics():
    scorecard = build_paper_burn_in_scorecard(_strong_paper_results())

    assert scorecard["ready_for_signoff"] is True
    assert scorecard["score"] == 1.0
    assert scorecard["blockers"] == []
    assert scorecard["metrics"]["trading_days"] == 75
    assert scorecard["metrics"]["total_trades"] == 140


def test_build_paper_burn_in_scorecard_not_ready_with_blockers():
    scorecard = build_paper_burn_in_scorecard(
        {
            "trading_days": 12,
            "total_trades": 20,
            "max_drawdown": -0.25,
            "reconciliation_pass_rate": 0.94,
            "operational_error_rate": 0.12,
            "execution_quality_score": 55.0,
            "paper_live_shadow_drift": 0.30,
            "critical_slo_breaches": 3,
        }
    )

    assert scorecard["ready_for_signoff"] is False
    assert scorecard["score"] < 0.5
    assert any("burn_in_trading_days" in blocker for blocker in scorecard["blockers"])
    assert any("burn_in_shadow_drift" in blocker for blocker in scorecard["blockers"])


def test_build_paper_burn_in_scorecard_manual_signoff_requirement():
    criteria = BurnInCriteria(
        min_trading_days=30,
        min_trades=50,
        require_manual_signoff=True,
    )

    missing_signoff = build_paper_burn_in_scorecard(
        _strong_paper_results(),
        criteria=criteria,
    )
    assert missing_signoff["ready_for_signoff"] is False
    assert any("burn_in_manual_signoff" in blocker for blocker in missing_signoff["blockers"])

    approved_signoff = build_paper_burn_in_scorecard(
        {
            **_strong_paper_results(),
            "manual_signoff_approved": True,
        },
        criteria=criteria,
    )
    assert approved_signoff["ready_for_signoff"] is True


def test_format_paper_burn_in_markdown_contains_sections():
    scorecard = build_paper_burn_in_scorecard(_strong_paper_results())
    markdown = format_paper_burn_in_markdown(scorecard)

    assert "# Paper Burn-In Scorecard" in markdown
    assert "## Checks" in markdown
    assert "## Blockers" in markdown
