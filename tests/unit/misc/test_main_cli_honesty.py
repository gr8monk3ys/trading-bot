"""CLI honesty tests for main.py.

A 2026-08 audit found the primary commands lied about their own defaults:

- `main.py live` defaulted to --strategy auto, whose 30-day evaluation
  window can never warm up the 50-bar slow MA -> zero signals -> score
  exactly 0.0 -> nothing clears --min-score 0.5 -> guaranteed no-op after
  minutes of work.
- `main.py backtest` defaulted to "auto" too, which run_backtest doesn't
  even recognize -> immediate "Strategy 'auto' not found".
- --min-momentum was parsed and printed but never used by the scan.
- --position-size / --max-position-size / --stop-loss / --take-profit were
  computed into risk overrides and then dropped: only max_daily_loss ever
  reached anything (the circuit breaker).
- --plot only fired for multi-strategy runs, silently doing nothing for
  the common single-strategy invocation.

These tests pin the corrected surface: working defaults, no dead flags,
and a plot helper that works for any number of strategies.
"""

import numpy as np
import pytest

from main import _apply_risk_profile, _build_parser, _plot_equity_curves


def _parse(argv):
    return _build_parser().parse_args(argv)


def test_live_defaults_to_adaptive_not_auto():
    args = _parse(["live"])
    assert args.strategy == "adaptive"


def test_backtest_defaults_to_backtest_variant():
    args = _parse(["backtest"])
    assert args.strategy == "MomentumStrategyBacktest"


def test_min_momentum_flag_is_gone():
    with pytest.raises(SystemExit):
        _parse(["live", "--min-momentum", "2.0"])


@pytest.mark.parametrize(
    "flag", ["--position-size", "--max-position-size", "--stop-loss", "--take-profit"]
)
def test_dead_risk_flags_are_gone(flag):
    with pytest.raises(SystemExit):
        _parse(["live", flag, "0.05"])


def test_max_daily_loss_flag_still_works():
    args = _parse(["live", "--max-daily-loss", "0.02"])
    assert args.max_daily_loss == 0.02


def test_risk_profile_yields_only_what_is_wired():
    args = _parse(["live", "--risk-profile", "conservative"])
    profile = _apply_risk_profile(args)
    assert profile == {"max_daily_loss": 0.025}


def test_risk_profile_cli_override_wins():
    args = _parse(["live", "--risk-profile", "conservative", "--max-daily-loss", "0.05"])
    assert _apply_risk_profile(args)["max_daily_loss"] == 0.05


def test_plot_helper_writes_file_for_single_strategy(tmp_path):
    out = tmp_path / "curves.png"
    results = {"OnlyStrategy": {"equity_curve": list(np.linspace(100_000, 110_000, 50))}}

    _plot_equity_curves(results, path=str(out))

    assert out.exists() and out.stat().st_size > 0
