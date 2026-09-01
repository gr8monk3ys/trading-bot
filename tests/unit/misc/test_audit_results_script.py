"""`scripts/audit_results.py` is a gate, so its exit codes are the contract.

The failure this file exists to prevent: an artifact the command was told to
audit and could not open being reported as a skip, so a typo in a filename
drains the run and it still exits 0 having audited nothing.

Nothing here asserts a verdict on a committed artifact — that is the script's
job, not the suite's. What is asserted is which outcome each input produces.
"""

from __future__ import annotations

import re

import pytest

from scripts.audit_results import RESULTS_DIR, main

# The A/B roll-up summarises two runs rather than being one, so it carries no
# trade log. A real committed example beats a synthetic one.
SUMMARY_ONLY = RESULTS_DIR / "bollinger_filter_ab_2020-2024.json"


def _counts(out: str) -> tuple[int, int, int]:
    """Pull (audited, skipped, unreadable) out of the summary line."""
    m = re.search(r"(\d+) audited, (\d+) skipped, (\d+) unreadable\.", out)
    assert m, f"no summary line in output:\n{out}"
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def test_missing_artifact_is_an_error_not_a_skip(tmp_path, capsys):
    code = main([str(tmp_path / "does_not_exist.json")])
    out = capsys.readouterr().out

    assert code == 2, "a file that cannot be opened must not exit 0"
    assert "ERROR" in out
    assert _counts(out) == (0, 0, 1)


def test_corrupt_artifact_is_an_error(tmp_path, capsys):
    bad = tmp_path / "truncated.json"
    bad.write_text('{"trades": [')

    code = main([str(bad)])
    out = capsys.readouterr().out

    assert code == 2
    assert _counts(out) == (0, 0, 1)


def test_summary_only_artifact_is_skipped(capsys):
    code = main([str(SUMMARY_ONLY)])
    out = capsys.readouterr().out

    assert code == 0, "an artifact with no trade log is not a failure"
    assert "skipped" in out
    assert _counts(out) == (0, 1, 0)


def test_all_widens_the_default_set(capsys):
    code_default = main([])
    audited_default = _counts(capsys.readouterr().out)[0]

    code_all = main(["--all"])
    out_all = capsys.readouterr().out
    audited_all = _counts(out_all)[0]

    assert audited_all > audited_default, "--all must cover more than the default set"
    assert "honest_backtest_2020-2024.json" in out_all
    # Either is legitimate; 2 would mean an artifact went missing.
    assert code_default in (0, 1)
    assert code_all in (0, 1)


def test_all_with_explicit_paths_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--all", str(SUMMARY_ONLY)])
    assert exc.value.code == 2
