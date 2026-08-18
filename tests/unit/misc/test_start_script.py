from __future__ import annotations

import subprocess
import sys

import start


def test_dashboard_command_targets_web_app():
    assert start._dashboard_command("8123") == [
        sys.executable,
        "-m",
        "uvicorn",
        "web.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8123",
    ]


def test_bot_command_invokes_main_adaptive_live():
    assert start._bot_command() == [
        sys.executable,
        "main.py",
        "live",
        "--strategy",
        "adaptive",
    ]


# --- Supervisor restart policy -------------------------------------------------
# The trading child must NOT be relaunched forever on any exit. A clean exit
# (code 0) is intentional (market closed without --force, operator stop); a crash
# (non-zero) is retried with exponential backoff up to a cap, then abandoned.


def test_should_not_restart_on_clean_exit():
    assert start._should_restart(0, consecutive_failures=0, max_failures=5) is False


def test_should_restart_on_failure_below_cap():
    assert start._should_restart(1, consecutive_failures=2, max_failures=5) is True


def test_should_stop_restarting_at_failure_cap():
    assert start._should_restart(1, consecutive_failures=5, max_failures=5) is False


def test_should_restart_treats_negative_exit_as_failure():
    # Signals surface as negative return codes; those are crashes, not clean stops.
    assert start._should_restart(-15, consecutive_failures=0, max_failures=5) is True


def test_backoff_grows_exponentially_from_base():
    assert start._restart_backoff_seconds(1, base=5.0, cap=300.0) == 5.0
    assert start._restart_backoff_seconds(2, base=5.0, cap=300.0) == 10.0
    assert start._restart_backoff_seconds(3, base=5.0, cap=300.0) == 20.0


def test_backoff_is_capped():
    assert start._restart_backoff_seconds(20, base=5.0, cap=300.0) == 300.0


# --- Graceful shutdown ---------------------------------------------------------
# A stuck child must never prevent the rest of shutdown from running; on timeout
# the supervisor force-kills it. The helper is None-safe and exception-safe.


class _FakeProc:
    def __init__(self, *, wait_raises: bool = False):
        self.terminated = False
        self.killed = False
        self._wait_raises = wait_raises
        self.wait_calls = 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._wait_raises and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
        return 0


def test_terminate_process_is_none_safe():
    # Must not raise when there is no child process.
    start._terminate_process(None)


def test_terminate_process_terminates_gracefully():
    proc = _FakeProc()
    start._terminate_process(proc, timeout=1)
    assert proc.terminated is True
    assert proc.killed is False


def test_terminate_process_force_kills_on_timeout():
    proc = _FakeProc(wait_raises=True)
    start._terminate_process(proc, timeout=1)
    assert proc.terminated is True
    assert proc.killed is True
