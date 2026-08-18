#!/usr/bin/env python3
"""
Startup script for Railway deployment.

Runs the web dashboard and the trading bot under a small supervisor. Children
are restarted on *failure* with exponential backoff and a consecutive-failure
cap; a clean bot exit (code 0) is treated as intentional and is not respawned.
When the local cap is exceeded the process exits non-zero so the platform's own
restart policy (railway.toml ``restartPolicyMaxRetries``) can take over with a
fresh container.
"""

import logging
import os
import signal
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("startup")

# Supervisor policy.
MAX_CONSECUTIVE_FAILURES = 5
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_CAP_SECONDS = 300.0
# A child that stays up at least this long is considered healthy; its failure
# streak resets so occasional restarts over a long run don't accumulate.
HEALTHY_UPTIME_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 2.0
TERMINATE_TIMEOUT_SECONDS = 10.0


def _dashboard_command(port: str) -> list[str]:
    return [sys.executable, "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", port]


def _bot_command() -> list[str]:
    return [sys.executable, "main.py", "live", "--strategy", "adaptive"]


def _should_restart(returncode: int, consecutive_failures: int, max_failures: int) -> bool:
    """Decide whether to relaunch the trading child after it exits.

    A clean exit (``returncode == 0``) is intentional — the runtime stopped on
    purpose (market closed without ``--force``, operator interrupt) — so it must
    not be respawned. Any non-zero (or signal, which surfaces negative) exit is a
    crash, restarted until the consecutive-failure count reaches ``max_failures``.
    """
    if returncode == 0:
        return False
    return consecutive_failures < max_failures


def _restart_backoff_seconds(
    consecutive_failures: int,
    *,
    base: float = BACKOFF_BASE_SECONDS,
    cap: float = BACKOFF_CAP_SECONDS,
) -> float:
    """Exponential backoff between restart attempts: base, 2*base, 4*base ... capped."""
    exponent = max(0, consecutive_failures - 1)
    return min(cap, base * (2**exponent))


def _terminate_process(proc, *, timeout: float = TERMINATE_TIMEOUT_SECONDS) -> None:
    """Terminate a subprocess, force-killing it if it does not exit in time.

    None-safe and exception-safe: a stuck or already-dead child can never prevent
    the rest of shutdown from running.
    """
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except Exception:
            pass
    except Exception:
        pass


def main():
    port = os.environ.get("PORT", "8000")

    def launch_dashboard():
        return subprocess.Popen(_dashboard_command(port), stdout=sys.stdout, stderr=sys.stderr)

    def launch_bot():
        return subprocess.Popen(_bot_command(), stdout=sys.stdout, stderr=sys.stderr)

    dashboard_proc = launch_dashboard()
    bot_proc = launch_bot()

    dashboard_failures = 0
    bot_failures = 0
    dashboard_started_at = time.monotonic()
    bot_started_at = time.monotonic()
    bot_supervised = True

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        _terminate_process(dashboard_proc)
        _terminate_process(bot_proc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        # Dashboard is a long-running server: bring it back on any exit, with
        # backoff, escalating to the platform once the local cap is exceeded.
        dashboard_status = dashboard_proc.poll()
        if dashboard_status is not None:
            if time.monotonic() - dashboard_started_at >= HEALTHY_UPTIME_SECONDS:
                dashboard_failures = 0
            if dashboard_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.critical(
                    "Dashboard failed %d consecutive times; exiting for platform restart.",
                    dashboard_failures,
                )
                _terminate_process(bot_proc)
                sys.exit(1)
            dashboard_failures += 1
            delay = _restart_backoff_seconds(dashboard_failures)
            logger.error(
                "Dashboard exited (code %s); restarting in %.0fs (failure %d/%d)",
                dashboard_status,
                delay,
                dashboard_failures,
                MAX_CONSECUTIVE_FAILURES,
            )
            time.sleep(delay)
            dashboard_proc = launch_dashboard()
            dashboard_started_at = time.monotonic()

        # Trading bot: only real crashes are restarted; a clean exit is honored
        # so we never tight-loop respawn an intentionally-stopped runtime.
        if bot_supervised:
            bot_status = bot_proc.poll()
            if bot_status is not None:
                if time.monotonic() - bot_started_at >= HEALTHY_UPTIME_SECONDS:
                    bot_failures = 0
                if _should_restart(bot_status, bot_failures, MAX_CONSECUTIVE_FAILURES):
                    bot_failures += 1
                    delay = _restart_backoff_seconds(bot_failures)
                    logger.warning(
                        "Trading bot crashed (code %s); restarting in %.0fs (failure %d/%d)",
                        bot_status,
                        delay,
                        bot_failures,
                        MAX_CONSECUTIVE_FAILURES,
                    )
                    time.sleep(delay)
                    bot_proc = launch_bot()
                    bot_started_at = time.monotonic()
                elif bot_status == 0:
                    logger.info(
                        "Trading bot exited cleanly (code 0); not restarting. "
                        "Dashboard stays up."
                    )
                    bot_supervised = False
                else:
                    logger.critical(
                        "Trading bot failed %d consecutive times; exiting for platform restart.",
                        bot_failures,
                    )
                    _terminate_process(dashboard_proc)
                    sys.exit(1)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
