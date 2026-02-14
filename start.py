#!/usr/bin/env python3
"""
Startup script for Railway deployment.
Runs the trading bot and web dashboard concurrently.
"""

import logging
import os
import signal
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("startup")


def main():
    port = os.environ.get("PORT", "8000")

    # Start the web dashboard (FastAPI)
    dashboard_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", port],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Start the trading bot
    bot_proc = subprocess.Popen(
        [sys.executable, "run_adaptive.py", "--force"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Handle shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down...")
        dashboard_proc.terminate()
        bot_proc.terminate()
        dashboard_proc.wait(timeout=10)
        bot_proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Wait for either process to exit
    while True:
        dashboard_status = dashboard_proc.poll()
        bot_status = bot_proc.poll()

        if dashboard_status is not None:
            logger.error(f"Dashboard exited with code {dashboard_status}")
            # Restart dashboard
            dashboard_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "web.app:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    port,
                ],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

        if bot_status is not None:
            logger.warning(f"Trading bot exited with code {bot_status}")
            # Restart bot
            bot_proc = subprocess.Popen(
                [sys.executable, "run_adaptive.py", "--force"],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

        import time

        time.sleep(5)


if __name__ == "__main__":
    main()
