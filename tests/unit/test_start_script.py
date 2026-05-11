from __future__ import annotations

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
