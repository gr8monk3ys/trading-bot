from __future__ import annotations

import json
import subprocess
import sys

from scripts import replay_notification_dead_letters as replay
from utils.run_artifacts import read_jsonl


class _Notifier:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def notify(self, breach):
        self.calls.append(dict(breach))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _dead_letter_record(channel: str = "slo_alert") -> dict:
    return {
        "event_type": "notification_dead_letter",
        "channel": channel,
        "reason": "notifier_returned_false",
        "breach": {
            "name": "data_quality_errors",
            "severity": "critical",
            "message": "threshold breached",
        },
    }


def test_replay_dead_letters_success():
    alert_notifier = _Notifier(True)
    report, remaining = replay._replay_dead_letters(
        [_dead_letter_record("slo_alert")],
        alert_notifier=alert_notifier,
        incident_ticket_notifier=None,
    )

    assert report["records_total"] == 1
    assert report["records_processed"] == 1
    assert report["records_replayed"] == 1
    assert report["records_remaining"] == 0
    assert remaining == []
    assert len(alert_notifier.calls) == 1


def test_replay_dead_letters_notifier_unavailable():
    report, remaining = replay._replay_dead_letters(
        [_dead_letter_record("slo_alert")],
        alert_notifier=None,
        incident_ticket_notifier=None,
    )

    assert report["records_total"] == 1
    assert report["records_processed"] == 1
    assert report["records_skipped"] == 1
    assert report["records_remaining"] == 1
    assert report["reasons"]["notifier_unavailable"] == 1
    assert remaining[0]["last_replay_error"] == "notifier_unavailable"
    assert remaining[0]["replay_attempts"] == 1


def test_replay_notification_dead_letters_script_in_place(tmp_path):
    dead_letter_path = tmp_path / "notification_dead_letters.jsonl"
    dead_letter_path.write_text(
        json.dumps(_dead_letter_record("unknown_channel")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "replay_report.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/replay_notification_dead_letters.py",
            "--dead-letter-path",
            str(dead_letter_path),
            "--in-place",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "NOTIFICATION DEAD-LETTER REPLAY" in proc.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["records_total"] == 1
    assert payload["records_remaining"] == 1
    assert payload["reasons"]["unknown_channel"] == 1

    remaining = read_jsonl(dead_letter_path)
    assert len(remaining) == 1
    assert remaining[0]["last_replay_error"] == "unknown_channel"
    assert remaining[0]["replay_attempts"] == 1
