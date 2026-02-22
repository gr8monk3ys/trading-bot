from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _CaptureWebhookHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw_body": body}
        self.__class__.payloads.append(payload)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


def test_staging_incident_ticket_drill_script_delivers_webhook(tmp_path):
    _CaptureWebhookHandler.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureWebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    report_path = tmp_path / "staging_ticket_drill_report.json"
    webhook_url = f"http://127.0.0.1:{server.server_port}/incident-ticket"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/staging_incident_ticket_drill.py",
                "--webhook-url",
                webhook_url,
                "--artifact-dir",
                str(tmp_path),
                "--output",
                str(report_path),
                "--require-delivery",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert proc.returncode == 0
    assert "STAGING INCIDENT TICKET DRILL" in proc.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["delivery"]["attempted"] is True
    assert report["delivery"]["succeeded"] is True
    assert report["ticketing"]["created"] == 1
    assert report["dead_letters"]["queued"] == 0

    assert len(_CaptureWebhookHandler.payloads) >= 1
    webhook_payload = _CaptureWebhookHandler.payloads[0]
    assert webhook_payload["event_type"] == "incident_ticket"
    assert webhook_payload["breach"]["name"] == "incident_ack_sla_breach"


def test_staging_incident_ticket_drill_script_fails_when_non_test_target_required(tmp_path):
    _CaptureWebhookHandler.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureWebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    report_path = tmp_path / "staging_ticket_drill_report.json"
    webhook_url = f"http://127.0.0.1:{server.server_port}/incident-ticket"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/staging_incident_ticket_drill.py",
                "--webhook-url",
                webhook_url,
                "--artifact-dir",
                str(tmp_path),
                "--output",
                str(report_path),
                "--require-delivery",
                "--require-non-test-target",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert proc.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["webhook"]["is_non_test_target"] is False
