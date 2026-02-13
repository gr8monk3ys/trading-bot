#!/usr/bin/env python3
"""
Unit tests for SLO external alerting hooks.
"""

from __future__ import annotations

import json

from utils.slo_alerting import WebhookSLOAlertNotifier, build_slo_alert_notifier


class _DummyResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_webhook_notifier_skips_below_min_severity(monkeypatch):
    notifier = WebhookSLOAlertNotifier(
        webhook_url="https://example.com/hook",
        min_severity="critical",
    )

    called = {"value": False}

    def _unexpected_call(*args, **kwargs):
        called["value"] = True
        raise AssertionError("urlopen should not be called for skipped warnings")

    monkeypatch.setattr("utils.slo_alerting.request.urlopen", _unexpected_call)
    result = notifier.notify({"severity": "warning", "name": "warn", "message": "test"})

    assert result is None
    assert called["value"] is False


def test_webhook_notifier_posts_payload(monkeypatch):
    notifier = WebhookSLOAlertNotifier(
        webhook_url="https://example.com/hook",
        min_severity="warning",
        timeout_seconds=5,
        source="test-suite",
    )
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["content_type"] = req.get_header("Content-type")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _DummyResponse(status=200)

    monkeypatch.setattr("utils.slo_alerting.request.urlopen", _fake_urlopen)
    result = notifier.notify(
        {
            "severity": "critical",
            "name": "order_reconciliation_consecutive_mismatch_runs",
            "message": "threshold breached",
        }
    )

    assert result is True
    assert captured["url"] == "https://example.com/hook"
    assert captured["timeout"] == 5
    assert captured["content_type"] == "application/json"
    assert captured["body"]["event_type"] == "slo_breach"
    assert captured["body"]["source"] == "test-suite"
    assert captured["body"]["breach"]["name"] == "order_reconciliation_consecutive_mismatch_runs"


def test_build_notifier_from_risk_params():
    notifier = build_slo_alert_notifier(
        {
            "SLO_PAGING_ENABLED": True,
            "SLO_PAGING_WEBHOOK_URL": "https://example.com/hook",
            "SLO_PAGING_MIN_SEVERITY": "warning",
            "SLO_PAGING_TIMEOUT_SECONDS": 4,
        },
        source="unit-test",
    )
    assert notifier is not None
    assert notifier.webhook_url == "https://example.com/hook"
    assert notifier.min_severity == "warning"
    assert notifier.timeout_seconds == 4
    assert notifier.source == "unit-test"


def test_build_notifier_returns_none_when_disabled():
    notifier = build_slo_alert_notifier(
        {
            "SLO_PAGING_ENABLED": False,
            "SLO_PAGING_WEBHOOK_URL": "https://example.com/hook",
        }
    )
    assert notifier is None
