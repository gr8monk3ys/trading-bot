#!/usr/bin/env python3
"""
Unit tests for SLO external alerting hooks.
"""

from __future__ import annotations

import json
from urllib import error

from utils.slo_alerting import (
    WebhookIncidentTicketNotifier,
    WebhookSLOAlertNotifier,
    build_incident_ticket_notifier,
    build_slo_alert_notifier,
)


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
        auth_token="secret-token",
    )
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["content_type"] = req.get_header("Content-type")
        captured["authorization"] = req.get_header("Authorization")
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
    assert captured["authorization"] == "Bearer secret-token"
    assert captured["body"]["event_type"] == "slo_breach"
    assert captured["body"]["source"] == "test-suite"
    assert captured["body"]["breach"]["name"] == "order_reconciliation_consecutive_mismatch_runs"


def test_webhook_notifier_retries_transient_failure(monkeypatch):
    notifier = WebhookSLOAlertNotifier(
        webhook_url="https://example.com/hook",
        min_severity="warning",
        max_retries=2,
        retry_backoff_seconds=0.25,
    )
    attempts = {"count": 0}
    sleep_calls = []

    def _fake_urlopen(req, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise error.URLError("temporary network outage")
        return _DummyResponse(status=200)

    monkeypatch.setattr("utils.slo_alerting.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("utils.slo_alerting.time.sleep", lambda delay: sleep_calls.append(delay))

    result = notifier.notify({"severity": "critical", "name": "critical_breach", "message": "x"})

    assert result is True
    assert attempts["count"] == 2
    assert sleep_calls == [0.25]


def test_webhook_notifier_does_not_retry_on_http_400(monkeypatch):
    notifier = WebhookSLOAlertNotifier(
        webhook_url="https://example.com/hook",
        min_severity="warning",
        max_retries=3,
        retry_backoff_seconds=0.1,
    )
    attempts = {"count": 0}

    def _fake_urlopen(req, timeout):
        attempts["count"] += 1
        return _DummyResponse(status=400)

    monkeypatch.setattr("utils.slo_alerting.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("utils.slo_alerting.time.sleep", lambda delay: None)

    result = notifier.notify({"severity": "critical", "name": "critical_breach", "message": "x"})

    assert result is False
    assert attempts["count"] == 1


def test_build_notifier_from_risk_params():
    notifier = build_slo_alert_notifier(
        {
            "SLO_PAGING_ENABLED": True,
            "SLO_PAGING_WEBHOOK_URL": "https://example.com/hook",
            "SLO_PAGING_MIN_SEVERITY": "warning",
            "SLO_PAGING_TIMEOUT_SECONDS": 4,
            "SLO_PAGING_MAX_RETRIES": 3,
            "SLO_PAGING_RETRY_BACKOFF_SECONDS": 1.25,
            "SLO_PAGING_AUTH_TOKEN": "abc",
            "SLO_PAGING_AUTH_SCHEME": "Token",
        },
        source="unit-test",
    )
    assert notifier is not None
    assert notifier.webhook_url == "https://example.com/hook"
    assert notifier.min_severity == "warning"
    assert notifier.timeout_seconds == 4
    assert notifier.max_retries == 3
    assert notifier.retry_backoff_seconds == 1.25
    assert notifier.auth_token == "abc"
    assert notifier.auth_scheme == "Token"
    assert notifier.source == "unit-test"


def test_build_notifier_returns_none_when_disabled():
    notifier = build_slo_alert_notifier(
        {
            "SLO_PAGING_ENABLED": False,
            "SLO_PAGING_WEBHOOK_URL": "https://example.com/hook",
        }
    )
    assert notifier is None


def test_incident_ticket_notifier_posts_payload(monkeypatch):
    notifier = WebhookIncidentTicketNotifier(
        webhook_url="https://example.com/ticket",
        timeout_seconds=7,
        source="test-suite",
        auth_token="ticket-token",
        runbook_url="https://ops.example.com/runbooks/trading",
        escalation_roster_url="https://ops.example.com/escalation/trading",
    )
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["authorization"] = req.get_header("Authorization")
        captured["idempotency_key"] = req.get_header("Idempotency-key")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _DummyResponse(status=200)

    monkeypatch.setattr("utils.slo_alerting.request.urlopen", _fake_urlopen)
    result = notifier.notify(
        {
            "name": "incident_ack_sla_breach",
            "severity": "critical",
            "context": {"incident": {"incident_id": "inc_123"}},
        }
    )

    assert result is True
    assert captured["url"] == "https://example.com/ticket"
    assert captured["timeout"] == 7
    assert captured["authorization"] == "Bearer ticket-token"
    assert captured["idempotency_key"] == "test-suite:inc_123:incident_ack_sla_breach"
    assert captured["body"]["event_type"] == "incident_ticket"
    assert captured["body"]["incident"]["incident_id"] == "inc_123"
    assert (
        captured["body"]["response_links"]["runbook_url"]
        == "https://ops.example.com/runbooks/trading"
    )
    assert (
        captured["body"]["response_links"]["escalation_roster_url"]
        == "https://ops.example.com/escalation/trading"
    )


def test_build_incident_ticket_notifier_from_risk_params():
    notifier = build_incident_ticket_notifier(
        {
            "INCIDENT_TICKETING_ENABLED": True,
            "INCIDENT_TICKETING_WEBHOOK_URL": "https://example.com/ticket",
            "INCIDENT_TICKETING_TIMEOUT_SECONDS": 5,
            "INCIDENT_TICKETING_MAX_RETRIES": 4,
            "INCIDENT_TICKETING_RETRY_BACKOFF_SECONDS": 2.0,
            "INCIDENT_TICKETING_AUTH_TOKEN": "token-2",
            "INCIDENT_TICKETING_AUTH_SCHEME": "Token",
            "INCIDENT_RESPONSE_RUNBOOK_URL": "https://ops.example.com/runbooks/trading",
            "INCIDENT_ESCALATION_ROSTER_URL": "https://ops.example.com/escalation/trading",
        },
        source="unit-test",
    )
    assert notifier is not None
    assert notifier.webhook_url == "https://example.com/ticket"
    assert notifier.timeout_seconds == 5
    assert notifier.max_retries == 4
    assert notifier.retry_backoff_seconds == 2.0
    assert notifier.auth_token == "token-2"
    assert notifier.auth_scheme == "Token"
    assert notifier.source == "unit-test"
    assert notifier.runbook_url == "https://ops.example.com/runbooks/trading"
    assert notifier.escalation_roster_url == "https://ops.example.com/escalation/trading"
