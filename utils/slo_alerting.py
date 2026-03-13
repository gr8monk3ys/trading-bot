"""
External paging hooks for operational SLO breaches.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib import error, parse, request

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {
    "warning": 1,
    "critical": 2,
}


def _rank(severity: str) -> int:
    return _SEVERITY_RANK.get(str(severity or "").strip().lower(), 0)


def _parse_int_clamped(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return min(max(parsed, minimum), maximum)


def _parse_float_clamped(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return min(max(parsed, minimum), maximum)


def _authorization_header_value(auth_scheme: str, auth_token: str) -> str | None:
    token = str(auth_token or "").strip()
    if not token:
        return None
    scheme = str(auth_scheme or "").strip()
    if not scheme:
        return token
    return f"{scheme} {token}"


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def _deliver_json_webhook(
    *,
    webhook_url: str,
    payload: Mapping[str, Any],
    timeout_seconds: int,
    user_agent: str,
    description: str,
    max_retries: int,
    retry_backoff_seconds: float,
    authorization_header: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> bool:
    parsed_url = parse.urlsplit(str(webhook_url or "").strip())
    if parsed_url.scheme.lower() == "file":
        output_path = Path(request.url2pathname(parsed_url.path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload_record = {
            "description": description,
            "payload": dict(payload),
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload_record))
            handle.write("\n")
        return True

    body = json.dumps(dict(payload)).encode("utf-8")
    attempts = max(1, int(max_retries) + 1)

    for attempt_index in range(attempts):
        attempt = attempt_index + 1
        req = request.Request(webhook_url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", user_agent)
        if authorization_header:
            req.add_header("Authorization", authorization_header)
        for header_name, header_value in (extra_headers or {}).items():
            if str(header_name).strip() and str(header_value).strip():
                req.add_header(str(header_name), str(header_value))

        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", 200) or 200)
                if status < 400:
                    return True

                retryable = _is_retryable_status(status)
                if retryable and attempt < attempts:
                    delay = min(float(retry_backoff_seconds) * (2**attempt_index), 10.0)
                    logger.warning(
                        "%s webhook non-success status=%s (attempt %s/%s), retrying in %.2fs",
                        description,
                        status,
                        attempt,
                        attempts,
                        delay,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue

                logger.warning(
                    "%s webhook non-success status=%s (attempt %s/%s)",
                    description,
                    status,
                    attempt,
                    attempts,
                )
                return False
        except error.URLError as exc:
            if attempt < attempts:
                delay = min(float(retry_backoff_seconds) * (2**attempt_index), 10.0)
                logger.warning(
                    "%s webhook delivery failed (attempt %s/%s): %s; retrying in %.2fs",
                    description,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            logger.warning("%s webhook delivery failed: %s", description, exc)
            return False
        except Exception as exc:
            if attempt < attempts:
                delay = min(float(retry_backoff_seconds) * (2**attempt_index), 10.0)
                logger.warning(
                    "Unexpected %s webhook error (attempt %s/%s): %s; retrying in %.2fs",
                    description.lower(),
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            logger.warning("Unexpected %s webhook error: %s", description.lower(), exc)
            return False

    return False


@dataclass
class WebhookSLOAlertNotifier:
    """
    Delivers SLO breach notifications to an external webhook endpoint.

    Compatible with common webhook-based paging tools (Slack, PagerDuty relay,
    incident intake proxies) as long as they accept JSON POST payloads.
    """

    webhook_url: str
    min_severity: str = "critical"
    timeout_seconds: int = 3
    source: str = "trading-bot"
    max_retries: int = 0
    retry_backoff_seconds: float = 0.5
    auth_token: str = ""
    auth_scheme: str = "Bearer"

    def notify(self, breach: Mapping[str, Any]) -> Optional[bool]:
        """
        Send one SLO breach alert.

        Returns:
            - None: alert intentionally skipped (below min severity)
            - True: delivered
            - False: attempted but failed
        """
        severity = str(breach.get("severity", "warning")).strip().lower()
        if _rank(severity) < _rank(self.min_severity):
            return None

        payload = {
            "event_type": "slo_breach",
            "source": self.source,
            "sent_at": datetime.utcnow().isoformat(),
            "breach": dict(breach),
        }
        authorization_header = _authorization_header_value(self.auth_scheme, self.auth_token)
        return _deliver_json_webhook(
            webhook_url=self.webhook_url,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            user_agent="trading-bot-slo-monitor/1.0",
            description="SLO alert",
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            authorization_header=authorization_header,
        )


@dataclass
class WebhookIncidentTicketNotifier:
    """Creates incident-ticket payloads for acknowledgment SLA breaches."""

    webhook_url: str
    timeout_seconds: int = 3
    source: str = "trading-bot"
    max_retries: int = 0
    retry_backoff_seconds: float = 0.5
    auth_token: str = ""
    auth_scheme: str = "Bearer"
    runbook_url: str = ""
    escalation_roster_url: str = ""

    def notify(self, breach: Mapping[str, Any]) -> Optional[bool]:
        if str(breach.get("name", "")).strip().lower() != "incident_ack_sla_breach":
            return None

        context = dict(breach.get("context", {}) or {})
        incident = dict(context.get("incident", {}) or {})
        incident_id = str(incident.get("incident_id") or context.get("incident_id") or "").strip()
        incident_name = str(breach.get("name", "incident_ack_sla_breach")).strip()
        idempotency_key = f"{self.source}:{incident_id}:{incident_name}" if incident_id else ""

        payload = {
            "event_type": "incident_ticket",
            "source": self.source,
            "sent_at": datetime.utcnow().isoformat(),
            "breach": dict(breach),
            "incident": incident,
        }
        response_links = {}
        if self.runbook_url:
            response_links["runbook_url"] = self.runbook_url
        if self.escalation_roster_url:
            response_links["escalation_roster_url"] = self.escalation_roster_url
        if response_links:
            payload["response_links"] = response_links
        authorization_header = _authorization_header_value(self.auth_scheme, self.auth_token)
        extra_headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return _deliver_json_webhook(
            webhook_url=self.webhook_url,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            user_agent="trading-bot-incident-ticketing/1.0",
            description="Incident ticket",
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            authorization_header=authorization_header,
            extra_headers=extra_headers,
        )


def build_slo_alert_notifier(
    risk_params: Mapping[str, Any] | None,
    *,
    source: str = "trading-bot",
) -> Optional[WebhookSLOAlertNotifier]:
    """
    Build notifier from runtime risk configuration.
    """
    params = risk_params or {}
    enabled = bool(params.get("SLO_PAGING_ENABLED", False))
    webhook_url = str(params.get("SLO_PAGING_WEBHOOK_URL", "") or "").strip()
    if not enabled or not webhook_url:
        return None

    min_severity = str(params.get("SLO_PAGING_MIN_SEVERITY", "critical") or "critical")
    min_severity = min_severity.strip().lower()
    if min_severity not in _SEVERITY_RANK:
        min_severity = "critical"

    timeout_seconds = _parse_int_clamped(
        params.get("SLO_PAGING_TIMEOUT_SECONDS", 3), 3, minimum=1, maximum=30
    )
    max_retries = _parse_int_clamped(
        params.get("SLO_PAGING_MAX_RETRIES", 0),
        0,
        minimum=0,
        maximum=5,
    )
    retry_backoff_seconds = _parse_float_clamped(
        params.get("SLO_PAGING_RETRY_BACKOFF_SECONDS", 0.5),
        0.5,
        minimum=0.0,
        maximum=10.0,
    )
    auth_token = str(params.get("SLO_PAGING_AUTH_TOKEN", "") or "").strip()
    auth_scheme = str(params.get("SLO_PAGING_AUTH_SCHEME", "Bearer") or "").strip()

    return WebhookSLOAlertNotifier(
        webhook_url=webhook_url,
        min_severity=min_severity,
        timeout_seconds=timeout_seconds,
        source=source,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        auth_token=auth_token,
        auth_scheme=auth_scheme,
    )


def build_incident_ticket_notifier(
    risk_params: Mapping[str, Any] | None,
    *,
    source: str = "trading-bot",
) -> Optional[WebhookIncidentTicketNotifier]:
    """Build incident-ticket notifier from runtime risk configuration."""
    params = risk_params or {}
    enabled = bool(params.get("INCIDENT_TICKETING_ENABLED", False))
    webhook_url = str(params.get("INCIDENT_TICKETING_WEBHOOK_URL", "") or "").strip()
    if not enabled or not webhook_url:
        return None

    timeout_seconds = _parse_int_clamped(
        params.get("INCIDENT_TICKETING_TIMEOUT_SECONDS", 3),
        3,
        minimum=1,
        maximum=30,
    )
    max_retries = _parse_int_clamped(
        params.get("INCIDENT_TICKETING_MAX_RETRIES", 0),
        0,
        minimum=0,
        maximum=5,
    )
    retry_backoff_seconds = _parse_float_clamped(
        params.get("INCIDENT_TICKETING_RETRY_BACKOFF_SECONDS", 0.5),
        0.5,
        minimum=0.0,
        maximum=10.0,
    )
    auth_token = str(params.get("INCIDENT_TICKETING_AUTH_TOKEN", "") or "").strip()
    auth_scheme = str(params.get("INCIDENT_TICKETING_AUTH_SCHEME", "Bearer") or "").strip()
    runbook_url = str(params.get("INCIDENT_RESPONSE_RUNBOOK_URL", "") or "").strip()
    escalation_roster_url = str(params.get("INCIDENT_ESCALATION_ROSTER_URL", "") or "").strip()

    return WebhookIncidentTicketNotifier(
        webhook_url=webhook_url,
        timeout_seconds=timeout_seconds,
        source=source,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        auth_token=auth_token,
        auth_scheme=auth_scheme,
        runbook_url=runbook_url,
        escalation_roster_url=escalation_roster_url,
    )
