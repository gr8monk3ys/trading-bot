"""
External paging hooks for operational SLO breaches.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional
from urllib import error, request

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {
    "warning": 1,
    "critical": 2,
}


def _rank(severity: str) -> int:
    return _SEVERITY_RANK.get(str(severity or "").strip().lower(), 0)


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
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(self.webhook_url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "trading-bot-slo-monitor/1.0")

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200) or 200)
                if status >= 400:
                    logger.warning("SLO alert webhook non-success status=%s", status)
                    return False
                return True
        except error.URLError as exc:
            logger.warning("SLO alert webhook delivery failed: %s", exc)
            return False
        except Exception as exc:
            logger.warning("Unexpected SLO alert webhook error: %s", exc)
            return False


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

    try:
        timeout_seconds = int(params.get("SLO_PAGING_TIMEOUT_SECONDS", 3))
    except (TypeError, ValueError):
        timeout_seconds = 3
    timeout_seconds = min(max(timeout_seconds, 1), 30)

    return WebhookSLOAlertNotifier(
        webhook_url=webhook_url,
        min_severity=min_severity,
        timeout_seconds=timeout_seconds,
        source=source,
    )
