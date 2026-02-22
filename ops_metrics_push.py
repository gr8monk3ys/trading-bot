"""
External delivery helpers for operational Prometheus metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib import error, parse, request


@dataclass
class MetricsPushResult:
    attempted: bool
    delivered: bool
    status_code: int | None
    endpoint: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "delivered": self.delivered,
            "status_code": self.status_code,
            "endpoint": self.endpoint,
            "message": self.message,
        }


def _build_pushgateway_endpoint(base_url: str, job: str, instance: str = "") -> str:
    normalized_base = str(base_url or "").strip().rstrip("/")
    if not normalized_base:
        return ""

    encoded_job = parse.quote(str(job or "trading_bot"), safe="")
    endpoint = f"{normalized_base}/metrics/job/{encoded_job}"
    normalized_instance = str(instance or "").strip()
    if normalized_instance:
        encoded_instance = parse.quote(normalized_instance, safe="")
        endpoint = f"{endpoint}/instance/{encoded_instance}"
    return endpoint


def _authorization_header(auth_scheme: str, auth_token: str) -> str | None:
    token = str(auth_token or "").strip()
    if not token:
        return None
    scheme = str(auth_scheme or "").strip()
    if not scheme:
        return token
    return f"{scheme} {token}"


def push_prometheus_metrics(
    *,
    pushgateway_url: str,
    metrics_text: str,
    job: str = "trading_bot",
    instance: str = "",
    timeout_seconds: int = 5,
    method: str = "PUT",
    auth_token: str = "",
    auth_scheme: str = "Bearer",
) -> MetricsPushResult:
    """
    Push Prometheus text metrics to a Pushgateway-compatible endpoint.
    """
    endpoint = _build_pushgateway_endpoint(pushgateway_url, job=job, instance=instance)
    if not endpoint:
        return MetricsPushResult(
            attempted=False,
            delivered=False,
            status_code=None,
            endpoint="",
            message="pushgateway_url is empty",
        )

    normalized_method = str(method or "PUT").strip().upper()
    if normalized_method not in {"PUT", "POST"}:
        return MetricsPushResult(
            attempted=False,
            delivered=False,
            status_code=None,
            endpoint=endpoint,
            message=f"Unsupported HTTP method: {normalized_method}",
        )

    body = str(metrics_text or "").encode("utf-8")
    req = request.Request(endpoint, data=body, method=normalized_method)
    req.add_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
    req.add_header("User-Agent", "trading-bot-ops-metrics/1.0")

    auth_header = _authorization_header(auth_scheme, auth_token)
    if auth_header:
        req.add_header("Authorization", auth_header)

    try:
        with request.urlopen(req, timeout=max(1, int(timeout_seconds))) as response:  # noqa: S310
            status = int(getattr(response, "status", 200) or 200)
        delivered = 200 <= status < 300
        return MetricsPushResult(
            attempted=True,
            delivered=delivered,
            status_code=status,
            endpoint=endpoint,
            message=(
                f"Metrics push delivered (HTTP {status})"
                if delivered
                else f"Metrics push failed with HTTP {status}"
            ),
        )
    except error.HTTPError as exc:
        return MetricsPushResult(
            attempted=True,
            delivered=False,
            status_code=int(exc.code),
            endpoint=endpoint,
            message=f"Metrics push HTTP error: {exc.code}",
        )
    except Exception as exc:
        return MetricsPushResult(
            attempted=True,
            delivered=False,
            status_code=None,
            endpoint=endpoint,
            message=f"Metrics push failed: {exc}",
        )
