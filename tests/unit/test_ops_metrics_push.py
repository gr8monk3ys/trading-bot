from __future__ import annotations

from urllib import error

from utils.ops_metrics_push import _build_pushgateway_endpoint, push_prometheus_metrics


class _DummyResponse:
    def __init__(self, status: int = 202):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_build_pushgateway_endpoint_encodes_job_and_instance():
    endpoint = _build_pushgateway_endpoint(
        "https://push.example.com/",
        job="trading bot/prod",
        instance="pi-01",
    )
    assert endpoint == "https://push.example.com/metrics/job/trading%20bot%2Fprod/instance/pi-01"


def test_push_prometheus_metrics_skips_without_url():
    result = push_prometheus_metrics(
        pushgateway_url="",
        metrics_text="metric 1\n",
    )
    assert result.attempted is False
    assert result.delivered is False


def test_push_prometheus_metrics_success(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["content_type"] = req.get_header("Content-type")
        captured["body"] = req.data.decode("utf-8")
        return _DummyResponse(status=200)

    monkeypatch.setattr("utils.ops_metrics_push.request.urlopen", _fake_urlopen)

    result = push_prometheus_metrics(
        pushgateway_url="https://push.example.com",
        metrics_text="trade_ops_metric 1\n",
        job="trade_ops",
        instance="pi-01",
        timeout_seconds=7,
        auth_token="secret-token",
    )
    assert result.attempted is True
    assert result.delivered is True
    assert result.status_code == 200
    assert captured["url"].endswith("/metrics/job/trade_ops/instance/pi-01")
    assert captured["timeout"] == 7
    assert captured["auth"] == "Bearer secret-token"
    assert captured["content_type"] == "text/plain; version=0.0.4; charset=utf-8"
    assert captured["body"] == "trade_ops_metric 1\n"


def test_push_prometheus_metrics_http_error(monkeypatch):
    def _fake_urlopen(req, timeout):
        raise error.HTTPError(req.full_url, 500, "boom", hdrs=None, fp=None)

    monkeypatch.setattr("utils.ops_metrics_push.request.urlopen", _fake_urlopen)

    result = push_prometheus_metrics(
        pushgateway_url="https://push.example.com",
        metrics_text="x 1\n",
    )
    assert result.attempted is True
    assert result.delivered is False
    assert result.status_code == 500
