import json
import logging

import pytest

import web.app as web_app


class _FailingBroker:
    async def get_account(self):
        raise RuntimeError("super-secret-account")

    async def get_positions(self):
        raise RuntimeError("super-secret-positions")

    async def get_market_status(self):
        raise RuntimeError("super-secret-market-status")


class _FailingDatabase:
    async def get_trades(self, *, limit):
        raise RuntimeError(f"super-secret-trades-{limit}")

    async def get_summary_stats(self):
        raise RuntimeError("super-secret-performance")

    async def get_daily_metrics(self, start, end):
        raise RuntimeError(f"super-secret-daily-metrics-{start}-{end}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "kwargs", "expected_payload", "secret_fragment"),
    [
        (
            web_app.get_account,
            {},
            {"error": "Unable to fetch account data", "paper_mode": True},
            "super-secret-account",
        ),
        (
            web_app.get_positions,
            {},
            {"positions": [], "count": 0, "error": "Unable to fetch positions"},
            "super-secret-positions",
        ),
        (
            web_app.get_trades,
            {"limit": 20},
            {"trades": [], "count": 0, "error": "Unable to fetch trades"},
            "super-secret-trades",
        ),
        (
            web_app.get_performance,
            {},
            {"error": "Unable to fetch performance data"},
            "super-secret-performance",
        ),
        (
            web_app.get_daily_metrics,
            {"days": 30},
            {"metrics": [], "count": 0, "error": "Unable to fetch daily metrics"},
            "super-secret-daily-metrics",
        ),
    ],
)
async def test_error_responses_do_not_expose_exception_details(
    endpoint,
    kwargs,
    expected_payload,
    secret_fragment,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(web_app, "_broker", _FailingBroker())
    monkeypatch.setattr(web_app, "_db", _FailingDatabase())
    monkeypatch.setattr(web_app, "_paper_mode", True)

    with caplog.at_level(logging.ERROR):
        response = await endpoint(**kwargs)

    assert response.status_code == 500
    assert json.loads(response.body) == expected_payload
    assert secret_fragment not in caplog.text


@pytest.mark.asyncio
async def test_market_status_error_response_is_sanitized(monkeypatch, caplog):
    monkeypatch.setattr(web_app, "_broker", _FailingBroker())

    with caplog.at_level(logging.ERROR):
        response = await web_app.get_market_status()

    assert response == {
        "is_open": False,
        "error": "Unable to fetch market status",
    }
    assert "super-secret-market-status" not in caplog.text
