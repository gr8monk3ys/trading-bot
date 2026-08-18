"""Tests for dashboard authentication and CORS posture.

The dashboard binds 0.0.0.0 (start.py) and previously served full account
data (/api/account, /api/positions, /api/trades) with no authentication,
plus allow_origins=["*"] with credentials. These tests pin the fixed
behavior: a DASHBOARD_TOKEN env var gates every route except /api/health;
with no token configured the API fails closed; CORS reflection is gone.
"""

import pytest
from fastapi.testclient import TestClient

import web.app as web_app

TOKEN = "test-dashboard-token"


@pytest.fixture
def client(monkeypatch):
    # https base so the secure=True session cookie is sent back by the client.
    with TestClient(web_app.app, base_url="https://testserver") as c:
        # The lifespan wires the real broker/database when .env has credentials;
        # auth tests must exercise the default (offline) endpoint paths, never
        # the network.
        monkeypatch.setattr(web_app, "_broker", None)
        monkeypatch.setattr(web_app, "_db", None)
        yield c


@pytest.fixture
def token_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", TOKEN)


def test_api_fails_closed_when_no_token_configured(monkeypatch, client):
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)

    assert client.get("/api/account").status_code == 503
    assert client.get("/").status_code == 503


def test_health_is_always_reachable(monkeypatch, client):
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    assert client.get("/api/health").status_code == 200


def test_missing_credentials_rejected(token_env, client):
    assert client.get("/api/account").status_code == 401
    assert client.get("/").status_code == 401


def test_wrong_bearer_rejected(token_env, client):
    r = client.get("/api/account", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_correct_bearer_accepted(token_env, client):
    r = client.get("/api/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200
    r = client.get("/api/account", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_query_token_sets_cookie_and_redirects_to_stripped_url(token_env, client):
    # The token must not linger in history/logs/Referer: a valid ?token= visit
    # sets the cookie and 302-redirects to the same path without the query.
    r = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 302
    assert "token" not in r.headers["location"]
    assert client.cookies.get("dashboard_token") == TOKEN
    set_cookie = r.headers.get("set-cookie", "").lower()
    assert "httponly" in set_cookie
    assert "secure" in set_cookie

    # Following the redirect lands on the page; subsequent same-origin
    # requests authenticate via the cookie alone.
    assert client.get(r.headers["location"]).status_code == 200
    assert client.get("/api/account").status_code == 200


def test_wrong_query_token_rejected_and_no_cookie(token_env, client):
    r = client.get("/?token=wrong")
    assert r.status_code == 401
    assert client.cookies.get("dashboard_token") is None


def test_no_cors_reflection_for_foreign_origin(token_env, client):
    r = client.get(
        "/api/account",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
