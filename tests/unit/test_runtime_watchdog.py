from __future__ import annotations

import asyncio
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from urllib import error

from utils import runtime_watchdog


def test_check_alpaca_connectivity_fails_without_credentials():
    check = runtime_watchdog._check_alpaca_connectivity(  # noqa: SLF001
        api_key="",
        secret_key="",
        paper=True,
    )

    assert check.passed is False
    assert "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY" in check.message


def test_check_alpaca_connectivity_passes_with_account(monkeypatch):
    class _FakeAccount:
        id = "acct-1"
        account_number = "PA1234"
        status = "ACTIVE"

    class _FakeTradingClient:
        def __init__(self, api_key, secret_key, paper):  # noqa: ANN001
            self.api_key = api_key
            self.secret_key = secret_key
            self.paper = paper

        def get_account(self):
            return _FakeAccount()

    monkeypatch.setattr(runtime_watchdog, "TradingClient", _FakeTradingClient)

    check = runtime_watchdog._check_alpaca_connectivity(  # noqa: SLF001
        api_key="key",
        secret_key="secret",
        paper=True,
    )

    assert check.passed is True
    assert check.details["account_id"] == "acct-1"
    assert check.details["paper"] is True


def test_check_alpaca_connectivity_handles_client_exception(monkeypatch):
    class _FailingTradingClient:
        def __init__(self, api_key, secret_key, paper):  # noqa: ANN001
            raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(runtime_watchdog, "TradingClient", _FailingTradingClient)

    check = runtime_watchdog._check_alpaca_connectivity(  # noqa: SLF001
        api_key="key",
        secret_key="secret",
        paper=True,
    )

    assert check.passed is False
    assert "alpaca unavailable" in check.message


def test_check_incident_webhook_passes_on_accepted_response(monkeypatch):
    class _Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    def _urlopen(req, timeout):  # noqa: ANN001
        assert req.get_method() == "POST"
        assert timeout == 3
        return _Response()

    monkeypatch.setattr(runtime_watchdog.request, "urlopen", _urlopen)

    check = runtime_watchdog._check_incident_webhook(  # noqa: SLF001
        webhook_url="https://hooks.ops.example.com/incident-ticket",
        timeout_seconds=3,
    )

    assert check.passed is True
    assert check.details["status_code"] == 202


def test_check_incident_webhook_includes_auth_header(monkeypatch):
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    def _urlopen(req, timeout):  # noqa: ANN001
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["authorization"] == "Bearer token-123"
        return _Response()

    monkeypatch.setattr(runtime_watchdog.request, "urlopen", _urlopen)

    check = runtime_watchdog._check_incident_webhook(  # noqa: SLF001
        webhook_url="https://hooks.ops.example.com/incident-ticket",
        timeout_seconds=3,
        auth_token="token-123",
        auth_scheme="Bearer",
    )

    assert check.passed is True


def test_check_incident_webhook_fails_on_http_error(monkeypatch):
    def _urlopen(req, timeout):  # noqa: ANN001
        raise error.HTTPError(req.full_url, 500, "server error", hdrs=None, fp=None)

    monkeypatch.setattr(runtime_watchdog.request, "urlopen", _urlopen)

    check = runtime_watchdog._check_incident_webhook(  # noqa: SLF001
        webhook_url="https://hooks.ops.example.com/incident-ticket",
        timeout_seconds=2,
    )

    assert check.passed is False
    assert check.details["status_code"] == 500


def test_check_incident_webhook_fails_on_generic_exception(monkeypatch):
    def _urlopen(req, timeout):  # noqa: ANN001
        raise TimeoutError("timed out")

    monkeypatch.setattr(runtime_watchdog.request, "urlopen", _urlopen)

    check = runtime_watchdog._check_incident_webhook(  # noqa: SLF001
        webhook_url="https://hooks.ops.example.com/incident-ticket",
        timeout_seconds=2,
    )

    assert check.passed is False
    assert "timed out" in check.message


def test_check_ib_socket_passes_when_connection_available(monkeypatch):
    def _create_connection(address, timeout):  # noqa: ANN001
        assert address == ("127.0.0.1", 7497)
        assert timeout == 1.5
        return nullcontext()

    monkeypatch.setattr(runtime_watchdog.socket, "create_connection", _create_connection)

    check = runtime_watchdog._check_ib_socket(  # noqa: SLF001
        host="127.0.0.1",
        port=7497,
        timeout_seconds=1.5,
    )

    assert check.passed is True
    assert "IB socket reachable" in check.message


def test_check_ib_socket_fails_when_connection_unavailable(monkeypatch):
    def _create_connection(address, timeout):  # noqa: ANN001
        raise OSError("connection refused")

    monkeypatch.setattr(runtime_watchdog.socket, "create_connection", _create_connection)

    check = runtime_watchdog._check_ib_socket(  # noqa: SLF001
        host="127.0.0.1",
        port=7497,
        timeout_seconds=1.5,
    )

    assert check.passed is False
    assert "connection refused" in check.message


def test_check_ib_api_session_passes_when_handshake_succeeds(monkeypatch):
    class _FakeIB:
        def __init__(self):
            self._connected = False

        def connect(self, host, port, clientId, timeout, readonly):  # noqa: N803, ANN001
            self._connected = True

        def isConnected(self):  # noqa: N802
            return self._connected

        def managedAccounts(self):  # noqa: N802
            return ["DU123456"]

        def disconnect(self):
            self._connected = False

    monkeypatch.setitem(sys.modules, "ib_insync", SimpleNamespace(IB=_FakeIB))

    check = runtime_watchdog._check_ib_api_session(  # noqa: SLF001
        host="127.0.0.1",
        port=7497,
        client_id=1,
        timeout_seconds=2.0,
    )

    assert check.passed is True
    assert check.details["accounts"] == ["DU123456"]


def test_check_ib_api_session_fails_without_accounts(monkeypatch):
    class _FakeIB:
        def connect(self, host, port, clientId, timeout, readonly):  # noqa: N803, ANN001
            return None

        def isConnected(self):  # noqa: N802
            return True

        def managedAccounts(self):  # noqa: N802
            return []

        def disconnect(self):
            return None

    monkeypatch.setitem(sys.modules, "ib_insync", SimpleNamespace(IB=_FakeIB))

    check = runtime_watchdog._check_ib_api_session(  # noqa: SLF001
        host="127.0.0.1",
        port=7497,
        client_id=1,
        timeout_seconds=2.0,
    )

    assert check.passed is False
    assert "no managed accounts" in check.message


def test_check_ib_api_session_fails_when_import_unavailable(monkeypatch):
    monkeypatch.delitem(sys.modules, "ib_insync", raising=False)

    import builtins

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "ib_insync":
            raise ImportError("missing ib_insync")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    check = runtime_watchdog._check_ib_api_session(  # noqa: SLF001
        host="127.0.0.1",
        port=7497,
        client_id=1,
        timeout_seconds=2.0,
    )

    assert check.passed is False
    assert "ib_insync import failed" in check.message


def test_check_ib_api_session_handles_connect_and_disconnect_exceptions(monkeypatch):
    class _FakeIB:
        def connect(self, host, port, clientId, timeout, readonly):  # noqa: N803, ANN001
            raise RuntimeError("connect blew up")

        def disconnect(self):
            raise RuntimeError("disconnect blew up")

    monkeypatch.setitem(sys.modules, "ib_insync", SimpleNamespace(IB=_FakeIB))

    check = runtime_watchdog._check_ib_api_session(  # noqa: SLF001
        host="127.0.0.1",
        port=7497,
        client_id=1,
        timeout_seconds=2.0,
    )

    assert check.passed is False
    assert "connect blew up" in check.message


def test_run_runtime_watchdog_ready_when_only_warnings():
    report = asyncio.run(
        runtime_watchdog.run_runtime_watchdog(
            check_alpaca=False,
            check_ticket_webhook=False,
            check_ib_port=False,
            check_ib_api=False,
        )
    )

    assert report["ready"] is True
    assert all(check["severity"] == "warning" for check in report["checks"])


def test_run_runtime_watchdog_not_ready_when_critical_check_fails(monkeypatch):
    monkeypatch.setattr(
        runtime_watchdog,
        "_check_alpaca_connectivity",
        lambda **kwargs: runtime_watchdog.RuntimeWatchdogCheck(
            name="alpaca_connectivity",
            passed=False,
            message="forced failure",
        ),
    )

    report = asyncio.run(
        runtime_watchdog.run_runtime_watchdog(
            check_alpaca=True,
            check_ticket_webhook=False,
            check_ib_port=False,
            check_ib_api=False,
            alpaca_api_key="key",
            alpaca_secret_key="secret",
        )
    )

    assert report["ready"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["alpaca_connectivity"]["passed"] is False


def test_run_runtime_watchdog_invokes_webhook_and_ib_branches(monkeypatch):
    monkeypatch.setattr(
        runtime_watchdog,
        "_check_incident_webhook",
        lambda **kwargs: runtime_watchdog.RuntimeWatchdogCheck(
            name="incident_ticket_webhook",
            passed=True,
            message="ok",
        ),
    )
    monkeypatch.setattr(
        runtime_watchdog,
        "_check_ib_socket",
        lambda **kwargs: runtime_watchdog.RuntimeWatchdogCheck(
            name="ib_socket_port",
            passed=True,
            message="ok",
        ),
    )
    monkeypatch.setattr(
        runtime_watchdog,
        "_check_ib_api_session",
        lambda **kwargs: runtime_watchdog.RuntimeWatchdogCheck(
            name="ib_api_session",
            passed=True,
            message="ok",
        ),
    )

    report = asyncio.run(
        runtime_watchdog.run_runtime_watchdog(
            check_alpaca=False,
            check_ticket_webhook=True,
            check_ib_port=True,
            check_ib_api=True,
        )
    )

    assert report["ready"] is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["incident_ticket_webhook"]["passed"] is True
    assert checks["ib_socket_port"]["passed"] is True
    assert checks["ib_api_session"]["passed"] is True
