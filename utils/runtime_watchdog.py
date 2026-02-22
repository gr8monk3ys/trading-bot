"""
Runtime watchdog checks for broker and incident-webhook connectivity.
"""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib import error, request

from alpaca.trading.client import TradingClient


@dataclass
class RuntimeWatchdogCheck:
    name: str
    passed: bool
    message: str
    severity: str = "critical"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
        }


def _check_alpaca_connectivity(
    *,
    api_key: str,
    secret_key: str,
    paper: bool,
) -> RuntimeWatchdogCheck:
    key = str(api_key or "").strip()
    secret = str(secret_key or "").strip()
    if not key or not secret:
        return RuntimeWatchdogCheck(
            name="alpaca_connectivity",
            passed=False,
            message="Missing ALPACA_API_KEY or ALPACA_SECRET_KEY",
        )

    try:
        client = TradingClient(api_key=key, secret_key=secret, paper=paper)
        account = client.get_account()
        return RuntimeWatchdogCheck(
            name="alpaca_connectivity",
            passed=True,
            message="Alpaca account connectivity verified",
            details={
                "account_id": str(getattr(account, "id", "") or ""),
                "account_number": str(getattr(account, "account_number", "") or ""),
                "status": str(getattr(account, "status", "") or ""),
                "paper": bool(paper),
            },
        )
    except Exception as exc:
        return RuntimeWatchdogCheck(
            name="alpaca_connectivity",
            passed=False,
            message=f"Alpaca connectivity check failed: {exc}",
        )


def _check_incident_webhook(
    *,
    webhook_url: str,
    timeout_seconds: int,
    auth_token: str = "",
    auth_scheme: str = "Bearer",
) -> RuntimeWatchdogCheck:
    url = str(webhook_url or "").strip()
    if not url:
        return RuntimeWatchdogCheck(
            name="incident_ticket_webhook",
            passed=False,
            message="Missing INCIDENT_TICKETING_WEBHOOK_URL",
        )

    payload = {
        "event_type": "runtime_watchdog_ping",
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "source": "scripts.runtime_watchdog",
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "trading-bot-runtime-watchdog/1.0",
    }
    token = str(auth_token or "").strip()
    scheme = str(auth_scheme or "").strip()
    if token:
        headers["Authorization"] = f"{scheme} {token}" if scheme else token

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=max(1, int(timeout_seconds))) as response:  # noqa: S310
            status = int(getattr(response, "status", 200) or 200)
        passed = 200 <= status < 400
        return RuntimeWatchdogCheck(
            name="incident_ticket_webhook",
            passed=passed,
            message=(
                f"Incident webhook accepted watchdog ping (HTTP {status})"
                if passed
                else f"Incident webhook returned HTTP {status}"
            ),
            details={"status_code": status, "webhook_url": url},
        )
    except error.HTTPError as exc:
        return RuntimeWatchdogCheck(
            name="incident_ticket_webhook",
            passed=False,
            message=f"Incident webhook HTTP error: {exc.code}",
            details={"status_code": int(exc.code), "webhook_url": url},
        )
    except Exception as exc:
        return RuntimeWatchdogCheck(
            name="incident_ticket_webhook",
            passed=False,
            message=f"Incident webhook connectivity check failed: {exc}",
            details={"webhook_url": url},
        )


def _check_ib_socket(
    *,
    host: str,
    port: int,
    timeout_seconds: float,
) -> RuntimeWatchdogCheck:
    socket_host = str(host or "127.0.0.1").strip() or "127.0.0.1"
    socket_port = int(port)
    try:
        with socket.create_connection(
            (socket_host, socket_port),
            timeout=max(0.1, float(timeout_seconds)),
        ):
            pass
        return RuntimeWatchdogCheck(
            name="ib_socket_port",
            passed=True,
            message=f"IB socket reachable at {socket_host}:{socket_port}",
            details={"host": socket_host, "port": socket_port},
        )
    except Exception as exc:
        return RuntimeWatchdogCheck(
            name="ib_socket_port",
            passed=False,
            message=f"IB socket connectivity failed at {socket_host}:{socket_port}: {exc}",
            details={"host": socket_host, "port": socket_port},
        )


def _check_ib_api_session(
    *,
    host: str,
    port: int,
    client_id: int,
    timeout_seconds: float,
) -> RuntimeWatchdogCheck:
    socket_host = str(host or "127.0.0.1").strip() or "127.0.0.1"
    socket_port = int(port)
    session_client_id = int(client_id)
    created_loop = None
    try:
        # ib_insync expects an event loop in the current thread.
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            created_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(created_loop)

        from ib_insync import IB  # type: ignore
    except Exception as exc:
        return RuntimeWatchdogCheck(
            name="ib_api_session",
            passed=False,
            message=f"IB API readiness check unavailable (ib_insync import failed): {exc}",
            details={"host": socket_host, "port": socket_port, "client_id": session_client_id},
        )

    ib = IB()
    try:
        ib.connect(
            host=socket_host,
            port=socket_port,
            clientId=session_client_id,
            timeout=max(0.1, float(timeout_seconds)),
            readonly=True,
        )
        if not bool(ib.isConnected()):
            return RuntimeWatchdogCheck(
                name="ib_api_session",
                passed=False,
                message=f"IB API session failed to connect at {socket_host}:{socket_port}",
                details={"host": socket_host, "port": socket_port, "client_id": session_client_id},
            )

        accounts = []
        managed_accounts = getattr(ib, "managedAccounts", None)
        if callable(managed_accounts):
            accounts = [str(account).strip() for account in list(managed_accounts() or [])]
            accounts = [account for account in accounts if account]

        if not accounts:
            return RuntimeWatchdogCheck(
                name="ib_api_session",
                passed=False,
                message="IB API connected but no managed accounts were returned",
                details={
                    "host": socket_host,
                    "port": socket_port,
                    "client_id": session_client_id,
                },
            )

        return RuntimeWatchdogCheck(
            name="ib_api_session",
            passed=True,
            message=f"IB API session established at {socket_host}:{socket_port}",
            details={
                "host": socket_host,
                "port": socket_port,
                "client_id": session_client_id,
                "accounts": accounts,
            },
        )
    except Exception as exc:
        return RuntimeWatchdogCheck(
            name="ib_api_session",
            passed=False,
            message=f"IB API session check failed at {socket_host}:{socket_port}: {exc}",
            details={"host": socket_host, "port": socket_port, "client_id": session_client_id},
        )
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
        if created_loop is not None:
            try:
                created_loop.close()
            except Exception:
                pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass


async def run_runtime_watchdog(
    *,
    check_alpaca: bool = True,
    check_ticket_webhook: bool = True,
    check_ib_port: bool = True,
    check_ib_api: bool = True,
    alpaca_api_key: str = "",
    alpaca_secret_key: str = "",
    alpaca_paper: bool = True,
    ticket_webhook_url: str = "",
    ticket_timeout_seconds: int = 5,
    ticket_auth_token: str = "",
    ticket_auth_scheme: str = "Bearer",
    ib_host: str = "127.0.0.1",
    ib_port: int = 7497,
    ib_client_id: int = 1,
    ib_timeout_seconds: float = 2.0,
) -> Dict[str, Any]:
    checks: List[RuntimeWatchdogCheck] = []

    if check_alpaca:
        checks.append(
            _check_alpaca_connectivity(
                api_key=alpaca_api_key,
                secret_key=alpaca_secret_key,
                paper=bool(alpaca_paper),
            )
        )
    else:
        checks.append(
            RuntimeWatchdogCheck(
                name="alpaca_connectivity",
                passed=True,
                severity="warning",
                message="Alpaca connectivity check skipped by configuration",
            )
        )

    if check_ticket_webhook:
        checks.append(
            _check_incident_webhook(
                webhook_url=ticket_webhook_url,
                timeout_seconds=max(1, int(ticket_timeout_seconds)),
                auth_token=ticket_auth_token,
                auth_scheme=ticket_auth_scheme,
            )
        )
    else:
        checks.append(
            RuntimeWatchdogCheck(
                name="incident_ticket_webhook",
                passed=True,
                severity="warning",
                message="Incident webhook check skipped by configuration",
            )
        )

    if check_ib_port:
        checks.append(
            _check_ib_socket(
                host=ib_host,
                port=max(1, int(ib_port)),
                timeout_seconds=max(0.1, float(ib_timeout_seconds)),
            )
        )
    else:
        checks.append(
            RuntimeWatchdogCheck(
                name="ib_socket_port",
                passed=True,
                severity="warning",
                message="IB socket check skipped by configuration",
            )
        )

    if check_ib_api:
        checks.append(
            await asyncio.to_thread(
                _check_ib_api_session,
                host=ib_host,
                port=max(1, int(ib_port)),
                client_id=max(1, int(ib_client_id)),
                timeout_seconds=max(0.1, float(ib_timeout_seconds)),
            )
        )
    else:
        checks.append(
            RuntimeWatchdogCheck(
                name="ib_api_session",
                passed=True,
                severity="warning",
                message="IB API session check skipped by configuration",
            )
        )

    ready = all(check.passed for check in checks if check.severity == "critical")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "checks": [check.to_dict() for check in checks],
    }
