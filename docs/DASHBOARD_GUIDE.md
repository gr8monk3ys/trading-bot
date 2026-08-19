# Dashboard Guide

Three real monitoring options exist. (A 2026-08 audit found the previous
version of this guide recommended `scripts/enhanced_dashboard.py`, a file
that does not exist; this rewrite documents only what ships.)

## 1. Web dashboard (`web/app.py`)

FastAPI dashboard with account, positions, and trade views.

```bash
python start.py            # runs the bot supervisor + web dashboard
# or standalone:
uvicorn web.app:app --host 127.0.0.1 --port 8000
```

**Authentication is required and fails closed.** Set `DASHBOARD_TOKEN` in
`.env` (generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`).
Without it every route except `/api/health` returns 503.

- Browser: open `http://host:8000/?token=<value>` once — the token is moved
  into a secure HttpOnly cookie and stripped from the URL.
- API: send `Authorization: Bearer <value>`.

## 2. Terminal dashboard (`scripts/dashboard.py`)

Simple text-based account/position display, refreshed in place.

```bash
python scripts/dashboard.py
```

## 3. Monitor script (`scripts/monitor_bot.py`)

Real-time monitoring loop for a running bot.

```bash
python scripts/monitor_bot.py
```

## Running alongside the bot

Use two terminals (or tmux panes): one for `python main.py live --strategy
adaptive`, one for the dashboard of your choice. All three read the same
Alpaca paper account, so they can run independently of the bot process.

## Circuit breaker alerts

When the circuit breaker trips (daily-loss limit or economic-event window),
the bot halts entries and the audit log (`audit_logs/`) records the event.
The web dashboard's positions view reflects any auto-liquidation.
