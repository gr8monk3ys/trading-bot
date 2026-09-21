"""
FastAPI Web Dashboard for Trading Bot

Provides real-time visibility into trading activity via:
- JSON API endpoints for account, positions, trades, and performance data
- HTML dashboard with auto-refreshing charts and tables

Usage:
    uvicorn web.app:app --host 0.0.0.0 --port 8000
"""

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state – populated during lifespan startup
# ---------------------------------------------------------------------------
_broker = None
_history = None
_start_time: float = 0.0
_paper_mode: bool = True

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

BOT_VERSION = "3.0.0"


# ---------------------------------------------------------------------------
# Lifespan – initialize broker (optional) and database on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize broker and database connections on startup, clean up on shutdown."""
    global _broker, _history, _start_time, _paper_mode
    _start_time = time.time()

    # --- Trade history (always attempt) ---
    try:
        from engine.trade_history import SqliteStore, TradeHistory

        _history = TradeHistory(SqliteStore("data/trading_bot.db"))
        logger.info("Dashboard trade history opened")
    except Exception as exc:
        _log_internal_error(
            "Dashboard trade history initialization",
            exc,
            level=logging.WARNING,
        )
        _history = None

    # --- Broker (optional – dashboard still works without it) ---
    try:
        from brokers.alpaca_broker import AlpacaBroker
        from config import ALPACA_CREDS

        api_key = ALPACA_CREDS.get("API_KEY", "")
        if api_key:
            _paper_mode = ALPACA_CREDS.get("PAPER", True)
            _broker = AlpacaBroker(paper=_paper_mode)
            logger.info("Dashboard broker connected")
        else:
            logger.warning("No Alpaca API key configured – broker endpoints will return defaults")
    except Exception as exc:
        _log_internal_error(
            "Dashboard broker initialization",
            exc,
            level=logging.WARNING,
        )
        _broker = None

    yield  # ---- app is running ----

    # --- Shutdown ---
    if _history:
        try:
            _history.close()
        except Exception:
            pass
    logger.info("Dashboard shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Trading Bot Dashboard",
    version=BOT_VERSION,
    lifespan=lifespan,
)

# NOTE: no CORS middleware on purpose. The dashboard is a same-origin app
# (HTML + /api consumed by its own page); allow_origins=["*"] with
# credentials let any page in the operator's browser read account data
# cross-origin.


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
# The dashboard binds 0.0.0.0 (start.py) and serves full account state, so
# every route except the platform healthcheck requires DASHBOARD_TOKEN.
# Fail-closed: with no token configured, the app refuses to serve rather
# than exposing account data unauthenticated.
_AUTH_EXEMPT_PATHS = {"/api/health"}
_AUTH_COOKIE = "dashboard_token"


@app.middleware("http")
async def _require_dashboard_token(request: Request, call_next):
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    configured = os.environ.get("DASHBOARD_TOKEN", "")
    if not configured:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Dashboard auth not configured. Set DASHBOARD_TOKEN in the "
                "environment; requests must send it as 'Authorization: Bearer <token>' "
                "(or open /?token=<token> once in a browser)."
            },
        )

    supplied = ""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        supplied = auth_header[7:]
    elif "token" in request.query_params:
        supplied = request.query_params["token"]
    else:
        supplied = request.cookies.get(_AUTH_COOKIE, "")

    if not supplied or not hmac.compare_digest(supplied, configured):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    # Browser flow: a valid ?token= visit gets an HttpOnly session cookie so
    # the page's own /api fetches authenticate without embedding the token.
    # Redirect to the stripped URL immediately so the token does not linger
    # in browser history, server logs, or Referer headers. secure=True is
    # safe for local use too — browsers treat localhost as a secure context.
    if request.query_params.get("token"):
        stripped = request.url.remove_query_params("token")
        response = RedirectResponse(url=str(stripped), status_code=302)
        response.set_cookie(_AUTH_COOKIE, configured, httponly=True, samesite="strict", secure=True)
        return response

    return await call_next(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_timestamp(dt: Any) -> Optional[str]:
    """Format a datetime-like object to ISO string."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    try:
        return dt.isoformat()
    except AttributeError:
        return str(dt)


def _log_internal_error(context: str, exc: Exception, *, level: int = logging.ERROR) -> None:
    """Log a sanitized internal failure without leaking exception content."""
    logger.log(level, "%s failed with %s", context, type(exc).__name__)


def _server_error_response(message: str, **payload: Any) -> JSONResponse:
    """Return a generic API error without exposing exception details."""
    content = dict(payload)
    content["error"] = message
    return JSONResponse(content=content, status_code=500)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard HTML page."""
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "paper_mode": _paper_mode,
            "version": BOT_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check endpoint with uptime and connection status."""
    uptime_seconds = time.time() - _start_time if _start_time else 0
    return {
        "status": "ok",
        "version": BOT_VERSION,
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime_human": _format_uptime(uptime_seconds),
        "broker_connected": _broker is not None,
        "database_connected": _history is not None,
        "paper_mode": _paper_mode,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


@app.get("/api/account")
async def get_account():
    """Account information from Alpaca broker."""
    if not _broker:
        return JSONResponse(
            content={
                "equity": 0,
                "cash": 0,
                "buying_power": 0,
                "portfolio_value": 0,
                "day_pnl": 0,
                "day_pnl_pct": 0,
                "paper_mode": _paper_mode,
                "error": "Broker not connected",
            },
            status_code=200,
        )

    try:
        account = await _broker.get_account()
        equity = _safe_float(account.equity)
        last_equity = _safe_float(account.last_equity)
        day_pnl = equity - last_equity if last_equity > 0 else 0
        day_pnl_pct = (day_pnl / last_equity * 100) if last_equity > 0 else 0

        return {
            "equity": equity,
            "cash": _safe_float(account.cash),
            "buying_power": _safe_float(account.buying_power),
            "portfolio_value": _safe_float(account.portfolio_value),
            "last_equity": last_equity,
            "day_pnl": round(day_pnl, 2),
            "day_pnl_pct": round(day_pnl_pct, 4),
            "paper_mode": _paper_mode,
        }
    except Exception as exc:
        _log_internal_error("Account fetch", exc)
        return _server_error_response(
            "Unable to fetch account data",
            paper_mode=_paper_mode,
        )


@app.get("/api/positions")
async def get_positions():
    """Current open positions from Alpaca with unrealized P&L."""
    if not _broker:
        return {"positions": [], "count": 0, "error": "Broker not connected"}

    try:
        positions = await _broker.get_positions()
        result = []
        for pos in positions:
            result.append(
                {
                    "symbol": str(pos.symbol),
                    "qty": _safe_float(pos.qty),
                    "side": str(pos.side) if hasattr(pos, "side") else "long",
                    "avg_entry_price": _safe_float(pos.avg_entry_price),
                    "current_price": _safe_float(pos.current_price),
                    "market_value": _safe_float(pos.market_value),
                    "unrealized_pl": _safe_float(pos.unrealized_pl),
                    "unrealized_plpc": _safe_float(pos.unrealized_plpc),
                }
            )
        return {"positions": result, "count": len(result)}
    except Exception as exc:
        _log_internal_error("Positions fetch", exc)
        return _server_error_response(
            "Unable to fetch positions",
            positions=[],
            count=0,
        )


@app.get("/api/trades")
async def get_trades(limit: int = Query(default=20, ge=1, le=500)):
    """Recent completed trades from trade history (newest first)."""
    if not _history:
        return {"trades": [], "count": 0, "error": "Trade history not available"}

    try:
        result = [t.to_dict() for t in _history.trades(limit=limit)]
        return {"trades": result, "count": len(result)}
    except Exception as exc:
        _log_internal_error("Trades fetch", exc)
        return _server_error_response("Unable to fetch trades", trades=[], count=0)


@app.get("/api/performance")
async def get_performance():
    """Performance from trade history: summary plus 90-day daily-P&L statistics."""
    if not _history:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "error": "Trade history not available",
        }
    try:
        summary = _history.summary()
        response: Dict[str, Any] = {
            "total_trades": summary["total_trades"],
            "winning_trades": summary["winning_trades"],
            "win_rate": round(summary["win_rate"] * 100, 2),
            "total_pnl": round(summary["total_pnl"], 2),
            "profit_factor": summary["profit_factor"],
            "avg_win": summary["avg_win"],
            "avg_loss": summary["avg_loss"],
            "unique_symbols": summary["unique_symbols"],
            "unique_strategies": summary["unique_strategies"],
            "first_trade": summary["first_trade"].isoformat() if summary["first_trade"] else None,
            "last_trade": summary["last_trade"].isoformat() if summary["last_trade"] else None,
            "sharpe_ratio": None,
            "max_drawdown": None,
        }
        end = date.today()
        daily = _history.daily(end - timedelta(days=90), end)
        pnls = [d["pnl"] for d in daily]
        if len(pnls) >= 2:
            import statistics

            std = statistics.stdev(pnls)
            response["sharpe_ratio"] = (
                round(statistics.mean(pnls) / std * (252**0.5), 2) if std > 0 else None
            )
            peak = running = 0.0
            max_dd = 0.0
            for pnl in pnls:
                running += pnl
                peak = max(peak, running)
                max_dd = min(max_dd, running - peak)
            response["max_drawdown"] = round(max_dd, 2)
        return response
    except Exception as exc:
        _log_internal_error("Performance fetch", exc)
        return _server_error_response("Unable to fetch performance data")


@app.get("/api/daily-metrics")
async def get_daily_metrics(days: int = Query(default=30, ge=1, le=365)):
    """Daily realised P&L for charting."""
    if not _history:
        return {"metrics": [], "count": 0, "error": "Trade history not available"}

    try:
        end = date.today()
        result = _history.daily(end - timedelta(days=days), end)
        return {"metrics": result, "count": len(result)}
    except Exception as exc:
        _log_internal_error("Daily metrics fetch", exc)
        return _server_error_response("Unable to fetch daily metrics", metrics=[], count=0)


@app.get("/api/market-status")
async def get_market_status():
    """Current market open/closed status."""
    if not _broker:
        return {"is_open": False, "error": "Broker not connected"}

    try:
        status = await _broker.get_market_status()
        return {
            "is_open": status.get("is_open", False),
            "next_open": _format_timestamp(status.get("next_open")),
            "next_close": _format_timestamp(status.get("next_close")),
        }
    except Exception as exc:
        _log_internal_error("Market status fetch", exc)
        return {"is_open": False, "error": "Unable to fetch market status"}


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
