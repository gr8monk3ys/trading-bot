"""
FastAPI Web Dashboard for Trading Bot

Provides real-time visibility into trading activity via:
- JSON API endpoints for account, positions, trades, and performance data
- HTML dashboard with auto-refreshing charts and tables

Usage:
    uvicorn web.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state – populated during lifespan startup
# ---------------------------------------------------------------------------
_broker = None
_db = None
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
    global _broker, _db, _start_time, _paper_mode
    _start_time = time.time()

    # --- Database (always attempt) ---
    try:
        from utils.database import TradingDatabase

        _db = TradingDatabase("data/trading_bot.db")
        await _db.initialize()
        logger.info("Dashboard database connected")
    except Exception as exc:
        logger.warning(f"Database initialization failed (dashboard will show limited data): {exc}")
        _db = None

    # --- Broker (optional – dashboard still works without it) ---
    try:
        from config import ALPACA_CREDS
        from brokers.alpaca_broker import AlpacaBroker

        api_key = ALPACA_CREDS.get("API_KEY", "")
        if api_key:
            _paper_mode = ALPACA_CREDS.get("PAPER", True)
            _broker = AlpacaBroker(paper=_paper_mode)
            logger.info(f"Dashboard broker connected (paper={_paper_mode})")
        else:
            logger.warning("No Alpaca API key configured – broker endpoints will return defaults")
    except Exception as exc:
        logger.warning(f"Broker initialization failed (dashboard will use database only): {exc}")
        _broker = None

    yield  # ---- app is running ----

    # --- Shutdown ---
    if _db:
        try:
            await _db.close()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard HTML page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
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
        "database_connected": _db is not None,
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
        logger.error(f"Error fetching account: {exc}")
        return JSONResponse(
            content={"error": str(exc), "paper_mode": _paper_mode},
            status_code=500,
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
            result.append({
                "symbol": str(pos.symbol),
                "qty": _safe_float(pos.qty),
                "side": str(pos.side) if hasattr(pos, "side") else "long",
                "avg_entry_price": _safe_float(pos.avg_entry_price),
                "current_price": _safe_float(pos.current_price),
                "market_value": _safe_float(pos.market_value),
                "unrealized_pl": _safe_float(pos.unrealized_pl),
                "unrealized_plpc": _safe_float(pos.unrealized_plpc),
            })
        return {"positions": result, "count": len(result)}
    except Exception as exc:
        logger.error(f"Error fetching positions: {exc}")
        return JSONResponse(
            content={"positions": [], "count": 0, "error": str(exc)},
            status_code=500,
        )


@app.get("/api/trades")
async def get_trades(limit: int = Query(default=20, ge=1, le=500)):
    """Recent trades from database (paginated)."""
    if not _db:
        return {"trades": [], "count": 0, "error": "Database not connected"}

    try:
        trades = await _db.get_trades(limit=limit)
        result = [t.to_dict() for t in trades]
        return {"trades": result, "count": len(result)}
    except Exception as exc:
        logger.error(f"Error fetching trades: {exc}")
        return JSONResponse(
            content={"trades": [], "count": 0, "error": str(exc)},
            status_code=500,
        )


@app.get("/api/performance")
async def get_performance():
    """Performance metrics from database summary stats."""
    if not _db:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "error": "Database not connected",
        }

    try:
        summary = await _db.get_summary_stats()
        latest = await _db.get_latest_metrics()

        # Build performance response from available data
        response: Dict[str, Any] = {
            "total_trades": summary.get("total_trades", 0),
            "winning_trades": summary.get("winning_trades", 0) if "winning_trades" in summary else None,
            "win_rate": round(summary.get("win_rate", 0) * 100, 2),
            "total_pnl": round(summary.get("total_pnl", 0), 2),
            "open_positions": summary.get("open_positions", 0),
            "unique_symbols": summary.get("unique_symbols", 0),
            "unique_strategies": summary.get("unique_strategies", 0),
            "first_trade": summary.get("first_trade"),
            "last_trade": summary.get("last_trade"),
        }

        # Add daily metrics if available
        if latest:
            response["max_drawdown"] = round(latest.max_drawdown * 100, 2) if latest.max_drawdown else None
            response["latest_pnl"] = round(latest.pnl, 2) if latest.pnl else None
            response["latest_pnl_pct"] = round(latest.pnl_pct * 100, 2) if latest.pnl_pct else None

        # Try to compute extended metrics from daily data
        try:
            end = date.today()
            start = end - timedelta(days=90)
            daily = await _db.get_daily_metrics(start, end)
            if daily and len(daily) >= 2:
                returns = []
                for dm in daily:
                    if dm.starting_equity and dm.starting_equity > 0:
                        returns.append(dm.pnl / dm.starting_equity)

                if returns:
                    import statistics

                    mean_ret = statistics.mean(returns)
                    std_ret = statistics.stdev(returns) if len(returns) > 1 else 0

                    # Sharpe (annualized, assuming 252 trading days)
                    if std_ret > 0:
                        response["sharpe_ratio"] = round(
                            (mean_ret / std_ret) * (252 ** 0.5), 2
                        )

                    # Sortino (annualized, downside deviation)
                    downside = [r for r in returns if r < 0]
                    if downside:
                        downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
                        if downside_std > 0:
                            response["sortino_ratio"] = round(
                                (mean_ret / downside_std) * (252 ** 0.5), 2
                            )

                    # Profit factor from daily returns
                    gross_profit = sum(r for r in returns if r > 0)
                    gross_loss = abs(sum(r for r in returns if r < 0))
                    if gross_loss > 0:
                        response["profit_factor"] = round(gross_profit / gross_loss, 2)

                    # Max drawdown from daily equity
                    equities = [dm.ending_equity for dm in daily if dm.ending_equity]
                    if equities:
                        peak = equities[0]
                        max_dd = 0
                        for eq in equities:
                            if eq > peak:
                                peak = eq
                            dd = (peak - eq) / peak if peak > 0 else 0
                            max_dd = max(max_dd, dd)
                        response["max_drawdown"] = round(max_dd * 100, 2)
        except Exception as exc:
            logger.debug(f"Extended metrics calculation skipped: {exc}")

        return response
    except Exception as exc:
        logger.error(f"Error fetching performance: {exc}")
        return JSONResponse(
            content={"error": str(exc)},
            status_code=500,
        )


@app.get("/api/daily-metrics")
async def get_daily_metrics(days: int = Query(default=30, ge=1, le=365)):
    """Daily equity/P&L history for charting."""
    if not _db:
        return {"metrics": [], "count": 0, "error": "Database not connected"}

    try:
        end = date.today()
        start = end - timedelta(days=days)
        daily = await _db.get_daily_metrics(start, end)
        result = [dm.to_dict() for dm in daily]
        return {"metrics": result, "count": len(result)}
    except Exception as exc:
        logger.error(f"Error fetching daily metrics: {exc}")
        return JSONResponse(
            content={"metrics": [], "count": 0, "error": str(exc)},
            status_code=500,
        )


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
        logger.error(f"Error fetching market status: {exc}")
        return {"is_open": False, "error": str(exc)}


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
