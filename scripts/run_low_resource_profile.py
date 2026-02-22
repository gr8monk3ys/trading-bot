#!/usr/bin/env python3
"""Launch a conservative low-resource paper-trading profile."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a low-resource paper-trading profile suitable for Raspberry Pi-class hardware"
    )
    parser.add_argument(
        "--asset-class",
        default="crypto",
        choices=["crypto", "stock"],
        help="Preferred asset class. Crypto mode enables true 24/7 operation.",
    )
    parser.add_argument(
        "--strategy",
        default="momentum",
        choices=["momentum", "mean_reversion", "bracket_momentum"],
        help="Strategy name passed to live_trader.py",
    )
    parser.add_argument(
        "--risk-profile",
        default="conservative",
        choices=["custom", "conservative", "balanced", "aggressive"],
        help="Risk preset forwarded to live_trader.py",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Small symbol set to keep CPU/network usage low",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=None,
        help="Override position size fraction",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=None,
        help="Override max position size fraction",
    )
    parser.add_argument("--stop-loss", type=float, default=None, help="Override stop-loss fraction")
    parser.add_argument(
        "--take-profit", type=float, default=None, help="Override take-profit fraction"
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=None,
        help="Override circuit-breaker max daily loss fraction",
    )
    parser.add_argument(
        "--max-intraday-drawdown",
        type=float,
        default=None,
        help="Override hard intraday drawdown kill-switch fraction",
    )
    parser.add_argument(
        "--crypto-buy-score-threshold",
        type=float,
        default=None,
        help="Override momentum crypto long-only buy score threshold",
    )
    parser.add_argument(
        "--crypto-dip-rsi-max",
        type=float,
        default=None,
        help="Override momentum crypto dip-buy RSI ceiling",
    )
    parser.add_argument(
        "--crypto-dip-min-macd-hist-delta",
        type=float,
        default=None,
        help="Override momentum crypto dip-buy minimum MACD histogram delta",
    )
    parser.add_argument(
        "--crypto-dip-min-rebound-pct",
        type=float,
        default=None,
        help="Override momentum crypto dip-buy minimum rebound percentage (decimal)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print environment + command without starting live_trader.py",
    )
    return parser.parse_args(argv)


def _resolve_float_env_override(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return float(raw.strip())


def _build_command(args: argparse.Namespace) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    symbols = args.symbols
    if not symbols:
        if args.asset_class == "crypto":
            configured = os.getenv(
                "LOW_RESOURCE_CRYPTO_SYMBOLS",
                "BTC/USD,ETH/USD,SOL/USD,LTC/USD,DOGE/USD,AVAX/USD,LINK/USD,BCH/USD,DOT/USD,UNI/USD",
            )
            symbols = [token.strip() for token in configured.split(",") if token.strip()]
            if not symbols:
                symbols = [
                    "BTC/USD",
                    "ETH/USD",
                    "SOL/USD",
                    "LTC/USD",
                    "DOGE/USD",
                    "AVAX/USD",
                    "LINK/USD",
                    "BCH/USD",
                    "DOT/USD",
                    "UNI/USD",
                ]
        else:
            configured = os.getenv("LOW_RESOURCE_STOCK_SYMBOLS", "AAPL,MSFT")
            symbols = [token.strip() for token in configured.split(",") if token.strip()]
            if not symbols:
                symbols = ["AAPL", "MSFT"]
    command = [
        sys.executable,
        str(root / "live_trader.py"),
        "--strategy",
        args.strategy,
        "--risk-profile",
        args.risk_profile,
        "--symbols",
        *symbols,
    ]
    if args.position_size is not None:
        command.extend(["--position-size", str(args.position_size)])
    if args.max_position_size is not None:
        command.extend(["--max-position-size", str(args.max_position_size)])
    if args.stop_loss is not None:
        command.extend(["--stop-loss", str(args.stop_loss)])
    if args.take_profit is not None:
        command.extend(["--take-profit", str(args.take_profit)])
    if args.max_daily_loss is not None:
        command.extend(["--max-daily-loss", str(args.max_daily_loss)])
    if args.max_intraday_drawdown is not None:
        command.extend(["--max-intraday-drawdown", str(args.max_intraday_drawdown)])

    if args.asset_class == "crypto" and args.strategy == "momentum":
        buy_score = args.crypto_buy_score_threshold
        if buy_score is None and args.risk_profile == "aggressive":
            buy_score = _resolve_float_env_override("LOW_RESOURCE_CRYPTO_BUY_SCORE_THRESHOLD")
            if buy_score is None:
                buy_score = 0.75
        if buy_score is not None:
            command.extend(["--crypto-buy-score-threshold", str(buy_score)])

        dip_rsi_max = args.crypto_dip_rsi_max
        if dip_rsi_max is None and args.risk_profile == "aggressive":
            dip_rsi_max = _resolve_float_env_override("LOW_RESOURCE_CRYPTO_DIP_RSI_MAX")
            if dip_rsi_max is None:
                dip_rsi_max = 45.0
        if dip_rsi_max is not None:
            command.extend(["--crypto-dip-rsi-max", str(dip_rsi_max)])

        dip_macd_delta = args.crypto_dip_min_macd_hist_delta
        if dip_macd_delta is None and args.risk_profile == "aggressive":
            dip_macd_delta = _resolve_float_env_override(
                "LOW_RESOURCE_CRYPTO_DIP_MIN_MACD_HIST_DELTA"
            )
            if dip_macd_delta is None:
                dip_macd_delta = 0.005
        if dip_macd_delta is not None:
            command.extend(["--crypto-dip-min-macd-hist-delta", str(dip_macd_delta)])

        dip_rebound_pct = args.crypto_dip_min_rebound_pct
        if dip_rebound_pct is None and args.risk_profile == "aggressive":
            dip_rebound_pct = _resolve_float_env_override("LOW_RESOURCE_CRYPTO_DIP_MIN_REBOUND_PCT")
            if dip_rebound_pct is None:
                dip_rebound_pct = 0.0005
        if dip_rebound_pct is not None:
            command.extend(["--crypto-dip-min-rebound-pct", str(dip_rebound_pct)])

    return command


def _build_environment(args: argparse.Namespace | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PAPER", "true")
    env.setdefault("SLO_PAGING_ENABLED", "false")
    env.setdefault("INCIDENT_ACK_SLA_MINUTES", "15")
    env.setdefault("PAPER_LIVE_SHADOW_DRIFT_WARNING", "0.10")
    env.setdefault("PAPER_LIVE_SHADOW_DRIFT_MAX", "0.12")
    # Crypto on aggressive profile needs wider risk caps than equity defaults.
    if args and args.asset_class == "crypto" and args.risk_profile == "aggressive":
        env.setdefault("MAX_PORTFOLIO_RISK", "0.30")
        env.setdefault("MAX_POSITION_RISK", "0.15")
    return env


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    env = _build_environment(args)
    cmd = _build_command(args)

    print("Low-resource profile settings:")
    print(f"  PAPER={env['PAPER']}")
    print(f"  SLO_PAGING_ENABLED={env['SLO_PAGING_ENABLED']}")
    print(f"  INCIDENT_TICKETING_ENABLED={env.get('INCIDENT_TICKETING_ENABLED', 'false')}")
    print(f"  INCIDENT_ACK_SLA_MINUTES={env['INCIDENT_ACK_SLA_MINUTES']}")
    print(f"  MULTI_BROKER_ENABLED={env.get('MULTI_BROKER_ENABLED', 'false')}")
    if env.get("MULTI_BROKER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}:
        print(f"  IB_HOST={env.get('IB_HOST', '127.0.0.1')}")
        print(f"  IB_PAPER_PORT={env.get('IB_PAPER_PORT', '7497')}")
        print(f"  IB_CLIENT_ID={env.get('IB_CLIENT_ID', '1')}")
    print(f"  PAPER_LIVE_SHADOW_DRIFT_WARNING={env['PAPER_LIVE_SHADOW_DRIFT_WARNING']}")
    print(f"  PAPER_LIVE_SHADOW_DRIFT_MAX={env['PAPER_LIVE_SHADOW_DRIFT_MAX']}")
    if "MAX_PORTFOLIO_RISK" in env:
        print(f"  MAX_PORTFOLIO_RISK={env['MAX_PORTFOLIO_RISK']}")
    if "MAX_POSITION_RISK" in env:
        print(f"  MAX_POSITION_RISK={env['MAX_POSITION_RISK']}")
    print("Command:")
    print("  " + shlex.join(cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
