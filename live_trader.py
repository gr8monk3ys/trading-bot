#!/usr/bin/env python3
"""
Live Trading Launcher

Launches strategies in live paper trading with real-time monitoring.

Features:
- Real-time trade execution
- Live performance tracking
- Position monitoring
- P/L updates
- Risk monitoring (circuit breaker)
- Trade logging

Usage:
    python live_trader.py --strategy momentum --symbols AAPL MSFT GOOGL
    python live_trader.py --strategy mean_reversion --capital 50000
"""

import argparse
import asyncio
import logging
import signal
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import RISK_PARAMS, SYMBOLS
from strategies.bracket_momentum_strategy import BracketMomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.risk_manager import RiskManager
from utils.audit_log import AuditEventType, AuditLog
from utils.circuit_breaker import CircuitBreaker
from utils.data_quality import (
    should_halt_trading_for_data_quality,
    summarize_quality_reports,
    validate_ohlcv_frame,
)
from utils.incident_tracker import IncidentTracker
from utils.live_broker_factory import create_live_broker, shutdown_live_broker_failover
from utils.order_gateway import OrderGateway
from utils.order_reconciliation import OrderReconciler
from utils.position_manager import PositionManager
from utils.reconciliation import PositionReconciler
from utils.run_artifacts import JsonlWriter, ensure_run_directory, generate_run_id, write_json
from utils.runtime_state import RuntimeStateStore
from utils.slo_alerting import build_incident_ticket_notifier, build_slo_alert_notifier
from utils.slo_monitor import SLOMonitor
from utils.symbol_scope import build_symbol_scope

# Set up logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f'trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    ],
)

logger = logging.getLogger(__name__)

RISK_PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "custom": {},
    "conservative": {
        "position_size": 0.02,
        "max_position_size": 0.03,
        "stop_loss": 0.01,
        "take_profit": 0.02,
        "max_daily_loss": 0.025,
        "max_intraday_drawdown_pct": 0.03,
        "kill_switch_cooldown_minutes": 60,
        "drawdown_soft_limit_pct": 0.015,
        "drawdown_soft_scale": 0.8,
        "drawdown_medium_limit_pct": 0.0225,
        "drawdown_medium_scale": 0.6,
        "drawdown_hard_limit_pct": 0.03,
    },
    "balanced": {
        "position_size": 0.05,
        "max_position_size": 0.08,
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "max_daily_loss": 0.03,
        "max_intraday_drawdown_pct": 0.05,
        "kill_switch_cooldown_minutes": 90,
        "drawdown_soft_limit_pct": 0.03,
        "drawdown_soft_scale": 0.75,
        "drawdown_medium_limit_pct": 0.04,
        "drawdown_medium_scale": 0.55,
        "drawdown_hard_limit_pct": 0.05,
    },
    "aggressive": {
        "position_size": 0.1,
        "max_position_size": 0.15,
        "stop_loss": 0.03,
        "take_profit": 0.08,
        "max_daily_loss": 0.04,
        "max_intraday_drawdown_pct": 0.07,
        "kill_switch_cooldown_minutes": 120,
        "drawdown_soft_limit_pct": 0.04,
        "drawdown_soft_scale": 0.7,
        "drawdown_medium_limit_pct": 0.055,
        "drawdown_medium_scale": 0.45,
        "drawdown_hard_limit_pct": 0.07,
    },
}


def _resolve_runtime_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """Build runtime parameters from CLI arguments and selected risk profile."""
    profile_name = str(getattr(args, "risk_profile", "custom") or "custom").strip().lower()
    profile = dict(RISK_PROFILE_DEFAULTS.get(profile_name, RISK_PROFILE_DEFAULTS["custom"]))

    position_size = (
        float(args.position_size)
        if args.position_size is not None
        else float(profile.get("position_size", 0.10))
    )
    stop_loss = (
        float(args.stop_loss)
        if args.stop_loss is not None
        else float(profile.get("stop_loss", 0.02))
    )
    take_profit = (
        float(args.take_profit)
        if args.take_profit is not None
        else float(profile.get("take_profit", 0.05))
    )
    max_position_size = (
        float(args.max_position_size)
        if args.max_position_size is not None
        else float(profile.get("max_position_size", max(position_size, 0.05)))
    )
    max_daily_loss = (
        float(args.max_daily_loss)
        if args.max_daily_loss is not None
        else float(profile.get("max_daily_loss", 0.03))
    )
    hard_drawdown = (
        float(args.max_intraday_drawdown)
        if args.max_intraday_drawdown is not None
        else float(profile.get("drawdown_hard_limit_pct", profile.get("max_intraday_drawdown_pct", 0.07)))
    )
    soft_drawdown = (
        float(args.drawdown_soft_limit)
        if args.drawdown_soft_limit is not None
        else float(profile.get("drawdown_soft_limit_pct", 0.03))
    )
    medium_drawdown = (
        float(args.drawdown_medium_limit)
        if args.drawdown_medium_limit is not None
        else float(profile.get("drawdown_medium_limit_pct", 0.05))
    )
    soft_scale = (
        float(args.drawdown_soft_scale)
        if args.drawdown_soft_scale is not None
        else float(profile.get("drawdown_soft_scale", 0.75))
    )
    medium_scale = (
        float(args.drawdown_medium_scale)
        if args.drawdown_medium_scale is not None
        else float(profile.get("drawdown_medium_scale", 0.50))
    )
    cooldown_minutes = (
        int(args.kill_switch_cooldown_minutes)
        if args.kill_switch_cooldown_minutes is not None
        else int(profile.get("kill_switch_cooldown_minutes", 60))
    )

    parameters: dict[str, Any] = {
        "position_size": position_size,
        "max_position_size": max_position_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_daily_loss": max_daily_loss,
        "max_intraday_drawdown_pct": hard_drawdown,
        "kill_switch_cooldown_minutes": cooldown_minutes,
        "drawdown_soft_limit_pct": soft_drawdown,
        "drawdown_soft_scale": soft_scale,
        "drawdown_medium_limit_pct": medium_drawdown,
        "drawdown_medium_scale": medium_scale,
        "drawdown_hard_limit_pct": hard_drawdown,
    }
    if args.crypto_buy_score_threshold is not None:
        parameters["crypto_long_only_buy_score_threshold"] = float(args.crypto_buy_score_threshold)
    if args.crypto_dip_rsi_max is not None:
        parameters["crypto_long_only_dip_rsi_max"] = float(args.crypto_dip_rsi_max)
    if args.crypto_dip_min_macd_hist_delta is not None:
        parameters["crypto_long_only_dip_min_macd_hist_delta"] = float(
            args.crypto_dip_min_macd_hist_delta
        )
    if args.crypto_dip_min_rebound_pct is not None:
        parameters["crypto_long_only_dip_min_rebound_pct"] = float(args.crypto_dip_min_rebound_pct)
    _validate_runtime_risk_parameters(parameters)
    return parameters


def _validate_runtime_risk_parameters(parameters: dict[str, Any]) -> None:
    """Validate runtime risk controls for sane bounds and ordering."""
    position_size = float(parameters.get("position_size", 0.10))
    max_position_size = float(parameters.get("max_position_size", 0.10))
    stop_loss = float(parameters.get("stop_loss", 0.02))
    take_profit = float(parameters.get("take_profit", 0.05))
    max_daily_loss = float(parameters.get("max_daily_loss", 0.03))
    soft = float(parameters.get("drawdown_soft_limit_pct", 0.03))
    medium = float(parameters.get("drawdown_medium_limit_pct", 0.05))
    hard = float(parameters.get("drawdown_hard_limit_pct", 0.07))
    soft_scale = float(parameters.get("drawdown_soft_scale", 0.75))
    medium_scale = float(parameters.get("drawdown_medium_scale", 0.5))
    cooldown = int(parameters.get("kill_switch_cooldown_minutes", 60))
    crypto_buy_score_threshold = float(parameters.get("crypto_long_only_buy_score_threshold", 1.0))
    crypto_dip_rsi_max = float(parameters.get("crypto_long_only_dip_rsi_max", 35.0))
    crypto_dip_min_macd_hist_delta = float(
        parameters.get("crypto_long_only_dip_min_macd_hist_delta", 0.02)
    )
    crypto_dip_min_rebound_pct = float(parameters.get("crypto_long_only_dip_min_rebound_pct", 0.001))

    if position_size <= 0 or position_size > 1:
        raise ValueError(f"position_size must be between 0 and 1, got {position_size}")
    if max_position_size <= 0 or max_position_size > 1:
        raise ValueError(f"max_position_size must be between 0 and 1, got {max_position_size}")
    if max_position_size < position_size:
        raise ValueError(
            f"max_position_size ({max_position_size}) must be >= position_size ({position_size})"
        )
    if stop_loss <= 0 or stop_loss > 0.5:
        raise ValueError(f"stop_loss must be between 0 and 0.5, got {stop_loss}")
    if take_profit <= 0 or take_profit > 1:
        raise ValueError(f"take_profit must be between 0 and 1, got {take_profit}")
    if max_daily_loss <= 0 or max_daily_loss > 0.2:
        raise ValueError(f"max_daily_loss must be between 0 and 0.2, got {max_daily_loss}")
    if not (0 < soft < medium < hard < 1):
        raise ValueError(
            "drawdown ladder must satisfy 0 < soft < medium < hard < 1 "
            f"(got soft={soft}, medium={medium}, hard={hard})"
        )
    if not (0 < medium_scale <= soft_scale <= 1):
        raise ValueError(
            "drawdown scales must satisfy 0 < medium_scale <= soft_scale <= 1 "
            f"(got soft_scale={soft_scale}, medium_scale={medium_scale})"
        )
    if cooldown < 1 or cooldown > 1440:
        raise ValueError(
            f"kill_switch_cooldown_minutes must be between 1 and 1440, got {cooldown}"
        )
    if crypto_buy_score_threshold < 0 or crypto_buy_score_threshold > 5:
        raise ValueError(
            "crypto_long_only_buy_score_threshold must be between 0 and 5, "
            f"got {crypto_buy_score_threshold}"
        )
    if crypto_dip_rsi_max <= 0 or crypto_dip_rsi_max > 100:
        raise ValueError(
            f"crypto_long_only_dip_rsi_max must be between 0 and 100, got {crypto_dip_rsi_max}"
        )
    if crypto_dip_min_macd_hist_delta < 0 or crypto_dip_min_macd_hist_delta > 5:
        raise ValueError(
            "crypto_long_only_dip_min_macd_hist_delta must be between 0 and 5, "
            f"got {crypto_dip_min_macd_hist_delta}"
        )
    if crypto_dip_min_rebound_pct < 0 or crypto_dip_min_rebound_pct > 0.5:
        raise ValueError(
            "crypto_long_only_dip_min_rebound_pct must be between 0 and 0.5, "
            f"got {crypto_dip_min_rebound_pct}"
        )


def _determine_position_scale(drawdown_pct: float, parameters: dict[str, Any]) -> float:
    """Return target position scale (0-1) based on drawdown ladder."""
    hard = float(parameters.get("drawdown_hard_limit_pct", 0.07))
    medium = float(parameters.get("drawdown_medium_limit_pct", 0.05))
    soft = float(parameters.get("drawdown_soft_limit_pct", 0.03))
    medium_scale = float(parameters.get("drawdown_medium_scale", 0.5))
    soft_scale = float(parameters.get("drawdown_soft_scale", 0.75))

    if drawdown_pct >= hard:
        return 0.0
    if drawdown_pct >= medium:
        return medium_scale
    if drawdown_pct >= soft:
        return soft_scale
    return 1.0


class LiveTrader:
    """
    Live trading manager with real-time monitoring.

    Handles:
    - Strategy initialization
    - WebSocket connection
    - Real-time monitoring
    - Graceful shutdown
    - Performance tracking
    """

    def __init__(self, strategy_name: str, symbols: list, parameters: dict):
        """
        Initialize live trader.

        Args:
            strategy_name: Name of strategy to run
            symbols: List of symbols to trade
            parameters: Strategy parameters
        """
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.parameters = parameters

        self.broker = None
        self.failover_manager = None
        self.strategy = None
        self.circuit_breaker = None
        self.order_gateway = None
        self.audit_log = None
        self.position_manager = None
        self.reconciler = None
        self.order_reconciler = None
        self.slo_monitor = None
        self._data_quality_writer = None
        self.state_store = RuntimeStateStore("data/live_trader_state.json")
        self._pending_strategy_state = {}
        self._crypto_only = False
        self._symbol_scope = build_symbol_scope(self.symbols)
        self.running = False
        self.session_run_id = generate_run_id("live")
        self.session_run_dir = ensure_run_directory("results/runs", self.session_run_id)

        _validate_runtime_risk_parameters(self.parameters)
        self._position_scale = 1.0
        self._peak_equity = None
        self._base_position_size = None
        self._base_short_position_size = None
        self._base_max_position_size = None

        # Performance tracking
        self.start_time = None
        self.start_equity = None
        self.trades_executed = 0

        # Shutdown handling
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize broker and strategy."""
        try:
            logger.info("=" * 80)
            logger.info("🚀 LIVE TRADING INITIALIZATION")
            logger.info("=" * 80)
            logger.info(f"Session Run ID: {self.session_run_id}")
            logger.info(f"Session Artifacts: {self.session_run_dir}")

            # Initialize broker
            logger.info("1. Connecting to Alpaca (Paper Trading)...")
            self.broker, self.failover_manager = await create_live_broker(
                paper=True,
                source="live_trader.initialize",
            )
            if self.failover_manager:
                logger.info("✅ Multi-broker failover manager enabled (Alpaca + IB backup)")

            # Validate symbol universe for a single websocket session and normalize crypto pairs.
            if hasattr(self.broker, "is_crypto"):
                crypto_flags = [bool(self.broker.is_crypto(symbol)) for symbol in self.symbols]
                if any(crypto_flags) and not all(crypto_flags):
                    raise ValueError(
                        "Mixed stock and crypto symbols are not supported in one live session. "
                        "Run separate live processes per asset class."
                    )
                self._crypto_only = bool(crypto_flags) and all(crypto_flags)
                if self._crypto_only and hasattr(self.broker, "normalize_crypto_symbol"):
                    self.symbols = [self.broker.normalize_crypto_symbol(symbol) for symbol in self.symbols]
                    logger.info("✅ Crypto-only session detected (24/7): %s", ", ".join(self.symbols))
                    if self.strategy_name == "momentum" and "enable_short_selling" not in self.parameters:
                        # Spot crypto shorting is generally unavailable; keep momentum strategy long-only.
                        self.parameters["enable_short_selling"] = False
                        logger.info("Configured momentum strategy as long-only for crypto session")
            self._symbol_scope = build_symbol_scope(self.symbols)

            # Get account info
            account = await self.broker.get_account()
            self.start_equity = float(account.equity)

            logger.info(f"✅ Connected to account: {account.id}")
            logger.info(f"   Starting Capital: ${self.start_equity:,.2f}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}\n")

            # Initialize circuit breaker (CRITICAL SAFETY FEATURE)
            logger.info("1.5. Initializing circuit breaker...")
            configured_daily_loss = float(self.parameters.get("max_daily_loss", 0.03))
            self.circuit_breaker = CircuitBreaker(
                max_daily_loss=configured_daily_loss,
                auto_close_positions=True,  # Automatically close positions on trigger
            )
            await self.circuit_breaker.initialize(self.broker)
            logger.info("✅ Circuit breaker armed")
            logger.info("   Max Daily Loss: %.2f%%", configured_daily_loss * 100)
            logger.info("   Auto-close enabled: YES\n")

            # Initialize audit log
            self.audit_log = AuditLog(log_dir="./audit_logs", auto_verify=True)
            self.audit_log.log(
                AuditEventType.SYSTEM_START,
                {"component": "LiveTrader", "strategy": self.strategy_name},
            )
            if hasattr(self.broker, "set_audit_log"):
                self.broker.set_audit_log(self.audit_log)
            self._data_quality_writer = JsonlWriter(
                self.session_run_dir / "data_quality_events.jsonl"
            )

            # Initialize order gateway (CRITICAL SAFETY FEATURE)
            logger.info("1.6. Initializing OrderGateway...")
            position_manager = PositionManager()
            risk_manager = RiskManager(
                max_portfolio_risk=RISK_PARAMS.get("MAX_PORTFOLIO_RISK", 0.02),
                max_position_risk=RISK_PARAMS.get("MAX_POSITION_RISK", 0.01),
            )
            self.position_manager = position_manager
            self.order_gateway = OrderGateway(
                broker=self.broker,
                circuit_breaker=self.circuit_breaker,
                position_manager=position_manager,
                risk_manager=risk_manager,
                audit_log=self.audit_log,
                enforce_gateway=True,
                max_intraday_drawdown_pct=float(
                    self.parameters.get("max_intraday_drawdown_pct", 0.07)
                ),
                kill_switch_cooldown_minutes=int(
                    self.parameters.get("kill_switch_cooldown_minutes", 60)
                ),
            )
            if hasattr(self.circuit_breaker, "set_order_gateway"):
                self.circuit_breaker.set_order_gateway(self.order_gateway)
            logger.info("✅ OrderGateway initialized (mandatory routing enabled)\n")
            if hasattr(self.broker, "set_position_manager"):
                self.broker.set_position_manager(self.position_manager)

            # Load persisted runtime state (if any) and sync with broker
            state = await self.state_store.load()
            if state:
                await self.position_manager.import_state(state.position_manager)
                if state.lifecycle and self.order_gateway and self.order_gateway.lifecycle_tracker:
                    self.order_gateway.lifecycle_tracker.import_state(state.lifecycle)
                if state.gateway_state and self.order_gateway:
                    self.order_gateway.import_runtime_state(state.gateway_state)
                self._pending_strategy_state = state.strategy_states or {}
                logger.info("Runtime state restored into PositionManager")
            await self.position_manager.sync_with_broker(
                self.broker,
                default_strategy="recovered",
                symbol_scope=self._symbol_scope,
            )

            # Initialize reconciler
            self.reconciler = PositionReconciler(
                broker=self.broker,
                internal_tracker=self.position_manager,
                halt_on_mismatch=False,
                sync_to_broker=True,
                symbol_scope=self.symbols,
                audit_log=self.audit_log,
                events_path=self.session_run_dir / "position_reconciliation_events.jsonl",
                run_id=self.session_run_id,
            )
            self.order_reconciler = OrderReconciler(
                broker=self.broker,
                lifecycle_tracker=self.order_gateway.lifecycle_tracker,
                audit_log=self.audit_log,
                mismatch_halt_threshold=RISK_PARAMS.get("ORDER_RECON_MISMATCH_HALT_RUNS", 3),
                events_path=self.session_run_dir / "order_reconciliation_events.jsonl",
                run_id=self.session_run_id,
            )
            incident_tracker = IncidentTracker(
                events_path=self.session_run_dir / "incident_events.jsonl",
                run_id=self.session_run_id,
                ack_sla_minutes=RISK_PARAMS.get("INCIDENT_ACK_SLA_MINUTES", 15),
            )
            slo_alert_notifier = build_slo_alert_notifier(RISK_PARAMS, source="live_trader")
            incident_ticket_notifier = build_incident_ticket_notifier(
                RISK_PARAMS,
                source="live_trader",
            )
            if slo_alert_notifier:
                logger.info(
                    "SLO paging alerts enabled (severity>=%s)",
                    RISK_PARAMS.get("SLO_PAGING_MIN_SEVERITY", "critical"),
                )
            if incident_ticket_notifier:
                logger.info("Incident ticketing enabled for ack-SLA breaches")
            self.slo_monitor = SLOMonitor(
                audit_log=self.audit_log,
                events_path=self.session_run_dir / "ops_slo_events.jsonl",
                notification_dead_letter_path=(
                    self.session_run_dir / "notification_dead_letters.jsonl"
                ),
                alert_notifier=slo_alert_notifier,
                incident_ticket_notifier=incident_ticket_notifier,
                incident_tracker=incident_tracker,
                recon_mismatch_halt_runs=RISK_PARAMS.get("ORDER_RECON_MISMATCH_HALT_RUNS", 3),
                max_data_quality_errors=RISK_PARAMS.get("DATA_QUALITY_MAX_ERRORS", 0),
                max_stale_data_warnings=RISK_PARAMS.get("DATA_QUALITY_MAX_STALE_WARNINGS", 0),
                shadow_drift_warning_threshold=RISK_PARAMS.get(
                    "PAPER_LIVE_SHADOW_DRIFT_WARNING", 0.12
                ),
                shadow_drift_critical_threshold=RISK_PARAMS.get(
                    "PAPER_LIVE_SHADOW_DRIFT_MAX", 0.15
                ),
                notification_dead_letter_warning_threshold=RISK_PARAMS.get(
                    "NOTIFICATION_DEAD_LETTER_WARNING_THRESHOLD",
                    10,
                ),
                notification_dead_letter_critical_threshold=RISK_PARAMS.get(
                    "NOTIFICATION_DEAD_LETTER_CRITICAL_THRESHOLD",
                    25,
                ),
                notification_dead_letter_persist_minutes=RISK_PARAMS.get(
                    "NOTIFICATION_DEAD_LETTER_PERSIST_MINUTES",
                    5,
                ),
            )

            # Initialize strategy
            logger.info(f"2. Initializing {self.strategy_name} strategy...")

            strategy_class = self._get_strategy_class()
            self.strategy = strategy_class(
                broker=self.broker,
                parameters={"symbols": self.symbols, **self.parameters},
                order_gateway=self.order_gateway,
            )

            # Initialize strategy
            success = await self.strategy.initialize()
            if not success:
                raise RuntimeError("Strategy initialization failed")

            # Restore strategy state if available
            if hasattr(self.strategy, "import_state"):
                saved = self._pending_strategy_state.get(self.strategy_name)
                if saved:
                    await self._restore_strategy_state(self.strategy, saved)

            self._capture_strategy_risk_baselines()

            logger.info("✅ Strategy initialized")
            logger.info(f"   Trading: {', '.join(self.symbols)}")
            logger.info(f"   Parameters: {self.parameters}\n")
            logger.info(
                "   Dynamic De-risk Ladder: soft>=%.2f%% (x%.2f), medium>=%.2f%% (x%.2f), hard>=%.2f%% (halt)",
                float(self.parameters.get("drawdown_soft_limit_pct", 0.03)) * 100,
                float(self.parameters.get("drawdown_soft_scale", 0.75)),
                float(self.parameters.get("drawdown_medium_limit_pct", 0.05)) * 100,
                float(self.parameters.get("drawdown_medium_scale", 0.5)),
                float(self.parameters.get("drawdown_hard_limit_pct", 0.07)) * 100,
            )

            logger.info("3. Market Status:")
            if self._crypto_only:
                logger.info("   Session Type: Crypto 24/7 ✅")
                logger.info("   Market-Open Gate: DISABLED (crypto trades around the clock)\n")
            else:
                clock = await self.broker.get_clock()
                logger.info(f"   Market Open: {'YES ✅' if clock.is_open else 'NO ❌'}")
                logger.info(f"   Next Open: {clock.next_open}")
                logger.info(f"   Next Close: {clock.next_close}\n")

            logger.info("=" * 80)
            logger.info("✅ INITIALIZATION COMPLETE - READY TO TRADE")
            logger.info("=" * 80 + "\n")

            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False

    def _get_strategy_class(self):
        """Get strategy class by name."""
        strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "bracket_momentum": BracketMomentumStrategy,
        }

        if self.strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. Available: {list(strategies.keys())}"
            )

        return strategies[self.strategy_name]

    def _capture_strategy_risk_baselines(self) -> None:
        """Capture baseline strategy sizing used for dynamic de-risk scaling."""
        if self.strategy is None:
            return

        if hasattr(self.strategy, "position_size"):
            try:
                self._base_position_size = float(self.strategy.position_size)
            except (TypeError, ValueError):
                self._base_position_size = None
        if hasattr(self.strategy, "short_position_size"):
            try:
                self._base_short_position_size = float(self.strategy.short_position_size)
            except (TypeError, ValueError):
                self._base_short_position_size = None
        if hasattr(self.strategy, "max_position_size"):
            try:
                self._base_max_position_size = float(self.strategy.max_position_size)
            except (TypeError, ValueError):
                self._base_max_position_size = None

        self._apply_strategy_position_scale(1.0, reason="baseline")

    def _apply_strategy_position_scale(self, target_scale: float, *, reason: str) -> None:
        """Apply position-size multiplier to strategy sizing controls."""
        if self.strategy is None:
            return
        if abs(target_scale - self._position_scale) < 1e-9:
            return

        self._position_scale = target_scale
        params = getattr(self.strategy, "parameters", None)
        if not isinstance(params, dict):
            params = None

        if self._base_position_size is not None and hasattr(self.strategy, "position_size"):
            scaled_position = self._base_position_size * target_scale
            self.strategy.position_size = scaled_position
            if params is not None:
                params["position_size"] = scaled_position

        if self._base_short_position_size is not None and hasattr(self.strategy, "short_position_size"):
            scaled_short = self._base_short_position_size * target_scale
            self.strategy.short_position_size = scaled_short
            if params is not None:
                params["short_position_size"] = scaled_short

        if self._base_max_position_size is not None and hasattr(self.strategy, "max_position_size"):
            scaled_max = self._base_max_position_size * target_scale
            if self._base_position_size is not None:
                scaled_max = max(scaled_max, self._base_position_size * target_scale)
            self.strategy.max_position_size = scaled_max
            if params is not None:
                params["max_position_size"] = scaled_max

        logger.warning(
            "Adjusted strategy risk scale to x%.2f (%s)",
            target_scale,
            reason,
        )

    def _apply_drawdown_controls(self, current_equity: float) -> float:
        """Scale or halt based on peak-to-trough drawdown. Returns current drawdown."""
        if current_equity <= 0:
            return 0.0

        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity
        if not self._peak_equity:
            return 0.0

        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        target_scale = _determine_position_scale(drawdown, self.parameters)

        if target_scale <= 0:
            reason = (
                f"Dynamic drawdown hard stop breached: {drawdown:.2%} >= "
                f"{float(self.parameters.get('drawdown_hard_limit_pct', 0.07)):.2%}"
            )
            if self.order_gateway:
                self.order_gateway.activate_kill_switch(
                    reason=reason,
                    cooldown_minutes=int(self.parameters.get("kill_switch_cooldown_minutes", 60)),
                    source="live_monitor",
                )
            logger.critical(reason)
            self.running = False
            self.shutdown_event.set()
            return drawdown

        if abs(target_scale - self._position_scale) > 1e-9:
            self._apply_strategy_position_scale(
                target_scale,
                reason=f"drawdown={drawdown:.2%}",
            )

        return drawdown

    async def start_trading(self):
        """Start live trading."""
        try:
            self.running = True
            self.start_time = datetime.now()

            logger.info("\n" + "=" * 80)
            logger.info("📈 STARTING LIVE TRADING")
            logger.info("=" * 80)
            logger.info(f"Strategy: {self.strategy_name}")
            logger.info(f"Symbols: {', '.join(self.symbols)}")
            logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("\nPress Ctrl+C to stop trading gracefully")
            logger.info("=" * 80 + "\n")

            # Start WebSocket for real-time data
            await self.broker.start_websocket(self.symbols)

            # Start monitoring loop
            monitor_task = asyncio.create_task(self.monitor_performance())
            housekeeping_task = asyncio.create_task(self._housekeeping_loop())

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            # Cancel monitoring
            monitor_task.cancel()
            housekeeping_task.cancel()

        except Exception as e:
            logger.error(f"Error during trading: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def monitor_performance(self):
        """Monitor and log performance periodically."""
        try:
            while self.running:
                await asyncio.sleep(60)  # Check every minute

                # CRITICAL: Check circuit breaker FIRST
                if await self.circuit_breaker.check_and_halt():
                    logger.critical("=" * 80)
                    logger.critical("🚨 CIRCUIT BREAKER TRIGGERED 🚨")
                    logger.critical("Daily loss limit exceeded!")
                    logger.critical("All positions will be closed automatically.")
                    logger.critical("Trading will HALT for the rest of the day.")
                    logger.critical("=" * 80)

                    # Trigger shutdown
                    self.running = False
                    self.shutdown_event.set()
                    break

                # Get current account state
                account = await self.broker.get_account()
                current_equity = float(account.equity)
                drawdown_pct = self._apply_drawdown_controls(current_equity)
                if not self.running:
                    break

                # Calculate P/L
                pnl = current_equity - self.start_equity
                pnl_pct = (pnl / self.start_equity) * 100

                # Get positions
                positions = await self.broker.get_positions()

                # Log status
                runtime = datetime.now() - self.start_time

                logger.info("\n" + "-" * 80)
                logger.info(f"📊 PERFORMANCE UPDATE - Runtime: {str(runtime).split('.')[0]}")
                logger.info("-" * 80)
                logger.info(f"Equity: ${current_equity:,.2f} (P/L: ${pnl:+,.2f} / {pnl_pct:+.2f}%)")
                logger.info(
                    "Peak Drawdown: %.2f%% | Risk Scale: x%.2f",
                    drawdown_pct * 100,
                    self._position_scale,
                )
                logger.info(f"Positions: {len(positions)}")

                if positions:
                    logger.info("\nOpen Positions:")
                    for pos in positions:
                        logger.info(
                            f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} "
                            f"(Current: ${float(pos.current_price):.2f}, "
                            f"P/L: ${float(pos.unrealized_pl):+,.2f} / {float(pos.unrealized_plpc)*100:+.2f}%)"
                        )

                logger.info("-" * 80 + "\n")

        except asyncio.CancelledError:
            logger.info("Performance monitoring stopped")
        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}", exc_info=True)

    async def shutdown(self):
        """Graceful shutdown."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("🛑 SHUTTING DOWN LIVE TRADING")
            logger.info("=" * 80)

            self.running = False

            # Get final performance
            account = await self.broker.get_account()
            final_equity = float(account.equity)
            total_pnl = final_equity - self.start_equity
            total_pnl_pct = (total_pnl / self.start_equity) * 100

            runtime = datetime.now() - self.start_time

            logger.info("\n📊 FINAL PERFORMANCE SUMMARY:")
            logger.info(f"   Runtime: {str(runtime).split('.')[0]}")
            logger.info(f"   Starting Equity: ${self.start_equity:,.2f}")
            logger.info(f"   Final Equity: ${final_equity:,.2f}")
            logger.info(f"   Total P/L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")

            # Get final positions
            positions = await self.broker.get_positions()
            if positions:
                logger.info(f"\n   Open Positions: {len(positions)}")
                for pos in positions:
                    logger.info(
                        f"     {pos.symbol}: {pos.qty} shares, "
                        f"P/L: ${float(pos.unrealized_pl):+,.2f}"
                    )

            logger.info("\n✅ Shutdown complete")
            logger.info("=" * 80 + "\n")

            if self.position_manager:
                strategy_states = {}
                if self.strategy is not None:
                    strategy_states[self.strategy_name] = await self._build_strategy_state_snapshot(
                        self.strategy
                    )
                await self.state_store.save(
                    self.position_manager,
                    lifecycle=(
                        self.order_gateway.lifecycle_tracker.export_state()
                        if self.order_gateway and self.order_gateway.lifecycle_tracker
                        else {}
                    ),
                    gateway_state=(
                        self.order_gateway.export_runtime_state()
                        if self.order_gateway
                        and hasattr(self.order_gateway, "export_runtime_state")
                        else {}
                    ),
                    strategy_states=strategy_states,
                )

            if self.audit_log:
                self.audit_log.log(
                    AuditEventType.SYSTEM_STOP,
                    {"component": "LiveTrader", "strategy": self.strategy_name},
                )
                self.audit_log.close()
            if self.reconciler:
                self.reconciler.close()
            if self.order_reconciler:
                self.order_reconciler.close()
            if self.slo_monitor:
                self.slo_monitor.close()
            if self._data_quality_writer:
                self._data_quality_writer.close()
            await shutdown_live_broker_failover(self.failover_manager)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def _housekeeping_loop(self):
        """Persist runtime state and run reconciliation periodically."""
        try:
            state_interval = 60
            reconciliation_interval = 300
            counter = 0
            while self.running:
                await asyncio.sleep(1)
                counter += 1
                if counter % state_interval == 0 and self.position_manager:
                    strategy_states = {}
                    if self.strategy is not None:
                        strategy_states[self.strategy_name] = (
                            await self._build_strategy_state_snapshot(self.strategy)
                        )
                    await self.state_store.save(
                        self.position_manager,
                        lifecycle=(
                            self.order_gateway.lifecycle_tracker.export_state()
                            if self.order_gateway and self.order_gateway.lifecycle_tracker
                            else {}
                        ),
                        gateway_state=(
                            self.order_gateway.export_runtime_state()
                            if self.order_gateway
                            and hasattr(self.order_gateway, "export_runtime_state")
                            else {}
                        ),
                        strategy_states=strategy_states,
                    )
                    if self.slo_monitor:
                        self.slo_monitor.check_incident_ack_sla()
                        backlog = (
                            self.slo_monitor.get_status_snapshot()
                            .get("dead_letters", {})
                            .get("queued", 0)
                        )
                        self.slo_monitor.record_notification_dead_letter_backlog(backlog)
                if counter % reconciliation_interval == 0 and self.reconciler is not None:
                    try:
                        await self.reconciler.reconcile()
                    except Exception as e:
                        logger.error(f"Reconciliation error: {e}")
                    try:
                        await self._run_data_quality_gate()
                    except Exception as e:
                        logger.error(f"Data quality gate error: {e}")
                if counter % 120 == 0 and self.order_reconciler is not None:
                    try:
                        await self.order_reconciler.reconcile()
                        if self.slo_monitor:
                            breaches = self.slo_monitor.record_order_reconciliation_health(
                                self.order_reconciler.get_health_snapshot()
                            )
                            if self.order_gateway and SLOMonitor.has_critical_breach(breaches):
                                self.order_gateway.activate_kill_switch(
                                    reason="Critical order reconciliation SLO breach",
                                    source="slo_monitor",
                                )
                        if self.order_reconciler.should_halt_trading() and self.order_gateway:
                            health = self.order_reconciler.get_health_snapshot()
                            reason = (
                                health.get("halt_reason")
                                or "Order reconciliation drift threshold breached"
                            )
                            self.order_gateway.activate_kill_switch(
                                reason=reason,
                                source="order_reconciliation",
                            )
                    except Exception as e:
                        logger.error(f"Order reconciliation error: {e}")
        except asyncio.CancelledError:
            return

    async def _run_data_quality_gate(self) -> None:
        """Run live data quality checks and halt entries on severe issues."""
        if not self.broker or not self.order_gateway:
            return

        reference_time = datetime.now(timezone.utc)
        stock_start = (reference_time - timedelta(days=30)).strftime("%Y-%m-%d")
        stock_end = reference_time.strftime("%Y-%m-%d")
        crypto_start = reference_time - timedelta(days=30)
        crypto_end = reference_time
        stale_after_days = RISK_PARAMS.get("DATA_QUALITY_STALE_AFTER_DAYS", 3)
        reports = []

        for symbol in self.symbols:
            try:
                is_crypto = bool(
                    hasattr(self.broker, "is_crypto") and self.broker.is_crypto(symbol)
                )
                if is_crypto and hasattr(self.broker, "get_crypto_bars"):
                    bars = await self.broker.get_crypto_bars(
                        symbol,
                        start=crypto_start,
                        end=crypto_end,
                        timeframe="1Day",
                    )
                else:
                    bars = await self.broker.get_bars(
                        symbol,
                        start=stock_start,
                        end=stock_end,
                        timeframe="1Day",
                    )
                if not bars:
                    reports.append(
                        {
                            "symbol": symbol,
                            "rows": 0,
                            "error_count": 1,
                            "warning_count": 0,
                            "issues": [
                                {
                                    "severity": "error",
                                    "code": "no_data",
                                    "message": "No bars returned in data quality gate",
                                }
                            ],
                        }
                    )
                    continue

                if is_crypto:
                    frame = pd.DataFrame(
                        {
                            "open": [float(b["open"]) for b in bars],
                            "high": [float(b["high"]) for b in bars],
                            "low": [float(b["low"]) for b in bars],
                            "close": [float(b["close"]) for b in bars],
                            "volume": [float(b["volume"]) for b in bars],
                        },
                        index=pd.DatetimeIndex([b["timestamp"] for b in bars]),
                    )
                else:
                    frame = pd.DataFrame(
                        {
                            "open": [float(b.open) for b in bars],
                            "high": [float(b.high) for b in bars],
                            "low": [float(b.low) for b in bars],
                            "close": [float(b.close) for b in bars],
                            "volume": [float(b.volume) for b in bars],
                        },
                        index=pd.DatetimeIndex([b.timestamp for b in bars]),
                    )
                reports.append(
                    validate_ohlcv_frame(
                        frame,
                        symbol=symbol,
                        stale_after_days=stale_after_days,
                        reference_time=reference_time,
                    )
                )
            except Exception as e:
                reports.append(
                    {
                        "symbol": symbol,
                        "rows": 0,
                        "error_count": 1,
                        "warning_count": 0,
                        "issues": [
                            {
                                "severity": "error",
                                "code": "data_quality_exception",
                                "message": str(e),
                            }
                        ],
                    }
                )

        summary = summarize_quality_reports(reports)
        payload = {
            "event_type": "data_quality_snapshot",
            "run_id": self.session_run_id,
            "timestamp": reference_time.isoformat(),
            **summary,
        }
        write_json(self.session_run_dir / "data_quality_latest.json", payload)
        if self._data_quality_writer:
            self._data_quality_writer.write(payload)

        if self.slo_monitor:
            breaches = self.slo_monitor.record_data_quality_summary(summary)
            if SLOMonitor.has_critical_breach(breaches):
                self.order_gateway.activate_kill_switch(
                    reason="Critical data quality SLO breach",
                    source="slo_monitor",
                )
                return

        should_halt, reason = should_halt_trading_for_data_quality(
            summary,
            max_errors=RISK_PARAMS.get("DATA_QUALITY_MAX_ERRORS", 0),
            max_stale_warnings=RISK_PARAMS.get("DATA_QUALITY_MAX_STALE_WARNINGS", 0),
        )
        if should_halt and reason:
            self.order_gateway.activate_kill_switch(
                reason=reason,
                source="data_quality",
            )

    async def _build_strategy_state_snapshot(self, strategy) -> dict:
        """Capture strategy state with explicit export + bounded internals."""
        exported_state = {}
        if hasattr(strategy, "export_state"):
            try:
                candidate = await strategy.export_state()
                if isinstance(candidate, dict):
                    exported_state = candidate
            except Exception as e:
                logger.warning(f"Failed to export strategy state: {e}")
                exported_state = {"__export_error__": str(e)}

        internal = {}
        for field in ("price_history", "current_prices", "signals", "indicators"):
            if hasattr(strategy, field):
                internal[field] = self._serialize_state_value(getattr(strategy, field))

        return {
            "version": 2,
            "captured_at": datetime.utcnow().isoformat(),
            "exported_state": self._serialize_state_value(exported_state),
            "internal_state": internal,
        }

    def _restore_internal_strategy_state(self, strategy, internal_state: dict) -> None:
        """Restore bounded internal state fields for restart continuity."""
        if not isinstance(internal_state, dict):
            return

        for field in ("current_prices", "signals", "indicators"):
            if field in internal_state and hasattr(strategy, field):
                setattr(strategy, field, internal_state[field])

        if "price_history" in internal_state and hasattr(strategy, "price_history"):
            current_history = getattr(strategy, "price_history", {})
            restored = internal_state.get("price_history")
            if isinstance(current_history, dict) and isinstance(restored, dict):
                normalized = {}
                # Preserve the currently configured symbol universe first.
                for symbol, template in current_history.items():
                    rows = restored.get(symbol, [])
                    maxlen = getattr(template, "maxlen", None)
                    if isinstance(rows, list):
                        normalized[symbol] = deque(rows, maxlen=maxlen)
                    else:
                        normalized[symbol] = deque([], maxlen=maxlen)
                # Keep any legacy/restored-only symbols as best-effort extras.
                for symbol, rows in restored.items():
                    if symbol in normalized:
                        continue
                    if isinstance(rows, list):
                        normalized[symbol] = deque(rows)
                strategy.price_history = normalized

    async def _restore_strategy_state(self, strategy, saved) -> None:
        """Restore legacy or v2 strategy checkpoints."""
        if not hasattr(strategy, "import_state"):
            return

        if not isinstance(saved, dict):
            await strategy.import_state(saved)
            return

        if "exported_state" in saved:
            await strategy.import_state(saved.get("exported_state", {}))
            self._restore_internal_strategy_state(strategy, saved.get("internal_state", {}))
            return

        await strategy.import_state(saved)

    @classmethod
    def _serialize_state_value(
        cls,
        value,
        *,
        depth: int = 0,
        max_depth: int = 4,
        max_items: int = 200,
    ):
        """Convert state values into bounded JSON-safe primitives."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "isoformat") and callable(value.isoformat):
            try:
                return value.isoformat()
            except Exception:
                pass
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                pass
        if depth >= max_depth:
            return str(value)

        if isinstance(value, dict):
            serialized = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= max_items:
                    serialized["__truncated__"] = f"{len(value) - max_items} entries omitted"
                    break
                serialized[str(k)] = cls._serialize_state_value(
                    v,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                )
            return serialized

        if isinstance(value, deque):
            value = list(value)

        if isinstance(value, (list, tuple, set)):
            items = list(value)
            if len(items) > max_items:
                items = items[-max_items:]
            return [
                cls._serialize_state_value(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                )
                for item in items
            ]

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return cls._serialize_state_value(
                    value.to_dict(),
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                )
            except Exception:
                pass

        return str(value)

    def handle_shutdown_signal(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\n\n⚠️  Shutdown signal received...")
        self.shutdown_event.set()


async def main():
    """Main entry point for live trading."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Live Trading Bot")
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        choices=["momentum", "mean_reversion", "bracket_momentum"],
        help="Strategy to run",
    )
    parser.add_argument(
        "--risk-profile",
        type=str,
        default="custom",
        choices=["custom", "conservative", "balanced", "aggressive"],
        help="Runtime risk profile preset (custom uses only explicit CLI values)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None, help="Symbols to trade (default: from config)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=None,
        help="Position size as fraction of capital (overrides profile)",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=None,
        help="Hard cap per-position as fraction of capital (overrides profile)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss percentage (overrides profile)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit percentage (overrides profile)",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=None,
        help="Circuit-breaker daily loss threshold (decimal, overrides profile)",
    )
    parser.add_argument(
        "--max-intraday-drawdown",
        type=float,
        default=None,
        help="Hard intraday drawdown kill-switch threshold (decimal, overrides profile)",
    )
    parser.add_argument(
        "--drawdown-soft-limit",
        type=float,
        default=None,
        help="Soft drawdown threshold for first de-risk step (decimal)",
    )
    parser.add_argument(
        "--drawdown-soft-scale",
        type=float,
        default=None,
        help="Position multiplier at soft drawdown threshold",
    )
    parser.add_argument(
        "--drawdown-medium-limit",
        type=float,
        default=None,
        help="Medium drawdown threshold for second de-risk step (decimal)",
    )
    parser.add_argument(
        "--drawdown-medium-scale",
        type=float,
        default=None,
        help="Position multiplier at medium drawdown threshold",
    )
    parser.add_argument(
        "--kill-switch-cooldown-minutes",
        type=int,
        default=None,
        help="Minutes to keep kill-switch active once triggered",
    )
    parser.add_argument(
        "--crypto-buy-score-threshold",
        type=float,
        default=None,
        help="Momentum crypto long-only buy score threshold override",
    )
    parser.add_argument(
        "--crypto-dip-rsi-max",
        type=float,
        default=None,
        help="Momentum crypto dip-buy RSI ceiling override",
    )
    parser.add_argument(
        "--crypto-dip-min-macd-hist-delta",
        type=float,
        default=None,
        help="Momentum crypto dip-buy minimum MACD histogram delta override",
    )
    parser.add_argument(
        "--crypto-dip-min-rebound-pct",
        type=float,
        default=None,
        help="Momentum crypto dip-buy minimum rebound percent override (decimal)",
    )

    args = parser.parse_args()

    # Use provided symbols or default from config
    symbols = args.symbols if args.symbols else SYMBOLS[:3]  # Default to first 3

    # Strategy + runtime risk parameters
    parameters = _resolve_runtime_parameters(args)

    # Create live trader
    trader = LiveTrader(strategy_name=args.strategy, symbols=symbols, parameters=parameters)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, trader.handle_shutdown_signal)
    signal.signal(signal.SIGTERM, trader.handle_shutdown_signal)

    # Initialize
    if not await trader.initialize():
        logger.error("Failed to initialize. Exiting.")
        return 1

    # Start trading
    await trader.start_trading()

    return 0


if __name__ == "__main__":
    # Create logs directory if needed
    import os

    os.makedirs("logs", exist_ok=True)

    # Run
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
