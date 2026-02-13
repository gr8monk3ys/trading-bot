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
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from brokers.alpaca_broker import AlpacaBroker
from config import RISK_PARAMS, SYMBOLS
from strategies.bracket_momentum_strategy import BracketMomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.risk_manager import RiskManager
from utils.audit_log import AuditEventType, AuditLog
from utils.circuit_breaker import CircuitBreaker
from utils.incident_tracker import IncidentTracker
from utils.order_gateway import OrderGateway
from utils.order_reconciliation import OrderReconciler
from utils.position_manager import PositionManager
from utils.reconciliation import PositionReconciler
from utils.run_artifacts import JsonlWriter, ensure_run_directory, generate_run_id, write_json
from utils.runtime_state import RuntimeStateStore
from utils.slo_alerting import build_slo_alert_notifier
from utils.slo_monitor import SLOMonitor
from utils.data_quality import (
    should_halt_trading_for_data_quality,
    summarize_quality_reports,
    validate_ohlcv_frame,
)

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
        self.running = False
        self.session_run_id = generate_run_id("live")
        self.session_run_dir = ensure_run_directory("results/runs", self.session_run_id)

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
            self.broker = AlpacaBroker(paper=True)

            # Get account info
            account = await self.broker.get_account()
            self.start_equity = float(account.equity)

            logger.info(f"✅ Connected to account: {account.id}")
            logger.info(f"   Starting Capital: ${self.start_equity:,.2f}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}\n")

            # Initialize circuit breaker (CRITICAL SAFETY FEATURE)
            logger.info("1.5. Initializing circuit breaker...")
            self.circuit_breaker = CircuitBreaker(
                max_daily_loss=0.03,  # 3% max daily loss
                auto_close_positions=True,  # Automatically close positions on trigger
            )
            await self.circuit_breaker.initialize(self.broker)
            logger.info("✅ Circuit breaker armed")
            logger.info("   Max Daily Loss: 3%")
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
            )
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
                self.broker, default_strategy="recovered"
            )

            # Initialize reconciler
            self.reconciler = PositionReconciler(
                broker=self.broker,
                internal_tracker=self.position_manager,
                halt_on_mismatch=False,
                sync_to_broker=True,
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
            if slo_alert_notifier:
                logger.info(
                    "SLO paging alerts enabled (severity>=%s)",
                    RISK_PARAMS.get("SLO_PAGING_MIN_SEVERITY", "critical"),
                )
            self.slo_monitor = SLOMonitor(
                audit_log=self.audit_log,
                events_path=self.session_run_dir / "ops_slo_events.jsonl",
                alert_notifier=slo_alert_notifier,
                incident_tracker=incident_tracker,
                recon_mismatch_halt_runs=RISK_PARAMS.get("ORDER_RECON_MISMATCH_HALT_RUNS", 3),
                max_data_quality_errors=RISK_PARAMS.get("DATA_QUALITY_MAX_ERRORS", 0),
                max_stale_data_warnings=RISK_PARAMS.get("DATA_QUALITY_MAX_STALE_WARNINGS", 0),
                shadow_drift_warning_threshold=RISK_PARAMS.get("PAPER_LIVE_SHADOW_DRIFT_WARNING", 0.12),
                shadow_drift_critical_threshold=RISK_PARAMS.get("PAPER_LIVE_SHADOW_DRIFT_MAX", 0.15),
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

            logger.info("✅ Strategy initialized")
            logger.info(f"   Trading: {', '.join(self.symbols)}")
            logger.info(f"   Parameters: {self.parameters}\n")

            # Check market status
            clock = await self.broker.get_clock()
            logger.info("3. Market Status:")
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
            await self.broker.start_websocket()

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
                        if self.order_gateway and hasattr(self.order_gateway, "export_runtime_state")
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
                            if self.order_gateway and hasattr(self.order_gateway, "export_runtime_state")
                            else {}
                        ),
                        strategy_states=strategy_states,
                    )
                    if self.slo_monitor:
                        self.slo_monitor.check_incident_ack_sla()
                if (
                    counter % reconciliation_interval == 0
                    and self.reconciler is not None
                ):
                    try:
                        await self.reconciler.reconcile()
                    except Exception as e:
                        logger.error(f"Reconciliation error: {e}")
                    try:
                        await self._run_data_quality_gate()
                    except Exception as e:
                        logger.error(f"Data quality gate error: {e}")
                if (
                    counter % 120 == 0
                    and self.order_reconciler is not None
                ):
                    try:
                        await self.order_reconciler.reconcile()
                        if self.slo_monitor:
                            breaches = self.slo_monitor.record_order_reconciliation_health(
                                self.order_reconciler.get_health_snapshot()
                            )
                            if (
                                self.order_gateway
                                and SLOMonitor.has_critical_breach(breaches)
                            ):
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

        reference_time = datetime.utcnow()
        start = (reference_time - timedelta(days=30)).strftime("%Y-%m-%d")
        end = reference_time.strftime("%Y-%m-%d")
        stale_after_days = RISK_PARAMS.get("DATA_QUALITY_STALE_AFTER_DAYS", 3)
        reports = []

        for symbol in self.symbols:
            try:
                bars = await self.broker.get_bars(
                    symbol,
                    start=start,
                    end=end,
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
                for symbol, rows in restored.items():
                    template = current_history.get(symbol)
                    maxlen = getattr(template, "maxlen", None) if template is not None else None
                    if isinstance(rows, list):
                        normalized[symbol] = deque(rows, maxlen=maxlen)
                    else:
                        normalized[symbol] = deque([], maxlen=maxlen)
                setattr(strategy, "price_history", normalized)

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
        "--symbols", nargs="+", default=None, help="Symbols to trade (default: from config)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.10,
        help="Position size as fraction of capital (default: 0.10)",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=0.02, help="Stop loss percentage (default: 0.02)"
    )
    parser.add_argument(
        "--take-profit", type=float, default=0.05, help="Take profit percentage (default: 0.05)"
    )

    args = parser.parse_args()

    # Use provided symbols or default from config
    symbols = args.symbols if args.symbols else SYMBOLS[:3]  # Default to first 3

    # Strategy parameters
    parameters = {
        "position_size": args.position_size,
        "stop_loss": args.stop_loss,
        "take_profit": args.take_profit,
    }

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
