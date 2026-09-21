"""
BaseStrategy core module.

Contains the ``BaseStrategy`` abstract class — initialization, parameter
management, lifecycle (initialize/run/cleanup/shutdown), state import/export,
order submission stubs, and the abstract ``analyze_symbol`` /
``execute_trade`` interface that concrete strategies implement.

Risk-permission and position-sizing helpers (Kelly, volatility/streak
adjustments, position queries) sit at the end of the class.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Optional

# NOTE: Removed lumibot.strategies.Strategy import - it crashes at import time
# We don't actually need it - we'll create our own simple base class
from engine.order_submission import OrderIntent, OrderOutcome
from engine.position_sizing import PositionSizer
from strategies.params import BaseParams
from utils.kelly_criterion import KellyCriterion
from utils.volatility_regime import VolatilityRegimeDetector

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    Params = BaseParams
    """
    Base class for all trading strategies.

    This is a simplified version that doesn't depend on lumibot's Strategy class,
    which has import-time initialization issues that crash the bot.
    """

    def __init__(self, name=None, broker=None, parameters=None, order_submission=None):
        """Initialize the strategy.

        Args:
            name: Strategy name (defaults to class name)
            broker: Broker instance for data queries
            parameters: Strategy parameters dict
            order_submission: OrderSubmission that turns intents into outcomes
                          If not provided, orders will fail when gateway enforcement is enabled
        """
        # Basic attributes
        self.name = name or self.__class__.__name__
        self.broker = broker
        self.order_submission = order_submission
        # Every parameter is declared once on ``Params``; unknown keys are an error
        # and every declared key is present with its default (ADR 0010).
        parameters = self.Params.check(parameters or {})

        self.parameters = parameters
        self.interval = parameters["interval"]  # Default to 60 seconds
        self.symbols = parameters["symbols"]
        self._shutdown_event = asyncio.Event()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.price_history = {}

        # P1 FIX: Initialize running flag and tasks list for cleanup()
        self.running = False
        self.tasks = []

        # KELLY CRITERION: Initialize for optimal position sizing
        use_kelly = parameters["use_kelly_criterion"]
        if use_kelly:
            kelly_fraction = parameters["kelly_fraction"]  # Half Kelly by default
            self.kelly = KellyCriterion(
                kelly_fraction=kelly_fraction,
                min_trades_required=parameters["kelly_min_trades"],
                max_position_size=parameters["max_position_size"],
                min_position_size=parameters["min_position_size"],
                lookback_trades=parameters["kelly_lookback"],
            )
            self.logger.info(f"✅ Kelly Criterion enabled: {kelly_fraction} Kelly fraction")
        else:
            self.kelly = None

        # VOLATILITY REGIME: Initialize for adaptive risk management
        use_volatility_regime = parameters["use_volatility_regime"]
        if use_volatility_regime:
            self.volatility_regime = None  # Initialized in async initialize()
            self.logger.info("✅ Volatility Regime Detection enabled")
        else:
            self.volatility_regime = None

        # Multi-timeframe analysis lives on the concrete strategies (see
        # MomentumStrategy.mtf_analyzer / MeanReversionStrategy.mtf_analyzer).
        # The base-class wiring previously here imported a parallel
        # MultiTimeframeAnalyzer implementation with an .analyze() method that
        # the canonical utils.multi_timeframe.MultiTimeframeAnalyzer does not
        # expose; it was removed in the 2026-05 form cleanup.

        # Sentiment filtering removed in the 2026-05 cleanup. The FinBERT-based
        # NewsSentimentAnalyzer it depended on had no validated edge; placeholder
        # attributes are kept neutral so legacy callers do not crash.
        self.sentiment_analyzer = None
        self.sentiment_block_threshold = -0.3
        self.sentiment_boost_threshold = 0.3
        self.sentiment_max_multiplier = 1.0
        self.sentiment_min_multiplier = 1.0

    async def initialize(self, **kwargs):
        """Initialize strategy parameters."""
        try:
            # Update parameters
            self.parameters.update(kwargs)

            # Set up strategy parameters
            self.interval = self.parameters["interval"]
            self.symbols = self.parameters["symbols"]

            # Initialize any other strategy-specific parameters
            self._initialize_parameters()

            # Feed the Kelly estimator from real fills (ADR 0006).
            recorder = getattr(getattr(self, "order_submission", None), "recorder", None)
            if recorder is not None and getattr(self, "kelly", None) is not None:
                recorder.subscribe(
                    getattr(self, "name", self.__class__.__name__), self.kelly.add_trade
                )

            # VOLATILITY REGIME: Initialize detector with broker
            if self.parameters["use_volatility_regime"] and self.broker:
                self.volatility_regime = VolatilityRegimeDetector(self.broker)
                regime, adjustments = await self.volatility_regime.get_current_regime()
                self.logger.info(
                    f"✅ Volatility regime detector initialized: "
                    f"{regime.upper()} (Position: {adjustments['pos_mult']:.1f}x, "
                    f"Stop: {adjustments['stop_mult']:.1f}x)"
                )

            return True

        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}", exc_info=True)
            return False

    def _initialize_parameters(self):
        """Initialize strategy-specific parameters. Override in subclass."""
        self.sentiment_threshold = self.parameters["sentiment_threshold"]
        self.position_size = self.parameters["position_size"]
        self.max_position_size = self.parameters["max_position_size"]  # SAFETY: 5% max per position
        self.stop_loss_pct = self.parameters["stop_loss_pct"]
        self.take_profit_pct = self.parameters["take_profit_pct"]
        self.portfolio_risk_limit = self.parameters["portfolio_risk_limit"]
        self.position_risk_limit = self.parameters["position_risk_limit"]
        self.max_correlation = self.parameters["max_correlation"]
        self.var_confidence = self.parameters["var_confidence"]
        self.price_history_window = self.parameters["price_history_window"]
        self.volatility_threshold = self.parameters["volatility_threshold"]
        self.var_threshold = self.parameters["var_threshold"]
        self.es_threshold = self.parameters["es_threshold"]
        self.drawdown_threshold = self.parameters["drawdown_threshold"]

    async def on_trading_iteration(self):
        """Main trading logic. Must be implemented by subclasses."""
        raise NotImplementedError

    async def export_state(self) -> dict:
        """Export minimal strategy state for persistence."""
        return {}

    async def import_state(self, state: dict) -> None:
        """Restore strategy state from persistence."""
        return None

    def get_parameters(self):
        """Get strategy parameters."""
        return self.parameters

    def set_parameters(self, parameters):
        """Replace the parameters; every key must be declared on ``Params``."""
        self.parameters = self.Params.check(parameters or {})
        self._initialize_parameters()

    def on_bot_crash(self, error):
        """Called when the bot crashed."""
        self.logger.error(f"Bot crashed: {error}")

    async def cleanup(self):
        """Cleanup resources."""
        self.running = False
        tasks = [t for t in self.tasks if not t.done()]
        if tasks:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_broker_positions(self):
        """Positions in the protocol's shape (brokers/protocol.py), from either broker."""
        return await self.broker.get_positions()

    async def submit_entry_order(self, intent: OrderIntent) -> Optional[OrderOutcome]:
        """Hand an entry intent to order submission; None only when none is wired."""
        submission = getattr(self, "order_submission", None)
        strategy_logger = getattr(self, "logger", logger)
        if submission is None:
            strategy_logger.error("No OrderSubmission wired; entry for %s blocked.", intent.symbol)
            return None
        if not intent.strategy_name:
            intent = replace(intent, strategy_name=getattr(self, "name", self.__class__.__name__))
        outcome = await submission.submit(intent)
        if outcome.ok:
            strategy_logger.info(
                f"ENTRY ORDER: {intent.reason} - {outcome.side.upper()} {outcome.qty_requested} "
                f"{intent.symbol} ({outcome.status.value}, id={outcome.order_id})"
            )
        else:
            strategy_logger.warning(
                f"ENTRY ORDER {outcome.status.value.upper()} for {intent.symbol}: {outcome.reason}"
            )
        return outcome

    async def submit_exit_order(
        self, symbol: str, qty: float, side: str = "sell", reason: str = "exit"
    ) -> Optional[OrderOutcome]:
        """Close (part of) a held position. Returns None when nothing is held or no
        submission is wired; otherwise the outcome, which may be rejected."""
        strategy_logger = getattr(self, "logger", logger)
        submission = getattr(self, "order_submission", None)
        try:
            positions = await self._fetch_broker_positions()
        except Exception as e:
            strategy_logger.error(f"EXIT ORDER ERROR for {symbol}: {e}")
            return None
        held = next((p for p in positions if p.symbol == symbol), None)
        if held is None:
            strategy_logger.warning(f"EXIT REJECTED: No position found for {symbol}")
            return None
        actual_qty = abs(float(held.qty))
        if qty > actual_qty * 1.01:
            strategy_logger.warning(
                f"EXIT ADJUSTED: Requested {qty} but only have {actual_qty} {symbol}"
            )
            qty = actual_qty
        if submission is None:
            strategy_logger.error("No OrderSubmission wired; exit for %s blocked.", symbol)
            return None
        outcome = await submission.submit(
            OrderIntent(
                symbol=symbol,
                side=side,
                qty=qty,
                is_exit=True,
                strategy_name=getattr(self, "name", self.__class__.__name__),
                reason=reason,
            )
        )
        if outcome.ok:
            strategy_logger.info(
                f"EXIT ORDER: {reason} - {side.upper()} {qty:.4f} {symbol} (id={outcome.order_id})"
            )
        else:
            strategy_logger.warning(
                f"EXIT ORDER {outcome.status.value.upper()} for {symbol}: {outcome.reason}"
            )
        return outcome

    # ------------------------------------------------------------------
    # Decider interface: the session calls these; strategies never submit.
    # ------------------------------------------------------------------

    async def prepare(self, when, histories) -> None:
        """Compute this session's signals from bar histories (bars strictly before ``when``).

        The default feeds the legacy per-symbol structures and runs
        ``generate_signals`` so existing signal code keeps working unchanged.
        """
        if not hasattr(self, "current_data"):
            self.current_data = {}
        prices = getattr(self, "current_prices", None)
        for symbol, df in histories.items():
            if len(df) == 0:
                continue
            self.current_data[symbol] = df
            if hasattr(self, "price_history"):
                self.price_history[symbol] = df["close"].tolist()[-30:]
            if isinstance(prices, dict):
                # The last close before this session: what strategy-managed
                # exits (trailing stops etc.) compare against in daily mode.
                prices[symbol] = float(df["close"].iloc[-1])
        generate = getattr(self, "generate_signals", None)
        if callable(generate):
            await generate()

    async def decide(self, symbol, when, portfolio) -> list:
        """Order intents for ``symbol`` this session. Default: none."""
        return []

    def _signal_action(self, symbol) -> str:
        signal = getattr(self, "signals", {}).get(symbol, "neutral")
        if isinstance(signal, dict):
            signal = signal.get("action", "neutral")
        return signal or "neutral"

    async def _daily_intents(
        self, symbol, action, portfolio, *, size_pct, sizing_basis, reason, when=None
    ):
        """Fixed-fraction daily execution: enter when flat, exit on the opposite signal.

        Reproduces the 2020-2024 baseline semantics exactly (integer shares,
        equity- or cash-based sizing, no stop-and-reverse).
        """
        if action in ("neutral", "hold", None):
            return []
        held = portfolio.position(symbol)
        cash = float(portfolio.cash)
        price = await portfolio.ask_price(symbol)
        if sizing_basis == "cash":
            position_value = cash * size_pct
        else:
            position_value = (float(portfolio.equity) or cash) * size_pct
        qty = int(position_value / price)
        if action == "buy":
            qty = min(qty, int(cash / price))
        if qty <= 0:
            return []
        pos_qty = int(held.qty) if held is not None else 0
        name = getattr(self, "name", self.__class__.__name__)

        def entry(side):
            return OrderIntent(
                symbol=symbol, side=side, qty=qty, strategy_name=name, reason=f"{reason}_entry"
            )

        def exit_(side, q):
            return OrderIntent(
                symbol=symbol,
                side=side,
                qty=q,
                is_exit=True,
                strategy_name=name,
                reason=f"{reason}_exit",
            )

        if action in ("buy", "short") and held is None:
            self._record_entry(symbol, price, when, stop_loss=None, take_profit=None)
            return [entry("buy" if action == "buy" else "sell")]
        if action == "short" and pos_qty > 0:
            self._clear_entry(symbol)
            return [exit_("sell", pos_qty)]
        if action == "buy" and pos_qty < 0:
            self._clear_entry(symbol)
            return [exit_("buy", -pos_qty)]
        if action == "sell" and pos_qty > 0:
            self._clear_entry(symbol)
            return [exit_("sell", pos_qty)]
        return []

    # ------------------------------------------------------------------
    # Live execution as intents: brackets, cooldown, risk haircut, hard cap.
    # ------------------------------------------------------------------

    def _record_entry(self, symbol, price, when, *, stop_loss, take_profit):
        for attr, val in (
            ("stop_prices", stop_loss),
            ("target_prices", take_profit),
            ("entry_prices", price),
            ("peak_prices", price),
            ("lowest_prices", price),
            ("highest_prices", price),
            ("last_signal_time", when),
        ):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store[symbol] = val
        entries = getattr(self, "position_entries", None)
        if isinstance(entries, dict):
            entries[symbol] = {
                "time": when,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

    def _clear_entry(self, symbol):
        for attr in (
            "stop_prices",
            "target_prices",
            "entry_prices",
            "peak_prices",
            "lowest_prices",
            "highest_prices",
            "position_entries",
        ):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(symbol, None)

    async def _live_intents(self, symbol, when, view):
        """Signal → intents under the live rules. Exits from _exit_intents run first."""
        intents = list(await self._exit_intents(symbol, when, view))
        action = self._signal_action(symbol)
        if action in ("neutral", "hold", None):
            return intents
        last = getattr(self, "last_signal_time", {}).get(symbol)
        if last and (when - last).total_seconds() < 3600:
            return intents
        held = view.position(symbol)
        price = getattr(self, "current_prices", {}).get(symbol) or await view.ask_price(symbol)
        name = getattr(self, "name", self.__class__.__name__)
        max_positions = int(getattr(self, "max_positions", self.parameters["max_positions"]))
        if action in ("buy", "short") and held is None:
            is_short = action == "short"
            if is_short and not getattr(
                self, "enable_short_selling", self.parameters["enable_short_selling"]
            ):
                return intents
            if len(view.positions) >= max_positions:
                logger.info(
                    f"Max positions reached ({max_positions}), skipping {action} for {symbol}"
                )
                return intents
            sizing = self.sizer().size(
                symbol,
                price,
                view,
                self._extract_close_price_history(symbol),
                is_short=is_short,
                held_closes={
                    held_symbol: self._extract_close_price_history(held_symbol)
                    for held_symbol in view.positions
                },
            )
            if not sizing.tradeable:
                logger.info(f"No tradeable size for {symbol}: {'; '.join(sizing.steps)}")
                return intents
            qty = sizing.qty
            take_profit = float(getattr(self, "take_profit", self.parameters["take_profit"]))
            stop_loss = float(
                getattr(self, "short_stop_loss", self.parameters["short_stop_loss"])
                if is_short
                else getattr(self, "stop_loss", self.parameters["stop_loss"])
            )
            if is_short:
                tp_price, sl_price = price * (1 - take_profit), price * (1 + stop_loss)
            else:
                tp_price, sl_price = price * (1 + take_profit), price * (1 - stop_loss)
            self._record_entry(symbol, price, when, stop_loss=sl_price, take_profit=tp_price)
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side="sell" if is_short else "buy",
                    qty=qty,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    time_in_force="gtc",
                    strategy_name=name,
                    reason=f"{name}_{'short_' if is_short else ''}entry",
                )
            )
        elif action == "sell" and held is not None and float(held.qty) > 0:
            self._clear_entry(symbol)
            store = getattr(self, "last_signal_time", None)
            if isinstance(store, dict):
                store[symbol] = when
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side="sell",
                    qty=float(held.qty),
                    is_exit=True,
                    strategy_name=name,
                    reason="signal_exit",
                )
            )
        return intents

    def sizer(self) -> PositionSizer:
        """The one sizing decision, built from this strategy's parameters."""
        use_kelly = bool(self.parameters["use_kelly_criterion"])
        return PositionSizer(
            base_fraction=float(getattr(self, "position_size", self.parameters["position_size"])),
            short_fraction=float(
                getattr(self, "short_position_size", self.parameters["short_position_size"])
            ),
            max_position_fraction=float(
                getattr(self, "max_position_size", self.parameters["max_position_size"])
            ),
            kelly=self.kelly if use_kelly and getattr(self, "kelly", None) is not None else None,
            risk_manager=getattr(self, "risk_manager", None),
            regime_multiplier=getattr(self, "regime_multiplier", None),
        )

    async def _exit_intents(self, symbol, when, view):
        """Strategy-managed exits (trailing stops etc.). Default: none."""
        return []

    async def _decide(self, symbol, when, view, **daily):
        """Dispatch: live rules under a LiveSession, daily baseline rules otherwise."""
        if getattr(self, "execution_mode", "daily") == "live":
            return await self._live_intents(symbol, when, view)
        intents = []
        if self.parameters["daily_exits"]:
            intents.extend(await self._exit_intents(symbol, when, view))
        intents.extend(
            await self._daily_intents(symbol, self._signal_action(symbol), view, **daily)
        )
        return intents

    def _extract_close_price_history(self, symbol: str) -> list[float]:
        """
        Normalize symbol history into a close-price series for risk calculations.

        Supports strategy histories stored as:
        - list/deque of OHLCV dict bars (expects `close`)
        - list/deque of numeric close prices
        """
        raw_history = self.price_history.get(symbol, [])
        if raw_history is None:
            return []

        normalized_history = list(raw_history)
        closes: list[float] = []
        for item in normalized_history:
            if isinstance(item, dict):
                close = item.get("close")
            else:
                close = item
            if close is None:
                continue
            try:
                closes.append(float(close))
            except (TypeError, ValueError):
                continue
        return closes

    @abstractmethod
    async def analyze_symbol(self, symbol):
        """Analyze a symbol and return trading signals."""
        pass

    async def shutdown(self):
        """Shutdown the strategy."""
        self._shutdown_event.set()
        await self.cleanup()
