"""
BaseStrategy core module.

Contains the ``BaseStrategy`` abstract class — initialization, parameter
management, lifecycle (initialize/run/cleanup/shutdown), state import/export,
order submission stubs, and the abstract ``analyze_symbol`` /
``execute_trade`` interface that concrete strategies implement.

Risk-permission and position-sizing helpers (Kelly, volatility/streak
adjustments, position queries) live in
``strategies/base/position_sizing.py`` and are mixed in by the concrete
``BaseStrategy`` class to keep this module focused on lifecycle / order
plumbing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod

# NOTE: Removed lumibot.strategies.Strategy import - it crashes at import time
# We don't actually need it - we'll create our own simple base class
import numpy as np

from utils.circuit_breaker import CircuitBreaker
from utils.kelly_criterion import KellyCriterion
from utils.streak_sizing import StreakSizer
from utils.volatility_regime import VolatilityRegimeDetector

from strategies.base.position_sizing import BasePositionSizingMixin

logger = logging.getLogger(__name__)


class BaseStrategy(BasePositionSizingMixin, ABC):
    """
    Base class for all trading strategies.

    This is a simplified version that doesn't depend on lumibot's Strategy class,
    which has import-time initialization issues that crash the bot.
    """

    def __init__(self, name=None, broker=None, parameters=None, order_gateway=None):
        """Initialize the strategy.

        Args:
            name: Strategy name (defaults to class name)
            broker: Broker instance for data queries
            parameters: Strategy parameters dict
            order_gateway: OrderGateway instance for order submission (recommended)
                          If not provided, orders will fail when gateway enforcement is enabled
        """
        # Basic attributes
        self.name = name or self.__class__.__name__
        self.broker = broker
        self.order_gateway = order_gateway  # INSTITUTIONAL: All orders should go through gateway
        parameters = parameters or {}

        # No parent class to initialize anymore - we're independent!

        # Initialize our parameters
        self.parameters = parameters
        self.interval = parameters.get("interval", 60)  # Default to 60 seconds
        self.symbols = parameters.get("symbols", [])
        self._shutdown_event = asyncio.Event()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.price_history = {}

        # P1 FIX: Initialize running flag and tasks list for cleanup()
        self.running = False
        self.tasks = []

        # CRITICAL SAFETY: Initialize circuit breaker for daily loss protection
        max_daily_loss = parameters.get("max_daily_loss", 0.03)  # Default 3%
        self.circuit_breaker = CircuitBreaker(
            max_daily_loss=max_daily_loss, auto_close_positions=True
        )

        # KELLY CRITERION: Initialize for optimal position sizing
        use_kelly = parameters.get("use_kelly_criterion", False)
        if use_kelly:
            kelly_fraction = parameters.get("kelly_fraction", 0.5)  # Half Kelly by default
            self.kelly = KellyCriterion(
                kelly_fraction=kelly_fraction,
                min_trades_required=parameters.get("kelly_min_trades", 30),
                max_position_size=parameters.get("max_position_size", 0.20),
                min_position_size=parameters.get("min_position_size", 0.01),
                lookback_trades=parameters.get("kelly_lookback", 50),
            )
            self.logger.info(f"✅ Kelly Criterion enabled: {kelly_fraction} Kelly fraction")
        else:
            self.kelly = None

        # Track closed positions for Kelly Criterion
        self.closed_positions = {}  # {symbol: {'entry_price': float, 'entry_time': datetime}}

        # VOLATILITY REGIME: Initialize for adaptive risk management
        use_volatility_regime = parameters.get("use_volatility_regime", False)
        if use_volatility_regime:
            self.volatility_regime = None  # Initialized in async initialize()
            self.logger.info("✅ Volatility Regime Detection enabled")
        else:
            self.volatility_regime = None

        # STREAK SIZING: Initialize for dynamic position sizing based on recent performance
        use_streak_sizing = parameters.get("use_streak_sizing", False)
        if use_streak_sizing:
            self.streak_sizer = StreakSizer(
                lookback_trades=parameters.get("streak_lookback", 10),
                hot_streak_threshold=parameters.get("hot_streak_threshold", 7),
                cold_streak_threshold=parameters.get("cold_streak_threshold", 3),
                hot_multiplier=parameters.get("hot_multiplier", 1.2),
                cold_multiplier=parameters.get("cold_multiplier", 0.7),
                reset_after_trades=parameters.get("streak_reset_after", 5),
            )
            self.logger.info(
                f"✅ Streak-based position sizing enabled: lookback={parameters.get('streak_lookback', 10)} trades"
            )
        else:
            self.streak_sizer = None

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
            self.interval = self.parameters.get("interval", 60)
            self.symbols = self.parameters.get("symbols", [])

            # Initialize any other strategy-specific parameters
            self._initialize_parameters()

            # CRITICAL SAFETY: Initialize circuit breaker with broker
            if self.broker:
                await self.circuit_breaker.initialize(self.broker)
                self.logger.info(
                    f"✅ Circuit breaker armed: max daily loss = {self.circuit_breaker.max_daily_loss:.1%}"
                )

            # VOLATILITY REGIME: Initialize detector with broker
            if self.parameters.get("use_volatility_regime", False) and self.broker:
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
        self.sentiment_threshold = self.parameters.get("sentiment_threshold", 0.6)
        self.position_size = self.parameters.get("position_size", 0.1)
        self.max_position_size = self.parameters.get(
            "max_position_size", 0.05
        )  # SAFETY: 5% max per position
        self.stop_loss_pct = self.parameters.get("stop_loss_pct", 0.02)
        self.take_profit_pct = self.parameters.get("take_profit_pct", 0.05)
        self.portfolio_risk_limit = self.parameters.get("portfolio_risk_limit", 0.02)
        self.position_risk_limit = self.parameters.get("position_risk_limit", 0.01)
        self.max_correlation = self.parameters.get("max_correlation", 0.7)
        self.var_confidence = self.parameters.get("var_confidence", 0.95)
        self.price_history_window = self.parameters.get("price_history_window", 30)
        self.volatility_threshold = self.parameters.get("volatility_threshold", 0.4)
        self.var_threshold = self.parameters.get("var_threshold", 0.03)
        self.es_threshold = self.parameters.get("es_threshold", 0.04)
        self.drawdown_threshold = self.parameters.get("drawdown_threshold", 0.3)

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
        """Set strategy parameters."""
        self.parameters = parameters
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

    async def submit_exit_order(
        self,
        symbol: str,
        qty: float,
        side: str = "sell",
        reason: str = "exit",
    ):
        """
        Submit an exit order with appropriate safety checks.

        INSTITUTIONAL SAFETY: Exit orders MUST route through OrderGateway for
        audit trail, consistent controls, and kill-switch behavior.

        Args:
            symbol: Stock symbol
            qty: Quantity to exit
            side: 'sell' for long exit, 'buy' for short exit
            reason: Reason for exit (for logging)

        Returns:
            Order result or None on failure
        """
        try:
            strategy_name = getattr(self, "name", self.__class__.__name__)
            order_gateway = getattr(self, "order_gateway", None)
            strategy_logger = getattr(self, "logger", logger)

            # Verify we own this position
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            if not current_position:
                strategy_logger.warning(f"EXIT REJECTED: No position found for {symbol}")
                return None

            actual_qty = abs(float(current_position.qty))
            if qty > actual_qty * 1.01:  # Allow 1% tolerance for fractional shares
                strategy_logger.warning(
                    f"EXIT ADJUSTED: Requested {qty} but only have {actual_qty} {symbol}"
                )
                qty = actual_qty

            if not order_gateway:
                strategy_logger.error(
                    "No OrderGateway configured. Exit order blocked; "
                    "gateway-only routing is mandatory."
                )
                return None

            result = await order_gateway.submit_exit_order(
                symbol=symbol,
                quantity=qty,
                strategy_name=strategy_name,
                side=side,
                reason=reason,
            )
            if result.success:
                strategy_logger.info(
                    f"EXIT ORDER: {reason} - {side.upper()} {qty:.4f} {symbol} "
                    f"(Order ID: {result.order_id})"
                )
                return result

            strategy_logger.warning(f"EXIT ORDER FAILED for {symbol}: {result.rejection_reason}")
            return None

        except Exception as e:
            strategy_logger = getattr(self, "logger", logger)
            strategy_logger.error(f"EXIT ORDER ERROR for {symbol}: {e}")
            return None

    async def submit_entry_order(
        self,
        order_request,
        reason: str = "entry",
        max_positions: int = None,
    ):
        """
        Submit an entry order through the OrderGateway with full safety checks.

        INSTITUTIONAL SAFETY: ALL entry orders MUST route through OrderGateway
        for circuit breaker, position conflict, and risk limit enforcement.

        Args:
            order_request: Order request from OrderBuilder
            reason: Reason for entry (for logging)
            max_positions: Maximum number of positions allowed (optional)

        Returns:
            OrderResult with success status and details, or None on error

        Raises:
            RuntimeError: If no OrderGateway is configured and gateway enforcement is enabled
        """
        order_gateway = getattr(self, "order_gateway", None)
        strategy_name = getattr(self, "name", self.__class__.__name__)
        strategy_logger = getattr(self, "logger", logger)

        if not order_gateway:
            strategy_logger.error(
                "No OrderGateway configured. Entry order blocked; "
                "gateway-only routing is mandatory."
            )
            return None

        try:
            # Extract symbol for logging
            symbol = getattr(order_request, "symbol", "UNKNOWN")
            if hasattr(order_request, "build"):
                built = order_request.build()
                symbol = getattr(built, "symbol", symbol)

            risk_price_history = self._extract_close_price_history(symbol)

            result = await order_gateway.submit_order(
                order_request=order_request,
                strategy_name=strategy_name,
                max_positions=max_positions,
                price_history=risk_price_history,
                is_exit_order=False,
            )

            if result.success:
                strategy_logger.info(
                    f"ENTRY ORDER: {reason} - {result.side.upper()} {result.quantity} {symbol} "
                    f"(Order ID: {result.order_id})"
                )
            else:
                strategy_logger.warning(
                    f"ENTRY ORDER REJECTED for {symbol}: {result.rejection_reason}"
                )

            return result

        except Exception as e:
            strategy_logger.error(f"Entry order error: {e}")
            return None

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

    async def run(self):
        """Run the strategy."""
        try:
            while not self._shutdown_event.is_set():
                # Get current positions
                positions = await self.get_positions()

                # Update stop losses for existing positions
                for position in positions:
                    await self._update_stop_loss(position)

                # Get trading signals for each symbol
                for symbol in self.symbols:
                    try:
                        signal = await self.get_signal(symbol)
                        if signal:
                            await self.execute_trade(symbol, signal)
                    except Exception as e:
                        logger.error(f"Error processing signal for {symbol}: {e}", exc_info=True)

                # Sleep before next iteration
                await asyncio.sleep(self.interval)

        except Exception as e:
            logger.error(f"Error in strategy {self.__class__.__name__}: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def backtest(self, *args, **kwargs):
        """Run backtesting."""
        try:
            self.running = True
            # NOTE: Removed super().backtest() call - we no longer inherit from lumibot.Strategy
            # Backtesting is now handled by engine/backtest_engine.py instead
            raise NotImplementedError(
                "Backtesting should be done via BacktestEngine, not directly on strategies"
            )
        except Exception as e:
            logger.error(f"Error in backtesting {self.name}: {e}")
            raise
        finally:
            await self.cleanup()

    @abstractmethod
    async def analyze_symbol(self, symbol):
        """Analyze a symbol and return trading signals."""
        pass

    @abstractmethod
    async def execute_trade(self, symbol, signal):
        """Execute a trade based on the signal.

        P1 FIX: Added async to match implementations in subclasses.
        """
        pass

    def create_order(
        self, symbol, quantity, side, type="market", limit_price=None, stop_price=None
    ):
        """
        Create an order object.

        Args:
            symbol (str): The symbol to trade.
            quantity (float): The quantity to trade.
            side (str): 'buy' or 'sell'.
            type (str): 'market', 'limit', or 'stop'.
            limit_price (float, optional): The limit price for limit orders.
            stop_price (float, optional): The stop price for stop orders.

        Returns:
            dict: The order object.
        """
        order = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "type": type,
        }
        if limit_price:
            order["limit_price"] = limit_price
        if stop_price:
            order["stop_price"] = stop_price
        return order

    async def _update_stop_loss(self, position):
        """
        Update the stop-loss level for a position based on volatility.

        Uses the wider (more protective) of:
        - Volatility-based stop (2 standard deviations)
        - Configured stop_loss_pct parameter

        Note: This calculates the optimal stop loss but does not automatically
        update broker orders. Subclasses should override to implement
        broker-specific order modification if needed.
        """
        try:
            symbol = position.symbol
            float(position.current_price)
            avg_entry_price = float(position.avg_entry_price)
            volatility = self._calculate_volatility(symbol)

            # Calculate volatility-based stop (2 standard deviations below entry)
            vol_stop_loss = avg_entry_price * (1 - 2 * volatility) if volatility > 0 else 0

            # Calculate parameter-based stop loss
            param_stop_loss = avg_entry_price * (1 - self.stop_loss_pct)

            # Use the wider stop loss (higher price = less likely to be triggered)
            # This provides better protection in volatile conditions
            stop_loss = max(vol_stop_loss, param_stop_loss)

            # Only log if stop loss is meaningful (not zero or negative)
            if stop_loss > 0:
                self.logger.debug(
                    f"Stop-loss for {symbol}: ${stop_loss:.2f} "
                    f"(vol-based: ${vol_stop_loss:.2f}, param-based: ${param_stop_loss:.2f})"
                )

            # Note: Broker order updates should be handled by strategy subclasses
            # as order modification APIs vary by broker and order type

        except Exception as e:
            self.logger.error(f"Error updating stop-loss for {symbol}: {e}", exc_info=True)

    def _calculate_volatility(self, symbol):
        """Calculate the historical volatility for a symbol."""
        try:
            # Assuming self.price_history is available and populated by the strategy
            if (
                symbol not in self.price_history
                or len(self.price_history[symbol]) < self.price_history_window
            ):
                self.logger.warning(
                    f"Insufficient price history for {symbol} to calculate volatility"
                )
                return 0  # Or some default value

            prices = np.array(self.price_history[symbol])
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualize
            return volatility

        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}", exc_info=True)
            return 0  # Or some default value

    async def shutdown(self):
        """Shutdown the strategy."""
        self._shutdown_event.set()
        await self.cleanup()
