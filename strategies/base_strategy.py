from abc import ABC, abstractmethod
import logging
import asyncio
from typing import Optional
# NOTE: Removed lumibot.strategies.Strategy import - it crashes at import time
# We don't actually need it - we'll create our own simple base class
import numpy as np
from datetime import datetime
from utils.circuit_breaker import CircuitBreaker
from utils.kelly_criterion import KellyCriterion, Trade
from utils.volatility_regime import VolatilityRegimeDetector
from utils.streak_sizing import StreakSizer
from utils.multi_timeframe_analyzer import MultiTimeframeAnalyzer

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    This is a simplified version that doesn't depend on lumibot's Strategy class,
    which has import-time initialization issues that crash the bot.
    """
    def __init__(self, name=None, broker=None, parameters=None):
        """Initialize the strategy."""
        # Basic attributes
        self.name = name or self.__class__.__name__
        self.broker = broker
        parameters = parameters or {}

        # No parent class to initialize anymore - we're independent!

        # Initialize our parameters
        self.parameters = parameters
        self.interval = parameters.get('interval', 60)  # Default to 60 seconds
        self.symbols = parameters.get('symbols', [])
        self._shutdown_event = asyncio.Event()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.price_history = {}

        # P1 FIX: Initialize running flag and tasks list for cleanup()
        self.running = False
        self.tasks = []

        # CRITICAL SAFETY: Initialize circuit breaker for daily loss protection
        max_daily_loss = parameters.get('max_daily_loss', 0.03)  # Default 3%
        self.circuit_breaker = CircuitBreaker(
            max_daily_loss=max_daily_loss,
            auto_close_positions=True
        )

        # KELLY CRITERION: Initialize for optimal position sizing
        use_kelly = parameters.get('use_kelly_criterion', False)
        if use_kelly:
            kelly_fraction = parameters.get('kelly_fraction', 0.5)  # Half Kelly by default
            self.kelly = KellyCriterion(
                kelly_fraction=kelly_fraction,
                min_trades_required=parameters.get('kelly_min_trades', 30),
                max_position_size=parameters.get('max_position_size', 0.20),
                min_position_size=parameters.get('min_position_size', 0.01),
                lookback_trades=parameters.get('kelly_lookback', 50)
            )
            self.logger.info(f"✅ Kelly Criterion enabled: {kelly_fraction} Kelly fraction")
        else:
            self.kelly = None

        # Track closed positions for Kelly Criterion
        self.closed_positions = {}  # {symbol: {'entry_price': float, 'entry_time': datetime}}

        # VOLATILITY REGIME: Initialize for adaptive risk management
        use_volatility_regime = parameters.get('use_volatility_regime', False)
        if use_volatility_regime:
            self.volatility_regime = None  # Initialized in async initialize()
            self.logger.info("✅ Volatility Regime Detection enabled")
        else:
            self.volatility_regime = None

        # STREAK SIZING: Initialize for dynamic position sizing based on recent performance
        use_streak_sizing = parameters.get('use_streak_sizing', False)
        if use_streak_sizing:
            self.streak_sizer = StreakSizer(
                lookback_trades=parameters.get('streak_lookback', 10),
                hot_streak_threshold=parameters.get('hot_streak_threshold', 7),
                cold_streak_threshold=parameters.get('cold_streak_threshold', 3),
                hot_multiplier=parameters.get('hot_multiplier', 1.2),
                cold_multiplier=parameters.get('cold_multiplier', 0.7),
                reset_after_trades=parameters.get('streak_reset_after', 5)
            )
            self.logger.info(f"✅ Streak-based position sizing enabled: lookback={parameters.get('streak_lookback', 10)} trades")
        else:
            self.streak_sizer = None

        # MULTI-TIMEFRAME ANALYSIS: Initialize for trend confirmation across timeframes
        use_multi_timeframe = parameters.get('use_multi_timeframe', False)
        if use_multi_timeframe:
            self.multi_timeframe = None  # Initialized in async initialize()
            self.mtf_min_confidence = parameters.get('mtf_min_confidence', 0.70)
            self.mtf_require_daily = parameters.get('mtf_require_daily_alignment', True)
            self.logger.info(f"✅ Multi-timeframe analysis enabled: min_confidence={self.mtf_min_confidence:.0%}")
        else:
            self.multi_timeframe = None

    async def initialize(self, **kwargs):
        """Initialize strategy parameters."""
        try:
            # Update parameters
            self.parameters.update(kwargs)

            # Set up strategy parameters
            self.interval = self.parameters.get('interval', 60)
            self.symbols = self.parameters.get('symbols', [])

            # Initialize any other strategy-specific parameters
            await self._initialize_parameters()

            # CRITICAL SAFETY: Initialize circuit breaker with broker
            if self.broker:
                await self.circuit_breaker.initialize(self.broker)
                self.logger.info(f"✅ Circuit breaker armed: max daily loss = {self.circuit_breaker.max_daily_loss:.1%}")

            # VOLATILITY REGIME: Initialize detector with broker
            if self.parameters.get('use_volatility_regime', False) and self.broker:
                self.volatility_regime = VolatilityRegimeDetector(self.broker)
                regime, adjustments = await self.volatility_regime.get_current_regime()
                self.logger.info(
                    f"✅ Volatility regime detector initialized: "
                    f"{regime.upper()} (Position: {adjustments['pos_mult']:.1f}x, "
                    f"Stop: {adjustments['stop_mult']:.1f}x)"
                )

            # MULTI-TIMEFRAME ANALYSIS: Initialize analyzer with broker
            if self.parameters.get('use_multi_timeframe', False) and self.broker:
                self.multi_timeframe = MultiTimeframeAnalyzer(self.broker)
                self.logger.info(
                    f"✅ Multi-timeframe analyzer initialized: "
                    f"min_confidence={self.mtf_min_confidence:.0%}, "
                    f"require_daily_alignment={self.mtf_require_daily}"
                )

            return True

        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}", exc_info=True)
            return False

    async def _initialize_parameters(self):
        """Initialize strategy-specific parameters. Override in subclass."""
        self.sentiment_threshold = self.parameters.get('sentiment_threshold', 0.6)
        self.position_size = self.parameters.get('position_size', 0.1)
        self.max_position_size = self.parameters.get('max_position_size', 0.05)  # SAFETY: 5% max per position
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.05)
        self.portfolio_risk_limit = self.parameters.get('portfolio_risk_limit', 0.02)
        self.position_risk_limit = self.parameters.get('position_risk_limit', 0.01)
        self.max_correlation = self.parameters.get('max_correlation', 0.7)
        self.var_confidence = self.parameters.get('var_confidence', 0.95)
        self.price_history_window = self.parameters.get('price_history_window', 30)
        self.volatility_threshold = self.parameters.get('volatility_threshold', 0.4)
        self.var_threshold = self.parameters.get('var_threshold', 0.03)
        self.es_threshold = self.parameters.get('es_threshold', 0.04)
        self.drawdown_threshold = self.parameters.get('drawdown_threshold', 0.3)


    async def on_trading_iteration(self):
        """Main trading logic. Must be implemented by subclasses."""
        raise NotImplementedError

    def before_market_opens(self):
        """Actions to take before market opens."""
        pass

    def before_starting(self):
        """Initialize strategy before starting."""
        pass

    def after_market_closes(self):
        """Actions to take after market closes."""
        pass

    def on_abrupt_closing(self):
        """Handle abrupt closings."""
        pass

    def trace_stats(self, context, snapshot_before):
        """Record strategy statistics."""
        pass

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

    async def check_trading_allowed(self) -> bool:
        """
        CRITICAL SAFETY: Check if trading is allowed (circuit breaker not triggered).

        ALL strategies must call this before executing any trades.

        Returns:
            True if trading is allowed, False if halted

        Example:
            if not await self.check_trading_allowed():
                logger.warning("Trading halted by circuit breaker")
                return
        """
        is_halted = await self.circuit_breaker.check_and_halt()
        return not is_halted

    async def enforce_position_size_limit(self, symbol, desired_position_value, current_price):
        """
        CRITICAL SAFETY: Enforce maximum position size limit.

        Prevents over-concentration in a single position which could lead to
        catastrophic losses. Default limit is 5% of portfolio value.

        Args:
            symbol: Stock symbol
            desired_position_value: Dollar value of desired position
            current_price: Current stock price

        Returns:
            Tuple of (capped_position_value, capped_quantity) that respects limits

        Raises:
            ValueError: If account information cannot be retrieved
        """
        # P0 FIX: Validate current_price to prevent division by zero
        if not current_price or current_price <= 0:
            self.logger.error(
                f"SAFETY: Invalid current_price for {symbol}: {current_price}. "
                "Returning 0 to prevent division by zero."
            )
            return 0, 0

        try:
            # Get current account value
            account = await self.broker.get_account()
            account_value = float(account.equity)

            # Calculate maximum allowed position value
            max_position_value = account_value * self.max_position_size

            # Check if desired position exceeds limit
            if desired_position_value > max_position_value:
                self.logger.warning(
                    f"POSITION SIZE LIMIT ENFORCED for {symbol}: "
                    f"Requested ${desired_position_value:,.2f} exceeds "
                    f"max ${max_position_value:,.2f} ({self.max_position_size:.1%} of ${account_value:,.2f})"
                )
                capped_value = max_position_value
            else:
                capped_value = desired_position_value

            # Calculate capped quantity
            capped_quantity = capped_value / current_price

            self.logger.debug(
                f"Position size check for {symbol}: "
                f"${capped_value:,.2f} ({capped_quantity:.2f} shares) "
                f"= {(capped_value/account_value):.1%} of portfolio"
            )

            return capped_value, capped_quantity

        except Exception as e:
            self.logger.error(f"Error enforcing position size limit for {symbol}: {e}")
            # FAIL SAFE: Return 0 to prevent trading on error
            return 0, 0

    async def calculate_kelly_position_size(self, symbol, current_price):
        """
        Calculate optimal position size using Kelly Criterion.

        If Kelly is enabled and we have sufficient trade history, uses Kelly formula
        for optimal position sizing. Otherwise falls back to fixed position_size parameter.

        Args:
            symbol: Stock symbol
            current_price: Current stock price

        Returns:
            Tuple of (position_value, position_fraction, quantity)
        """
        # P0 FIX: Validate current_price to prevent division by zero
        if not current_price or current_price <= 0:
            self.logger.error(
                f"SAFETY: Invalid current_price for {symbol}: {current_price}. "
                "Returning 0 position size to prevent division by zero."
            )
            return 0, 0, 0

        try:
            # Get current account value
            account = await self.broker.get_account()
            account_value = float(account.equity)

            # If Kelly not enabled, use fixed position sizing
            if not self.kelly:
                position_fraction = self.position_size
                position_value = account_value * position_fraction
                quantity = position_value / current_price
                self.logger.debug(
                    f"Fixed position sizing: {position_fraction:.1%} = ${position_value:,.2f}"
                )
                return position_value, position_fraction, quantity

            # Use Kelly Criterion for optimal sizing
            position_value, position_fraction = self.kelly.calculate_position_size(
                current_capital=account_value,
                current_price=current_price
            )

            quantity = position_value / current_price

            self.logger.info(
                f"📊 Kelly position size for {symbol}: "
                f"{position_fraction:.1%} = ${position_value:,.2f} ({quantity:.2f} shares) "
                f"[Win rate: {self.kelly.win_rate:.1%}, Profit factor: {self.kelly.profit_factor:.2f}]"
            )

            return position_value, position_fraction, quantity

        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}")
            # P0 FIX: Safe fallback - return zeros if we can't calculate
            # account_value may not be defined if error occurred early
            return 0, 0, 0

    def track_position_entry(self, symbol, entry_price, entry_time=None):
        """
        Track position entry for Kelly Criterion trade recording.

        Call this when opening a position. Will be used to calculate P/L when position closes.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            entry_time: Entry timestamp (defaults to now)
        """
        if entry_time is None:
            entry_time = datetime.now()

        self.closed_positions[symbol] = {
            'entry_price': entry_price,
            'entry_time': entry_time
        }

        self.logger.debug(f"Tracking entry for {symbol} at ${entry_price:.2f}")

    def record_completed_trade(self, symbol, exit_price, exit_time, quantity, side='long'):
        """
        Record a completed trade for Kelly Criterion analysis.

        Call this when closing a position. Calculates P/L and adds to Kelly trade history.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            quantity: Number of shares traded
            side: 'long' or 'short'
        """
        if not self.kelly:
            return  # Kelly not enabled

        # Check if we tracked the entry
        if symbol not in self.closed_positions:
            self.logger.warning(f"No entry tracked for {symbol}, cannot record trade for Kelly")
            return

        entry_info = self.closed_positions[symbol]
        entry_price = entry_info['entry_price']
        entry_time = entry_info['entry_time']

        # Calculate P/L
        if side == 'long':
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price

        is_winner = pnl > 0

        # Create Trade object
        trade = Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_winner=is_winner
        )

        # Add to Kelly history
        self.kelly.add_trade(trade)

        # Add to streak sizer history
        if self.streak_sizer:
            self.streak_sizer.record_trade(is_winner=is_winner, pnl_pct=pnl_pct, symbol=symbol)

        self.logger.info(
            f"📈 Trade recorded: {symbol} {side.upper()} "
            f"{'WIN' if is_winner else 'LOSS'} {pnl_pct:+.2%} "
            f"(Total trades: {len(self.kelly.trades)})"
        )

        # Remove from tracking
        del self.closed_positions[symbol]

    async def apply_volatility_adjustments(self, base_position_size: float, base_stop_loss: float):
        """
        Apply volatility regime adjustments to position sizing and stop-loss.

        If volatility regime detection is enabled, adjusts parameters based on current market conditions.
        Otherwise, returns base values unchanged.

        Args:
            base_position_size: Base position size (e.g., 0.10 for 10%)
            base_stop_loss: Base stop-loss (e.g., 0.03 for 3%)

        Returns:
            Tuple of (adjusted_position_size, adjusted_stop_loss, regime_name)
        """
        if not self.volatility_regime:
            return base_position_size, base_stop_loss, 'normal'

        try:
            # Get current regime
            regime, adjustments = await self.volatility_regime.get_current_regime()

            # Apply adjustments
            adjusted_position_size = self.volatility_regime.adjust_position_size(
                base_position_size,
                adjustments['pos_mult']
            )

            adjusted_stop_loss = self.volatility_regime.adjust_stop_loss(
                base_stop_loss,
                adjustments['stop_mult']
            )

            self.logger.debug(
                f"Volatility adjustments ({regime.upper()}): "
                f"Position: {base_position_size:.1%} → {adjusted_position_size:.1%}, "
                f"Stop: {base_stop_loss:.1%} → {adjusted_stop_loss:.1%}"
            )

            return adjusted_position_size, adjusted_stop_loss, regime

        except Exception as e:
            self.logger.error(f"Error applying volatility adjustments: {e}", exc_info=True)
            return base_position_size, base_stop_loss, 'normal'

    def apply_streak_adjustments(self, base_position_size: float) -> float:
        """
        Apply streak-based adjustments to position sizing.

        If streak sizing is enabled, adjusts position size based on recent win/loss performance.
        Otherwise, returns base value unchanged.

        Args:
            base_position_size: Base position size (e.g., 0.10 for 10%)

        Returns:
            Adjusted position size
        """
        if not self.streak_sizer:
            return base_position_size

        try:
            # Get adjusted size based on streak
            adjusted_position_size = self.streak_sizer.adjust_for_streak(base_position_size)

            if adjusted_position_size != base_position_size:
                self.logger.debug(
                    f"Streak adjustments ({self.streak_sizer.current_streak.upper()}): "
                    f"Position: {base_position_size:.1%} → {adjusted_position_size:.1%}"
                )

            return adjusted_position_size

        except Exception as e:
            self.logger.error(f"Error applying streak adjustments: {e}", exc_info=True)
            return base_position_size

    async def check_multi_timeframe_signal(self, symbol: str) -> Optional[str]:
        """
        Check multi-timeframe analysis before entering a trade.

        Professional standard: ALL timeframes should align before entering trades.
        This dramatically reduces false signals and improves win rate.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            'buy', 'sell', or None (skip trade)

        Usage in strategies:
            # In analyze_symbol() or execute_trade():
            if self.multi_timeframe:
                mtf_signal = await self.check_multi_timeframe_signal(symbol)
                if not mtf_signal:
                    return 'neutral'  # Skip trade
                # mtf_signal is 'buy' or 'sell', proceed with trade
        """
        if not self.multi_timeframe:
            # Multi-timeframe not enabled, allow trade
            return None  # Means "no opinion", let strategy decide

        try:
            analysis = await self.multi_timeframe.analyze(
                symbol,
                min_confidence=self.mtf_min_confidence,
                require_daily_alignment=self.mtf_require_daily
            )

            if not analysis:
                self.logger.warning(f"Multi-timeframe analysis failed for {symbol}")
                return None  # Skip trade on analysis failure

            if analysis['should_enter']:
                self.logger.info(
                    f"✅ Multi-timeframe CONFIRMS {analysis['signal'].upper()} signal for {symbol} "
                    f"(Confidence: {analysis['confidence']:.0%})"
                )
                return analysis['signal']  # 'buy' or 'sell'
            else:
                self.logger.info(
                    f"⏭️  Multi-timeframe REJECTS trade for {symbol} "
                    f"(Confidence: {analysis['confidence']:.0%}, "
                    f"Signal: {analysis['signal']})"
                )
                return None  # Skip trade

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe check for {symbol}: {e}", exc_info=True)
            return None  # Skip trade on error

    async def is_short_position(self, symbol):
        """
        Check if we currently have a short position in a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if we have a short position (negative quantity), False otherwise
        """
        try:
            positions = await self.broker.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if position:
                qty = float(position.qty)
                return qty < 0

            return False

        except Exception as e:
            self.logger.error(f"Error checking short position for {symbol}: {e}")
            return False

    async def get_position_pnl(self, symbol):
        """
        Get current P/L for a position (works for both long and short).

        Args:
            symbol: Stock symbol

        Returns:
            Dict with unrealized_pl (dollar amount) and unrealized_plpc (percentage)
            or None if no position exists
        """
        try:
            positions = await self.broker.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if position:
                return {
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'is_short': float(position.qty) < 0
                }

            return None

        except Exception as e:
            self.logger.error(f"Error getting position P/L for {symbol}: {e}")
            return None

    async def cleanup(self):
        """Cleanup resources."""
        self.running = False
        tasks = [t for t in self.tasks if not t.done()]
        if tasks:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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
            raise NotImplementedError("Backtesting should be done via BacktestEngine, not directly on strategies")
        except Exception as e:
            logger.error(f"Error in backtesting {self.name}: {e}")
            raise
        finally:
            await self.cleanup()

    def _legacy_initialize(self, symbols=None, cash_at_risk=0.5, max_positions=3,
                           stop_loss_pct=0.05, take_profit_pct=0.20, max_drawdown=0.15):
        """Legacy synchronous initialize for backward compatibility with old scripts.
        DEPRECATED: Use async initialize() instead."""
        self.symbols = symbols or []
        self.cash_at_risk = cash_at_risk
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.positions_dict = {}

        # Performance tracking
        self.trades_made = 0
        self.successful_trades = 0
        self.total_profit_loss = 0
        self.peak_portfolio_value = self.portfolio_value
        self.current_drawdown = 0

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

    def create_order(self, symbol, quantity, side, type="market", limit_price=None, stop_price=None):
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

    def check_risk_limits(self):
        """Check if any risk limits have been breached."""
        try:
            current_value = self.portfolio_value
            self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
            self.current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value

            if self.current_drawdown > self.max_drawdown:
                self.logger.warning(f"Maximum drawdown limit reached: {self.current_drawdown:.2%}")
                return False

            current_positions = len(self.get_positions())
            if current_positions >= self.max_positions:
                self.logger.warning(f"Maximum positions limit reached: {current_positions}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in risk limit check: {str(e)}")
            return False

    def position_sizing(self, symbol, strategy_type="stock"):
        """Calculate position size using Kelly Criterion and volatility."""
        try:
            cash = self.get_cash()
            last_price = self.get_last_price(symbol)

            # Calculate historical volatility
            historical_data = self.get_historical_prices(symbol, 30)
            returns = np.diff(np.log(historical_data))
            volatility = np.std(returns) * np.sqrt(252)

            # Adjust position size based on volatility
            volatility_scalar = 1 / (1 + volatility)

            # Apply Kelly Criterion with safety factor
            win_rate = self.successful_trades / max(1, self.trades_made)
            kelly_fraction = max(0.0, (win_rate - (1 - win_rate)) / 1)
            safe_kelly = kelly_fraction * 0.5  # Half-Kelly for safety

            # Adjust sizing based on strategy type
            if strategy_type == "option":
                # More conservative sizing for options
                safe_kelly *= 0.5

            risk_adjusted_cash = cash * self.cash_at_risk * safe_kelly * volatility_scalar
            quantity = round(risk_adjusted_cash / last_price, 0)

            self.logger.info(f"Position sizing for {symbol} - Cash: {cash}, Quantity: {quantity}, Kelly: {safe_kelly:.2f}")
            return cash, last_price, quantity

        except Exception as e:
            self.logger.error(f"Error in position sizing for {symbol}: {str(e)}")
            return cash, last_price, 0
    async def _update_stop_loss(self, position):
        """Update the stop-loss level for a position based on volatility."""
        try:
            symbol = position.symbol
            current_price = position.current_price  # Assuming Position object has current_price
            volatility = self._calculate_volatility(symbol)

            # Example: Set stop-loss at 2 standard deviations below the current price
            stop_loss = current_price - (2 * volatility * current_price)

            #TODO: compare to the parameter and take the greater of the two.

            self.logger.info(f"Updating stop-loss for {symbol} to {stop_loss:.2f}")

            #TODO: Implement logic to update the stop-loss order with the broker

        except Exception as e:
            self.logger.error(f"Error updating stop-loss for {symbol}: {e}", exc_info=True)

    def _calculate_volatility(self, symbol):
        """Calculate the historical volatility for a symbol."""
        try:
            # Assuming self.price_history is available and populated by the strategy
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.price_history_window:
                self.logger.warning(f"Insufficient price history for {symbol} to calculate volatility")
                return 0  # Or some default value

            prices = np.array(self.price_history[symbol])
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualize
            return volatility

        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}", exc_info=True)
            return 0  # Or some default value

    def update_performance_metrics(self, trade_result, symbol):
        """Update performance tracking metrics after each trade."""
        try:
            self.trades_made += 1
            if trade_result > 0:
                self.successful_trades += 1
            self.total_profit_loss += trade_result

            win_rate = self.successful_trades / self.trades_made
            avg_profit_loss = self.total_profit_loss / self.trades_made

            self.logger.info(f"Performance metrics for {symbol} - Win rate: {win_rate:.2%}, Avg P/L: {avg_profit_loss:.2f}")

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    async def shutdown(self):
        """Shutdown the strategy."""
        self._shutdown_event.set()
        await self.cleanup()
