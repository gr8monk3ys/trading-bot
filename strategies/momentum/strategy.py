"""
MomentumStrategy core module.

Holds the concrete ``MomentumStrategy`` class: parameter defaults,
initialization, the ``on_bar`` event handler, signal-execution dispatch
(buy / short / sell), position-value math, state import/export, the
``analyze_symbol`` / ``execute_trade`` / ``generate_signals`` / ``get_orders``
public API, and small utility helpers (crypto-symbol classifier, previous
close lookup, position tracking cleanup, cached positions).

The TA-Lib indicator pipeline and the signal-generation / exit-condition
checks live in ``strategies/momentum/indicators.py`` and
``strategies/momentum/signals.py`` and are mixed in to keep this module
focused on lifecycle / order plumbing.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta

from brokers.order_builder import OrderBuilder
from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
from utils.multi_timeframe import MultiTimeframeAnalyzer

from strategies.momentum.indicators import MomentumIndicatorsMixin
from strategies.momentum.signals import MomentumSignalsMixin

logger = logging.getLogger(__name__)


class MomentumStrategy(MomentumIndicatorsMixin, MomentumSignalsMixin, BaseStrategy):
    """
    Momentum-based trading strategy that uses technical indicators to identify
    trend strength and momentum for making buy/sell decisions. This strategy combines
    multiple momentum indicators including MACD, RSI, and ADX to generate signals.
    """

    NAME = "MomentumStrategy"

    def default_parameters(self):
        """
        Return default parameters for the strategy.

        CONSERVATIVE DEFAULTS (reduced overfitting risk):
        Features are disabled by default until validated with sufficient trades.

        VALIDATED FEATURES (enabled by default):
        - RSI-14 standard mode (proven, well-studied)
        - MACD confirmation (standard momentum indicator)
        - ADX trend strength (standard)
        - Trailing stops (captures extended moves)
        - Volatility regime detection (adaptive risk)

        EXPERIMENTAL FEATURES (disabled by default, enable after 100+ trades):
        - Kelly Criterion (requires win rate/payoff data)
        - Streak sizing (requires trade history)
        - Multi-timeframe (not useful for daily data)
        - RSI-2 aggressive mode (high win rate but needs validation)
        - Short selling (requires separate validation)
        - Bollinger Band filter (mean reversion overlay)

        To enable experimental features after validation:
            strategy = MomentumStrategy(broker, symbols, config={
                "use_kelly_criterion": True,
                "use_streak_sizing": True,
                ...
            })
        """
        return {
            # === CORE PARAMETERS (VALIDATED) ===
            "position_size": 0.05,  # 5% per position (conservative start)
            "max_positions": 5,  # Maximum concurrent positions
            "max_portfolio_risk": 0.02,  # 2% max portfolio risk
            "stop_loss": 0.03,  # 3% stop loss
            "take_profit": 0.05,  # 5% take profit (if not using trailing)
            # === MOMENTUM INDICATORS (VALIDATED - CORE STRATEGY) ===
            # Using standard RSI-14 by default for proven reliability
            "rsi_mode": "standard",  # Use proven RSI-14 (enable 'aggressive' after validation)
            "rsi_period": 14,  # Standard RSI period
            "rsi_overbought": 70,  # Standard overbought
            "rsi_oversold": 30,  # Standard oversold
            "macd_fast_period": 12,  # Standard MACD
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "adx_period": 14,  # Standard ADX
            "adx_threshold": 25,  # ADX > 25 = trending
            # === VOLUME PARAMETERS (VALIDATED) ===
            "volume_ma_period": 20,
            "volume_factor": 1.5,
            # === PRICE MA PARAMETERS (VALIDATED) ===
            "fast_ma_period": 10,
            "medium_ma_period": 20,
            "slow_ma_period": 50,
            # === VOLATILITY PARAMETERS (VALIDATED) ===
            "atr_period": 14,
            "atr_multiplier": 2.0,
            # === RISK MANAGEMENT (VALIDATED) ===
            "max_correlation": 0.7,
            "max_sector_exposure": 0.3,
            # === TRAILING STOPS (VALIDATED - let winners run) ===
            "use_trailing_stop": True,  # ENABLED - proven to capture extended moves
            "trailing_stop_pct": 0.02,  # 2% trailing distance
            "trailing_activation_pct": 0.02,  # Activate after 2% profit
            # === VOLATILITY REGIME (VALIDATED - adaptive risk) ===
            "use_volatility_regime": True,  # ENABLED - adjusts to market conditions
            # === EXPERIMENTAL FEATURES (DISABLED until validated) ===
            # These require 100+ trades before enabling to avoid overfitting
            "use_bollinger_filter": False,  # DISABLED - enable after validation
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_buy_threshold": 0.3,
            "bb_sell_threshold": 0.7,
            "use_multi_timeframe": False,  # DISABLED - not useful for daily data
            "mtf_timeframes": ["5Min", "15Min", "1Hour"],
            "mtf_require_alignment": True,
            "enable_short_selling": False,  # DISABLED - requires separate validation
            # In long-only spot crypto sessions, allow moderately bullish setups
            # (score >= 1) to avoid remaining neutral for extended bearish regimes.
            "crypto_long_only_relaxed_entry": True,
            "crypto_long_only_buy_score_threshold": 1.0,
            # Controlled dip-buy mode for long-only crypto:
            # require oversold RSI + improving MACD histogram + price rebound.
            "crypto_long_only_dip_buy_enabled": True,
            "crypto_long_only_dip_rsi_max": 35.0,
            "crypto_long_only_dip_min_macd_hist_delta": 0.02,
            "crypto_long_only_dip_min_rebound_pct": 0.001,
            "short_position_size": 0.08,
            "short_stop_loss": 0.04,
            "use_kelly_criterion": False,  # DISABLED - requires 100+ trades for win rate data
            "kelly_fraction": 0.5,
            "kelly_min_trades": 100,  # Increased from 30 to 100 for statistical significance
            "kelly_lookback": 50,
            "use_streak_sizing": False,  # DISABLED - requires trade history validation
        }

    async def initialize(self, **kwargs):
        """Initialize the momentum strategy."""
        try:
            # Initialize the base strategy
            await super().initialize(**kwargs)

            # Set strategy-specific parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Extract parameters
            self.position_size = self.parameters["position_size"]
            self.max_positions = self.parameters["max_positions"]
            self.stop_loss = self.parameters["stop_loss"]
            self.take_profit = self.parameters["take_profit"]

            # Technical indicator parameters
            # RSI-2 Optimization: Apply aggressive settings if mode is 'aggressive'
            # Research: RSI-2 with extreme thresholds (10/90) achieves 91% win rate
            self.rsi_mode = self.parameters.get("rsi_mode", "standard")

            if self.rsi_mode == "aggressive":
                # RSI-2 Strategy (Larry Connors style)
                self.rsi_period = 2  # Very short period for rapid signals
                self.rsi_overbought = 90  # Extreme overbought for exits
                self.rsi_oversold = 10  # Extreme oversold for entries
                logger.info("✅ RSI-2 AGGRESSIVE mode enabled (period=2, thresholds: 10/90)")
                logger.info("   Expected improvement: ~91% win rate (vs 55% for RSI-14)")
            else:
                # Standard RSI-14
                self.rsi_period = self.parameters["rsi_period"]
                self.rsi_overbought = self.parameters["rsi_overbought"]
                self.rsi_oversold = self.parameters["rsi_oversold"]
                logger.info(
                    f"RSI standard mode (period={self.rsi_period}, thresholds: {self.rsi_oversold}/{self.rsi_overbought})"
                )
            self.macd_fast = self.parameters["macd_fast_period"]
            self.macd_slow = self.parameters["macd_slow_period"]
            self.macd_signal = self.parameters["macd_signal_period"]
            self.adx_period = self.parameters["adx_period"]
            self.adx_threshold = self.parameters["adx_threshold"]
            self.volume_ma_period = self.parameters["volume_ma_period"]
            self.volume_factor = self.parameters["volume_factor"]
            self.fast_ma = self.parameters["fast_ma_period"]
            self.medium_ma = self.parameters["medium_ma_period"]
            self.slow_ma = self.parameters["slow_ma_period"]
            self.atr_period = self.parameters["atr_period"]
            self.atr_multiplier = self.parameters["atr_multiplier"]

            # Initialize tracking dictionaries
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.signals = dict.fromkeys(self.symbols, "neutral")
            self.last_signal_time = dict.fromkeys(self.symbols)
            self.stop_prices = {}
            self._last_macd_hist = {}

            # Trailing stop parameters (NEW FEATURE - let winners run)
            self.use_trailing_stop = self.parameters.get("use_trailing_stop", True)
            self.trailing_stop_pct = self.parameters.get("trailing_stop_pct", 0.02)  # 2% trail
            self.trailing_activation_pct = self.parameters.get(
                "trailing_activation_pct", 0.02
            )  # Activate after 2% profit
            self.peak_prices = {}  # Track highest price since entry for trailing stops
            self.entry_prices = {}  # Track entry prices for profit calculation

            if self.use_trailing_stop:
                logger.info(
                    f"Trailing stops enabled: {self.trailing_stop_pct:.1%} trail, activates at {self.trailing_activation_pct:.1%} profit"
                )

            # Short selling parameters (NEW FEATURE)
            # Default matches default_parameters() which has True for maximum profit mode
            self.enable_short_selling = self.parameters.get("enable_short_selling", True)
            self.short_position_size = self.parameters.get("short_position_size", 0.08)
            self.short_stop_loss = self.parameters.get("short_stop_loss", 0.04)
            self.crypto_long_only_relaxed_entry = bool(
                self.parameters.get("crypto_long_only_relaxed_entry", True)
            )
            self.crypto_long_only_buy_score_threshold = float(
                self.parameters.get("crypto_long_only_buy_score_threshold", 1.0)
            )
            self.crypto_long_only_dip_buy_enabled = bool(
                self.parameters.get("crypto_long_only_dip_buy_enabled", True)
            )
            self.crypto_long_only_dip_rsi_max = float(
                self.parameters.get("crypto_long_only_dip_rsi_max", 35.0)
            )
            self.crypto_long_only_dip_min_macd_hist_delta = float(
                self.parameters.get("crypto_long_only_dip_min_macd_hist_delta", 0.02)
            )
            self.crypto_long_only_dip_min_rebound_pct = float(
                self.parameters.get("crypto_long_only_dip_min_rebound_pct", 0.001)
            )

            if self.enable_short_selling:
                logger.info("✅ Short selling enabled - can profit from bear markets!")
            elif self.crypto_long_only_relaxed_entry:
                logger.info(
                    "Crypto long-only relaxed entry enabled: buy threshold score >= %.2f",
                    self.crypto_long_only_buy_score_threshold,
                )
            if not self.enable_short_selling and self.crypto_long_only_dip_buy_enabled:
                logger.info(
                    "Crypto long-only dip-buy enabled: RSI<=%.1f, MACD-hist delta>=%.3f, rebound>=%.2f%%",
                    self.crypto_long_only_dip_rsi_max,
                    self.crypto_long_only_dip_min_macd_hist_delta,
                    self.crypto_long_only_dip_min_rebound_pct * 100.0,
                )

            # Multi-timeframe analysis (NEW FEATURE)
            # Default matches default_parameters() which has True for maximum profit mode
            self.use_multi_timeframe = self.parameters.get("use_multi_timeframe", True)
            self.mtf_require_alignment = self.parameters.get("mtf_require_alignment", True)
            self.mtf_analyzer = None

            if self.use_multi_timeframe:
                mtf_timeframes = self.parameters.get("mtf_timeframes", ["5Min", "15Min", "1Hour"])
                self.mtf_analyzer = MultiTimeframeAnalyzer(
                    timeframes=mtf_timeframes, history_length=200
                )
                logger.info(f"✅ Multi-timeframe filtering enabled: {', '.join(mtf_timeframes)}")

            # Bollinger Band mean reversion filter (NEW FEATURE)
            self.use_bollinger_filter = self.parameters.get("use_bollinger_filter", True)
            self.bb_period = self.parameters.get("bb_period", 20)
            self.bb_std = self.parameters.get("bb_std", 2.0)
            self.bb_buy_threshold = self.parameters.get("bb_buy_threshold", 0.3)
            self.bb_sell_threshold = self.parameters.get("bb_sell_threshold", 0.7)

            if self.use_bollinger_filter:
                logger.info(
                    f"✅ Bollinger Band filter enabled (period={self.bb_period}, std={self.bb_std})"
                )
            self.target_prices = {}
            self.current_prices = {}

            # Performance optimization: Pre-calculate max history size for deque
            self.max_history = (
                max(
                    self.slow_ma,
                    self.rsi_period,
                    self.macd_slow + self.macd_signal,
                    self.adx_period,
                )
                + 10  # Extra buffer
            )

            # Performance optimization: Use deque with maxlen for O(1) append and auto-trimming
            # This avoids memory churn from list slicing
            self.price_history = {symbol: deque(maxlen=self.max_history) for symbol in self.symbols}

            # Performance optimization: Position caching to reduce API calls
            self._positions_cache = None
            self._positions_cache_time = None
            self._positions_cache_ttl = timedelta(seconds=1)

            # Risk manager initialization
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters["max_portfolio_risk"],
                max_position_risk=self.parameters.get("max_position_risk", 0.01),
                max_correlation=self.parameters["max_correlation"],
            )

            # Add strategy as subscriber to broker
            if hasattr(self.broker, "_add_subscriber"):
                self.broker._add_subscriber(self)

            logger.info(f"Initialized {self.NAME} with {len(self.symbols)} symbols")
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False

    async def export_state(self) -> dict:
        """Export lightweight state for restart recovery."""

        def _dt(v):
            return v.isoformat() if hasattr(v, "isoformat") else v

        return {
            "last_signal_time": {k: _dt(v) for k, v in self.last_signal_time.items() if v},
            "stop_prices": self.stop_prices,
            "target_prices": self.target_prices,
            "entry_prices": self.entry_prices,
            "peak_prices": self.peak_prices,
        }

    async def import_state(self, state: dict) -> None:
        """Restore lightweight state after restart."""
        from datetime import datetime

        def _parse_dt(v):
            return datetime.fromisoformat(v) if isinstance(v, str) else v

        self.stop_prices = state.get("stop_prices", {})
        self.target_prices = state.get("target_prices", {})
        self.entry_prices = state.get("entry_prices", {})
        self.peak_prices = state.get("peak_prices", {})
        self.last_signal_time = {
            k: _parse_dt(v) for k, v in state.get("last_signal_time", {}).items()
        }

    async def on_bar(
        self, symbol, open_price, high_price, low_price, close_price, volume, timestamp
    ):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return

            # Store latest price
            self.current_prices[symbol] = close_price

            # Update multi-timeframe analyzer (if enabled)
            if self.use_multi_timeframe and self.mtf_analyzer:
                await self.mtf_analyzer.update(symbol, timestamp, close_price, volume)

            # Update price history (deque auto-trims to max_history via maxlen)
            # Performance optimization: O(1) append, no list slicing needed
            self.price_history[symbol].append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

            # Update technical indicators
            await self._update_indicators(symbol)

            # Check for signals
            signal = await self._generate_signal(symbol)
            self.signals[symbol] = signal

            # Execute trades if needed
            # Note: 'buy', 'short', and 'sell' are all valid signals
            if signal != "neutral":
                await self._execute_signal(symbol, signal)

            # Check stop losses and take profits for existing positions
            await self._check_exit_conditions(symbol)

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    @staticmethod
    def _is_crypto_symbol(symbol: str) -> bool:
        """Best-effort symbol classification for crypto pairs."""
        if not symbol:
            return False
        normalized = str(symbol).upper().replace("-", "/")
        if "/" in normalized:
            base, quote = normalized.split("/", 1)
            return bool(base) and quote in {"USD", "USDT", "USDC", "BTC", "ETH"}
        return normalized.endswith(("USD", "USDT", "USDC")) and len(normalized) >= 6

    def _get_previous_close(self, symbol: str):
        """Return previous bar close from in-memory history when available."""
        history = getattr(self, "price_history", {}).get(symbol)
        if not history or len(history) < 2:
            return None
        previous_bar = history[-2]
        if not isinstance(previous_bar, dict):
            return None
        previous_close = previous_bar.get("close")
        return float(previous_close) if previous_close is not None else None

    def _build_current_positions_dict(self, positions):
        """Build a dictionary of current positions with price history for risk analysis."""
        current_positions = {}
        for pos in positions:
            pos_symbol = pos.symbol
            if pos_symbol in self.price_history:
                price_history = self.price_history[pos_symbol]
                close_prices = [bar["close"] for bar in price_history]
                current_positions[pos_symbol] = {
                    "value": abs(float(pos.market_value)),
                    "price_history": close_prices,
                    "risk": None,
                }
        return current_positions

    async def _calculate_position_value(self, symbol, price, buying_power, is_short=False):
        """
        Calculate position value using Kelly criterion or fixed sizing.

        Returns:
            float: The calculated position value
        """
        use_kelly = self.parameters.get("use_kelly_criterion", True)

        if use_kelly and hasattr(self, "kelly") and self.kelly is not None:
            position_value, position_fraction, _ = await self.calculate_kelly_position_size(
                symbol, price
            )
            if is_short:
                # Apply short position reduction (shorts use 80% of Kelly size)
                position_value = position_value * 0.8
                logger.info(
                    f"📊 KELLY SHORT: {symbol} position = {position_fraction * 0.8:.1%} (${position_value:,.2f})"
                )
            else:
                logger.info(
                    f"📊 KELLY SIZING: {symbol} position = {position_fraction:.1%} (${position_value:,.2f})"
                )
        else:
            # Use fixed position sizing
            size_pct = self.short_position_size if is_short else self.position_size
            position_value = buying_power * size_pct

        return position_value

    async def _apply_risk_adjustments(self, symbol, position_value, positions):
        """Apply risk manager adjustments to position value."""
        current_positions = self._build_current_positions_dict(positions)

        if len(self.price_history[symbol]) > 20:
            close_prices = [bar["close"] for bar in self.price_history[symbol]]
            position_value = self.risk_manager.adjust_position_size(
                symbol, position_value, close_prices, current_positions
            )

        return position_value

    async def _execute_buy_signal(self, symbol, positions, buying_power, current_time):
        """Execute a buy signal for the given symbol."""
        if len(positions) >= self.max_positions:
            logger.info(f"Max positions reached ({self.max_positions}), skipping buy for {symbol}")
            return

        price = self.current_prices[symbol]
        position_value = await self._calculate_position_value(symbol, price, buying_power)
        position_value = await self._apply_risk_adjustments(symbol, position_value, positions)

        if position_value <= 0:
            logger.info(f"Risk manager rejected position for {symbol}")
            return

        # Enforce maximum position size limit
        position_value, quantity = await self.enforce_position_size_limit(
            symbol, position_value, price
        )

        if quantity < 0.01:
            logger.info(f"Position size too small for {symbol}, need at least 0.01 shares")
            return

        # Calculate take-profit and stop-loss levels
        take_profit_price = price * (1 + self.take_profit)
        stop_loss_price = price * (1 - self.stop_loss)

        if self._is_crypto_symbol(symbol):
            # Alpaca crypto does not support advanced order classes (e.g. bracket/OTOCO).
            logger.info(f"Creating market entry order for {symbol} (crypto):")
            logger.info(f"  Entry: ${price:.2f} x {quantity:.4f} units")
            logger.info(f"  Managed TP: ${take_profit_price:.2f} (+{self.take_profit:.1%})")
            logger.info(f"  Managed SL: ${stop_loss_price:.2f} (-{self.stop_loss:.1%})")
            order = OrderBuilder(symbol, "buy", quantity).market().gtc().build()
        else:
            logger.info(f"Creating bracket order for {symbol}:")
            logger.info(f"  Entry: ${price:.2f} x {quantity:.4f} shares")
            logger.info(f"  Take-profit: ${take_profit_price:.2f} (+{self.take_profit:.1%})")
            logger.info(f"  Stop-loss: ${stop_loss_price:.2f} (-{self.stop_loss:.1%})")
            order = (
                OrderBuilder(symbol, "buy", quantity)
                .market()
                .bracket(take_profit=take_profit_price, stop_loss=stop_loss_price)
                .gtc()
                .build()
            )

        result = await self.submit_entry_order(
            order,
            reason="momentum_entry",
            max_positions=self.max_positions,
        )

        if result and (not hasattr(result, "success") or result.success):
            logger.info(
                f"BUY bracket order submitted for {symbol}: {quantity:.4f} shares at ~${price:.2f}"
            )
            self.stop_prices[symbol] = stop_loss_price
            self.target_prices[symbol] = take_profit_price
            self.entry_prices[symbol] = price
            self.peak_prices[symbol] = price
            self.last_signal_time[symbol] = current_time

    async def _execute_short_signal(self, symbol, positions, buying_power, current_time):
        """Execute a short signal for the given symbol."""
        if len(positions) >= self.max_positions:
            logger.info(
                f"Max positions reached ({self.max_positions}), skipping short for {symbol}"
            )
            return

        price = self.current_prices[symbol]
        position_value = await self._calculate_position_value(
            symbol, price, buying_power, is_short=True
        )
        position_value = await self._apply_risk_adjustments(symbol, position_value, positions)

        if position_value <= 0:
            logger.info(f"Risk manager rejected SHORT position for {symbol}")
            return

        # Enforce maximum position size limit
        position_value, quantity = await self.enforce_position_size_limit(
            symbol, position_value, price
        )

        if quantity < 0.01:
            logger.info(f"Position size too small for {symbol}, need at least 0.01 shares")
            return

        # For shorts: profit when price DROPS, loss when price RISES
        take_profit_price = price * (1 - self.take_profit)
        stop_loss_price = price * (1 + self.short_stop_loss)

        if self._is_crypto_symbol(symbol):
            logger.info(f"🔻 Creating SHORT market order for {symbol} (crypto):")
            logger.info(f"  Entry: SELL ${price:.2f} x {quantity:.4f} units")
            logger.info(
                f"  Managed take-profit: BUY at ${take_profit_price:.2f} "
                f"(-{self.take_profit:.1%} price drop)"
            )
            logger.info(
                f"  Managed stop-loss: BUY at ${stop_loss_price:.2f} "
                f"(+{self.short_stop_loss:.1%} price rise)"
            )
            order = OrderBuilder(symbol, "sell", quantity).market().gtc().build()
        else:
            logger.info(f"🔻 Creating SHORT bracket order for {symbol}:")
            logger.info(f"  Entry: SELL ${price:.2f} x {quantity:.4f} shares (SHORT)")
            logger.info(
                f"  Take-profit: BUY at ${take_profit_price:.2f} (-{self.take_profit:.1%} price drop)"
            )
            logger.info(
                f"  Stop-loss: BUY at ${stop_loss_price:.2f} (+{self.short_stop_loss:.1%} price rise)"
            )
            order = (
                OrderBuilder(symbol, "sell", quantity)
                .market()
                .bracket(take_profit=take_profit_price, stop_loss=stop_loss_price)
                .gtc()
                .build()
            )

        result = await self.submit_entry_order(
            order,
            reason="momentum_short_entry",
            max_positions=self.max_positions,
        )

        if result and (not hasattr(result, "success") or result.success):
            logger.info(
                f"🔻 SHORT bracket order submitted for {symbol}: {quantity:.4f} shares at ~${price:.2f}"
            )
            logger.info(f"   (Will profit if price drops below ${take_profit_price:.2f})")
            self.stop_prices[symbol] = stop_loss_price
            self.target_prices[symbol] = take_profit_price
            self.entry_prices[symbol] = price
            self.peak_prices[symbol] = price
            self.last_signal_time[symbol] = current_time

    async def _execute_sell_signal(self, symbol, current_position, current_time):
        """Execute a sell signal to close an existing long position."""
        quantity = float(current_position.qty)
        price = self.current_prices[symbol]

        result = await self.submit_exit_order(
            symbol=symbol,
            qty=quantity,
            side="sell",
            reason="signal_exit",
        )

        if result:
            logger.info(f"SELL order submitted for {symbol}: {quantity} shares at ~${price:.2f}")
            self.stop_prices.pop(symbol, None)
            self.target_prices.pop(symbol, None)
            self.last_signal_time[symbol] = current_time

    async def _execute_signal(self, symbol, signal):
        """Execute a trading signal by dispatching to the appropriate handler."""
        try:
            # Check cooldown to avoid overtrading
            current_time = datetime.now()
            if (
                self.last_signal_time.get(symbol)
                and (current_time - self.last_signal_time[symbol]).total_seconds() < 3600
            ):
                return

            # Performance optimization: Fetch positions and account info in parallel
            positions, account = await asyncio.gather(
                self.broker.get_positions(), self.broker.get_account()
            )
            current_position = next((p for p in positions if p.symbol == symbol), None)
            buying_power = float(account.buying_power)

            # Dispatch to appropriate handler
            if signal == "buy" and not current_position:
                await self._execute_buy_signal(symbol, positions, buying_power, current_time)
            elif signal == "short" and not current_position and self.enable_short_selling:
                await self._execute_short_signal(symbol, positions, buying_power, current_time)
            elif signal == "sell" and current_position and float(current_position.qty) > 0:
                await self._execute_sell_signal(symbol, current_position, current_time)

        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}", exc_info=True)

    async def _get_cached_positions(self):
        """Get positions with 1-second cache to reduce API calls.

        Performance optimization: When checking exit conditions for multiple symbols,
        this prevents redundant API calls by caching position data for 1 second.
        """
        now = datetime.now()
        if (
            self._positions_cache is None
            or self._positions_cache_time is None
            or now - self._positions_cache_time > self._positions_cache_ttl
        ):
            self._positions_cache = await self.broker.get_positions()
            self._positions_cache_time = now
        return self._positions_cache

    def _cleanup_position_tracking(self, symbol: str):
        """Clean up all tracking data for a closed position."""
        for tracking_dict in [
            self.stop_prices,
            self.target_prices,
            self.entry_prices,
            self.peak_prices,
        ]:
            if symbol in tracking_dict:
                del tracking_dict[symbol]

    async def analyze_symbol(self, symbol):
        """Analyze a symbol and determine if we should trade it."""
        # This is already handled in _generate_signal
        return self.signals.get(symbol, "neutral")

    async def execute_trade(self, symbol, signal):
        """Execute a trade based on the signal."""
        # This is already handled in _execute_signal
        pass

    async def generate_signals(self):
        """Generate signals for all symbols (used in backtest mode)."""
        for symbol in self.symbols:
            if symbol in self.current_data:
                df = self.current_data[symbol]
                if len(df) < self.slow_ma:
                    continue

                # Extract price data and calculate indicators using shared method
                closes = df["close"].values
                highs = df["high"].values
                lows = df["low"].values
                volumes = df["volume"].values

                self.indicators[symbol] = self._calculate_indicators_from_arrays(
                    closes, highs, lows, volumes
                )

                # Generate signal
                signal = await self._generate_signal(symbol)
                self.signals[symbol] = signal

    def get_orders(self):
        """Get orders for backtest mode."""
        orders = []

        for symbol, signal in self.signals.items():
            if signal == "neutral":
                continue

            # Get current positions (for backtest)
            current_positions = getattr(self, "positions", {})
            has_position = symbol in current_positions

            # Current price
            price = self.indicators[symbol]["close"]
            if not price:
                continue

            # Buy signal
            if signal == "buy" and not has_position:
                # Calculate position size (simplified for backtest)
                capital = getattr(self, "capital", 100000)
                position_size = capital * self.position_size
                quantity = position_size / price

                # Allow fractional shares (minimum 0.01 shares)
                if quantity >= 0.01:
                    orders.append(
                        {
                            "symbol": symbol,
                            "quantity": quantity,  # Keep fractional quantity
                            "side": "buy",
                            "type": "market",
                        }
                    )

            # Sell signal
            elif signal == "sell" and has_position:
                position = current_positions[symbol]
                quantity = position.get("quantity", 0)

                if quantity > 0:
                    orders.append(
                        {"symbol": symbol, "quantity": quantity, "side": "sell", "type": "market"}
                    )

        return orders
