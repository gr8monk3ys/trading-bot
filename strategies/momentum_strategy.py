import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import talib

from brokers.order_builder import OrderBuilder
from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
from utils.multi_timeframe import MultiTimeframeAnalyzer

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
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

            if self.enable_short_selling:
                logger.info("✅ Short selling enabled - can profit from bear markets!")

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

    def _safe_last(self, arr):
        """Extract last value from array, returning None if empty or NaN."""
        if arr is None or len(arr) == 0:
            return None
        val = arr[-1]
        if isinstance(val, float) and np.isnan(val):
            return None
        return float(val) if not np.isnan(val) else None

    def _calculate_indicators_from_arrays(self, closes, highs, lows, volumes):
        """
        Calculate all technical indicators from price arrays.

        Args:
            closes: numpy array of close prices
            highs: numpy array of high prices
            lows: numpy array of low prices
            volumes: numpy array of volumes

        Returns:
            dict: Dictionary of calculated indicator values
        """
        # Calculate RSI
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)

        # Calculate MACD
        macd, signal, hist = talib.MACD(
            closes,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        # Calculate ADX
        adx = talib.ADX(highs, lows, closes, timeperiod=self.adx_period)

        # Calculate moving averages
        fast_ma = talib.SMA(closes, timeperiod=self.fast_ma)
        medium_ma = talib.SMA(closes, timeperiod=self.medium_ma)
        slow_ma = talib.SMA(closes, timeperiod=self.slow_ma)

        # Calculate volume moving average
        volume_ma = talib.SMA(volumes, timeperiod=self.volume_ma_period)

        # Calculate ATR for stop loss
        atr = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)

        # Calculate Bollinger Bands for mean reversion filter
        bb_upper, bb_middle, bb_lower = None, None, None
        bb_position = None

        if self.use_bollinger_filter and len(closes) >= self.bb_period:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                closes,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0,
            )

            # Calculate Bollinger Band position (0 = at lower band, 1 = at upper band)
            current_close = self._safe_last(closes)
            bb_lower_val = self._safe_last(bb_lower)
            bb_upper_val = self._safe_last(bb_upper)

            if current_close and bb_lower_val and bb_upper_val and bb_upper_val != bb_lower_val:
                bb_position = (current_close - bb_lower_val) / (bb_upper_val - bb_lower_val)

        return {
            "rsi": self._safe_last(rsi),
            "macd": self._safe_last(macd),
            "macd_signal": self._safe_last(signal),
            "macd_hist": self._safe_last(hist),
            "adx": self._safe_last(adx),
            "fast_ma": self._safe_last(fast_ma),
            "medium_ma": self._safe_last(medium_ma),
            "slow_ma": self._safe_last(slow_ma),
            "volume": self._safe_last(volumes),
            "volume_ma": self._safe_last(volume_ma),
            "atr": self._safe_last(atr),
            "close": self._safe_last(closes),
            "bb_upper": self._safe_last(bb_upper) if bb_upper is not None else None,
            "bb_middle": self._safe_last(bb_middle) if bb_middle is not None else None,
            "bb_lower": self._safe_last(bb_lower) if bb_lower is not None else None,
            "bb_position": bb_position,
        }

    async def _update_indicators(self, symbol):
        """Update technical indicators for a symbol."""
        try:
            # Ensure we have enough price history
            if len(self.price_history[symbol]) < self.slow_ma:
                return

            # Extract price data into arrays
            closes = np.array([bar["close"] for bar in self.price_history[symbol]])
            highs = np.array([bar["high"] for bar in self.price_history[symbol]])
            lows = np.array([bar["low"] for bar in self.price_history[symbol]])
            volumes = np.array([bar["volume"] for bar in self.price_history[symbol]])

            # Calculate and store indicators using shared method
            self.indicators[symbol] = self._calculate_indicators_from_arrays(
                closes, highs, lows, volumes
            )

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}", exc_info=True)

    async def _generate_signal(self, symbol):
        """Generate trading signal based on indicators."""
        try:
            # Check if indicators are available
            if not self.indicators.get(symbol) or self.indicators[symbol]["rsi"] is None:
                return "neutral"

            ind = self.indicators[symbol]

            # P2 Fix: Add null checks for all indicators
            # Get current indicator values with None handling
            rsi = ind.get("rsi")
            macd = ind.get("macd")
            macd_signal = ind.get("macd_signal")
            macd_hist = ind.get("macd_hist")
            adx = ind.get("adx")
            fast_ma = ind.get("fast_ma")
            medium_ma = ind.get("medium_ma")
            slow_ma = ind.get("slow_ma")
            volume = ind.get("volume")
            volume_ma = ind.get("volume_ma")

            # P2 Fix: Return neutral if any critical indicator is missing
            if any(
                v is None for v in [rsi, macd, macd_signal, macd_hist, fast_ma, medium_ma, slow_ma]
            ):
                logger.debug(f"{symbol}: Missing critical indicators, returning neutral")
                return "neutral"

            # Calculate strength factors
            momentum_score = 0

            # RSI factor (0-100)
            if rsi < self.rsi_oversold:
                momentum_score += 1  # Bullish
            elif rsi > self.rsi_overbought:
                momentum_score -= 1  # Bearish

            # MACD factor
            if macd > macd_signal and macd_hist > 0:
                momentum_score += 1  # Bullish
            elif macd < macd_signal and macd_hist < 0:
                momentum_score -= 1  # Bearish

            # ADX factor (trend strength)
            trend_strength = 0
            if adx > self.adx_threshold:
                trend_strength = 1  # Strong trend

            # Moving average setup
            ma_bullish = fast_ma > medium_ma > slow_ma
            ma_bearish = fast_ma < medium_ma < slow_ma

            if ma_bullish:
                momentum_score += 1
            elif ma_bearish:
                momentum_score -= 1

            # Volume confirmation
            volume_confirmation = volume > (volume_ma * self.volume_factor)

            # BOLLINGER BAND MEAN REVERSION FILTER (NEW FEATURE)
            # Research shows combining momentum with mean reversion achieves 73% win rate
            if self.use_bollinger_filter:
                bb_position = ind.get("bb_position")
                # P1 Fix: Removed unused bb_adjustment, kept momentum_score adjustments
                # P2 Fix: Added near-zero band width edge case check
                bb_width = ind.get("bb_upper", 0) - ind.get("bb_lower", 0)
                if (
                    bb_position is not None and bb_width > 0.001
                ):  # Avoid division issues with flat bands
                    # Apply adjustment to momentum score
                    if momentum_score > 0:
                        # For buy signals: boost when oversold, reduce when overbought
                        if bb_position < self.bb_buy_threshold:
                            momentum_score += 0.5  # Extra boost near lower band
                            logger.debug(
                                f"BB FILTER: {symbol} near lower band ({bb_position:.2f}), boosting buy signal"
                            )
                        elif bb_position > self.bb_sell_threshold:
                            momentum_score -= 0.5  # Reduce near upper band
                            logger.debug(
                                f"BB FILTER: {symbol} near upper band ({bb_position:.2f}), reducing buy signal"
                            )
                    elif momentum_score < 0:
                        # For sell/short signals: boost when overbought
                        if bb_position > self.bb_sell_threshold:
                            momentum_score -= 0.5  # Extra bearish near upper band
                            logger.debug(
                                f"BB FILTER: {symbol} near upper band ({bb_position:.2f}), boosting short signal"
                            )
                        elif bb_position < self.bb_buy_threshold:
                            momentum_score += 0.5  # Reduce near lower band
                            logger.debug(
                                f"BB FILTER: {symbol} near lower band ({bb_position:.2f}), reducing short signal"
                            )

            # MULTI-TIMEFRAME FILTERING (NEW FEATURE)
            # Only take trades that align with higher timeframe trends
            if self.use_multi_timeframe and self.mtf_analyzer:
                if self.mtf_require_alignment:
                    # STRICT: All timeframes must align
                    mtf_signal = self.mtf_analyzer.get_aligned_signal(symbol)
                    if mtf_signal == "neutral":
                        logger.debug(f"MTF: {symbol} - timeframes not aligned, signal filtered out")
                        return "neutral"
                    elif mtf_signal == "bearish" and momentum_score > 0:
                        logger.debug(
                            f"MTF: {symbol} - bullish signal rejected (higher TFs bearish)"
                        )
                        return "neutral"
                    elif mtf_signal == "bullish" and momentum_score < 0:
                        logger.debug(
                            f"MTF: {symbol} - bearish signal rejected (higher TFs bullish)"
                        )
                        return "neutral"
                else:
                    # SOFT: Just check if higher timeframe trend agrees
                    # Get highest timeframe trend (e.g., 1Hour)
                    mtf_timeframes = self.parameters.get(
                        "mtf_timeframes", ["5Min", "15Min", "1Hour"]
                    )
                    highest_tf = mtf_timeframes[-1]  # Last one is highest
                    higher_tf_trend = self.mtf_analyzer.get_trend(symbol, highest_tf)

                    # Filter signals that go against higher timeframe trend
                    if higher_tf_trend == "bearish" and momentum_score > 0:
                        logger.info(
                            f"MTF FILTER: {symbol} - BUY signal rejected (1Hour trend: {higher_tf_trend})"
                        )
                        return "neutral"
                    elif higher_tf_trend == "bullish" and momentum_score < 0:
                        logger.info(
                            f"MTF FILTER: {symbol} - SELL signal rejected (1Hour trend: {higher_tf_trend})"
                        )
                        return "neutral"

                    # Log when signal passes multi-timeframe filter
                    if momentum_score >= 2 or momentum_score <= -2:
                        signal_dir = "BUY" if momentum_score > 0 else "SELL"
                        logger.info(
                            f"✅ MTF PASS: {symbol} - {signal_dir} signal aligns with {highest_tf} trend ({higher_tf_trend})"
                        )

            # Determine final signal
            if momentum_score >= 2 and trend_strength and volume_confirmation:
                return "buy"
            elif momentum_score <= -2 and trend_strength and volume_confirmation:
                # SHORT SELLING FEATURE: Return 'short' instead of 'sell' for new positions
                if self.enable_short_selling:
                    return "short"  # Open short position (profit from price drop)
                else:
                    return "neutral"  # Skip if short selling disabled

            return "neutral"

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return "neutral"

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

    async def _check_exit_conditions(self, symbol):
        """
        Check exit conditions including TRAILING STOPS.

        Implements a hybrid exit strategy:
        1. Bracket orders handle basic stop-loss and take-profit at broker level
        2. This method implements TRAILING STOPS to let winners run beyond fixed take-profit

        Trailing Stop Logic:
        - Activates when position is in profit by trailing_activation_pct (default 2%)
        - Trails the peak price by trailing_stop_pct (default 2%)
        - If price drops 2% from peak, exit to lock in profits
        - This allows capturing 10%+ moves instead of always exiting at 5%
        """
        try:
            # Get current position (uses 1-second cache to reduce API calls)
            positions = await self._get_cached_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            if not current_position:
                # Position was closed (likely by bracket order), clean up tracking
                if symbol in self.stop_prices:
                    del self.stop_prices[symbol]
                if symbol in self.target_prices:
                    del self.target_prices[symbol]
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                if symbol in self.peak_prices:
                    del self.peak_prices[symbol]
                return

            current_price = self.current_prices.get(symbol)
            if not current_price:
                return

            # Get entry price for profit calculation
            entry_price = self.entry_prices.get(symbol)
            if not entry_price:
                return

            # Determine if this is a long or short position
            qty = float(current_position.qty)
            is_long = qty > 0

            # Calculate current profit/loss percentage
            if is_long:
                profit_pct = (current_price - entry_price) / entry_price
            else:  # Short position
                profit_pct = (entry_price - current_price) / entry_price

            # === TRAILING STOP LOGIC ===
            if self.use_trailing_stop:
                # Check if trailing stop should be activated (position is in profit)
                trailing_activated = profit_pct >= self.trailing_activation_pct

                if trailing_activated:
                    if is_long:
                        # LONG: Track highest price, sell if it drops trailing_stop_pct below peak
                        if (
                            symbol not in self.peak_prices
                            or current_price > self.peak_prices[symbol]
                        ):
                            self.peak_prices[symbol] = current_price
                            logger.debug(
                                f"{symbol} new peak: ${current_price:.2f} (profit: {profit_pct:.1%})"
                            )

                        peak = self.peak_prices[symbol]
                        trailing_stop_price = peak * (1 - self.trailing_stop_pct)

                        # Check if trailing stop triggered
                        if current_price <= trailing_stop_price:
                            # Calculate actual profit locked in
                            locked_profit_pct = (trailing_stop_price - entry_price) / entry_price

                            logger.info(
                                f"TRAILING STOP TRIGGERED for {symbol}! "
                                f"Peak: ${peak:.2f} -> Current: ${current_price:.2f} "
                                f"(locked profit: {locked_profit_pct:.1%})"
                            )

                            # Exit the position using safe exit method
                            result = await self.submit_exit_order(
                                symbol=symbol,
                                qty=abs(qty),
                                side="sell",
                                reason=f"trailing_stop_long (locked profit: {locked_profit_pct:.1%})",
                            )

                            if result:
                                logger.info(
                                    f"Trailing stop exit for {symbol}: sold {abs(qty):.4f} shares at ~${current_price:.2f}"
                                )
                                # Clean up tracking
                                self._cleanup_position_tracking(symbol)
                            return

                    else:  # SHORT position
                        # SHORT: Track lowest price, cover if it rises trailing_stop_pct above trough
                        if (
                            symbol not in self.peak_prices
                            or current_price < self.peak_prices[symbol]
                        ):
                            self.peak_prices[symbol] = (
                                current_price  # For shorts, track the lowest (best) price
                            )
                            logger.debug(
                                f"{symbol} SHORT new trough: ${current_price:.2f} (profit: {profit_pct:.1%})"
                            )

                        trough = self.peak_prices[symbol]
                        trailing_stop_price = trough * (1 + self.trailing_stop_pct)

                        # Check if trailing stop triggered (price rose above trailing stop)
                        if current_price >= trailing_stop_price:
                            locked_profit_pct = (entry_price - trailing_stop_price) / entry_price

                            logger.info(
                                f"TRAILING STOP TRIGGERED for SHORT {symbol}! "
                                f"Trough: ${trough:.2f} -> Current: ${current_price:.2f} "
                                f"(locked profit: {locked_profit_pct:.1%})"
                            )

                            # Cover the short position using safe exit method
                            result = await self.submit_exit_order(
                                symbol=symbol,
                                qty=abs(qty),
                                side="buy",
                                reason=f"trailing_stop_short (locked profit: {locked_profit_pct:.1%})",
                            )

                            if result:
                                logger.info(
                                    f"Trailing stop exit for SHORT {symbol}: bought {abs(qty):.4f} shares at ~${current_price:.2f}"
                                )
                                self._cleanup_position_tracking(symbol)
                            return

            # Log monitoring info (even without trailing stops)
            stop_price = self.stop_prices.get(symbol)
            target_price = self.target_prices.get(symbol)

            if stop_price and is_long and current_price <= stop_price * 1.01:
                logger.debug(
                    f"{symbol} approaching stop-loss: ${current_price:.2f} near ${stop_price:.2f}"
                )

            if target_price and is_long and current_price >= target_price * 0.99:
                logger.debug(
                    f"{symbol} approaching take-profit: ${current_price:.2f} near ${target_price:.2f}"
                )

        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}", exc_info=True)

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
