"""
MeanReversionStrategy core module.

Holds the concrete ``MeanReversionStrategy`` class: parameter defaults,
initialization, the ``on_bar`` event handler, signal-execution dispatch
(buy / short / sell), state import/export, and the
``analyze_symbol`` / ``execute_trade`` / ``generate_signals`` / ``get_orders``
public API.

Indicator updates, signal generation, and post-entry exit-condition checks
sit at the end of the class.
"""

import logging
from datetime import datetime

import talib

from engine.order_submission import OrderIntent
from strategies.base_strategy import BaseStrategy
from strategies.params import MeanReversionParams
from strategies.risk_manager import RiskManager
from utils.multi_timeframe import MultiTimeframeAnalyzer

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    Params = MeanReversionParams
    """
    Mean reversion strategy that identifies overbought/oversold conditions and
    trades on the expectation that prices will revert to the mean. Uses Bollinger
    Bands, RSI, and standard deviation to identify entry/exit points.
    """

    NAME = "MeanReversionStrategy"

    def default_parameters(self):
        """The defaults live once, on ``Params`` (strategies/params.py)."""
        return dict(self.Params.defaults())

    async def initialize(self, **kwargs):
        """Initialize the mean reversion strategy."""
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

            # Mean reversion parameters
            self.bb_period = self.parameters["bb_period"]
            self.bb_std = self.parameters["bb_std"]
            self.rsi_period = self.parameters["rsi_period"]
            self.rsi_overbought = self.parameters["rsi_overbought"]
            self.rsi_oversold = self.parameters["rsi_oversold"]
            self.sma_period = self.parameters["sma_period"]
            self.mean_lookback = self.parameters["mean_lookback"]
            self.std_threshold = self.parameters["std_threshold"]

            # Exit parameters
            self.profit_target_std = self.parameters["profit_target_std"]
            self.max_hold_days = self.parameters["max_hold_days"]
            self.trailing_stop = self.parameters["trailing_stop"]

            # Initialize tracking dictionaries
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.signals = dict.fromkeys(self.symbols, "neutral")
            self.last_signal_time = dict.fromkeys(self.symbols)
            self.position_entries = {}  # Track entry times and prices
            self.highest_prices = {}  # For trailing stops
            self.lowest_prices = {}  # For trailing stops
            self.current_prices = {}
            self.price_history = {symbol: [] for symbol in self.symbols}

            # Multi-timeframe analysis (NEW FEATURE)
            self.use_multi_timeframe = self.parameters["use_multi_timeframe"]
            self.mtf_require_alignment = self.parameters["mtf_require_alignment"]
            self.mtf_analyzer = None

            if self.use_multi_timeframe:
                mtf_timeframes = self.parameters["mtf_timeframes"]
                self.mtf_analyzer = MultiTimeframeAnalyzer(
                    timeframes=mtf_timeframes, history_length=200
                )
                logger.info(f"✅ Multi-timeframe filtering enabled: {', '.join(mtf_timeframes)}")

            # Short selling parameters (NEW FEATURE)
            self.enable_short_selling = self.parameters["enable_short_selling"]
            self.short_position_size = self.parameters["short_position_size"]
            self.short_stop_loss = self.parameters["short_stop_loss"]

            if self.enable_short_selling:
                logger.info("✅ Short selling enabled - profit from extreme overbought conditions!")

            # Risk manager initialization
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters["max_portfolio_risk"],
                max_position_risk=self.parameters["max_position_risk"],
                max_correlation=self.parameters["max_correlation"],
            )

            # Add strategy as subscriber to broker

            logger.info(f"Initialized {self.NAME} with {len(self.symbols)} symbols")
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False

    async def export_state(self) -> dict:
        """Export lightweight state for restart recovery."""

        def _dt(v):
            return v.isoformat() if hasattr(v, "isoformat") else v

        entries = {}
        for k, v in self.position_entries.items():
            entry = v.copy()
            if "time" in entry:
                entry["time"] = _dt(entry["time"])
            entries[k] = entry

        return {
            "last_signal_time": {k: _dt(v) for k, v in self.last_signal_time.items() if v},
            "position_entries": entries,
            "highest_prices": self.highest_prices,
            "lowest_prices": self.lowest_prices,
        }

    async def import_state(self, state: dict) -> None:
        """Restore lightweight state after restart."""

        def _parse_dt(v):
            return datetime.fromisoformat(v) if isinstance(v, str) else v

        self.highest_prices = state.get("highest_prices", {})
        self.lowest_prices = state.get("lowest_prices", {})
        self.last_signal_time = {
            k: _parse_dt(v) for k, v in state.get("last_signal_time", {}).items()
        }

        entries = {}
        for k, v in state.get("position_entries", {}).items():
            entry = v.copy()
            if "time" in entry:
                entry["time"] = _parse_dt(entry["time"])
            entries[k] = entry
        self.position_entries = entries

    async def analyze_symbol(self, symbol):
        """Analyze a symbol and return trading signal."""
        return self.signals.get(symbol, "neutral")

    async def decide(self, symbol, when, portfolio):
        """Daily execution of this session's signal. Before the session runner,
        mean reversion computed signals in backtests but never traded them."""
        return await self._decide(
            symbol,
            when,
            portfolio,
            size_pct=float(self.parameters["position_size_pct"]),
            sizing_basis=self.parameters["sizing_basis"],
            reason="mean_reversion_backtest",
        )

    async def _exit_intents(self, symbol, when, view):
        """Smart exits on top of the bracket: max holding period, return to the
        mean, and a trailing stop once in profit."""
        held = view.position(symbol)
        if held is None:
            self._clear_entry(symbol)
            return []
        price = self.current_prices.get(symbol)
        entry = self.position_entries.get(symbol)
        if not price or not entry:
            return []
        entry_price, entry_time = entry["price"], entry["time"]
        qty = float(held.qty)

        def exit_(reason):
            self._clear_entry(symbol)
            return [
                OrderIntent(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    is_exit=True,
                    strategy_name=getattr(self, "name", self.NAME),
                    reason=reason,
                )
            ]

        if (when - entry_time).days >= self.max_hold_days:
            return exit_("max_hold_exit")
        ind = self.indicators.get(symbol, {})
        sma, std = ind.get("sma"), ind.get("std")
        if sma and std and std > 0:
            if (entry_price < sma and price >= sma - self.profit_target_std * std) or (
                entry_price > sma and price <= sma + self.profit_target_std * std
            ):
                return exit_("mean_reversion_target_exit")
        self.highest_prices[symbol] = max(self.highest_prices.get(symbol, price), price)
        if price > entry_price and price <= self.highest_prices[symbol] * (1 - self.trailing_stop):
            return exit_("trailing_stop_exit")
        return []

    async def generate_signals(self):
        """Generate signals for all symbols (used in backtest mode)."""
        for symbol in self.symbols:
            if symbol in self.current_data:
                df = self.current_data[symbol]
                if len(df) < self.sma_period:
                    continue

                # Extract price data
                closes = df["close"].values
                highs = df["high"].values
                lows = df["low"].values

                # Calculate indicators
                upper, middle, lower = talib.BBANDS(
                    closes,
                    timeperiod=self.bb_period,
                    nbdevup=self.bb_std,
                    nbdevdn=self.bb_std,
                    matype=0,
                )

                rsi = talib.RSI(closes, timeperiod=self.rsi_period)
                sma = talib.SMA(closes, timeperiod=self.sma_period)
                std = talib.STDDEV(closes, timeperiod=self.mean_lookback)

                slowk, slowd = talib.STOCH(
                    highs,
                    lows,
                    closes,
                    fastk_period=14,
                    slowk_period=3,
                    slowk_matype=0,
                    slowd_period=3,
                    slowd_matype=0,
                )

                # Calculate z-score
                z_score = (closes[-1] - sma[-1]) / std[-1] if len(std) > 0 and std[-1] > 0 else 0

                # Calculate BB position
                bb_range = upper[-1] - lower[-1] if len(upper) > 0 else 0
                bb_position = (closes[-1] - lower[-1]) / bb_range if bb_range > 0 else 0.5

                # Store the indicators
                self.indicators[symbol] = {
                    "upper_band": upper[-1] if len(upper) > 0 else None,
                    "middle_band": middle[-1] if len(middle) > 0 else None,
                    "lower_band": lower[-1] if len(lower) > 0 else None,
                    "rsi": rsi[-1] if len(rsi) > 0 else None,
                    "sma": sma[-1] if len(sma) > 0 else None,
                    "std": std[-1] if len(std) > 0 else None,
                    "z_score": z_score,
                    "bb_position": bb_position,
                    "slowk": slowk[-1] if len(slowk) > 0 else None,
                    "slowd": slowd[-1] if len(slowd) > 0 else None,
                    "close": closes[-1] if len(closes) > 0 else None,
                }

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

    async def _generate_signal(self, symbol):
        """Generate trading signal based on indicators."""
        try:
            # Check if indicators are available
            if not self.indicators.get(symbol) or self.indicators[symbol]["rsi"] is None:
                return "neutral"

            ind = self.indicators[symbol]

            # Get current indicator values
            close = ind["close"]
            upper_band = ind["upper_band"]
            lower_band = ind["lower_band"]
            rsi = ind["rsi"]
            z_score = ind["z_score"]
            bb_position = ind["bb_position"]
            stoch_k = ind["slowk"]
            stoch_d = ind["slowd"]

            # Buy signal: Price is below lower Bollinger Band + RSI is oversold + far from mean
            buy_signal = (
                close < lower_band
                and rsi < self.rsi_oversold
                and z_score < -self.std_threshold
                and bb_position < 0.05  # Near bottom of BB
                and stoch_k < 20
                and stoch_k > stoch_d  # Stoch turning up
            )

            # Sell signal: Price is above upper Bollinger Band + RSI is overbought + far from mean
            sell_signal = (
                close > upper_band
                and rsi > self.rsi_overbought
                and z_score > self.std_threshold
                and bb_position > 0.95  # Near top of BB
                and stoch_k > 80
                and stoch_k < stoch_d  # Stoch turning down
            )

            # MULTI-TIMEFRAME FILTERING (NEW FEATURE)
            # Mean reversion works best when price is extended in ranging markets
            # Filter out mean reversion trades when higher timeframe has strong trend
            if self.use_multi_timeframe and self.mtf_analyzer:
                mtf_timeframes = self.parameters["mtf_timeframes"]
                highest_tf = mtf_timeframes[-1]  # Highest timeframe (e.g., 1Hour)
                higher_tf_trend = self.mtf_analyzer.get_trend(symbol, highest_tf)

                # Mean reversion works AGAINST trends, so we want ranging/neutral markets
                # Reject mean reversion signals if there's a strong trend on higher timeframe
                if buy_signal and higher_tf_trend == "bearish":
                    # Strong bearish trend: don't try to catch falling knife
                    logger.info(
                        f"MTF FILTER: {symbol} - Mean reversion BUY rejected ({highest_tf} strong downtrend)"
                    )
                    return "neutral"
                elif sell_signal and higher_tf_trend == "bullish":
                    # Strong bullish trend: don't fight the trend
                    logger.info(
                        f"MTF FILTER: {symbol} - Mean reversion SELL rejected ({highest_tf} strong uptrend)"
                    )
                    return "neutral"

                # Log when signal passes filter
                if buy_signal or sell_signal:
                    signal_dir = "BUY" if buy_signal else "SELL"
                    logger.info(
                        f"✅ MTF PASS: {symbol} - Mean reversion {signal_dir} (higher TF: {higher_tf_trend})"
                    )

            # Determine final signal
            if buy_signal:
                return "buy"
            elif sell_signal:
                # SHORT SELLING FEATURE: Return 'short' for extreme overbought
                if self.enable_short_selling:
                    return "short"  # Open short position (profit from mean reversion down)
                else:
                    return "neutral"  # Skip if short selling disabled

            return "neutral"

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return "neutral"
