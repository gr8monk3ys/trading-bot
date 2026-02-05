"""
Ensemble Strategy - Multi-Strategy Combination

Combines multiple strategies (mean reversion + momentum + trend following) with
intelligent weighting based on market regime detection.

Key Features:
- Regime detection (trending vs ranging vs volatile)
- Dynamic strategy weighting based on market conditions
- Confirmation from multiple strategies required
- Risk-adjusted position sizing
- Advanced exit management

Expected Sharpe Ratio: 0.95-1.25 (research target)
"""

import logging
import warnings
from datetime import datetime

import numpy as np

from brokers.order_builder import OrderBuilder
from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
from utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy combining multiple trading approaches.

    Sub-Strategies:
    1. Mean Reversion (best in ranging markets)
    2. Momentum (best in trending markets)
    3. Trend Following (best in strong trends)

    Market Regimes:
    - Trending: ADX > 25, directional movement
    - Ranging: ADX < 20, oscillating price
    - Volatile: High ATR, low ADX
    """

    NAME = "EnsembleStrategy"

    def default_parameters(self):
        """Return default parameters."""
        return {
            # Basic parameters
            "position_size": 0.10,  # 10% per position
            "max_positions": 5,
            "max_portfolio_risk": 0.02,
            "stop_loss": 0.025,  # 2.5% stop
            "take_profit": 0.05,  # 5% profit target
            # Regime detection parameters
            "adx_trending_threshold": 25,  # ADX > 25 = trending
            "adx_ranging_threshold": 20,  # ADX < 20 = ranging
            "atr_volatility_threshold": 0.02,  # 2% ATR = high volatility
            # Sub-strategy parameters
            # Mean reversion
            "mr_bb_period": 20,
            "mr_bb_std": 2.0,
            "mr_rsi_period": 14,
            "mr_rsi_oversold": 30,
            "mr_rsi_overbought": 70,
            # Momentum
            "mom_rsi_period": 14,
            "mom_macd_fast": 12,
            "mom_macd_slow": 26,
            "mom_macd_signal": 9,
            "mom_adx_threshold": 25,
            # Trend following
            "tf_fast_ma": 10,
            "tf_slow_ma": 30,
            "tf_ema_period": 20,
            # Ensemble weighting
            "min_agreement_pct": 0.60,  # Need 60% agreement to trade
            "regime_weight_boost": 1.5,  # Boost strategy that matches regime
            # Risk management
            "max_correlation": 0.7,
            "trailing_stop": 0.015,  # 1.5% trailing stop
        }

    async def initialize(self, **kwargs):
        """Initialize ensemble strategy."""
        warnings.warn(
            f"{self.__class__.__name__} is experimental and has not been fully validated. "
            "Use in production at your own risk.",
            category=UserWarning,
            stacklevel=2,
        )
        try:
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Extract parameters
            self.position_size = self.parameters["position_size"]
            self.max_positions = self.parameters["max_positions"]
            self.stop_loss = self.parameters["stop_loss"]
            self.take_profit = self.parameters["take_profit"]

            # Initialize tracking
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.market_regime = dict.fromkeys(self.symbols, "unknown")
            self.sub_strategy_signals = {symbol: {} for symbol in self.symbols}
            self.ensemble_signals = dict.fromkeys(self.symbols, "neutral")
            self.current_prices = {}
            self.price_history = {symbol: [] for symbol in self.symbols}
            self.position_entries = {}
            self.highest_prices = {}

            # Risk manager
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters["max_portfolio_risk"],
                max_position_risk=self.parameters.get("max_position_risk", 0.01),
                max_correlation=self.parameters["max_correlation"],
            )

            # Add as subscriber
            if hasattr(self.broker, "_add_subscriber"):
                self.broker._add_subscriber(self)

            logger.info(f"Initialized {self.NAME} with {len(self.symbols)} symbols")
            logger.info("  Sub-strategies: Mean Reversion, Momentum, Trend Following")
            logger.info(f"  Min agreement: {self.parameters['min_agreement_pct']:.0%}")

            return True

        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False

    async def on_bar(
        self, symbol, open_price, high_price, low_price, close_price, volume, timestamp
    ):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return

            # Store price
            self.current_prices[symbol] = close_price

            # Update price history
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

            # Keep history manageable
            max_history = 200
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]

            # Need minimum history
            if len(self.price_history[symbol]) < 50:
                return

            # Update indicators
            await self._update_indicators(symbol)

            # Detect market regime
            await self._detect_market_regime(symbol)

            # Generate sub-strategy signals
            await self._generate_sub_strategy_signals(symbol)

            # Combine into ensemble signal
            await self._combine_signals(symbol)

            # Execute if signal present
            signal = self.ensemble_signals[symbol]
            if signal != "neutral":
                await self._execute_signal(symbol, signal)

            # Check exits
            await self._check_exit_conditions(symbol)

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    async def _update_indicators(self, symbol):
        """Update all technical indicators."""
        try:
            history = self.price_history[symbol]
            if len(history) < 50:
                return

            # Extract arrays
            closes = np.array([bar["close"] for bar in history])
            highs = np.array([bar["high"] for bar in history])
            lows = np.array([bar["low"] for bar in history])
            volumes = np.array([bar["volume"] for bar in history])
            timestamps = [bar["timestamp"] for bar in history]

            # Create indicator calculator
            ind = TechnicalIndicators(
                high=highs,
                low=lows,
                close=closes,
                open_=None,
                volume=volumes,
                timestamps=timestamps,
            )

            # Calculate all indicators
            # Trend indicators
            adx, plus_di, minus_di = ind.adx_di(period=14)
            fast_ma = ind.sma(period=self.parameters["tf_fast_ma"])
            slow_ma = ind.sma(period=self.parameters["tf_slow_ma"])
            ema_20 = ind.ema(period=20)

            # Momentum indicators
            rsi = ind.rsi(period=14)
            macd, macd_signal, macd_hist = ind.macd()
            stoch_k, stoch_d = ind.stochastic()

            # Volatility indicators
            bb_upper, bb_middle, bb_lower = ind.bollinger_bands(period=20, std=2.0)
            atr = ind.atr(period=14)
            stddev = ind.stddev(period=20)

            # Volume indicators
            try:
                vwap = ind.vwap()
            except Exception as e:
                logger.debug(f"VWAP calculation failed, using closes as fallback: {e}")
                vwap = closes  # Fallback if VWAP fails

            volume_sma = ind.volume_sma(period=20)

            # Store all indicators
            self.indicators[symbol] = {
                # Trend
                "adx": adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 0,
                "plus_di": plus_di[-1] if len(plus_di) > 0 and not np.isnan(plus_di[-1]) else 0,
                "minus_di": minus_di[-1] if len(minus_di) > 0 and not np.isnan(minus_di[-1]) else 0,
                "fast_ma": (
                    fast_ma[-1] if len(fast_ma) > 0 and not np.isnan(fast_ma[-1]) else closes[-1]
                ),
                "slow_ma": (
                    slow_ma[-1] if len(slow_ma) > 0 and not np.isnan(slow_ma[-1]) else closes[-1]
                ),
                "ema_20": (
                    ema_20[-1] if len(ema_20) > 0 and not np.isnan(ema_20[-1]) else closes[-1]
                ),
                # Momentum
                "rsi": rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50,
                "macd": macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0,
                "macd_signal": (
                    macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
                ),
                "macd_hist": (
                    macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0
                ),
                "stoch_k": stoch_k[-1] if len(stoch_k) > 0 and not np.isnan(stoch_k[-1]) else 50,
                "stoch_d": stoch_d[-1] if len(stoch_d) > 0 and not np.isnan(stoch_d[-1]) else 50,
                # Volatility
                "bb_upper": (
                    bb_upper[-1]
                    if len(bb_upper) > 0 and not np.isnan(bb_upper[-1])
                    else closes[-1] * 1.02
                ),
                "bb_middle": (
                    bb_middle[-1]
                    if len(bb_middle) > 0 and not np.isnan(bb_middle[-1])
                    else closes[-1]
                ),
                "bb_lower": (
                    bb_lower[-1]
                    if len(bb_lower) > 0 and not np.isnan(bb_lower[-1])
                    else closes[-1] * 0.98
                ),
                "atr": atr[-1] if len(atr) > 0 and not np.isnan(atr[-1]) else 0,
                "stddev": stddev[-1] if len(stddev) > 0 and not np.isnan(stddev[-1]) else 0,
                # Volume
                "vwap": vwap[-1] if len(vwap) > 0 and not np.isnan(vwap[-1]) else closes[-1],
                "volume": volumes[-1],
                "volume_sma": (
                    volume_sma[-1]
                    if len(volume_sma) > 0 and not np.isnan(volume_sma[-1])
                    else volumes[-1]
                ),
                # Price
                "close": closes[-1],
            }

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}", exc_info=True)

    async def _detect_market_regime(self, symbol):
        """Detect current market regime."""
        try:
            ind = self.indicators[symbol]

            adx = ind["adx"]
            atr = ind["atr"]
            close = ind["close"]

            # Calculate ATR percentage
            atr_pct = atr / close if close > 0 else 0

            # Determine regime
            if adx > self.parameters["adx_trending_threshold"]:
                regime = "trending"
            elif adx < self.parameters["adx_ranging_threshold"]:
                regime = "ranging"
            else:
                regime = "transitional"

            # Add volatility flag
            if atr_pct > self.parameters["atr_volatility_threshold"]:
                regime += "_volatile"
            else:
                regime += "_normal"

            self.market_regime[symbol] = regime

            logger.debug(f"{symbol} regime: {regime} (ADX: {adx:.1f}, ATR: {atr_pct:.2%})")

        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            self.market_regime[symbol] = "unknown"

    async def _generate_sub_strategy_signals(self, symbol):
        """Generate signals from each sub-strategy."""
        try:
            ind = self.indicators[symbol]

            signals = {}

            # 1. MEAN REVERSION SIGNAL
            close = ind["close"]
            bb_upper = ind["bb_upper"]
            bb_lower = ind["bb_lower"]
            rsi = ind["rsi"]
            stoch_k = ind["stoch_k"]

            # Mean reversion: buy oversold, sell overbought
            mr_score = 0
            if close < bb_lower and rsi < self.parameters["mr_rsi_oversold"] and stoch_k < 20:
                mr_score = 1  # Buy signal
            elif close > bb_upper and rsi > self.parameters["mr_rsi_overbought"] and stoch_k > 80:
                mr_score = -1  # Sell signal

            signals["mean_reversion"] = {
                "signal": "buy" if mr_score > 0 else "sell" if mr_score < 0 else "neutral",
                "strength": abs(mr_score),
                "best_regime": "ranging",
            }

            # 2. MOMENTUM SIGNAL
            macd = ind["macd"]
            macd_signal_line = ind["macd_signal"]
            macd_hist = ind["macd_hist"]
            adx = ind["adx"]

            mom_score = 0
            # MACD crossover + strong trend
            if (
                macd > macd_signal_line
                and macd_hist > 0
                and adx > self.parameters["mom_adx_threshold"]
            ):
                mom_score = 1
            elif (
                macd < macd_signal_line
                and macd_hist < 0
                and adx > self.parameters["mom_adx_threshold"]
            ):
                mom_score = -1

            # RSI confirmation
            if rsi > 50 and mom_score > 0:
                mom_score += 0.5
            elif rsi < 50 and mom_score < 0:
                mom_score -= 0.5

            signals["momentum"] = {
                "signal": "buy" if mom_score > 0 else "sell" if mom_score < 0 else "neutral",
                "strength": abs(mom_score) / 1.5,  # Normalize to 0-1
                "best_regime": "trending",
            }

            # 3. TREND FOLLOWING SIGNAL
            fast_ma = ind["fast_ma"]
            slow_ma = ind["slow_ma"]
            ema_20 = ind["ema_20"]
            plus_di = ind["plus_di"]
            minus_di = ind["minus_di"]

            tf_score = 0
            # MA crossover + directional movement
            if fast_ma > slow_ma and close > ema_20 and plus_di > minus_di:
                tf_score = 1
            elif fast_ma < slow_ma and close < ema_20 and minus_di > plus_di:
                tf_score = -1

            signals["trend_following"] = {
                "signal": "buy" if tf_score > 0 else "sell" if tf_score < 0 else "neutral",
                "strength": abs(tf_score),
                "best_regime": "trending",
            }

            self.sub_strategy_signals[symbol] = signals

        except Exception as e:
            logger.error(f"Error generating sub-strategy signals for {symbol}: {e}", exc_info=True)
            self.sub_strategy_signals[symbol] = {}

    async def _combine_signals(self, symbol):
        """Combine sub-strategy signals into ensemble signal."""
        try:
            signals = self.sub_strategy_signals[symbol]
            regime = self.market_regime[symbol]

            if not signals:
                self.ensemble_signals[symbol] = "neutral"
                return

            # Count votes and calculate weighted score
            buy_votes = 0
            sell_votes = 0
            total_weight = 0

            for strategy_name, signal_data in signals.items():
                signal = signal_data["signal"]
                strength = signal_data["strength"]
                best_regime = signal_data["best_regime"]

                # Base weight from strength
                weight = strength

                # Boost weight if strategy matches current regime
                if best_regime in regime:
                    weight *= self.parameters["regime_weight_boost"]

                total_weight += weight

                if signal == "buy":
                    buy_votes += weight
                elif signal == "sell":
                    sell_votes += weight

            # Calculate agreement percentage
            if total_weight > 0:
                buy_agreement = buy_votes / total_weight
                sell_agreement = sell_votes / total_weight
            else:
                buy_agreement = 0
                sell_agreement = 0

            min_agreement = self.parameters["min_agreement_pct"]

            # Determine ensemble signal
            if buy_agreement >= min_agreement:
                self.ensemble_signals[symbol] = "buy"
                logger.debug(
                    f"{symbol} ENSEMBLE BUY (agreement: {buy_agreement:.1%}, regime: {regime})"
                )
            elif sell_agreement >= min_agreement:
                self.ensemble_signals[symbol] = "sell"
                logger.debug(
                    f"{symbol} ENSEMBLE SELL (agreement: {sell_agreement:.1%}, regime: {regime})"
                )
            else:
                self.ensemble_signals[symbol] = "neutral"

        except Exception as e:
            logger.error(f"Error combining signals for {symbol}: {e}", exc_info=True)
            self.ensemble_signals[symbol] = "neutral"

    async def _execute_signal(self, symbol, signal):
        """Execute trading signal."""
        try:
            # Get current positions
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            # Get account
            account = await self.broker.get_account()
            buying_power = float(account.buying_power)

            # BUY signal
            if signal == "buy" and not current_position:
                # Check max positions
                if len(positions) >= self.max_positions:
                    logger.info(f"Max positions reached, skipping {symbol}")
                    return

                # Calculate position size
                price = self.current_prices[symbol]
                position_value = buying_power * self.position_size

                # Risk adjustment
                current_positions = {}
                for pos in positions:
                    if pos.symbol in self.price_history:
                        close_prices = [bar["close"] for bar in self.price_history[pos.symbol]]
                        current_positions[pos.symbol] = {
                            "value": float(pos.market_value),
                            "price_history": close_prices,
                            "risk": None,
                        }

                if len(self.price_history[symbol]) > 20:
                    close_prices = [bar["close"] for bar in self.price_history[symbol]]
                    adjusted_value = self.risk_manager.adjust_position_size(
                        symbol, position_value, close_prices, current_positions
                    )
                    position_value = adjusted_value

                if position_value <= 0:
                    logger.info(f"Risk manager rejected {symbol}")
                    return

                # Enforce position size limit
                position_value, quantity = await self.enforce_position_size_limit(
                    symbol, position_value, price
                )

                if quantity < 0.01:
                    logger.info(f"Position too small for {symbol}")
                    return

                # Calculate bracket levels
                take_profit_price = price * (1 + self.take_profit)
                stop_loss_price = price * (1 - self.stop_loss)

                logger.info(f"ENSEMBLE BUY {symbol}:")
                logger.info(f"  Quantity: {quantity:.4f} @ ${price:.2f}")
                logger.info(f"  Regime: {self.market_regime[symbol]}")
                logger.info(f"  Sub-signals: {self.sub_strategy_signals[symbol]}")

                # Create bracket order
                order = (
                    OrderBuilder(symbol, "buy", quantity)
                    .market()
                    .bracket(take_profit=take_profit_price, stop_loss=stop_loss_price)
                    .gtc()
                    .build()
                )

                result = await self.submit_entry_order(
                    order,
                    reason="ensemble_entry",
                    max_positions=self.max_positions,
                )

                if result and (not hasattr(result, "success") or result.success):
                    logger.info(f"✅ Ensemble BUY order submitted: {symbol}")
                    self.position_entries[symbol] = {
                        "time": datetime.now(),
                        "price": price,
                        "quantity": quantity,
                        "regime": self.market_regime[symbol],
                    }
                    self.highest_prices[symbol] = price

            # SELL signal
            elif signal == "sell" and current_position:
                quantity = float(current_position.qty)
                price = self.current_prices[symbol]

                logger.info(f"ENSEMBLE SELL {symbol}:")
                logger.info(f"  Quantity: {quantity} @ ${price:.2f}")
                logger.info(f"  Regime: {self.market_regime[symbol]}")

                result = await self.submit_exit_order(
                    symbol=symbol,
                    qty=quantity,
                    side="sell",
                    reason="ensemble_exit",
                )

                if result:
                    logger.info(f"✅ Ensemble SELL order submitted: {symbol}")

                    if symbol in self.position_entries:
                        del self.position_entries[symbol]
                    if symbol in self.highest_prices:
                        del self.highest_prices[symbol]

        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}", exc_info=True)

    async def _check_exit_conditions(self, symbol):
        """Check for exit conditions (trailing stop, etc)."""
        try:
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            if not current_position:
                return

            current_price = self.current_prices.get(symbol)
            if not current_price:
                return

            entry = self.position_entries.get(symbol)
            if not entry:
                return

            entry_price = entry["price"]
            unrealized_pnl = (current_price - entry_price) / entry_price

            # Update highest price
            if symbol in self.highest_prices:
                self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
            else:
                self.highest_prices[symbol] = current_price

            # Trailing stop (only if in profit)
            if unrealized_pnl > 0:
                peak_price = self.highest_prices[symbol]
                trailing_stop_price = peak_price * (1 - self.parameters["trailing_stop"])

                if current_price <= trailing_stop_price:
                    logger.info(
                        f"TRAILING STOP triggered for {symbol}: ${current_price:.2f} "
                        f"(peak: ${peak_price:.2f}, profit: {unrealized_pnl:.1%})"
                    )

                    quantity = float(current_position.qty)
                    await self.submit_exit_order(
                        symbol=symbol,
                        qty=quantity,
                        side="sell",
                        reason="trailing_stop_exit",
                    )

        except Exception as e:
            logger.error(f"Error checking exits for {symbol}: {e}", exc_info=True)

    async def analyze_symbol(self, symbol):
        """Analyze symbol and return signal."""
        return self.ensemble_signals.get(symbol, "neutral")

    async def execute_trade(self, symbol, signal):
        """Execute trade - handled by _execute_signal."""
        pass

    async def generate_signals(self):
        """Generate signals for backtest mode."""
        # Similar to on_bar but for batch processing
        pass

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
            "position_entries": entries,
            "highest_prices": self.highest_prices,
            "ensemble_signals": self.ensemble_signals,
        }

    async def import_state(self, state: dict) -> None:
        """Restore lightweight state after restart."""
        from datetime import datetime

        def _parse_dt(v):
            return datetime.fromisoformat(v) if isinstance(v, str) else v

        entries = {}
        for k, v in state.get("position_entries", {}).items():
            entry = v.copy()
            if "time" in entry:
                entry["time"] = _parse_dt(entry["time"])
            entries[k] = entry

        self.position_entries = entries
        self.highest_prices = state.get("highest_prices", {})
        self.ensemble_signals = state.get("ensemble_signals", self.ensemble_signals)

    def get_orders(self):
        """Get orders for backtest."""
        orders = []

        for symbol, signal in self.ensemble_signals.items():
            if signal == "neutral":
                continue

            current_positions = getattr(self, "positions", {})
            has_position = symbol in current_positions

            price = self.indicators[symbol]["close"]
            if not price:
                continue

            if signal == "buy" and not has_position:
                capital = getattr(self, "capital", 100000)
                position_value = capital * self.position_size
                quantity = position_value / price

                if quantity >= 0.01:
                    orders.append(
                        {"symbol": symbol, "quantity": quantity, "side": "buy", "type": "market"}
                    )

            elif signal == "sell" and has_position:
                position = current_positions[symbol]
                quantity = position.get("quantity", 0)

                if quantity > 0:
                    orders.append(
                        {"symbol": symbol, "quantity": quantity, "side": "sell", "type": "market"}
                    )

        return orders
