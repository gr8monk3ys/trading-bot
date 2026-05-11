"""
MeanReversionStrategy signals mixin.

Contains the technical-indicator update logic, the signal-generation
heuristic, and the post-entry exit-condition checks for the mean-reversion
strategy.

These methods rely on attributes initialized by
``strategies/mean_reversion/strategy.py``
(``self.price_history``, ``self.indicators``, ``self.signals``,
``self.position_entries``, ``self.highest_prices``, ``self.lowest_prices``,
``self.current_prices``, ``self.bb_period``, ``self.bb_std``,
``self.rsi_period``, ``self.rsi_overbought``, ``self.rsi_oversold``,
``self.sma_period``, ``self.mean_lookback``, ``self.std_threshold``,
``self.profit_target_std``, ``self.max_hold_days``, ``self.trailing_stop``,
``self.stop_loss``, ``self.take_profit``, ``self.use_multi_timeframe``,
``self.mtf_analyzer``, ``self.enable_short_selling``) and therefore must be
mixed into the same concrete class.
"""

import logging
from datetime import datetime

import numpy as np
import talib

logger = logging.getLogger(__name__)


class MeanReversionSignalsMixin:
    """Indicator updates, signal generation, and exit-condition checks."""

    async def _update_indicators(self, symbol):
        """Update technical indicators for a symbol."""
        try:
            # Ensure we have enough price history
            if len(self.price_history[symbol]) < self.sma_period:
                return

            # Extract price data into arrays
            closes = np.array([bar["close"] for bar in self.price_history[symbol]])
            highs = np.array([bar["high"] for bar in self.price_history[symbol]])
            lows = np.array([bar["low"] for bar in self.price_history[symbol]])

            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                closes,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0,
            )

            # Calculate RSI
            rsi = talib.RSI(closes, timeperiod=self.rsi_period)

            # Calculate SMA (mean)
            sma = talib.SMA(closes, timeperiod=self.sma_period)

            # Calculate standard deviation
            std = talib.STDDEV(closes, timeperiod=self.mean_lookback)

            # Calculate Stochastic
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

            # Calculate ATR for stop loss
            atr = talib.ATR(highs, lows, closes, timeperiod=14)

            # Calculate distance from mean (z-score)
            z_score = (closes[-1] - sma[-1]) / std[-1] if std[-1] > 0 else 0

            # Calculate percent of BB range
            bb_range = upper[-1] - lower[-1]
            bb_position = (closes[-1] - lower[-1]) / bb_range if bb_range > 0 else 0.5

            # Store indicators
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
                "atr": atr[-1] if len(atr) > 0 else None,
                "close": closes[-1] if len(closes) > 0 else None,
            }

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}", exc_info=True)

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
                mtf_timeframes = self.parameters.get("mtf_timeframes", ["5Min", "15Min", "1Hour"])
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

    async def _check_exit_conditions(self, symbol):
        """
        Check advanced exit conditions for the given symbol.

        Note: Bracket orders automatically handle basic stop-loss (-2%) and take-profit (+4%)
        at the broker level. This method implements ADDITIONAL "smart exits" that can optimize
        returns beyond the bracket limits:

        1. Max holding period (5 days) - Frees capital from stale positions
        2. Mean reversion target - Exits when price returns to mean (strategy's core edge)
        3. Trailing stop (1.5%) - Locks in profits beyond 4% if stock keeps running

        These exits work IN ADDITION TO the bracket orders for maximum profit potential.
        """
        try:
            # Get current position
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            if not current_position:
                # Position was closed (likely by bracket order), clean up tracking
                if symbol in self.position_entries:
                    logger.debug(
                        f"Position {symbol} closed (likely by bracket order), cleaning up tracking"
                    )
                    del self.position_entries[symbol]
                if symbol in self.lowest_prices:
                    del self.lowest_prices[symbol]
                if symbol in self.highest_prices:
                    del self.highest_prices[symbol]
                return

            current_price = self.current_prices.get(symbol)
            if not current_price:
                return

            # Get entry details
            entry = self.position_entries.get(symbol)
            if not entry:
                return

            entry_price = entry["price"]
            entry_time = entry["time"]
            current_time = datetime.now()

            # Calculate unrealized profit/loss
            unrealized_pnl = (current_price - entry_price) / entry_price

            # SMART EXIT 1: Max holding period - Free up capital from stale positions
            holding_days = (current_time - entry_time).days
            if holding_days >= self.max_hold_days:
                logger.info(
                    f"SMART EXIT: Max holding period ({self.max_hold_days} days) reached for {symbol}, "
                    f"exiting position to free capital (P/L: {unrealized_pnl*100:.1f}%)"
                )
                quantity = float(current_position.qty)
                await self.submit_exit_order(
                    symbol=symbol,
                    qty=quantity,
                    side="sell",
                    reason="max_hold_exit",
                )
                return

            # SMART EXIT 2: Mean reversion target - Exit when price returns to mean (strategy's core edge)
            ind = self.indicators[symbol]
            sma = ind.get("sma")
            std = ind.get("std")

            if sma and std and std > 0:
                z_score = (current_price - sma) / std

                # Check if price has reverted to mean (or beyond)
                if (entry_price < sma and current_price >= sma - self.profit_target_std * std) or (
                    entry_price > sma and current_price <= sma + self.profit_target_std * std
                ):
                    logger.info(
                        f"SMART EXIT: Mean reversion target reached for {symbol} "
                        f"(z-score: {z_score:.2f}, P/L: {unrealized_pnl*100:.1f}%), exiting position"
                    )
                    quantity = float(current_position.qty)
                    await self.submit_exit_order(
                        symbol=symbol,
                        qty=quantity,
                        side="sell",
                        reason="mean_reversion_target_exit",
                    )
                    return

            # SMART EXIT 3: Trailing stop - Lock in profits beyond 4% take-profit
            # For LONG positions: track HIGHEST price (peak) and trail DOWN from it
            # Update peak price tracking
            if symbol in self.highest_prices:
                self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
            else:
                self.highest_prices[symbol] = current_price

            # Trailing stop triggers only if in profit (can capture more than bracket's fixed 4%)
            if unrealized_pnl > 0:
                # Calculate trailing stop price: peak price minus trailing percentage
                peak_price = self.highest_prices[symbol]
                trailing_stop_price = peak_price * (1 - self.trailing_stop)  # Trail DOWN by 1.5%

                if current_price <= trailing_stop_price:
                    logger.info(
                        f"SMART EXIT: Trailing stop triggered for {symbol} at ${current_price:.2f} "
                        f"(peak: ${peak_price:.2f}, trailing stop: ${trailing_stop_price:.2f}, "
                        f"profit locked: {unrealized_pnl*100:.1f}%)"
                    )
                    quantity = float(current_position.qty)
                    await self.submit_exit_order(
                        symbol=symbol,
                        qty=quantity,
                        side="sell",
                        reason="trailing_stop_exit",
                    )
                    return

            # Monitor bracket order levels for logging/debugging
            # Bracket order stop-loss at -2%, take-profit at +4%
            bracket_stop_loss = entry_price * (1 - self.stop_loss)  # -2%
            bracket_take_profit = entry_price * (1 + self.take_profit)  # +4%

            # Log when approaching bracket levels (for monitoring)
            if current_price <= bracket_stop_loss * 1.005:  # Within 0.5% of stop
                logger.debug(
                    f"{symbol} approaching bracket stop-loss: ${current_price:.2f} near ${bracket_stop_loss:.2f}"
                )

            if current_price >= bracket_take_profit * 0.995:  # Within 0.5% of take-profit
                logger.debug(
                    f"{symbol} approaching bracket take-profit: ${current_price:.2f} near ${bracket_take_profit:.2f}"
                )

            # Note: Basic stop-loss and take-profit are handled by bracket orders at broker level
            # No manual sell orders needed - bracket will execute automatically

        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}", exc_info=True)
