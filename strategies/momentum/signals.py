"""
MomentumStrategy signals mixin.

Contains the signal-generation heuristic and the post-entry trailing-stop /
exit-condition checks for the momentum strategy.

These methods rely on attributes initialized by
``strategies/momentum/strategy.py``
(``self.indicators``, ``self.price_history``, ``self.current_prices``,
``self.entry_prices``, ``self.peak_prices``, ``self.stop_prices``,
``self.target_prices``, ``self.rsi_overbought``, ``self.rsi_oversold``,
``self.adx_threshold``, ``self.volume_factor``, ``self.use_bollinger_filter``,
``self.bb_buy_threshold``, ``self.bb_sell_threshold``,
``self.use_multi_timeframe``, ``self.mtf_analyzer``,
``self.mtf_require_alignment``, ``self.enable_short_selling``,
``self._last_macd_hist``, ``self.crypto_long_only_*``,
``self.use_trailing_stop``, ``self.trailing_activation_pct``,
``self.trailing_stop_pct``, ``self._get_cached_positions``,
``self._is_crypto_symbol``, ``self._get_previous_close``,
``self.submit_exit_order``, ``self._cleanup_position_tracking``) and
therefore must be mixed into the same concrete class.
"""

import logging

logger = logging.getLogger(__name__)


class MomentumSignalsMixin:
    """Signal generation and exit-condition checks for the momentum strategy."""

    async def _generate_signal(self, symbol):
        """Generate trading signal based on indicators."""
        macd_hist = None
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
            # Some live crypto feeds may emit bars with zero/absent volume.
            # Treat non-positive baselines as "volume unavailable" and avoid
            # hard-blocking otherwise valid momentum entries.
            if volume is None or volume_ma is None:
                volume_confirmation = False
            elif volume_ma <= 0:
                volume_confirmation = True
            else:
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
            buy_score_threshold = 2.0
            is_crypto_long_only = not self.enable_short_selling and self._is_crypto_symbol(symbol)
            if is_crypto_long_only and bool(getattr(self, "crypto_long_only_relaxed_entry", True)):
                buy_score_threshold = float(
                    getattr(self, "crypto_long_only_buy_score_threshold", 1.0)
                )

            macd_hist_cache = getattr(self, "_last_macd_hist", {})
            if not isinstance(macd_hist_cache, dict):
                macd_hist_cache = {}
            previous_macd_hist = macd_hist_cache.get(symbol)
            previous_close = self._get_previous_close(symbol)
            close_price = ind.get("close")
            dip_buy_eligible = False
            if is_crypto_long_only and bool(
                getattr(self, "crypto_long_only_dip_buy_enabled", True)
            ):
                rebound_ok = (
                    previous_close is not None
                    and close_price is not None
                    and close_price
                    >= previous_close
                    * (1.0 + float(getattr(self, "crypto_long_only_dip_min_rebound_pct", 0.001)))
                )
                macd_hist_delta_ok = previous_macd_hist is not None and (
                    macd_hist - previous_macd_hist
                ) >= float(getattr(self, "crypto_long_only_dip_min_macd_hist_delta", 0.02))
                dip_buy_eligible = (
                    trend_strength
                    and volume_confirmation
                    and rsi <= float(getattr(self, "crypto_long_only_dip_rsi_max", 35.0))
                    and macd_hist_delta_ok
                    and rebound_ok
                )

            if momentum_score >= buy_score_threshold and trend_strength and volume_confirmation:
                return "buy"
            if dip_buy_eligible:
                logger.info(
                    "CRYPTO DIP-BUY: %s buy triggered (rsi=%.2f, macd_hist=%.4f, prev_hist=%.4f)",
                    symbol,
                    rsi,
                    macd_hist,
                    previous_macd_hist,
                )
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
        finally:
            if macd_hist is not None:
                if not isinstance(getattr(self, "_last_macd_hist", None), dict):
                    self._last_macd_hist = {}
                self._last_macd_hist[symbol] = macd_hist

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
