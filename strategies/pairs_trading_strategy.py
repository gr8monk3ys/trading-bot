"""
Pairs Trading Strategy - Market-Neutral Statistical Arbitrage

Trades the spread between two cointegrated stocks. When the spread widens beyond
normal levels, we bet on mean reversion by going long the underperformer and
short the outperformer.

Key Features:
- Cointegration testing (Engle-Granger method)
- Z-score based entry/exit signals
- Market-neutral (long one stock, short another)
- Lower correlation to market (hedged)
- Statistical edge from mean reversion

Expected Sharpe Ratio: 0.80-1.20 (highest potential from research)
Best For: All market conditions (market-neutral)
Risk: Medium (both positions can move against you)

Common Stock Pairs:
- KO/PEP (Coca-Cola / PepsiCo)
- GM/F (General Motors / Ford)
- WMT/TGT (Walmart / Target)
- JPM/BAC (JPMorgan / Bank of America)
- AAPL/MSFT (Apple / Microsoft)
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

from brokers.order_builder import OrderBuilder
from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy using statistical arbitrage.

    Methodology:
    1. Find cointegrated pairs (statistical relationship)
    2. Calculate spread (hedge ratio * stock1 - stock2)
    3. Calculate z-score of spread
    4. Enter when |z-score| > threshold (spread is wide)
    5. Exit when z-score approaches 0 (spread normalizes)

    Market-neutral: Long-short positions cancel out market risk.
    """

    NAME = "PairsTradingStrategy"

    def default_parameters(self):
        """Return default parameters."""
        return {
            # Basic parameters
            "position_size": 0.10,  # 10% per PAIR (split between long/short)
            "max_pairs": 3,  # Maximum concurrent pairs
            "max_portfolio_risk": 0.02,
            # Pair selection
            "lookback_period": 60,  # Days for cointegration test
            "min_correlation": 0.70,  # Minimum correlation to consider
            "cointegration_pvalue": 0.05,  # Max p-value for cointegration
            # Entry/exit signals
            "entry_z_score": 2.0,  # Enter when |z-score| > 2.0
            "exit_z_score": 0.5,  # Exit when |z-score| < 0.5
            "stop_loss_z_score": 3.5,  # Stop loss at |z-score| > 3.5
            # Position management
            "hedge_ratio_recalc_days": 7,  # Recalculate hedge ratio weekly
            "max_holding_days": 10,  # Maximum days to hold pair (fallback)
            "use_half_life_exit": True,  # Use half-life for dynamic holding period
            "half_life_multiplier": 3.0,  # Exit after 3x half-life (99% mean reversion)
            "take_profit_pct": 0.04,  # 4% profit target on pair
            "stop_loss_pct": 0.03,  # 3% stop loss on pair
            "min_hurst_threshold": 0.5,  # Maximum Hurst exponent (must be < this)
            # Risk management
            "max_correlation": 0.7,
        }

    async def initialize(self, **kwargs):
        """Initialize pairs trading strategy."""
        try:
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Extract parameters
            self.position_size = self.parameters["position_size"]
            self.max_pairs = self.parameters["max_pairs"]

            # Pairs must be provided as tuples
            # Example: symbols = [('AAPL', 'MSFT'), ('KO', 'PEP'), ('JPM', 'BAC')]
            if not self.symbols or not isinstance(self.symbols[0], (tuple, list)):
                raise ValueError(
                    "Pairs trading requires symbol pairs, e.g. [('AAPL', 'MSFT'), ('KO', 'PEP')]"
                )

            # Store pairs
            self.pairs = self.symbols
            self.all_symbols = list({s for pair in self.pairs for s in pair})

            # Initialize tracking
            self.price_history = {symbol: [] for symbol in self.all_symbols}
            self.current_prices = {}

            # Pair statistics
            self.pair_stats = {pair: {} for pair in self.pairs}
            self.pair_spreads = {pair: [] for pair in self.pairs}
            self.pair_signals = dict.fromkeys(self.pairs, "neutral")
            self.pair_positions = {}  # Track active pair trades

            # Cointegration results
            self.cointegration_results = dict.fromkeys(self.pairs)
            self.last_coint_check = dict.fromkeys(self.pairs)

            # Risk manager
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters["max_portfolio_risk"],
                max_position_risk=self.parameters.get("max_position_risk", 0.01),
                max_correlation=self.parameters["max_correlation"],
            )

            logger.info(f"Initialized {self.NAME}")
            logger.info(f"  Pairs: {len(self.pairs)}")
            for pair in self.pairs:
                logger.info(f"    {pair[0]} / {pair[1]}")
            logger.info(f"  Entry z-score: {self.parameters['entry_z_score']}")
            logger.info(f"  Exit z-score: {self.parameters['exit_z_score']}")

            return True

        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False

    async def on_bar(
        self, symbol, open_price, high_price, low_price, close_price, volume, timestamp
    ):
        """Handle incoming bar data."""
        try:
            if symbol not in self.all_symbols:
                return

            # Store price
            self.current_prices[symbol] = close_price

            # Update price history
            self.price_history[symbol].append(
                {"timestamp": timestamp, "close": close_price, "volume": volume}
            )

            # Keep history manageable
            max_history = 100
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]

            # Process each pair
            for pair in self.pairs:
                symbol1, symbol2 = pair

                # Wait until we have data for both symbols
                if len(self.price_history[symbol1]) < 30 or len(self.price_history[symbol2]) < 30:
                    continue

                # Check cointegration periodically
                await self._check_cointegration(pair)

                # Calculate spread and z-score
                await self._calculate_spread(pair)

                # Generate trading signals
                await self._generate_pair_signal(pair)

                # Execute trades
                signal = self.pair_signals[pair]
                if signal != "neutral":
                    await self._execute_pair_trade(pair, signal)

                # Check exits for active positions
                await self._check_pair_exit(pair)

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent to measure mean reversion.

        H < 0.5: Mean-reverting (good for pairs trading)
        H = 0.5: Random walk (no edge)
        H > 0.5: Trending (bad for pairs trading)

        Uses R/S analysis method.
        """
        try:
            if len(prices) < 20:
                return 0.5  # Not enough data, assume random walk

            # Calculate returns
            returns = np.diff(np.log(prices))

            # R/S analysis for different lag sizes
            lags = range(2, min(len(returns) // 2, 20))
            rs_values = []

            for lag in lags:
                # Split into subseries
                n_subseries = len(returns) // lag
                if n_subseries < 2:
                    continue

                rs_subseries = []
                for i in range(n_subseries):
                    subseries = returns[i * lag : (i + 1) * lag]
                    if len(subseries) < 2:
                        continue

                    # Calculate R/S for this subseries
                    mean_ret = np.mean(subseries)
                    cumsum = np.cumsum(subseries - mean_ret)
                    r = max(cumsum) - min(cumsum)  # Range
                    s = np.std(subseries)  # Standard deviation

                    if s > 0:
                        rs_subseries.append(r / s)

                if rs_subseries:
                    rs_values.append((lag, np.mean(rs_subseries)))

            if len(rs_values) < 3:
                return 0.5

            # Log-log regression to find H
            log_lags = np.log([x[0] for x in rs_values])
            log_rs = np.log([x[1] for x in rs_values])

            # Hurst exponent is the slope
            hurst = np.polyfit(log_lags, log_rs, 1)[0]

            # Bound to reasonable range
            return max(0.0, min(1.0, hurst))

        except Exception as e:
            logger.debug(f"Error calculating Hurst exponent: {e}")
            return 0.5

    def _calculate_half_life(self, spread: np.ndarray) -> Optional[float]:
        """
        Calculate half-life of mean reversion for optimal holding period.

        Half-life = time for spread to revert halfway to mean.
        Uses Ornstein-Uhlenbeck model: spread_t = α + β * spread_{t-1} + ε
        Half-life = -ln(2) / ln(β)

        Returns:
            Half-life in periods (bars), or None if not mean-reverting
        """
        try:
            if len(spread) < 10:
                return None

            # Lag the spread
            spread_lag = spread[:-1]
            spread_current = spread[1:]

            # Add constant for regression
            spread_lag_const = add_constant(spread_lag)

            # OLS regression: spread_t = α + β * spread_{t-1}
            model = OLS(spread_current, spread_lag_const)
            result = model.fit()

            # β coefficient (how much spread reverts each period)
            beta = result.params[1]

            # Half-life calculation
            if beta <= 0 or beta >= 1:
                # Not mean-reverting
                return None

            half_life = -np.log(2) / np.log(beta)

            # Sanity check
            if half_life <= 0 or half_life > 100:
                return None

            return half_life

        except Exception as e:
            logger.debug(f"Error calculating half-life: {e}")
            return None

    async def _check_cointegration(self, pair: Tuple[str, str]):
        """
        Test if pair is cointegrated (has stable long-run relationship).

        Uses Engle-Granger two-step method with additional validation:
        1. Correlation check
        2. Cointegration test (Engle-Granger)
        3. Stationarity test (ADF)
        4. Hurst exponent (mean reversion confirmation)
        5. Half-life calculation (optimal holding period)
        """
        try:
            symbol1, symbol2 = pair

            # Check if we need to recalculate
            last_check = self.last_coint_check.get(pair)
            if last_check:
                days_since = (datetime.now() - last_check).days
                if days_since < self.parameters["hedge_ratio_recalc_days"]:
                    return  # Still valid

            # Get price series
            prices1 = np.array([bar["close"] for bar in self.price_history[symbol1]])
            prices2 = np.array([bar["close"] for bar in self.price_history[symbol2]])

            # Need enough history
            if len(prices1) < 30 or len(prices2) < 30:
                return

            # Use same length
            min_len = min(len(prices1), len(prices2))
            prices1 = prices1[-min_len:]
            prices2 = prices2[-min_len:]

            # Step 1: Check correlation
            correlation = np.corrcoef(prices1, prices2)[0, 1]

            if correlation < self.parameters["min_correlation"]:
                logger.debug(f"{pair} correlation too low: {correlation:.2f}")
                self.cointegration_results[pair] = {
                    "cointegrated": False,
                    "reason": "low_correlation",
                    "correlation": correlation,
                }
                return

            # Step 2: Run cointegration test (Engle-Granger)
            score, pvalue, _ = coint(prices1, prices2)

            # Calculate hedge ratio (beta from regression)
            # spread = prices1 - hedge_ratio * prices2
            hedge_ratio = np.polyfit(prices2, prices1, 1)[0]

            # Test if pair is cointegrated
            is_cointegrated = pvalue < self.parameters["cointegration_pvalue"]

            # Calculate spread
            spread = prices1 - hedge_ratio * prices2

            # Test spread for stationarity (should be stationary if cointegrated)
            adf_result = adfuller(spread)
            adf_pvalue = adf_result[1]
            is_stationary = adf_pvalue < 0.05

            # Calculate Hurst exponent for mean reversion confirmation
            hurst = self._calculate_hurst_exponent(spread)
            is_mean_reverting = hurst < 0.5  # H < 0.5 = mean reverting

            # Calculate half-life for optimal holding period
            half_life = self._calculate_half_life(spread)

            # All conditions must be met for trading
            is_tradeable = is_cointegrated and is_stationary and is_mean_reverting

            self.cointegration_results[pair] = {
                "cointegrated": is_tradeable,
                "correlation": correlation,
                "hedge_ratio": hedge_ratio,
                "coint_pvalue": pvalue,
                "adf_pvalue": adf_pvalue,
                "hurst_exponent": hurst,
                "half_life": half_life,
                "spread_mean": np.mean(spread),
                "spread_std": np.std(spread),
                "timestamp": datetime.now(),
            }

            self.last_coint_check[pair] = datetime.now()

            if is_tradeable:
                logger.info(f"✅ {pair} is COINTEGRATED and MEAN-REVERTING:")
                logger.info(f"   Correlation: {correlation:.3f}")
                logger.info(f"   Hedge Ratio: {hedge_ratio:.4f}")
                logger.info(f"   Coint p-value: {pvalue:.4f}")
                logger.info(f"   ADF p-value: {adf_pvalue:.4f}")
                logger.info(f"   Hurst exponent: {hurst:.3f} (< 0.5 = mean-reverting)")
                if half_life:
                    logger.info(f"   Half-life: {half_life:.1f} bars")
                logger.info(f"   Spread std: {np.std(spread):.4f}")
            else:
                reasons = []
                if not is_cointegrated:
                    reasons.append(f"coint p={pvalue:.3f}")
                if not is_stationary:
                    reasons.append(f"adf p={adf_pvalue:.3f}")
                if not is_mean_reverting:
                    reasons.append(f"hurst={hurst:.3f}")
                logger.debug(f"❌ {pair} NOT tradeable: {', '.join(reasons)}")

        except Exception as e:
            logger.error(f"Error checking cointegration for {pair}: {e}", exc_info=True)

    async def _calculate_spread(self, pair: Tuple[str, str]):
        """Calculate spread and z-score for pair."""
        try:
            symbol1, symbol2 = pair

            # Check if pair is cointegrated
            coint_result = self.cointegration_results.get(pair)
            if not coint_result or not coint_result.get("cointegrated"):
                return

            # Get current prices
            price1 = self.current_prices.get(symbol1)
            price2 = self.current_prices.get(symbol2)

            if not price1 or not price2:
                return

            # Get hedge ratio
            hedge_ratio = coint_result["hedge_ratio"]

            # Calculate spread: stock1 - hedge_ratio * stock2
            spread = price1 - hedge_ratio * price2

            # Store spread
            self.pair_spreads[pair].append({"timestamp": datetime.now(), "spread": spread})

            # Keep limited history
            if len(self.pair_spreads[pair]) > 100:
                self.pair_spreads[pair] = self.pair_spreads[pair][-100:]

            # Calculate z-score
            recent_spreads = [s["spread"] for s in self.pair_spreads[pair][-30:]]  # Last 30 bars

            if len(recent_spreads) >= 5:
                spread_mean = np.mean(recent_spreads)
                spread_std = np.std(recent_spreads)

                if spread_std > 0:
                    z_score = (spread - spread_mean) / spread_std
                else:
                    z_score = 0

                self.pair_stats[pair] = {
                    "spread": spread,
                    "spread_mean": spread_mean,
                    "spread_std": spread_std,
                    "z_score": z_score,
                    "hedge_ratio": hedge_ratio,
                    "price1": price1,
                    "price2": price2,
                }

        except Exception as e:
            logger.error(f"Error calculating spread for {pair}: {e}", exc_info=True)

    async def _generate_pair_signal(self, pair: Tuple[str, str]):
        """Generate trading signal for pair based on z-score."""
        try:
            stats = self.pair_stats.get(pair)
            if not stats or "z_score" not in stats:
                self.pair_signals[pair] = "neutral"
                return

            z_score = stats["z_score"]
            entry_threshold = self.parameters["entry_z_score"]

            # Entry signals
            # z-score > 2: spread too wide, short spread (sell stock1, buy stock2)
            # z-score < -2: spread too narrow, long spread (buy stock1, sell stock2)

            if z_score > entry_threshold:
                # Spread is too wide - expect it to narrow
                # Short the spread: sell stock1 (expensive), buy stock2 (cheap)
                self.pair_signals[pair] = "short_spread"
                logger.debug(f"{pair} signal: SHORT spread (z={z_score:.2f})")

            elif z_score < -entry_threshold:
                # Spread is too narrow - expect it to widen
                # Long the spread: buy stock1 (cheap), sell stock2 (expensive)
                self.pair_signals[pair] = "long_spread"
                logger.debug(f"{pair} signal: LONG spread (z={z_score:.2f})")

            else:
                self.pair_signals[pair] = "neutral"

        except Exception as e:
            logger.error(f"Error generating signal for {pair}: {e}", exc_info=True)
            self.pair_signals[pair] = "neutral"

    async def _execute_pair_trade(self, pair: Tuple[str, str], signal: str):
        """Execute pair trade (long one stock, short the other)."""
        try:
            symbol1, symbol2 = pair

            # Check if we already have position in this pair
            if pair in self.pair_positions:
                logger.debug(f"Already have position in {pair}")
                return

            # Check max pairs
            if len(self.pair_positions) >= self.max_pairs:
                logger.info(f"Max pairs ({self.max_pairs}) reached")
                return

            # Get account
            account = await self.broker.get_account()
            buying_power = float(account.buying_power)

            # Calculate position sizes
            stats = self.pair_stats[pair]
            price1 = stats["price1"]
            price2 = stats["price2"]
            hedge_ratio = stats["hedge_ratio"]

            # Total capital for this pair (split between both positions)
            pair_capital = buying_power * self.position_size

            # For market-neutral pair:
            # We want: quantity1 * price1 = hedge_ratio * quantity2 * price2
            # And: quantity1 * price1 + quantity2 * price2 = pair_capital

            # Solve for quantities
            # Let value1 = quantity1 * price1
            # Then value2 = value1 / hedge_ratio
            # And value1 + value2 = pair_capital
            # So value1 = pair_capital / (1 + 1/hedge_ratio)

            value1 = pair_capital / (1 + 1 / hedge_ratio)
            value2 = pair_capital - value1

            quantity1 = value1 / price1
            quantity2 = value2 / price2

            # Minimum position check
            if quantity1 < 0.01 or quantity2 < 0.01:
                logger.info(f"Position sizes too small for {pair}")
                return

            # Determine sides based on signal
            if signal == "long_spread":
                # Long the spread: BUY stock1, SELL (short) stock2
                side1 = "buy"
                side2 = "sell"
                logger.info(f"LONG spread {pair}:")
            else:  # short_spread
                # Short the spread: SELL stock1, BUY stock2
                side1 = "sell"
                side2 = "buy"
                logger.info(f"SHORT spread {pair}:")

            logger.info(f"  {side1.upper()} {symbol1}: {quantity1:.4f} @ ${price1:.2f}")
            logger.info(f"  {side2.upper()} {symbol2}: {quantity2:.4f} @ ${price2:.2f}")
            logger.info(f"  Z-score: {stats['z_score']:.2f}")
            logger.info(f"  Hedge ratio: {hedge_ratio:.4f}")

            # Execute both orders
            order1 = OrderBuilder(symbol1, side1, quantity1).market().day().build()

            order2 = OrderBuilder(symbol2, side2, quantity2).market().day().build()

            result1 = await self.submit_entry_order(
                order1,
                reason="pairs_entry",
                max_positions=self.max_positions,
            )
            result2 = await self.submit_entry_order(
                order2,
                reason="pairs_entry",
                max_positions=self.max_positions,
            )

            success1 = result1 and (not hasattr(result1, "success") or result1.success)
            success2 = result2 and (not hasattr(result2, "success") or result2.success)

            if success1 and success2:
                logger.info(f"✅ Pair trade executed: {pair}")

                # Get half-life for exit timing
                coint_result = self.cointegration_results.get(pair, {})
                half_life = coint_result.get("half_life")

                # Track position
                self.pair_positions[pair] = {
                    "entry_time": datetime.now(),
                    "signal": signal,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "quantity1": quantity1,
                    "quantity2": quantity2,
                    "price1": price1,
                    "price2": price2,
                    "side1": side1,
                    "side2": side2,
                    "entry_z_score": stats["z_score"],
                    "hedge_ratio": hedge_ratio,
                    "half_life": half_life,
                }
            else:
                logger.error(f"❌ Failed to execute pair trade for {pair}")

        except Exception as e:
            logger.error(f"Error executing pair trade for {pair}: {e}", exc_info=True)

    async def _check_pair_exit(self, pair: Tuple[str, str]):
        """Check if we should exit pair position."""
        try:
            if pair not in self.pair_positions:
                return

            position = self.pair_positions[pair]
            stats = self.pair_stats.get(pair)

            if not stats or "z_score" not in stats:
                return

            z_score = stats["z_score"]
            entry_z_score = position["entry_z_score"]

            # Exit conditions:
            # 1. Z-score reverted to near zero (take profit)
            # 2. Z-score went even further out (stop loss)
            # 3. Maximum holding period reached
            # 4. P/L threshold reached

            should_exit = False
            exit_reason = ""

            # 1. Z-score reversion (profit target)
            if abs(z_score) < self.parameters["exit_z_score"]:
                should_exit = True
                exit_reason = f"z-score reverted ({z_score:.2f})"

            # 2. Z-score divergence (stop loss)
            if abs(z_score) > self.parameters["stop_loss_z_score"]:
                should_exit = True
                exit_reason = f"z-score diverged ({z_score:.2f})"

            # 3. Max holding period (use half-life if available)
            entry_time = position["entry_time"]
            holding_hours = (datetime.now() - entry_time).total_seconds() / 3600
            holding_days = holding_hours / 24

            # Determine max holding period
            half_life = position.get("half_life")
            if self.parameters.get("use_half_life_exit") and half_life:
                # Exit after 3x half-life (99.5% mean reversion expected)
                max_holding = half_life * self.parameters.get("half_life_multiplier", 3.0)
                max_holding_hours = max_holding  # half-life is in bars, assume hourly
                if holding_hours >= max_holding_hours:
                    should_exit = True
                    exit_reason = (
                        f"half-life exit ({holding_hours:.1f}h > {max_holding_hours:.1f}h)"
                    )
            elif holding_days >= self.parameters["max_holding_days"]:
                should_exit = True
                exit_reason = f"max holding period ({holding_days:.1f} days)"

            # 4. P/L check
            symbol1 = position["symbol1"]
            symbol2 = position["symbol2"]
            current_price1 = self.current_prices.get(symbol1)
            current_price2 = self.current_prices.get(symbol2)

            if current_price1 and current_price2:
                # Calculate P/L
                entry_price1 = position["price1"]
                entry_price2 = position["price2"]
                quantity1 = position["quantity1"]
                quantity2 = position["quantity2"]
                side1 = position["side1"]
                side2 = position["side2"]

                # P/L for each leg
                if side1 == "buy":
                    pnl1 = (current_price1 - entry_price1) * quantity1
                else:
                    pnl1 = (entry_price1 - current_price1) * quantity1

                if side2 == "buy":
                    pnl2 = (current_price2 - entry_price2) * quantity2
                else:
                    pnl2 = (entry_price2 - current_price2) * quantity2

                total_pnl = pnl1 + pnl2
                entry_value = entry_price1 * quantity1 + entry_price2 * quantity2
                pnl_pct = total_pnl / entry_value if entry_value > 0 else 0

                # Take profit
                if pnl_pct >= self.parameters["take_profit_pct"]:
                    should_exit = True
                    exit_reason = f"take profit ({pnl_pct:.1%})"

                # Stop loss
                if pnl_pct <= -self.parameters["stop_loss_pct"]:
                    should_exit = True
                    exit_reason = f"stop loss ({pnl_pct:.1%})"

                # Exit if conditions met
                if should_exit:
                    logger.info(f"Exiting pair {pair}: {exit_reason}")
                    logger.info(f"  P/L: ${total_pnl:+,.2f} ({pnl_pct:+.2%})")
                    logger.info(f"  Z-score: {entry_z_score:.2f} → {z_score:.2f}")

                    # Close both positions
                    # Reverse the original sides
                    exit_side1 = "sell" if side1 == "buy" else "buy"
                    exit_side2 = "sell" if side2 == "buy" else "buy"

                    result1 = await self.submit_exit_order(
                        symbol=symbol1,
                        qty=quantity1,
                        side=exit_side1,
                        reason="pairs_exit",
                    )
                    result2 = await self.submit_exit_order(
                        symbol=symbol2,
                        qty=quantity2,
                        side=exit_side2,
                        reason="pairs_exit",
                    )

                    if result1 and result2:
                        logger.info(f"✅ Pair position closed: {pair}")
                        del self.pair_positions[pair]

        except Exception as e:
            logger.error(f"Error checking pair exit for {pair}: {e}", exc_info=True)

    async def analyze_symbol(self, symbol):
        """Not used for pairs trading."""
        return "neutral"

    async def execute_trade(self, symbol, signal):
        """Not used for pairs trading."""
        pass

    async def generate_signals(self):
        """Generate signals for backtest mode."""
        pass

    def get_orders(self):
        """Get orders for backtest."""
        return []

    async def export_state(self) -> dict:
        """Export pair positions for restart recovery."""
        def _dt(v):
            return v.isoformat() if hasattr(v, "isoformat") else v

        positions = {}
        for pair, data in self.pair_positions.items():
            item = data.copy()
            if "entry_time" in item:
                item["entry_time"] = _dt(item["entry_time"])
            positions[str(pair)] = item
        return {"pair_positions": positions}

    async def import_state(self, state: dict) -> None:
        """Restore pair positions after restart."""
        from datetime import datetime

        def _parse_dt(v):
            return datetime.fromisoformat(v) if isinstance(v, str) else v

        positions = {}
        for key, data in state.get("pair_positions", {}).items():
            item = data.copy()
            if "entry_time" in item:
                item["entry_time"] = _parse_dt(item["entry_time"])
            try:
                pair = tuple(key.strip("()").replace("'", "").split(", "))
                if len(pair) == 2:
                    positions[(pair[0], pair[1])] = item
                else:
                    positions[key] = item
            except Exception:
                positions[key] = item

        self.pair_positions = positions
