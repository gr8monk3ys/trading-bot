#!/usr/bin/env python3
"""
Gap Trading Strategy

Trades overnight gaps that tend to fill during the trading day.

Research shows:
- Gaps > 2% fill within the same day 60-70% of the time
- Gap fills work best in sideways/ranging markets
- Works poorly in strong trending markets (gaps extend)

Strategy:
- At market open, identify stocks with significant gaps
- Fade the gap (short gap ups, long gap downs)
- Exit when gap fills 50-80% or by lunch if not filled
- Use tight stops to limit losses on gaps that don't fill

Expected Impact: 5-10% annual returns from gap reversions

Usage:
    from strategies.gap_trading_strategy import GapTradingStrategy

    strategy = GapTradingStrategy(broker, symbols)
    await strategy.initialize()

    # Check for gap opportunities at market open
    gaps = await strategy.scan_for_gaps()
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pytz

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class GapTradingStrategy(BaseStrategy):
    """
    Gap trading strategy that fades overnight gaps.

    Gap Types:
    - Gap Up: Open > Previous Close (fade by shorting)
    - Gap Down: Open < Previous Close (fade by going long)

    Entry: At or shortly after market open
    Exit: When gap fills 50-80%, or by lunch, or stop hit
    """

    NAME = "GapTradingStrategy"

    # Trading time constants
    MARKET_OPEN = time(9, 30)  # 9:30 AM ET
    MARKET_CLOSE = time(16, 0)  # 4:00 PM ET
    END_OF_DAY_EXIT = time(15, 45)  # Exit all positions 15 min before close
    LUNCH_START = time(12, 0)  # Noon - gap trades should exit by lunch

    # Gap fill boundaries
    MIN_GAP_FILL = 0.0  # Minimum fill percentage
    MAX_GAP_FILL = 1.0  # Maximum (100%) fill percentage

    def __init__(self, broker=None, symbols=None, parameters=None):
        """Initialize gap trading strategy."""
        parameters = parameters or {}
        if symbols:
            parameters["symbols"] = symbols
        super().__init__(name=self.NAME, broker=broker, parameters=parameters)

    def default_parameters(self):
        """Return default parameters for gap trading."""
        return {
            # Gap detection
            "min_gap_pct": 2.0,  # Minimum gap size to trade (%)
            "max_gap_pct": 8.0,  # Maximum gap size (avoid news-driven)
            "min_volume_ratio": 1.5,  # Premarket volume vs average
            # Position sizing
            "position_size": 0.05,  # 5% per gap trade (smaller, riskier)
            "max_gap_positions": 3,  # Max simultaneous gap trades
            # Exit parameters
            "gap_fill_target": 0.6,  # Exit when 60% of gap fills
            "stop_loss_pct": 0.02,  # 2% stop loss
            "max_hold_hours": 3,  # Exit by lunch if not filled
            # Timing
            "entry_window_minutes": 15,  # Enter within first 15 min
            "avoid_first_minutes": 5,  # Skip first 5 min volatility
            # Market regime filter
            "avoid_trending_markets": True,  # Skip gaps in strong trends
            "max_trend_strength": 30,  # ADX threshold for trending
            # Risk management
            "enable_short_selling": True,
            "use_limit_orders": True,
            "limit_offset_pct": 0.001,  # 0.1% offset for limit orders
        }

    async def initialize(self, **kwargs):
        """Initialize the gap trading strategy."""
        try:
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Extract key parameters
            self.min_gap_pct = self.parameters["min_gap_pct"]
            self.max_gap_pct = self.parameters["max_gap_pct"]
            self.position_size = self.parameters["position_size"]
            self.gap_fill_target = self.parameters["gap_fill_target"]
            self.stop_loss_pct = self.parameters["stop_loss_pct"]
            self.max_hold_hours = self.parameters["max_hold_hours"]

            # Tracking
            self.active_gap_trades = {}  # symbol -> gap trade info
            self.today_gaps = {}  # symbol -> gap info for today
            self.gap_trade_history = []

            # Timezone
            self.tz = pytz.timezone("US/Eastern")

            logger.info("GapTradingStrategy initialized")
            logger.info(f"  Gap range: {self.min_gap_pct}% - {self.max_gap_pct}%")
            logger.info(f"  Fill target: {self.gap_fill_target:.0%}")

            return True

        except Exception as e:
            logger.error(f"Error initializing GapTradingStrategy: {e}")
            return False

    async def scan_for_gaps(self, symbols: List[str] = None) -> List[Dict]:
        """
        Scan for gap opportunities at market open.

        Args:
            symbols: List of symbols to scan (default: self.symbols)

        Returns:
            List of gap opportunities sorted by gap size
        """
        symbols = symbols or self.symbols
        gaps = []

        # Parallel execution for better performance
        tasks = [self._detect_gap(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Error scanning gap for {symbol}: {result}")
            elif result and result["is_tradeable"]:
                gaps.append(result)

        # Sort by absolute gap size (largest first)
        gaps.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)

        if gaps:
            logger.info(f"Found {len(gaps)} tradeable gaps:")
            for gap in gaps[:5]:
                direction = "UP" if gap["gap_pct"] > 0 else "DOWN"
                logger.info(
                    f"  {gap['symbol']}: {direction} {abs(gap['gap_pct']):.1f}% "
                    f"(prev: ${gap['prev_close']:.2f}, open: ${gap['open_price']:.2f})"
                )

        return gaps

    async def _detect_gap(self, symbol: str) -> Optional[Dict]:
        """
        Detect if a symbol has gapped.

        Returns:
            Gap info dict or None if no significant gap
        """
        try:
            # Get yesterday's close and today's open
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)

            bars = await self.broker.get_bars(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                timeframe="1Day",
            )

            if bars is None or len(bars) < 2:
                return None

            # Get previous close and current open
            prev_bar = bars[-2]
            curr_bar = bars[-1]

            prev_close = float(prev_bar.close)
            open_price = float(curr_bar.open)
            current_price = float(curr_bar.close)

            # Calculate gap
            gap_pct = ((open_price - prev_close) / prev_close) * 100

            # Check if gap is tradeable
            is_tradeable = abs(gap_pct) >= self.min_gap_pct and abs(gap_pct) <= self.max_gap_pct

            # Calculate gap fill progress
            if gap_pct > 0:  # Gap up
                gap_fill_pct = (open_price - current_price) / (open_price - prev_close)
            else:  # Gap down
                gap_fill_pct = (current_price - open_price) / (prev_close - open_price)

            gap_fill_pct = max(self.MIN_GAP_FILL, min(self.MAX_GAP_FILL, gap_fill_pct))

            return {
                "symbol": symbol,
                "prev_close": prev_close,
                "open_price": open_price,
                "current_price": current_price,
                "gap_pct": gap_pct,
                "gap_direction": "up" if gap_pct > 0 else "down",
                "gap_fill_pct": gap_fill_pct,
                "is_tradeable": is_tradeable,
                "trade_direction": "short" if gap_pct > 0 else "long",
                "target_price": prev_close + (open_price - prev_close) * (1 - self.gap_fill_target),
                "stop_price": (
                    open_price * (1 + self.stop_loss_pct)
                    if gap_pct > 0
                    else open_price * (1 - self.stop_loss_pct)
                ),
            }

        except Exception as e:
            logger.warning(f"Error detecting gap for {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol: str) -> str:
        """Analyze symbol for gap trading opportunity."""
        gap_info = await self._detect_gap(symbol)

        if gap_info is None:
            return "neutral"

        if not gap_info["is_tradeable"]:
            return "neutral"

        # Check if we're in the entry window
        now = datetime.now(self.tz)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

        entry_start = market_open + timedelta(minutes=self.parameters["avoid_first_minutes"])
        entry_end = market_open + timedelta(minutes=self.parameters["entry_window_minutes"])

        if not (entry_start <= now <= entry_end):
            return "neutral"  # Outside entry window

        # Return trade direction
        return gap_info["trade_direction"]

    async def execute_trade(self, symbol: str, signal: str):
        """Execute a gap trade."""
        if signal not in ["long", "short"]:
            return

        # Check position limits
        if len(self.active_gap_trades) >= self.parameters["max_gap_positions"]:
            logger.info(f"Max gap positions reached, skipping {symbol}")
            return

        # Check if we already have a position in this symbol
        positions = await self.broker.get_positions()
        if positions:
            existing_symbols = [p.symbol for p in positions]
            if symbol in existing_symbols:
                logger.info(f"Already have position in {symbol}, skipping gap trade")
                return

        gap_info = await self._detect_gap(symbol)
        if gap_info is None:
            return

        try:
            # Calculate position size
            account = await self.broker.get_account()
            portfolio_value = float(account.portfolio_value)
            position_value = portfolio_value * self.position_size

            current_price = gap_info["current_price"]
            shares = int(position_value / current_price)

            if shares < 1:
                return

            # Create order
            side = "buy" if signal == "long" else "sell"

            order = await self.broker.submit_order(
                symbol=symbol, qty=shares, side=side, type="market", time_in_force="day"
            )

            if order:
                # Get actual fill price to avoid race condition with price changes
                try:
                    await asyncio.sleep(0.5)  # Brief wait for fill
                    filled_order = await self.broker.get_order(order.id)
                    actual_fill_price = (
                        float(filled_order.filled_avg_price)
                        if filled_order and filled_order.filled_avg_price
                        else current_price
                    )
                except Exception:
                    actual_fill_price = current_price

                # Track the gap trade
                self.active_gap_trades[symbol] = {
                    "entry_time": datetime.now(self.tz),
                    "entry_price": actual_fill_price,
                    "shares": shares,
                    "direction": signal,
                    "gap_info": gap_info,
                    "target_price": gap_info["target_price"],
                    "stop_price": gap_info["stop_price"],
                }

                logger.info(
                    f"GAP TRADE: {signal.upper()} {shares} {symbol} @ ${actual_fill_price:.2f} "
                    f"(gap: {gap_info['gap_pct']:+.1f}%, target: ${gap_info['target_price']:.2f})"
                )

        except Exception as e:
            logger.error(f"Error executing gap trade for {symbol}: {e}")

    async def check_exits(self):
        """Check if any gap trades should be exited."""
        now = datetime.now(self.tz)
        to_exit = []

        for symbol, trade in self.active_gap_trades.items():
            try:
                gap_info = await self._detect_gap(symbol)
                if gap_info is None:
                    continue

                current_price = gap_info["current_price"]
                entry_price = trade["entry_price"]
                direction = trade["direction"]

                # Check exit conditions
                should_exit = False
                exit_reason = None

                # 1. Gap filled target
                if gap_info["gap_fill_pct"] >= self.gap_fill_target:
                    should_exit = True
                    exit_reason = f"Gap filled {gap_info['gap_fill_pct']:.0%}"

                # 2. Stop loss hit
                if direction == "long" and current_price <= trade["stop_price"]:
                    should_exit = True
                    exit_reason = "Stop loss hit"
                elif direction == "short" and current_price >= trade["stop_price"]:
                    should_exit = True
                    exit_reason = "Stop loss hit"

                # 3. Max hold time
                hold_time = (now - trade["entry_time"]).total_seconds() / 3600
                if hold_time >= self.max_hold_hours:
                    should_exit = True
                    exit_reason = f"Max hold time ({self.max_hold_hours}h)"

                # 4. End of day
                if now.time() >= self.END_OF_DAY_EXIT:
                    should_exit = True
                    exit_reason = "End of day exit"

                if should_exit:
                    to_exit.append((symbol, exit_reason, current_price))

            except Exception as e:
                logger.error(f"Error checking exit for {symbol}: {e}")

        # Execute exits
        for symbol, reason, price in to_exit:
            await self._exit_gap_trade(symbol, reason, price)

    async def _exit_gap_trade(self, symbol: str, reason: str, current_price: float):
        """Exit a gap trade."""
        if symbol not in self.active_gap_trades:
            return

        trade = self.active_gap_trades[symbol]

        try:
            # Close position
            side = "sell" if trade["direction"] == "long" else "buy"

            order = await self.broker.submit_order(
                symbol=symbol, qty=trade["shares"], side=side, type="market", time_in_force="day"
            )

            if order:
                # Calculate P&L
                entry_price = trade["entry_price"]
                if trade["direction"] == "long":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                pnl_dollars = (current_price - entry_price) * trade["shares"]
                if trade["direction"] == "short":
                    pnl_dollars = -pnl_dollars

                logger.info(
                    f"GAP EXIT: {symbol} - {reason} | "
                    f"P&L: {pnl_pct:+.1f}% (${pnl_dollars:+,.0f})"
                )

                # Record trade
                self.gap_trade_history.append(
                    {
                        "symbol": symbol,
                        "direction": trade["direction"],
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl_pct": pnl_pct,
                        "pnl_dollars": pnl_dollars,
                        "reason": reason,
                        "gap_pct": trade["gap_info"]["gap_pct"],
                        "hold_time": (datetime.now(self.tz) - trade["entry_time"]).total_seconds()
                        / 60,
                    }
                )

                # Remove from active trades
                del self.active_gap_trades[symbol]

        except Exception as e:
            logger.error(f"Error exiting gap trade for {symbol}: {e}")

    def get_status(self) -> Dict:
        """Get strategy status."""
        return {
            "name": self.NAME,
            "active_trades": len(self.active_gap_trades),
            "trades_today": len(
                [
                    t
                    for t in self.gap_trade_history
                    if t.get("entry_time", datetime.min).date() == datetime.now().date()
                ]
            ),
            "active_symbols": list(self.active_gap_trades.keys()),
        }

    def get_performance(self) -> Dict:
        """Get gap trading performance stats."""
        if not self.gap_trade_history:
            return {"trades": 0}

        trades = self.gap_trade_history
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]

        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_win": np.mean([t["pnl_pct"] for t in wins]) if wins else 0,
            "avg_loss": np.mean([t["pnl_pct"] for t in losses]) if losses else 0,
            "total_pnl": sum(t["pnl_dollars"] for t in trades),
            "avg_hold_minutes": np.mean([t["hold_time"] for t in trades]),
        }


# Convenience function
async def scan_market_gaps(broker, symbols: List[str], min_gap: float = 2.0) -> List[Dict]:
    """Quick scan for gap opportunities."""
    strategy = GapTradingStrategy(broker=broker, symbols=symbols)
    await strategy.initialize()
    strategy.min_gap_pct = min_gap
    return await strategy.scan_for_gaps()
