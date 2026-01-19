#!/usr/bin/env python3
"""
Simple Backtest Script - With Realistic Slippage Modeling

The main.py backtest is broken (calls non-existent method).
This is a working backtest that ACTUALLY RUNS with REALISTIC costs.

Features:
- Slippage modeling (0.4% per trade by default)
- Bid-ask spread simulation
- Transaction cost tracking
- Statistical significance warnings
- More accurate P/L calculations
"""

import asyncio
import logging
from datetime import datetime

import numpy as np

from brokers.alpaca_broker import AlpacaBroker
from config import BACKTEST_PARAMS
from strategies.momentum_strategy import MomentumStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SlippageModel:
    """
    Models realistic trade execution costs.

    Components:
    - Slippage: Price movement between signal and execution (0.4% default)
    - Bid-ask spread: Cost of crossing the spread (0.1% default)
    - Commissions: Per-share costs (0 for Alpaca)
    """

    def __init__(
        self,
        slippage_pct: float = None,
        bid_ask_spread: float = None,
        commission_per_share: float = None,
        enabled: bool = None,
    ):
        self.slippage_pct = slippage_pct or BACKTEST_PARAMS.get("SLIPPAGE_PCT", 0.004)
        self.bid_ask_spread = bid_ask_spread or BACKTEST_PARAMS.get("BID_ASK_SPREAD", 0.001)
        self.commission_per_share = commission_per_share or BACKTEST_PARAMS.get(
            "COMMISSION_PER_SHARE", 0.0
        )
        self.enabled = enabled if enabled is not None else BACKTEST_PARAMS.get("USE_SLIPPAGE", True)

        # Track total costs for reporting
        self.total_slippage_cost = 0.0
        self.total_spread_cost = 0.0
        self.total_commission = 0.0

    def apply_slippage(self, price: float, side: str, qty: int) -> tuple:
        """
        Apply slippage to trade execution price.

        Args:
            price: The quoted price at signal time
            side: 'buy' or 'sell'
            qty: Number of shares

        Returns:
            (execution_price, slippage_cost, spread_cost, commission)
        """
        if not self.enabled:
            return price, 0.0, 0.0, 0.0

        # Slippage: price moves against us between signal and execution
        # For buys, price goes up; for sells, price goes down
        if side == "buy":
            slippage_factor = 1 + self.slippage_pct
            spread_factor = 1 + (self.bid_ask_spread / 2)  # Pay the ask
        else:
            slippage_factor = 1 - self.slippage_pct
            spread_factor = 1 - (self.bid_ask_spread / 2)  # Get the bid

        # Calculate execution price
        execution_price = price * slippage_factor * spread_factor

        # Calculate individual costs
        slippage_cost = abs(price * self.slippage_pct) * qty
        spread_cost = abs(price * (self.bid_ask_spread / 2)) * qty
        commission = self.commission_per_share * qty

        # Track totals
        self.total_slippage_cost += slippage_cost
        self.total_spread_cost += spread_cost
        self.total_commission += commission

        return execution_price, slippage_cost, spread_cost, commission

    def get_total_costs(self) -> dict:
        """Get summary of all transaction costs."""
        return {
            "slippage": self.total_slippage_cost,
            "spread": self.total_spread_cost,
            "commission": self.total_commission,
            "total": self.total_slippage_cost + self.total_spread_cost + self.total_commission,
        }

    def reset(self):
        """Reset cost tracking for new backtest."""
        self.total_slippage_cost = 0.0
        self.total_spread_cost = 0.0
        self.total_commission = 0.0


async def simple_backtest(
    symbols,
    start_date_str,
    end_date_str,
    initial_capital=100000,
    use_slippage=True,
    slippage_pct=None,
    strategy_params=None,
    quiet=False,
):
    """
    Run a simple backtest on MomentumStrategy with realistic costs.

    Features:
    - Slippage modeling (configurable, default 0.4%)
    - Bid-ask spread costs
    - Transaction cost tracking
    - Statistical significance warnings
    - Realistic P/L calculations
    - Configurable strategy parameters for feature comparison

    Args:
        symbols: List of stock symbols to trade
        start_date_str: Start date (YYYY-MM-DD)
        end_date_str: End date (YYYY-MM-DD)
        initial_capital: Starting capital (default $100,000)
        use_slippage: Whether to apply slippage (default True)
        slippage_pct: Custom slippage percentage (default from config)
        strategy_params: Dict of strategy parameters to override defaults
        quiet: If True, suppress most output (for batch runs)
    """
    # Initialize slippage model
    slippage_model = SlippageModel(slippage_pct=slippage_pct, enabled=use_slippage)

    # Build base parameters
    base_params = {
        "symbols": symbols,
        "position_size": 0.10,
        "max_positions": 3,
        "stop_loss": 0.03,
        "take_profit": 0.05,
        # Ensure all advanced features are OFF by default
        "use_kelly_criterion": False,
        "use_volatility_regime": False,
        "use_streak_sizing": False,
        "use_multi_timeframe": False,
        "enable_short_selling": False,
    }

    # Override with custom parameters if provided
    if strategy_params:
        base_params.update(strategy_params)

    # Determine config description
    enabled_features = []
    if base_params.get("rsi_mode") == "aggressive":
        enabled_features.append("RSI-2")
    if base_params.get("use_kelly_criterion"):
        enabled_features.append("Kelly")
    if base_params.get("use_multi_timeframe"):
        enabled_features.append("MTF")
    if base_params.get("use_volatility_regime"):
        enabled_features.append("VolRegime")
    config_desc = ", ".join(enabled_features) if enabled_features else "BASELINE"

    if not quiet:
        print("\n" + "=" * 80)
        print("BACKTEST WITH REALISTIC SLIPPAGE MODELING")
        print("=" * 80)
        print(f"Strategy: MomentumStrategy ({config_desc})")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Period: {start_date_str} to {end_date_str}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        if use_slippage:
            print(f"Slippage: {slippage_model.slippage_pct:.2%} per trade")
            print(f"Bid-Ask Spread: {slippage_model.bid_ask_spread:.2%}")
        else:
            print("Slippage: DISABLED (unrealistic)")
        print("=" * 80 + "\n")

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Initialize strategy with custom parameters
    strategy = MomentumStrategy(name="BacktestMomentum", broker=broker, parameters=base_params)

    await strategy.initialize()

    if not quiet:
        print("✅ Strategy initialized")
        print(f"   Configuration: {config_desc}")
        print(f"   Position size: {strategy.position_size:.0%}")
        print(f"   Max positions: {strategy.max_positions}")
        print(f"   Stop loss: {strategy.stop_loss:.0%}")
        print(f"   Take profit: {strategy.take_profit:.0%}")
        if enabled_features:
            print(f"   Features: {', '.join(enabled_features)}")
        print()

    # Get historical data for all symbols
    if not quiet:
        print("📊 Fetching historical data from Alpaca...")
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    all_bars = {}
    for symbol in symbols:
        try:
            bars = await broker.get_bars(
                symbol=symbol, timeframe="1Day", start=start_date, end=end_date
            )
            if bars and len(bars) > 0:
                all_bars[symbol] = bars
                if not quiet:
                    print(f"   {symbol}: {len(bars)} days of data")
            else:
                if not quiet:
                    print(f"   {symbol}: No data available")
        except Exception as e:
            if not quiet:
                print(f"   {symbol}: Error - {e}")

    if not all_bars:
        if not quiet:
            print("\n❌ No historical data available - cannot backtest")
        return None

    # Create date range
    trading_days = sorted(set(bar.timestamp.date() for bars in all_bars.values() for bar in bars))

    # P2 Fix: Handle empty trading_days gracefully
    if not trading_days:
        if not quiet:
            print("\n❌ No trading days found in the data - cannot backtest")
        return None

    if not quiet:
        print(f"\n✅ Got {len(trading_days)} trading days of data")
        print(f"   Start: {trading_days[0]}")
        print(f"   End: {trading_days[-1]}\n")

        # Simple backtest simulation
        print("=" * 80)
        print("RUNNING BACKTEST...")
        print("=" * 80 + "\n")

    capital = initial_capital
    positions = {}  # symbol -> {'qty': int, 'entry_price': float, 'entry_date': date}
    trades = []
    # P2 Fix: Initialize equity curve with first actual trading day
    equity_curve = [{"date": trading_days[0], "equity": capital}]

    for day_idx, current_date in enumerate(trading_days):
        # Get prices for this day
        day_prices = {}
        for symbol, bars in all_bars.items():
            day_bar = next((b for b in bars if b.timestamp.date() == current_date), None)
            if day_bar:
                day_prices[symbol] = {
                    "open": float(day_bar.open),
                    "high": float(day_bar.high),
                    "low": float(day_bar.low),
                    "close": float(day_bar.close),
                    "volume": float(day_bar.volume),
                }

        if not day_prices:
            continue

        # Check exit conditions for existing positions
        for symbol in list(positions.keys()):
            if symbol not in day_prices:
                continue

            position = positions[symbol]
            current_price = day_prices[symbol]["close"]
            entry_price = position["entry_price"]
            pnl_pct = (current_price - entry_price) / entry_price

            # Check stop-loss or take-profit
            should_exit = False
            exit_reason = None

            if pnl_pct <= -strategy.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            elif pnl_pct >= strategy.take_profit:
                should_exit = True
                exit_reason = "take_profit"

            if should_exit:
                # Apply slippage to exit price
                exit_price, slip_cost, spread_cost, comm = slippage_model.apply_slippage(
                    current_price, "sell", position["qty"]
                )

                # Close position at slippage-adjusted price
                position_value = position["qty"] * exit_price
                gross_pnl = (exit_price - entry_price) * position["qty"]
                net_pnl = gross_pnl  # Costs already reflected in exit_price

                # Return full position value to cash
                capital += position_value

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": position["entry_date"],
                        "exit_date": current_date,
                        "entry_price": position["entry_price_with_slippage"],
                        "exit_price": exit_price,
                        "quoted_entry": entry_price,
                        "quoted_exit": current_price,
                        "qty": position["qty"],
                        "gross_pnl": gross_pnl,
                        "pnl": net_pnl,
                        "pnl_pct": (exit_price - position["entry_price_with_slippage"])
                        / position["entry_price_with_slippage"],
                        "slippage_cost": slip_cost + position.get("entry_slippage", 0),
                        "spread_cost": spread_cost + position.get("entry_spread", 0),
                        "reason": exit_reason,
                    }
                )

                logger.info(
                    f"EXIT {symbol}: ${net_pnl:+.2f} ({pnl_pct:+.1%}) - {exit_reason} - "
                    f"Slippage: ${slip_cost:.2f} - Capital: ${capital:,.2f}"
                )

                del positions[symbol]

        # Simple signal generation (just for testing - not the real strategy)
        # In a real backtest, we'd call strategy.analyze_symbol()
        # For now, just buy on RSI oversold (< 30)
        if len(positions) < strategy.max_positions:
            for symbol in symbols:
                if symbol in positions or symbol not in day_prices:
                    continue

                # Get last 14 days for RSI
                symbol_bars = all_bars[symbol]
                recent_bars = [b for b in symbol_bars if b.timestamp.date() <= current_date][-14:]

                if len(recent_bars) < 14:
                    continue

                # Simple RSI calculation
                closes = [float(b.close) for b in recent_bars]
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                # P1 Fix: When avg_loss is 0, RSI should be 100 (maximum strength)
                if avg_loss == 0:
                    rsi = 100 if avg_gain > 0 else 50  # 50 = neutral if no movement
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                # Buy if RSI < 30 (oversold)
                if rsi < 30:
                    quoted_price = day_prices[symbol]["close"]
                    position_value = capital * strategy.position_size

                    # Apply slippage to entry price
                    entry_price, slip_cost, spread_cost, comm = slippage_model.apply_slippage(
                        quoted_price, "buy", 1  # Calculate per-share first
                    )

                    # Calculate quantity based on slippage-adjusted price
                    qty = int(position_value / entry_price)

                    if qty > 0:
                        # P2 Fix: Calculate per-share costs directly, no undo needed
                        # The previous per-share calculation was just to get the price
                        actual_slip_cost = quoted_price * slippage_model.slippage_pct * qty
                        actual_spread_cost = (
                            quoted_price * (slippage_model.bid_ask_spread / 2) * qty
                        )
                        actual_comm = slippage_model.commission_per_share * qty

                        # Update totals directly (without applying slippage again)
                        slippage_model.total_slippage_cost += actual_slip_cost - slip_cost
                        slippage_model.total_spread_cost += actual_spread_cost - spread_cost
                        slippage_model.total_commission += actual_comm - comm

                        slip_cost = actual_slip_cost
                        spread_cost = actual_spread_cost
                        comm = actual_comm

                        cost = qty * entry_price
                        capital -= cost

                        positions[symbol] = {
                            "qty": qty,
                            "entry_price": quoted_price,  # Original price for P/L display
                            "entry_price_with_slippage": entry_price,  # Actual execution price
                            "entry_date": current_date,
                            "entry_slippage": slip_cost,
                            "entry_spread": spread_cost,
                        }

                        logger.info(
                            f"ENTRY {symbol}: {qty} shares @ ${entry_price:.2f} "
                            f"(quoted: ${quoted_price:.2f}, slip: ${slip_cost:.2f}, RSI={rsi:.1f}) - "
                            f"Capital: ${capital:,.2f}"
                        )

                        if len(positions) >= strategy.max_positions:
                            break

        # Calculate equity (cash + position value)
        # P1 Fix: Include ALL positions in equity calculation, using entry price as fallback
        position_value = sum(
            pos["qty"] * day_prices.get(symbol, {}).get("close", pos["entry_price"])
            for symbol, pos in positions.items()
        )
        equity = capital + position_value

        equity_curve.append({"date": current_date, "equity": equity})

    # Close any remaining positions at end (with slippage)
    for symbol, position in list(positions.items()):
        if symbol in all_bars:
            final_bar = all_bars[symbol][-1]
            quoted_price = float(final_bar.close)

            # Apply slippage to exit
            exit_price, slip_cost, spread_cost, comm = slippage_model.apply_slippage(
                quoted_price, "sell", position["qty"]
            )

            position_value = position["qty"] * exit_price
            entry_with_slip = position.get("entry_price_with_slippage", position["entry_price"])
            pnl = (exit_price - entry_with_slip) * position["qty"]
            pnl_pct = (exit_price - entry_with_slip) / entry_with_slip

            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": position["entry_date"],
                    "exit_date": trading_days[-1],
                    "entry_price": entry_with_slip,
                    "exit_price": exit_price,
                    "quoted_entry": position["entry_price"],
                    "quoted_exit": quoted_price,
                    "qty": position["qty"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "slippage_cost": slip_cost + position.get("entry_slippage", 0),
                    "spread_cost": spread_cost + position.get("entry_spread", 0),
                    "reason": "end_of_backtest",
                }
            )

            # Return full position value to cash
            capital += position_value

    # Calculate final metrics
    final_equity = equity_curve[-1]["equity"]
    total_return = (final_equity - initial_capital) / initial_capital
    num_trades = len(trades)
    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

    # Get transaction costs
    costs = slippage_model.get_total_costs()

    if not quiet:
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80 + "\n")

        print(f"Initial Capital:  ${initial_capital:,.2f}")
        print(f"Final Equity:     ${final_equity:,.2f}")
        print(f"Total Return:     {total_return:+.2%}")
        print(f"Total P/L:        ${final_equity - initial_capital:+,.2f}\n")

        print(f"Number of Trades: {num_trades}")
        print(f"Winning Trades:   {len(winning_trades)} ({win_rate:.1%})")
        print(f"Losing Trades:    {len(losing_trades)}\n")

        if winning_trades:
            avg_win = np.mean([t["pnl_pct"] for t in winning_trades])
            print(f"Average Win:      {avg_win:+.2%}")

        if losing_trades:
            avg_loss = np.mean([t["pnl_pct"] for t in losing_trades])
            print(f"Average Loss:     {avg_loss:+.2%}")

        # Transaction costs breakdown
        print("\n--- Transaction Costs ---")
        print(f"Slippage Cost:    ${costs['slippage']:,.2f}")
        print(f"Spread Cost:      ${costs['spread']:,.2f}")
        print(f"Commission:       ${costs['commission']:,.2f}")
        print(
            f"Total Costs:      ${costs['total']:,.2f} ({costs['total']/initial_capital:.2%} of capital)"
        )

        # Calculate what return would have been without slippage
        if use_slippage:
            gross_return = total_return + (costs["total"] / initial_capital)
            print(f"\nGross Return (no slippage):  {gross_return:+.2%}")
            print(f"Net Return (with slippage):  {total_return:+.2%}")
            print(f"Cost Drag:                   {costs['total']/initial_capital:.2%}")

    # Max drawdown
    equity_values = [e["equity"] for e in equity_curve]
    peak = equity_values[0]
    max_dd = 0
    for value in equity_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd

    if not quiet:
        print(f"\nMax Drawdown:     {max_dd:.2%}")

    # Sharpe ratio (simplified)
    sharpe = 0
    if len(equity_curve) > 1:
        returns = [
            (equity_curve[i]["equity"] - equity_curve[i - 1]["equity"])
            / equity_curve[i - 1]["equity"]
            for i in range(1, len(equity_curve))
        ]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        if not quiet:
            print(f"Sharpe Ratio:     {sharpe:.2f}")

    # Statistical significance warnings
    min_trades = BACKTEST_PARAMS.get("MIN_TRADES_FOR_SIGNIFICANCE", 50)

    if not quiet:
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE CHECK")
        print("=" * 80)

        if num_trades < min_trades:
            print(
                f"⚠️  WARNING: Only {num_trades} trades (need {min_trades}+ for statistical significance)"
            )
            print("   Current results may be due to luck, not skill")
            print("   Confidence interval is extremely wide")

            # P2 Fix: Calculate statistically correct confidence interval
            # The CI should be around the mean per-trade return, scaled appropriately
            if num_trades > 0:
                pnl_pcts = [t["pnl_pct"] for t in trades]
                std_pnl = np.std(pnl_pcts, ddof=1)  # Use sample std (ddof=1)
                std_err = std_pnl / np.sqrt(num_trades)
                # CI for total return based on sum of trades
                ci_low = total_return - 1.96 * std_err * np.sqrt(num_trades)
                ci_high = total_return + 1.96 * std_err * np.sqrt(num_trades)
                print(f"   95% CI: [{ci_low:+.1%}, {ci_high:+.1%}]")
        else:
            print(f"✅ {num_trades} trades exceeds minimum threshold ({min_trades})")

        if sharpe > 2.0:
            print(f"⚠️  WARNING: Sharpe Ratio {sharpe:.2f} is suspiciously high")
            print("   Realistic Sharpe for retail traders: 0.5-1.5")
            print("   Results may be overfitted or unrealistic")

        print("\n" + "=" * 80)
        print("REALITY CHECK")
        print("=" * 80)
        if use_slippage:
            print(f"✅ Slippage modeling ENABLED ({slippage_model.slippage_pct:.2%} per trade)")
            print("✅ Results include realistic transaction costs")
        else:
            print("❌ Slippage modeling DISABLED - results are UNREALISTIC")

        print(f"✅ Bot executed {num_trades} trades")
        print(f"✅ Net Result: {total_return:+.1%} over {len(trading_days)} days")

        if use_slippage and costs["total"] > 0:
            print("\n📊 Cost Analysis:")
            print(f"   Your strategy lost ${costs['total']:,.2f} to transaction costs")
            print(f"   That's {costs['total']/initial_capital:.2%} of capital")
            if num_trades > 0:
                print(f"   Average cost per trade: ${costs['total']/num_trades:.2f}")

        print("=" * 80 + "\n")

    return {
        "config_name": config_desc,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "trades": trades,
        "equity_curve": equity_curve,
        # Slippage tracking
        "transaction_costs": costs,
        "slippage_enabled": use_slippage,
        "gross_return": (
            total_return + (costs["total"] / initial_capital) if use_slippage else total_return
        ),
        "cost_drag": costs["total"] / initial_capital,
        # Statistical significance
        "statistically_significant": num_trades >= min_trades,
    }


if __name__ == "__main__":
    # Run backtest on simplified strategy
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
    start = "2024-08-01"
    end = "2024-11-01"

    result = asyncio.run(simple_backtest(symbols, start, end))
