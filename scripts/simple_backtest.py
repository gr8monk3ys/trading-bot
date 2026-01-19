#!/usr/bin/env python3
"""
Simple Backtest Runner

A lightweight script to run backtests without the full Lumibot dependency
"""
import argparse
import logging
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import our mock strategies
from mock_strategies import MockMeanReversionStrategy, MockMomentumStrategy

# Define symbols
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]


class SimpleBacktester:
    """A simplified backtester that doesn't rely on Lumibot"""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.data = {}
        self.strategies = []

    def load_data(self, symbols, start_date, end_date):
        """Load or generate data for backtesting"""
        # Generate a random seed for reproducibility
        np.random.seed(42)

        self.data = {}
        for symbol in symbols:
            # Generate mock price data
            df = self.generate_mock_prices(symbol, start_date, end_date)

            # Add technical indicators
            # RSI (Relative Strength Index)
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # Moving Averages
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

            # MACD
            df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = df["ema_12"] - df["ema_26"]
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            df["bb_std"] = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
            df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

            # Store in data dictionary
            self.data[symbol] = df

        return self.data

    def generate_mock_prices(self, symbol, start_date, end_date):
        """Generate mock price data for backtesting"""
        # Create date range (including weekends for simplicity)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Base price (randomized by symbol)
        symbol_seed = sum([ord(c) for c in symbol])
        np.random.seed(symbol_seed)
        base_price = np.random.uniform(50, 500)

        # Set volatility based on symbol
        volatility = np.random.uniform(0.01, 0.03)

        # Generate price series with random walk
        np.random.seed(42 + symbol_seed)

        # Add trend and cyclical components
        prices = []
        price = base_price

        for i, date in enumerate(date_range):
            # Add random component
            daily_return = np.random.normal(0, volatility)

            # Add trend component (some stocks trend up, some down)
            trend = np.random.uniform(-0.0005, 0.001)

            # Add cyclical component with different phases
            cycle_period = np.random.uniform(20, 40)  # Different cycle periods
            cycle_amplitude = np.random.uniform(0.001, 0.006)
            cycle = cycle_amplitude * np.sin(2 * np.pi * i / cycle_period)

            # Add occasional price shocks
            shock = 0
            if np.random.random() < 0.02:  # 2% chance of price shock
                shock = np.random.normal(0, volatility * 5)

            # Calculate daily price change
            change = price * (daily_return + trend + cycle + shock)
            price += change

            # Ensure price is positive
            price = max(price, 0.1 * base_price)

            prices.append(price)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + np.random.uniform(0, volatility)) for p in prices],
                "low": [p * (1 - np.random.uniform(0, volatility)) for p in prices],
                "close": [p * (1 + np.random.normal(0, volatility / 3)) for p in prices],
                "volume": [int(np.random.uniform(100000, 10000000)) for _ in prices],
            },
            index=date_range,
        )

        # Add a bit more realistic behavior to closing prices
        df["close"] = df["open"].shift(-1)
        df.loc[df.index[-1], "close"] = df.loc[df.index[-1], "open"] * (
            1 + np.random.normal(0, volatility / 3)
        )

        return df

    def place_order(self, date, symbol, qty, side, price=None):
        """Place a simulated order"""
        if symbol not in self.data:
            logger.warning(f"Symbol {symbol} not in data")
            return False

        # Get price for the date
        if price is None:
            if date not in self.data[symbol].index:
                logger.warning(f"Date {date} not in data for {symbol}")
                return False
            price = self.data[symbol].loc[date, "close"]

        # Simple commission model
        commission = max(1.0, abs(qty * price * 0.0005))  # $1 minimum or 0.05%

        # Execute trade
        cost = qty * price + commission

        if side == "buy":
            if cost > self.capital:
                logger.warning(f"Insufficient capital: {self.capital} < {cost}")
                return False

            self.capital -= cost

            # Update position
            if symbol in self.positions:
                current_qty = self.positions[symbol]["qty"]
                current_cost = self.positions[symbol]["cost_basis"] * current_qty

                new_qty = current_qty + qty
                new_cost = (current_cost + (qty * price)) / new_qty

                self.positions[symbol] = {"qty": new_qty, "cost_basis": new_cost}
            else:
                self.positions[symbol] = {"qty": qty, "cost_basis": price}

        elif side == "sell":
            if symbol not in self.positions or self.positions[symbol]["qty"] < qty:
                logger.warning(f"Insufficient position for {symbol}")
                return False

            self.capital += price * qty - commission

            # Update position
            current_qty = self.positions[symbol]["qty"]
            new_qty = current_qty - qty

            if new_qty <= 0:
                del self.positions[symbol]
            else:
                self.positions[symbol]["qty"] = new_qty

        # Record trade
        self.trades.append(
            {
                "date": date,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "commission": commission,
            }
        )

        return True

    def calculate_portfolio_value(self, date):
        """Calculate total portfolio value at a given date"""
        value = self.capital

        for symbol, position in self.positions.items():
            if date in self.data[symbol].index:
                price = self.data[symbol].loc[date, "close"]
                value += position["qty"] * price

        return value

    async def run(self, strategy_classes, symbols, start_date, end_date):
        """Run the backtest"""
        # Make sure dates are datetime
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        # Load data
        self.load_data(symbols, start_date, end_date)

        # Get all trading dates
        all_dates = set()
        for symbol, df in self.data.items():
            symbol_dates = df.index
            all_dates.update(symbol_dates)
        trading_dates = sorted(list(all_dates))

        # Initialize strategies
        strategies = []
        for strategy_class in strategy_classes:
            strategy = strategy_class()
            strategy._legacy_initialize(symbols=symbols)
            strategies.append(strategy)

        # Track equity curve
        self.equity_curve = []

        # Run simulation day by day
        for date in trading_dates:
            # Skip if date is outside range
            if date < start_date or date > end_date:
                continue

            # Update data for each strategy
            for strategy in strategies:
                # Set current data and date
                strategy.current_date = date
                strategy.data = {symbol: df[df.index <= date] for symbol, df in self.data.items()}

                # Run strategy's trading logic
                strategy.on_trading_iteration()

                # Get and process signals
                for symbol in symbols:
                    try:
                        signal = strategy.get_signal(symbol, self.data[symbol].loc[date])

                        if signal:
                            logger.info(
                                f"[{date}] {strategy.__class__.__name__} signal for {symbol}: {signal['side']} {signal['qty']} @ {signal['price']:.2f} - {signal.get('reason', 'no reason')}"
                            )
                            side = signal["side"]
                            qty = signal["qty"]

                            if side and qty > 0:
                                success = self.place_order(date, symbol, qty, side)
                                if success:
                                    logger.info(
                                        f"[{date}] Order executed: {side} {qty} {symbol} @ {signal['price']:.2f}"
                                    )
                                    # If order was successful, update strategy's positions
                                    if side == "buy":
                                        strategy.positions[symbol] = qty
                                        # Track entry price for position
                                        strategy.position_entry_prices[symbol] = signal["price"]
                                    elif side == "sell":
                                        strategy.positions.pop(symbol, None)
                                        # Clear entry price when position closed
                                        strategy.position_entry_prices.pop(symbol, None)
                    except Exception as e:
                        logger.error(f"Error processing {symbol} on {date}: {e}")

            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(date)
            self.equity_curve.append({"date": date, "value": portfolio_value})

        # Calculate performance metrics
        return self.calculate_performance()

    def calculate_performance(self):
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0}

        # Convert to dataframe
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("date", inplace=True)

        # Calculate returns
        equity_df["daily_return"] = equity_df["value"].pct_change()

        # Total return
        total_return = (equity_df["value"].iloc[-1] / self.initial_capital) - 1

        # Calculate drawdown
        equity_df["peak"] = equity_df["value"].cummax()
        equity_df["drawdown"] = (equity_df["value"] - equity_df["peak"]) / equity_df["peak"]
        max_drawdown = equity_df["drawdown"].min()

        # Sharpe ratio (assuming 0% risk-free rate)
        if len(equity_df) > 1 and equity_df["daily_return"].std() > 0:
            sharpe = equity_df["daily_return"].mean() / equity_df["daily_return"].std() * (252**0.5)
        else:
            sharpe = 0

        results = {
            "equity_curve": equity_df,
            "total_return": total_return,
            "total_return_pct": f"{total_return * 100:.2f}%",
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": f"{max_drawdown * 100:.2f}%",
            "final_value": equity_df["value"].iloc[-1],
            "trade_count": len(self.trades),
        }

        return results

    def plot_results(self, title="Backtest Results"):
        """Plot equity curve and drawdowns"""
        if not self.equity_curve:
            logger.warning("No equity curve to plot")
            return

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("date", inplace=True)

        # Calculate drawdown
        equity_df["peak"] = equity_df["value"].cummax()
        equity_df["drawdown"] = (equity_df["value"] - equity_df["peak"]) / equity_df["peak"]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

        # Plot equity curve
        equity_df["value"].plot(ax=ax1, linewidth=2)
        ax1.set_title(title)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True)

        # Plot drawdown
        equity_df["drawdown"].plot(ax=ax2, linewidth=2, color="red")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True)

        # Format plot
        fig.tight_layout()

        # Save and show plot
        os.makedirs("results", exist_ok=True)
        plt.savefig(f'results/{title.replace(" ", "_").lower()}.png')

        return fig

    def run_backtest(self, strategies, symbols, start_date, end_date):
        """Run a backtest simulation"""
        # Load data
        self.load_data(symbols, start_date, end_date)

        # Store strategies
        self.strategies = strategies

        # Reset tracking variables
        self.cash = self.initial_capital
        self.portfolio = {}
        self.equity_curve = []
        self.trades = []

        # Initialize all strategies
        for strategy in strategies:
            strategy.data = self.data
            strategy._legacy_initialize(symbols=symbols)

        # Process daily
        trading_dates = sorted(self.data[symbols[0]].index)
        for date in trading_dates:
            # Skip if date is outside range
            if date < start_date or date > end_date:
                continue

            # Update strategy data references
            for strategy in strategies:
                strategy.current_date = date

                # Run strategy's trading logic
                strategy.on_trading_iteration()

                # Get and process signals
                for symbol in symbols:
                    try:
                        # Ensure data exists for this symbol on this date
                        if symbol in self.data and date in self.data[symbol].index:
                            signal = strategy.get_signal(symbol, self.data[symbol].loc[date])

                            if signal:
                                logger.info(
                                    f"[{date}] {strategy.__class__.__name__} signal for {symbol}: {signal['side']} {signal['qty']} @ {signal['price']:.2f} - {signal.get('reason', 'no reason')}"
                                )
                                side = signal["side"]
                                qty = signal["qty"]

                                if side and qty > 0:
                                    success = self.place_order(date, symbol, qty, side)
                                    if success:
                                        logger.info(
                                            f"[{date}] Order executed: {side} {qty} {symbol} @ {signal['price']:.2f}"
                                        )
                                        # Record trade
                                        self.trades.append(
                                            {
                                                "date": date,
                                                "symbol": symbol,
                                                "side": side,
                                                "qty": qty,
                                                "price": signal["price"],
                                                "reason": signal.get("reason", "no reason"),
                                            }
                                        )
                                        # If order was successful, update strategy's positions
                                        if side == "buy":
                                            strategy.positions[symbol] = qty
                                            # Track entry price for position
                                            strategy.position_entry_prices[symbol] = signal["price"]
                                        elif side == "sell":
                                            strategy.positions.pop(symbol, None)
                                            # Clear entry price when position closed
                                            strategy.position_entry_prices.pop(symbol, None)
                    except Exception as e:
                        logger.error(f"Error processing {symbol} on {date}: {e}")
                        import traceback

                        logger.error(traceback.format_exc())

            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(date)
            self.equity_curve.append({"date": date, "value": portfolio_value, "cash": self.cash})

        # Calculate performance metrics
        return self.calculate_performance()

    def calculate_performance(self):
        """Calculate performance metrics from equity curve"""
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("date", inplace=True)

        # Calculate return metrics
        initial_value = self.initial_capital
        final_value = equity_df["value"].iloc[-1] if not equity_df.empty else initial_value

        # Calculate percentage return
        pct_return = ((final_value / initial_value) - 1) * 100

        # Daily returns
        equity_df["daily_return"] = equity_df["value"].pct_change()

        # Annualized Sharpe Ratio (assuming 252 trading days/year)
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
        if len(equity_df) > 1 and equity_df["daily_return"].std() > 0:
            excess_return = equity_df["daily_return"] - daily_risk_free
            sharpe_ratio = excess_return.mean() / equity_df["daily_return"].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum Drawdown
        equity_df["cum_max"] = equity_df["value"].cummax()
        equity_df["drawdown"] = (equity_df["value"] - equity_df["cum_max"]) / equity_df["cum_max"]
        max_drawdown = equity_df["drawdown"].min() * 100 if not equity_df.empty else 0

        # Plot equity curve
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"results/backtest_{'_'.join(s.__class__.__name__.lower() for s in self.strategies)}_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)

        # Save equity curve data
        equity_df.to_csv(f"{result_dir}/equity_curve.csv")

        # Generate plots
        self.generate_performance_plots(equity_df, result_dir)

        # Return performance metrics
        return {
            "return": pct_return,
            "sharpe": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_value": final_value,
            "trades": len(self.trades),
            "equity_curve": equity_df,
            "result_dir": result_dir,
        }

    def generate_performance_plots(self, equity_df, result_dir):
        """Generate performance plots"""
        plt.figure(figsize=(12, 8))

        # Plot main equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_df.index, equity_df["value"], label="Portfolio Value")
        plt.title("Portfolio Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True)

        # Plot drawdowns
        plt.subplot(2, 1, 2)
        plt.plot(equity_df.index, equity_df["drawdown"] * 100)
        plt.title("Portfolio Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.fill_between(equity_df.index, equity_df["drawdown"] * 100, 0, color="red", alpha=0.3)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{result_dir}/equity_curve.png")

        # Plot daily returns
        plt.figure(figsize=(12, 6))
        equity_df["daily_return"].plot(kind="bar", title="Daily Returns", figsize=(12, 6))
        plt.axhline(y=0, color="r", linestyle="-")
        plt.xlabel("Date")
        plt.ylabel("Return (%)")
        plt.savefig(f"{result_dir}/daily_returns.png")

        # Plot trade analysis if we have trades
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            trade_df["value"] = trade_df["qty"] * trade_df["price"]

            plt.figure(figsize=(12, 6))
            grouped = trade_df.groupby("symbol")["value"].sum()
            grouped.plot(kind="bar", title="Trade Value by Symbol")
            plt.ylabel("Total Value ($)")
            plt.savefig(f"{result_dir}/trade_by_symbol.png")

            plt.figure(figsize=(12, 6))
            grouped = trade_df.groupby("reason")["value"].sum()
            grouped.plot(kind="bar", title="Trade Value by Reason")
            plt.ylabel("Total Value ($)")
            plt.savefig(f"{result_dir}/trade_by_reason.png")

    async def run(self, strategy_classes, symbols, start_date, end_date):
        """Run the backtest"""
        # Make sure dates are datetime
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        # Load data
        self.load_data(symbols, start_date, end_date)

        # Get all trading dates
        all_dates = set()
        for symbol, df in self.data.items():
            symbol_dates = df.index
            all_dates.update(symbol_dates)
        trading_dates = sorted(list(all_dates))

        # Initialize strategies
        strategies = []
        for strategy_class in strategy_classes:
            strategy = strategy_class()
            strategy._legacy_initialize(symbols=symbols)
            strategies.append(strategy)

        # Track equity curve
        self.equity_curve = []

        # Run simulation day by day
        for date in trading_dates:
            # Skip if date is outside range
            if date < start_date or date > end_date:
                continue

            # Update data for each strategy
            for strategy in strategies:
                # Set current data and date
                strategy.current_date = date
                strategy.data = {symbol: df[df.index <= date] for symbol, df in self.data.items()}

                # Run strategy's trading logic
                strategy.on_trading_iteration()

                # Get and process signals
                for symbol in symbols:
                    try:
                        signal = strategy.get_signal(symbol, self.data[symbol].loc[date])

                        if signal:
                            logger.info(
                                f"[{date}] {strategy.__class__.__name__} signal for {symbol}: {signal['side']} {signal['qty']} @ {signal['price']:.2f} - {signal.get('reason', 'no reason')}"
                            )
                            side = signal["side"]
                            qty = signal["qty"]

                            if side and qty > 0:
                                success = self.place_order(date, symbol, qty, side)
                                if success:
                                    logger.info(
                                        f"[{date}] Order executed: {side} {qty} {symbol} @ {signal['price']:.2f}"
                                    )
                                    # If order was successful, update strategy's positions
                                    if side == "buy":
                                        strategy.positions[symbol] = qty
                                        # Track entry price for position
                                        strategy.position_entry_prices[symbol] = signal["price"]
                                    elif side == "sell":
                                        strategy.positions.pop(symbol, None)
                                        # Clear entry price when position closed
                                        strategy.position_entry_prices.pop(symbol, None)
                    except Exception as e:
                        logger.error(f"Error processing {symbol} on {date}: {e}")

            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(date)
            self.equity_curve.append({"date": date, "value": portfolio_value})

        # Calculate performance metrics
        return self.calculate_performance()


def main():
    """Main entry point for the backtest application."""
    parser = argparse.ArgumentParser(description="Run a simple backtest.")
    parser.add_argument(
        "--strategies",
        type=str,
        default="momentum,mean_reversion",
        help="Comma-separated list of strategies to test",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,MSFT,AMZN,GOOGL,META",
        help="Comma-separated list of symbols to test",
    )
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")

    args = parser.parse_args()

    # Split strategy names and symbols
    strategy_names = [s.strip() for s in args.strategies.split(",")]
    symbols = [s.strip() for s in args.symbols.split(",")]

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Map strategy names to classes
    strategy_map = {"momentum": MockMomentumStrategy, "mean_reversion": MockMeanReversionStrategy}

    # Get strategy instances
    strategies = []
    for name in strategy_names:
        if name.lower() in strategy_map:
            strategies.append(strategy_map[name.lower()]())
        else:
            logger.warning(f"Strategy '{name}' not found, skipping")

    if not strategies:
        logger.error("No valid strategies specified")
        return

    logger.info(f"Running backtest for strategies: {', '.join(strategy_names)}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()} ({args.days} days)")
    logger.info(f"Symbols: {', '.join(symbols)}")

    # Create and run backtester
    backtester = SimpleBacktester()
    results = backtester.run_backtest(strategies, symbols, start_date, end_date)

    # Print results
    logger.info("Backtest complete:")
    logger.info(f"  Total Return: {results['return']:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['sharpe']:.2f}")
    logger.info(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
    logger.info(f"  Final Value: ${results['final_value']:.2f}")
    logger.info(f"  Trades: {results['trades']}")
    logger.info(f"  Equity curve saved to {results['result_dir']}/equity_curve.csv")

    return results


if __name__ == "__main__":
    main()
