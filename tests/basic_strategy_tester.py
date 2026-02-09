#!/usr/bin/env python3
"""
Basic Strategy Tester - Simple testing tool for trading strategies without dependencies

This script provides a simplified environment for testing basic trading strategies
using synthetic data and does not require external libraries like TA-Lib.
"""

import logging
import math
import os
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"strategy_test_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger(__name__)

# Create directories for data and results
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Basic indicator calculations (to replace TA-Lib functionality)
def calculate_sma(data, period=20):
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_ema(data, period=20):
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index."""
    # Calculate price changes
    delta = data.diff()

    # Create gain and loss series
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, sma, lower_band


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


class SimpleBacktester:
    """Simple backtesting framework for testing trading strategies."""

    def __init__(
        self,
        start_date,
        end_date,
        symbols=None,
        initial_capital=100000,
    ):
        """Initialize the backtester."""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []

        # Generate or load price data
        self.data = self._get_price_data()

    def _get_price_data(self):
        """Get or generate price data for the specified symbols and time period."""
        data = {}

        for symbol in self.symbols:
            # Check if data file exists
            filename = f"data/{symbol}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"

            if os.path.exists(filename):
                # Load existing data
                logger.info(f"Loading data for {symbol} from {filename}")
                df = pd.read_csv(filename, parse_dates=["date"])
                data[symbol] = df
            else:
                # Generate synthetic data
                logger.info(f"Generating synthetic data for {symbol}")
                df = self._generate_synthetic_data(symbol)

                # Save data for future use
                df.to_csv(filename, index=False)
                data[symbol] = df

        return data

    def _generate_synthetic_data(self, symbol):
        """Generate synthetic price data with realistic patterns."""
        # Calculate number of days
        days = (self.end_date - self.start_date).days + 1
        dates = [self.start_date + timedelta(days=x) for x in range(days)]

        # Filter for weekdays only (Monday = 0, Sunday = 6)
        dates = [date for date in dates if date.weekday() < 5]

        # Initial price between $50 and $200
        initial_price = random.uniform(50.0, 200.0)

        # Random trend and volatility parameters
        annual_return = random.uniform(-0.2, 0.4)  # Annual return between -20% and 40%
        daily_return = (1 + annual_return) ** (
            1 / 252
        ) - 1  # Convert to daily return (252 trading days)
        annual_volatility = random.uniform(0.15, 0.5)  # Annual volatility between 15% and 50%
        daily_volatility = annual_volatility / math.sqrt(252)  # Convert to daily volatility

        # Generate price series
        prices = [initial_price]
        for i in range(1, len(dates)):
            # Daily return with random component
            daily_random = random.gauss(daily_return, daily_volatility)

            # Add some autocorrelation (momentum) to make it more realistic
            momentum = 0.1 * (-1 if random.random() < 0.5 else 1)
            if i > 1 and prices[i - 1] > prices[i - 2]:
                daily_random += momentum
            else:
                daily_random -= momentum

            # Calculate new price
            new_price = prices[i - 1] * (1 + daily_random)
            prices.append(max(new_price, 0.01))  # Ensure price doesn't go negative

        # Create DataFrame with OHLCV data
        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [
                    p * (1 + random.uniform(0, 0.02)) for p in prices
                ],  # High up to 2% above open
                "low": [
                    p * (1 - random.uniform(0, 0.02)) for p in prices
                ],  # Low up to 2% below open
                "close": [
                    p * (1 + random.normalvariate(0, 0.005)) for p in prices
                ],  # Close normally distributed around open
                "volume": [int(random.uniform(100000, 5000000)) for _ in prices],  # Random volume
            }
        )

        # Ensure close prices are within high-low range
        df["close"] = df.apply(lambda row: min(max(row["close"], row["low"]), row["high"]), axis=1)

        return df

    def run_momentum_strategy(
        self, fast_ma=12, slow_ma=26, rsi_period=14, rsi_oversold=30, rsi_overbought=70
    ):
        """Run a basic momentum strategy."""
        logger.info(
            f"Running momentum strategy (Fast MA: {fast_ma}, Slow MA: {slow_ma}, RSI: {rsi_period})"
        )

        # Reset portfolio and positions
        self.current_capital = self.initial_capital
        self.positions = dict.fromkeys(self.symbols, 0)
        self.trades = []
        self.portfolio_history = []

        # Run the strategy for each day
        dates = sorted({date for symbol in self.symbols for date in self.data[symbol]["date"]})

        for date in dates:
            # Calculate portfolio value at the start of the day
            portfolio_value = self.current_capital + sum(
                self.positions.get(symbol, 0)
                * self.data[symbol].loc[self.data[symbol]["date"] == date, "open"].values[0]
                for symbol in self.symbols
                if not self.data[symbol][self.data[symbol]["date"] == date].empty
            )

            # Store portfolio value
            self.portfolio_history.append({"date": date, "value": portfolio_value})

            # Check for signals for each symbol
            for symbol in self.symbols:
                # Skip if no data for this date
                symbol_data = self.data[symbol][self.data[symbol]["date"] <= date].copy()
                if symbol_data.empty:
                    continue

                # Calculate indicators
                symbol_data["fast_ma"] = calculate_ema(symbol_data["close"], fast_ma)
                symbol_data["slow_ma"] = calculate_ema(symbol_data["close"], slow_ma)
                symbol_data["rsi"] = calculate_rsi(symbol_data["close"], rsi_period)

                # Skip if not enough data for indicators
                if len(symbol_data) <= slow_ma:
                    continue

                # Get current indicators
                current_data = symbol_data.iloc[-1]
                prev_data = symbol_data.iloc[-2] if len(symbol_data) > 1 else None

                current_price = current_data["close"]
                position = self.positions.get(symbol, 0)

                # Generate signals
                buy_signal = False
                sell_signal = False

                # MA Crossover (fast MA crosses above slow MA)
                if prev_data is not None:
                    ma_crossover_bullish = (
                        current_data["fast_ma"] > current_data["slow_ma"]
                        and prev_data["fast_ma"] <= prev_data["slow_ma"]
                    )
                    ma_crossover_bearish = (
                        current_data["fast_ma"] < current_data["slow_ma"]
                        and prev_data["fast_ma"] >= prev_data["slow_ma"]
                    )

                    # RSI conditions
                    rsi_oversold_bullish = (
                        current_data["rsi"] < rsi_oversold and prev_data["rsi"] >= rsi_oversold
                    )
                    rsi_overbought_bearish = (
                        current_data["rsi"] > rsi_overbought and prev_data["rsi"] <= rsi_overbought
                    )

                    # Combined signals
                    buy_signal = ma_crossover_bullish or rsi_oversold_bullish
                    sell_signal = ma_crossover_bearish or rsi_overbought_bearish

                # Execute trades
                if buy_signal and position <= 0:
                    # Calculate position size (10% of portfolio)
                    position_value = portfolio_value * 0.1
                    shares_to_buy = int(position_value / current_price)

                    if shares_to_buy > 0 and self.current_capital >= shares_to_buy * current_price:
                        # Update positions and capital
                        self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                        self.current_capital -= shares_to_buy * current_price

                        # Record trade
                        self.trades.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "action": "buy",
                                "price": current_price,
                                "shares": shares_to_buy,
                                "value": shares_to_buy * current_price,
                            }
                        )

                        logger.info(
                            f"{date}: BUY {shares_to_buy} shares of {symbol} at ${current_price:.2f}"
                        )

                elif sell_signal and position > 0:
                    # Sell all shares
                    shares_to_sell = self.positions[symbol]

                    if shares_to_sell > 0:
                        # Update positions and capital
                        self.positions[symbol] = 0
                        self.current_capital += shares_to_sell * current_price

                        # Record trade
                        self.trades.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "action": "sell",
                                "price": current_price,
                                "shares": shares_to_sell,
                                "value": shares_to_sell * current_price,
                            }
                        )

                        logger.info(
                            f"{date}: SELL {shares_to_sell} shares of {symbol} at ${current_price:.2f}"
                        )

        # Calculate final portfolio value
        final_portfolio_value = self.current_capital
        for symbol, shares in self.positions.items():
            if shares > 0:
                last_price = self.data[symbol]["close"].iloc[-1]
                final_portfolio_value += shares * last_price

        return self._calculate_performance(final_portfolio_value)

    def run_mean_reversion_strategy(self, lookback_period=20, entry_z=1.5, exit_z=0.5):
        """Run a basic mean reversion strategy."""
        logger.info(
            f"Running mean reversion strategy (Lookback: {lookback_period}, Entry Z: {entry_z}, Exit Z: {exit_z})"
        )

        # Reset portfolio and positions
        self.current_capital = self.initial_capital
        self.positions = dict.fromkeys(self.symbols, 0)
        self.trades = []
        self.portfolio_history = []

        # Run the strategy for each day
        dates = sorted({date for symbol in self.symbols for date in self.data[symbol]["date"]})

        for date in dates:
            # Calculate portfolio value at the start of the day
            portfolio_value = self.current_capital + sum(
                self.positions.get(symbol, 0)
                * self.data[symbol].loc[self.data[symbol]["date"] == date, "open"].values[0]
                for symbol in self.symbols
                if not self.data[symbol][self.data[symbol]["date"] == date].empty
            )

            # Store portfolio value
            self.portfolio_history.append({"date": date, "value": portfolio_value})

            # Check for signals for each symbol
            for symbol in self.symbols:
                # Skip if no data for this date
                symbol_data = self.data[symbol][self.data[symbol]["date"] <= date].copy()
                if symbol_data.empty:
                    continue

                # Skip if not enough data for lookback
                if len(symbol_data) <= lookback_period:
                    continue

                # Calculate indicators for mean reversion
                prices = symbol_data["close"].values

                # Calculate z-score (deviation from mean in terms of standard deviations)
                recent_prices = prices[-lookback_period:]
                price_mean = np.mean(recent_prices)
                price_std = np.std(recent_prices)
                current_price = prices[-1]

                if price_std > 0:  # Avoid division by zero
                    z_score = (current_price - price_mean) / price_std
                else:
                    z_score = 0

                # Get current position
                position = self.positions.get(symbol, 0)

                # Generate signals
                # Buy when price is significantly below mean (negative z-score)
                buy_signal = z_score < -entry_z

                # Sell when price reverts to mean or goes above
                sell_signal = z_score > exit_z

                # Execute trades
                if buy_signal and position <= 0:
                    # Calculate position size (10% of portfolio)
                    position_value = portfolio_value * 0.1
                    shares_to_buy = int(position_value / current_price)

                    if shares_to_buy > 0 and self.current_capital >= shares_to_buy * current_price:
                        # Update positions and capital
                        self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                        self.current_capital -= shares_to_buy * current_price

                        # Record trade
                        self.trades.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "action": "buy",
                                "price": current_price,
                                "shares": shares_to_buy,
                                "value": shares_to_buy * current_price,
                                "z_score": z_score,
                            }
                        )

                        logger.info(
                            f"{date}: BUY {shares_to_buy} shares of {symbol} at ${current_price:.2f} (z-score: {z_score:.2f})"
                        )

                elif sell_signal and position > 0:
                    # Sell all shares
                    shares_to_sell = self.positions[symbol]

                    if shares_to_sell > 0:
                        # Update positions and capital
                        self.positions[symbol] = 0
                        self.current_capital += shares_to_sell * current_price

                        # Record trade
                        self.trades.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "action": "sell",
                                "price": current_price,
                                "shares": shares_to_sell,
                                "value": shares_to_sell * current_price,
                                "z_score": z_score,
                            }
                        )

                        logger.info(
                            f"{date}: SELL {shares_to_sell} shares of {symbol} at ${current_price:.2f} (z-score: {z_score:.2f})"
                        )

        # Calculate final portfolio value
        final_portfolio_value = self.current_capital
        for symbol, shares in self.positions.items():
            if shares > 0:
                last_price = self.data[symbol]["close"].iloc[-1]
                final_portfolio_value += shares * last_price

        return self._calculate_performance(final_portfolio_value)

    def _calculate_performance(self, final_portfolio_value):
        """Calculate performance metrics."""
        # Calculate basic performance metrics
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital

        # Calculate daily returns
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if not portfolio_df.empty and len(portfolio_df) > 1:
            portfolio_df["daily_return"] = portfolio_df["value"].pct_change()

            # Calculate annualized metrics (assuming 252 trading days)
            days = len(portfolio_df)
            annualized_return = (1 + total_return) ** (252 / days) - 1

            # Calculate volatility (annualized)
            daily_volatility = portfolio_df["daily_return"].std()
            annualized_volatility = daily_volatility * np.sqrt(252)

            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            if annualized_volatility > 0:
                sharpe_ratio = annualized_return / annualized_volatility
            else:
                sharpe_ratio = 0

            # Calculate maximum drawdown
            portfolio_df["cum_max"] = portfolio_df["value"].cummax()
            portfolio_df["drawdown"] = (
                portfolio_df["cum_max"] - portfolio_df["value"]
            ) / portfolio_df["cum_max"]
            max_drawdown = portfolio_df["drawdown"].max()
        else:
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        # Calculate win rate and profit factor
        if self.trades:
            # Get closed trades (pairs of buy and sell)
            buy_trades = [t for t in self.trades if t["action"] == "buy"]
            sell_trades = [t for t in self.trades if t["action"] == "sell"]

            # Calculate P&L for closed positions
            profits = []
            for buy in buy_trades:
                symbol = buy["symbol"]
                buy_date = buy["date"]
                buy_price = buy["price"]
                buy_shares = buy["shares"]

                # Find corresponding sell trade
                matching_sells = [
                    s for s in sell_trades if s["symbol"] == symbol and s["date"] > buy_date
                ]
                if matching_sells:
                    # Use the first matching sell
                    sell = matching_sells[0]
                    sell_price = sell["price"]

                    # Calculate profit
                    profit = (sell_price - buy_price) * buy_shares
                    profits.append(profit)

            # Calculate win rate
            winning_trades = [p for p in profits if p > 0]
            win_rate = len(winning_trades) / len(profits) if profits else 0

            # Calculate profit factor
            gross_profit = sum([p for p in profits if p > 0]) if profits else 0
            gross_loss = abs(sum([p for p in profits if p < 0])) if profits else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Calculate average trade
            avg_trade = np.mean(profits) if profits else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0

        return {
            "final_value": final_portfolio_value,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": len(self.trades),
            "avg_trade": avg_trade,
            "portfolio_history": self.portfolio_history,
            "trades": self.trades,
        }

    def compare_strategies(self, plot=True):
        """Compare momentum and mean reversion strategies."""
        # Run both strategies
        momentum_result = self.run_momentum_strategy()
        mean_reversion_result = self.run_mean_reversion_strategy()

        # Create comparison table
        comparison = pd.DataFrame(
            {
                "Momentum": {
                    "Total Return": momentum_result["total_return"],
                    "Annualized Return": momentum_result["annualized_return"],
                    "Annualized Volatility": momentum_result["annualized_volatility"],
                    "Sharpe Ratio": momentum_result["sharpe_ratio"],
                    "Max Drawdown": momentum_result["max_drawdown"],
                    "Win Rate": momentum_result["win_rate"],
                    "Profit Factor": momentum_result["profit_factor"],
                    "Number of Trades": momentum_result["num_trades"],
                    "Average Trade": momentum_result["avg_trade"],
                },
                "Mean Reversion": {
                    "Total Return": mean_reversion_result["total_return"],
                    "Annualized Return": mean_reversion_result["annualized_return"],
                    "Annualized Volatility": mean_reversion_result["annualized_volatility"],
                    "Sharpe Ratio": mean_reversion_result["sharpe_ratio"],
                    "Max Drawdown": mean_reversion_result["max_drawdown"],
                    "Win Rate": mean_reversion_result["win_rate"],
                    "Profit Factor": mean_reversion_result["profit_factor"],
                    "Number of Trades": mean_reversion_result["num_trades"],
                    "Average Trade": mean_reversion_result["avg_trade"],
                },
            }
        ).T

        # Format the table
        print("\n--- Strategy Comparison ---")
        print(comparison.to_string(float_format=lambda x: f"{x:.2%}" if x < 10 else f"{x:.2f}"))

        # Plot equity curves if requested
        if plot:
            plt.figure(figsize=(12, 6))

            # Convert portfolio history to DataFrames
            momentum_df = pd.DataFrame(momentum_result["portfolio_history"])
            mean_reversion_df = pd.DataFrame(mean_reversion_result["portfolio_history"])

            plt.plot(momentum_df["date"], momentum_df["value"], label="Momentum")
            plt.plot(mean_reversion_df["date"], mean_reversion_df["value"], label="Mean Reversion")

            plt.title("Strategy Comparison - Equity Curves")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.legend()
            plt.grid(True)

            # Save plot
            plt.savefig("results/strategy_comparison.png")
            print("\nStrategy comparison plot saved as results/strategy_comparison.png")

            # Plot individual strategy equity curves with trades
            self._plot_strategy_with_trades("Momentum", momentum_result)
            self._plot_strategy_with_trades("Mean Reversion", mean_reversion_result)

        return {
            "momentum": momentum_result,
            "mean_reversion": mean_reversion_result,
            "comparison": comparison,
        }

    def _plot_strategy_with_trades(self, strategy_name, result):
        """Plot equity curve with buy/sell trades for a strategy."""
        try:
            plt.figure(figsize=(12, 6))

            # Convert portfolio history to DataFrame
            portfolio_df = pd.DataFrame(result["portfolio_history"])
            trades_df = pd.DataFrame(result["trades"])

            # Plot equity curve
            plt.plot(portfolio_df["date"], portfolio_df["value"])

            # Add buy and sell markers
            if not trades_df.empty:
                buys = trades_df[trades_df["action"] == "buy"]
                sells = trades_df[trades_df["action"] == "sell"]

                # Create a lookup dictionary for portfolio values by date
                portfolio_dict = {row["date"]: row["value"] for _, row in portfolio_df.iterrows()}

                # Plot buy points
                for _, buy in buys.iterrows():
                    if buy["date"] in portfolio_dict:
                        plt.scatter(
                            buy["date"],
                            portfolio_dict[buy["date"]],
                            marker="^",
                            color="green",
                            s=100,
                        )

                # Plot sell points
                for _, sell in sells.iterrows():
                    if sell["date"] in portfolio_dict:
                        plt.scatter(
                            sell["date"],
                            portfolio_dict[sell["date"]],
                            marker="v",
                            color="red",
                            s=100,
                        )

            plt.title(f"{strategy_name} Strategy - Equity Curve with Trades")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True)

            # Save plot
            plt.savefig(f"results/{strategy_name.lower()}_equity_curve.png")
            print(
                f"\n{strategy_name} equity curve saved as results/{strategy_name.lower()}_equity_curve.png"
            )

        except Exception as e:
            logger.error(f"Error plotting strategy with trades: {e}", exc_info=True)


def main():
    """Main function to run the strategy tester."""
    import argparse

    parser = argparse.ArgumentParser(description="Basic Strategy Tester")
    parser.add_argument(
        "--start-date", default="2022-01-01", help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2023-01-01", help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--symbols", default="AAPL,MSFT,GOOGL,AMZN", help="Comma-separated list of symbols to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=100000, help="Initial capital for backtest"
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")

    args = parser.parse_args()

    # Parse arguments
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    symbols = args.symbols.split(",")

    print(f"Running backtest from {start_date} to {end_date}")
    print(f"Trading symbols: {symbols}")
    print(f"Initial capital: ${args.capital:,.2f}")

    # Create and run backtester
    backtester = SimpleBacktester(
        start_date=start_date, end_date=end_date, symbols=symbols, initial_capital=args.capital
    )

    # Compare strategies
    backtester.compare_strategies(plot=not args.no_plot)


if __name__ == "__main__":
    main()
