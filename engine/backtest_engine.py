import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Type

import pandas as pd
import pytz

# Set up logging
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Engine for backtesting trading strategies using historical data.
    """

    def __init__(self, broker=None):
        """
        Initialize the backtest engine.

        Args:
            broker: The broker instance to use for market data. If None, create a new one.
        """
        self.broker = broker
        self.current_date = None
        self.strategies = []
        self.results = {}

    async def run(self, strategies, start_date, end_date):
        """
        Run backtest for strategies over the given period.

        Args:
            strategies: List of strategy instances to backtest
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            List of result DataFrames, one per strategy
        """
        self.strategies = strategies
        self.current_date = start_date

        # Initialize result tracking for each strategy
        results = []
        for strategy in strategies:
            # Create daily results dataframe with date index
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            result_df = pd.DataFrame(
                index=dates, columns=["equity", "cash", "holdings", "returns", "trades"]
            )
            result_df["trades"] = 0
            results.append(result_df)

        # Initialize strategies with broker
        for strategy in strategies:
            if not hasattr(strategy, "broker"):
                strategy.broker = self.broker

        # Run backtest day by day
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() > 4:  # Saturday = 5, Sunday = 6
                current_date += timedelta(days=1)
                continue

            logger.debug(f"Processing date: {current_date.date()}")
            self.current_date = current_date

            # Process each strategy for this day
            for i, strategy in enumerate(strategies):
                try:
                    # Run one iteration of the strategy
                    await self._run_strategy_iteration(strategy, current_date)

                    # Record daily results
                    result_df = results[i]
                    if current_date in result_df.index:
                        portfolio_value = self.broker.get_portfolio_value(current_date)
                        cash = self.broker.get_balance()
                        holdings = portfolio_value - cash

                        result_df.loc[current_date, "equity"] = portfolio_value
                        result_df.loc[current_date, "cash"] = cash
                        result_df.loc[current_date, "holdings"] = holdings

                        # Calculate daily returns
                        if current_date != start_date:
                            prev_date = current_date - timedelta(days=1)
                            while prev_date not in result_df.index and prev_date >= start_date:
                                prev_date = prev_date - timedelta(days=1)

                            if (
                                prev_date in result_df.index
                                and result_df.loc[prev_date, "equity"] > 0
                            ):
                                prev_equity = result_df.loc[prev_date, "equity"]
                                result_df.loc[current_date, "returns"] = (
                                    portfolio_value / prev_equity
                                ) - 1
                except Exception as e:
                    logger.error(
                        f"Error in strategy {strategy.__class__.__name__} on {current_date.date()}: {e}"
                    )

            # Move to next day
            current_date += timedelta(days=1)

        # Post-process results to fill missing values and calculate metrics
        for i, result_df in enumerate(results):
            # Forward fill equity values for non-trading days
            result_df.ffill(inplace=True)

            # Calculate cumulative returns
            result_df["cum_returns"] = (1 + result_df["returns"].fillna(0)).cumprod() - 1

            # Calculate additional metrics
            strategy_name = strategies[i].__class__.__name__
            self._calculate_performance_metrics(result_df, strategy_name)

        return results

    async def _run_strategy_iteration(self, strategy, current_date):
        """Run a single iteration of a strategy for the given date."""
        # Call the strategy's on_trading_iteration method if it exists
        if hasattr(strategy, "on_trading_iteration"):
            strategy.current_date = current_date  # Set current date for the strategy
            strategy.on_trading_iteration()

    def _calculate_performance_metrics(self, result_df, strategy_name):
        """Calculate performance metrics for a strategy."""
        # Skip if not enough data
        if len(result_df) < 2:
            return

        # Calculate daily, monthly, and annual returns
        result_df["daily_returns"] = result_df["returns"]

        # Calculate drawdowns
        result_df["peak"] = result_df["equity"].cummax()
        result_df["drawdown"] = (result_df["equity"] / result_df["peak"]) - 1

        # Maximum drawdown
        max_drawdown = result_df["drawdown"].min()

        # Annualized return (assuming 252 trading days in a year)
        days = (result_df.index[-1] - result_df.index[0]).days
        if days > 0:
            years = days / 365
            total_return = result_df["cum_returns"].iloc[-1]
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        if result_df["daily_returns"].std() > 0:
            sharpe_ratio = (
                result_df["daily_returns"].mean() / result_df["daily_returns"].std()
            ) * (252**0.5)
        else:
            sharpe_ratio = 0

        # Add metrics to dataframe
        result_df.attrs["strategy"] = strategy_name
        result_df.attrs["annualized_return"] = annualized_return
        result_df.attrs["max_drawdown"] = max_drawdown
        result_df.attrs["sharpe_ratio"] = sharpe_ratio

        logger.info(
            f"Strategy {strategy_name} - Annualized Return: {annualized_return:.2%}, "
            f"Max Drawdown: {max_drawdown:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}"
        )

    async def run_backtest(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date,
        end_date,
        initial_capital: float = 100000,
    ) -> Dict[str, Any]:
        """
        Run a comprehensive backtest for a strategy.

        Args:
            strategy_class: Strategy class to instantiate and test
            symbols: List of symbols to trade
            start_date: Start date for backtest (date or datetime)
            end_date: End date for backtest (date or datetime)
            initial_capital: Starting capital

        Returns:
            Dictionary with backtest results including equity_curve and trades
        """
        from brokers.alpaca_broker import AlpacaBroker
        from brokers.backtest_broker import BacktestBroker

        # Create backtest broker
        backtest_broker = BacktestBroker(initial_balance=initial_capital)

        # Use existing broker for data if available, otherwise create one
        data_broker = self.broker if self.broker else AlpacaBroker(paper=True)

        # Convert dates to datetime if they are date objects
        if hasattr(start_date, "strftime") and not hasattr(start_date, "hour"):
            start_dt = datetime.combine(start_date, datetime.min.time())
        else:
            start_dt = start_date

        if hasattr(end_date, "strftime") and not hasattr(end_date, "hour"):
            end_dt = datetime.combine(end_date, datetime.min.time())
        else:
            end_dt = end_date

        # Fetch historical data for all symbols
        logger.info(
            f"Loading historical data for {len(symbols)} symbols from {start_dt.date()} to {end_dt.date()}..."
        )

        for symbol in symbols:
            try:
                bars = await data_broker.get_bars(
                    symbol,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    timeframe="1Day",
                )

                if bars and len(bars) > 0:
                    # Convert to DataFrame format expected by BacktestBroker
                    # Note: volume must be float for talib SMA compatibility
                    data = pd.DataFrame(
                        {
                            "open": [float(b.open) for b in bars],
                            "high": [float(b.high) for b in bars],
                            "low": [float(b.low) for b in bars],
                            "close": [float(b.close) for b in bars],
                            "volume": [float(b.volume) for b in bars],
                        },
                        index=pd.DatetimeIndex([b.timestamp for b in bars]),
                    )

                    backtest_broker.set_price_data(symbol, data)
                    logger.debug(f"Loaded {len(bars)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")

        # Instantiate strategy with backtest broker and symbols
        strategy = strategy_class(broker=backtest_broker, parameters={"symbols": symbols})

        # Initialize the strategy if it has an initialize method
        if hasattr(strategy, "initialize"):
            try:
                await strategy.initialize()
            except Exception as e:
                logger.warning(f"Strategy initialization warning: {e}")

        # Track equity curve
        equity_curve = [initial_capital]

        # Generate trading days
        current_date = start_dt
        trading_days = []
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Skip weekends
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"Running backtest over {len(trading_days)} trading days...")

        # Run day by day
        for day_num, current_date in enumerate(trading_days):
            try:
                # Update the backtest broker's current date for price lookups
                backtest_broker._current_date = current_date

                # Populate price history and current_data for the strategy
                # Make current_date timezone-aware for comparison with UTC index
                current_date_utc = (
                    current_date.replace(tzinfo=pytz.UTC)
                    if current_date.tzinfo is None
                    else current_date
                )

                # Ensure strategy has current_data attribute
                if not hasattr(strategy, "current_data"):
                    strategy.current_data = {}

                for symbol in symbols:
                    if symbol in backtest_broker.price_data:
                        df = backtest_broker.price_data[symbol]
                        # Get prices up to current date
                        try:
                            historical = df[df.index <= current_date_utc]
                        except TypeError:
                            # If comparison fails, try normalizing the index
                            df_naive = df.copy()
                            df_naive.index = df_naive.index.tz_localize(None)
                            historical = df_naive[
                                df_naive.index <= current_date.replace(tzinfo=None)
                            ]

                        if len(historical) > 0:
                            prices = historical["close"].tolist()
                            if hasattr(strategy, "price_history"):
                                strategy.price_history[symbol] = prices[-30:]  # Keep last 30 days
                            # Also populate current_data with the historical DataFrame
                            strategy.current_data[symbol] = historical

                # Generate signals if the strategy has a generate_signals method
                if hasattr(strategy, "generate_signals"):
                    try:
                        await strategy.generate_signals()
                    except Exception as e:
                        logger.debug(f"Error in generate_signals: {e}")

                # Analyze and potentially trade each symbol
                for symbol in symbols:
                    if symbol not in backtest_broker.price_data:
                        continue

                    try:
                        signal = await strategy.analyze_symbol(symbol)
                        if signal:
                            # Handle both string and dict signal formats
                            if isinstance(signal, str):
                                action = signal
                            else:
                                action = (
                                    signal.get("action") if isinstance(signal, dict) else "neutral"
                                )

                            if day_num < 5:  # Log first few days for debugging
                                logger.debug(f"  {symbol} signal: {action}")

                            if action not in ["hold", "neutral", None]:
                                logger.info(f"  Trade signal: {symbol} - {action}")
                                # Convert string signal to dict for execute_trade
                                if isinstance(signal, str):
                                    signal = {"action": signal, "symbol": symbol}
                                await strategy.execute_trade(symbol, signal)
                    except Exception as e:
                        logger.warning(f"Error analyzing {symbol} on {current_date.date()}: {e}")

                # Record equity at end of day
                portfolio_value = backtest_broker.get_portfolio_value(current_date)
                equity_curve.append(portfolio_value)

                # Progress logging every 50 days
                if day_num % 50 == 0:
                    logger.info(
                        f"  Day {day_num}/{len(trading_days)}: Equity = ${portfolio_value:,.2f}"
                    )

            except Exception as e:
                logger.error(f"Error on {current_date.date()}: {e}")
                equity_curve.append(equity_curve[-1] if equity_curve else initial_capital)

        # Process trades to calculate P&L
        trades = backtest_broker.get_trades()
        trade_records = self._calculate_trade_pnl(trades)

        final_equity = equity_curve[-1] if equity_curve else initial_capital
        total_return = (final_equity / initial_capital) - 1

        logger.info(f"Backtest complete: Final equity = ${final_equity:,.2f} ({total_return:+.2%})")
        logger.info(f"Total trades: {len(trade_records)}")

        return {
            "equity_curve": equity_curve,
            "trades": trade_records,
            "start_date": start_dt,
            "end_date": end_dt,
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "positions": backtest_broker.get_positions(),
            "total_trades": len(trade_records),
        }

    def _calculate_trade_pnl(self, trades: List[Dict]) -> List[Dict]:
        """
        Calculate P&L for each trade by matching buys and sells.

        Args:
            trades: List of raw trade records

        Returns:
            List of trade records with P&L calculated
        """
        trade_records = []
        position_tracker = {}  # Track average entry price per symbol

        for trade in trades:
            symbol = trade["symbol"]
            side = trade["side"]
            quantity = trade["quantity"]
            price = trade["price"]

            if side == "buy":
                # Update position tracker with new buy
                if symbol not in position_tracker:
                    position_tracker[symbol] = {"qty": 0, "avg_price": 0}

                old_qty = position_tracker[symbol]["qty"]
                old_avg = position_tracker[symbol]["avg_price"]
                new_qty = old_qty + quantity

                if new_qty > 0:
                    position_tracker[symbol]["avg_price"] = (
                        old_qty * old_avg + quantity * price
                    ) / new_qty
                position_tracker[symbol]["qty"] = new_qty

                trade_records.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "timestamp": trade.get("timestamp"),
                        "pnl": 0,  # Buys don't have immediate P&L
                    }
                )

            else:  # sell
                pnl = 0
                if symbol in position_tracker and position_tracker[symbol]["qty"] > 0:
                    entry_price = position_tracker[symbol]["avg_price"]
                    pnl = (price - entry_price) * quantity
                    position_tracker[symbol]["qty"] -= quantity

                trade_records.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "timestamp": trade.get("timestamp"),
                        "pnl": pnl,
                    }
                )

        return trade_records
