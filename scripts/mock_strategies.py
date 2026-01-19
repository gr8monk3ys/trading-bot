"""
Mock Strategy Implementation

Contains simplified implementations of our trading strategies for backtesting
without the Lumibot dependency.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MockMomentumStrategy:
    """
    Momentum strategy implementation for backtesting.

    Buys assets showing upward momentum based on moving average crossovers
    and sells when momentum weakens.
    """

    def __init__(self):
        """Initialize the strategy"""
        self.symbols = []
        self.data = {}
        self.current_date = None
        self.positions = {}
        self.position_entry_prices = {}  # Track entry prices for positions
        self.fast_ma = 20
        self.slow_ma = 50
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.position_size = 0.05  # 5% of capital per position
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        self.max_positions = 5  # Maximum number of concurrent positions

    def initialize(self, symbols=None, **kwargs):
        """Initialize with configuration parameters"""
        self.symbols = symbols or []

        # Override default parameters if provided
        self.fast_ma = kwargs.get("fast_ma", self.fast_ma)
        self.slow_ma = kwargs.get("slow_ma", self.slow_ma)
        self.rsi_period = kwargs.get("rsi_period", self.rsi_period)
        self.rsi_overbought = kwargs.get("rsi_overbought", self.rsi_overbought)
        self.rsi_oversold = kwargs.get("rsi_oversold", self.rsi_oversold)
        self.position_size = kwargs.get("position_size", self.position_size)
        self.stop_loss_pct = kwargs.get("stop_loss_pct", self.stop_loss_pct)
        self.take_profit_pct = kwargs.get("take_profit_pct", self.take_profit_pct)
        self.max_positions = kwargs.get("max_positions", self.max_positions)

    def on_trading_iteration(self):
        """Run on each trading iteration"""
        # Nothing to do if no data
        if not self.data or not self.current_date:
            return

        # Process each symbol
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            # Get latest data point
            latest_data = self.get_latest_data(symbol)
            if latest_data is None:
                continue

            # Generate and process signals
            signal = self.get_signal(symbol, latest_data)

    def get_signal(self, symbol, bar_data):
        """Generate trading signal for a symbol"""
        if symbol not in self.data or bar_data is None:
            return None

        # Get full historical data
        df = self.data[symbol]

        # Skip if not enough data
        if len(df) < self.slow_ma + 5:
            return None

        # Check for exit based on stop loss or take profit first
        if symbol in self.positions and self.positions[symbol] > 0:
            entry_price = self.position_entry_prices.get(symbol, 0)
            current_price = bar_data["close"]

            if entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price

                # Stop loss - exit if loss exceeds threshold
                if pnl_pct < -self.stop_loss_pct:
                    qty = self.positions[symbol]
                    return {
                        "side": "sell",
                        "qty": qty,
                        "price": current_price,
                        "reason": "stop_loss",
                    }

                # Take profit - exit if gain exceeds threshold
                if pnl_pct > self.take_profit_pct:
                    qty = self.positions[symbol]
                    return {
                        "side": "sell",
                        "qty": qty,
                        "price": current_price,
                        "reason": "take_profit",
                    }

        # Calculate indicators
        try:
            # Moving averages
            close_prices = df["close"]
            fast_ma = close_prices.rolling(window=self.fast_ma).mean()
            slow_ma = close_prices.rolling(window=self.slow_ma).mean()

            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Get latest values
            current_price = bar_data["close"]
            current_fast_ma = fast_ma.iloc[-1]
            current_slow_ma = slow_ma.iloc[-1]
            prev_fast_ma = fast_ma.iloc[-2] if len(fast_ma) > 1 else None
            prev_slow_ma = slow_ma.iloc[-2] if len(slow_ma) > 1 else None
            current_rsi = rsi.iloc[-1]

            # Check if we're already at position limit for buy signals
            active_positions = len(self.positions)

            # Generate signal
            if (
                pd.notna(current_fast_ma)
                and pd.notna(current_slow_ma)
                and pd.notna(prev_fast_ma)
                and pd.notna(prev_slow_ma)
            ):
                # Buy signal: fast MA crosses above slow MA and RSI is not extreme
                if (
                    (prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma)
                    and current_rsi < 60
                    and active_positions < self.max_positions
                ):
                    # Calculate position size (fixed percentage of portfolio)
                    qty = int(10000 / current_price)  # Simplified for mock testing
                    return {
                        "side": "buy",
                        "qty": qty,
                        "price": current_price,
                        "reason": "momentum_crossover",
                    }

                # Sell signal: fast MA crosses below slow MA
                elif (
                    prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma
                ) and symbol in self.positions:
                    qty = self.positions[symbol]
                    return {
                        "side": "sell",
                        "qty": qty,
                        "price": current_price,
                        "reason": "momentum_reversal",
                    }

                # Sell signal: RSI overbought
                elif current_rsi > self.rsi_overbought and symbol in self.positions:
                    qty = self.positions[symbol]
                    return {
                        "side": "sell",
                        "qty": qty,
                        "price": current_price,
                        "reason": "overbought",
                    }

        except Exception as e:
            import logging

            logging.error(f"Error calculating signals for {symbol}: {e}")

        return None

    def get_latest_data(self, symbol):
        """Get the latest data point for a symbol"""
        if symbol not in self.data or self.data[symbol].empty:
            return None

        df = self.data[symbol]
        if self.current_date not in df.index:
            return None

        return df.loc[self.current_date]


class MockMeanReversionStrategy:
    """
    Mean reversion strategy implementation for backtesting.

    Buys oversold assets and sells overbought assets based on
    the assumption that prices will revert to their mean.
    """

    def __init__(self):
        """Initialize the strategy"""
        self.symbols = []
        self.data = {}
        self.current_date = None
        self.positions = {}
        self.position_entry_prices = {}  # Track entry prices for positions
        self.rsi_period = 14
        self.rsi_overbought = 65  # More sensitive overbought threshold
        self.rsi_oversold = 35  # More sensitive oversold threshold
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.position_size = 0.05  # 5% of capital per position
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.15  # 15% take profit (higher for mean reversion)
        self.max_positions = 5  # Maximum number of concurrent positions

    def initialize(self, symbols=None, **kwargs):
        """Initialize with configuration parameters"""
        self.symbols = symbols or []

        # Override default parameters if provided
        self.rsi_period = kwargs.get("rsi_period", self.rsi_period)
        self.rsi_overbought = kwargs.get("rsi_overbought", self.rsi_overbought)
        self.rsi_oversold = kwargs.get("rsi_oversold", self.rsi_oversold)
        self.bollinger_period = kwargs.get("bollinger_period", self.bollinger_period)
        self.bollinger_std = kwargs.get("bollinger_std", self.bollinger_std)
        self.position_size = kwargs.get("position_size", self.position_size)
        self.stop_loss_pct = kwargs.get("stop_loss_pct", self.stop_loss_pct)
        self.take_profit_pct = kwargs.get("take_profit_pct", self.take_profit_pct)
        self.max_positions = kwargs.get("max_positions", self.max_positions)

    def on_trading_iteration(self):
        """Run on each trading iteration"""
        pass

    def get_signal(self, symbol, bar_data):
        """Generate trading signal for a symbol"""
        if symbol not in self.data or bar_data is None:
            return None

        # Get full historical data
        df = self.data[symbol]

        # Skip if not enough data
        if len(df) < self.bollinger_period + 5:
            return None

        # Check for exit based on stop loss or take profit first
        if symbol in self.positions and self.positions[symbol] > 0:
            entry_price = self.position_entry_prices.get(symbol, 0)
            current_price = bar_data["close"]

            if entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price

                # Stop loss - exit if loss exceeds threshold
                if pnl_pct < -self.stop_loss_pct:
                    qty = self.positions[symbol]
                    return {
                        "side": "sell",
                        "qty": qty,
                        "price": current_price,
                        "reason": "stop_loss",
                    }

                # Take profit - exit if gain exceeds threshold
                if pnl_pct > self.take_profit_pct:
                    qty = self.positions[symbol]
                    return {
                        "side": "sell",
                        "qty": qty,
                        "price": current_price,
                        "reason": "take_profit",
                    }

        # Calculate indicators
        try:
            # RSI
            close_prices = df["close"]
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Bollinger Bands
            rolling_mean = close_prices.rolling(window=self.bollinger_period).mean()
            rolling_std = close_prices.rolling(window=self.bollinger_period).std()
            upper_band = rolling_mean + (rolling_std * self.bollinger_std)
            lower_band = rolling_mean - (rolling_std * self.bollinger_std)

            # Get latest values
            current_price = bar_data["close"]
            current_rsi = rsi.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_mean = rolling_mean.iloc[-1]

            # Check if we're already at position limit for buy signals
            active_positions = len(self.positions)

            # Generate signal
            if pd.notna(current_rsi) and pd.notna(current_lower) and pd.notna(current_upper):
                # Buy signal: price below lower band AND RSI is oversold
                # OR Price is significantly below mean and RSI showing momentum reversal
                if (
                    (current_price < current_lower and current_rsi < self.rsi_oversold)
                    or (
                        current_price < 0.95 * current_mean
                        and current_rsi > 30
                        and current_rsi < 40
                    )
                ) and active_positions < self.max_positions:
                    # Calculate position size with volatility adjustment
                    volatility = rolling_std.iloc[-1] / current_price
                    position_size_adj = max(
                        0.02, min(self.position_size, 0.05 / (volatility * 10))
                    )  # Reduce size for volatile assets
                    qty = max(1, int(10000 * position_size_adj / current_price))
                    return {"side": "buy", "qty": qty, "price": current_price, "reason": "oversold"}

                # Sell signal: price reaches or crosses above mean after being below
                elif current_price >= current_mean and symbol in self.positions:
                    entry_price = self.position_entry_prices.get(symbol, 0)
                    # Only sell if we're in profit or close to breakeven
                    if current_price > entry_price * 0.98:
                        qty = self.positions[symbol]
                        return {
                            "side": "sell",
                            "qty": qty,
                            "price": current_price,
                            "reason": "mean_reversion",
                        }

                # Sell signal: RSI overbought
                elif current_rsi > self.rsi_overbought and symbol in self.positions:
                    qty = self.positions[symbol]
                    return {
                        "side": "sell",
                        "qty": qty,
                        "price": current_price,
                        "reason": "overbought",
                    }

        except Exception as e:
            import logging

            logging.error(f"Error calculating signals for {symbol}: {e}")

        return None
