import pandas as pd
import numpy as np
from datetime import timedelta
import talib
from strategies.base_strategy import BaseStrategy
import os

class OptionsStrategy(BaseStrategy):
    def initialize(self, symbols=None, rsi_period=14, rsi_overbought=70, 
                  rsi_oversold=30, volatility_window=20, **kwargs):
        super().initialize(symbols, **kwargs)
        #self.api = REST( #Removed
        #    base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        #    key_id=os.getenv("ALPACA_API_KEY"),
        #    secret_key=os.getenv("ALPACA_SECRET_KEY")
        #)
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volatility_window = volatility_window
        #self.last_trade_dict = {symbol: None for symbol in self.symbols} #Removed
        self.option_positions = {}

    def get_technical_indicators(self, symbol):
        """Calculate technical indicators for the symbol."""
        try:
            # Get historical data
            end = self.get_datetime()
            start = end - timedelta(days=100)
            historical_data = self.get_historical_prices(symbol, 100)
            df = pd.DataFrame(historical_data, columns=['close'])
            
            # Calculate RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            
            # Calculate Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            
            # Calculate MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Calculate Implied Volatility (simplified)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
            
            return df.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
            return None

    def analyze_symbol(self, symbol):
        """Analyze a symbol using technical indicators for options trading."""
        indicators = self.get_technical_indicators(symbol)
        if indicators is None:
            return None
            
        signal = None
        strategy = None
        
        # Determine market conditions
        is_overbought = indicators['rsi'] > self.rsi_overbought
        is_oversold = indicators['rsi'] < self.rsi_oversold
        is_high_volatility = indicators['volatility'] > indicators['volatility'].mean()
        price_above_upper_bb = indicators['close'] > indicators['bb_upper']
        price_below_lower_bb = indicators['close'] < indicators['bb_lower']
        
        # Strategy selection based on market conditions
        if is_overbought and price_above_upper_bb:
            signal = "sell"
            if is_high_volatility:
                strategy = "put_debit_spread"  # Bearish with defined risk
            else:
                strategy = "covered_call"  # Bearish with income
                
        elif is_oversold and price_below_lower_bb:
            signal = "buy"
            if is_high_volatility:
                strategy = "call_debit_spread"  # Bullish with defined risk
            else:
                strategy = "naked_call"  # Bullish with leverage
                
        elif indicators['volatility'] > indicators['volatility'].mean() * 1.5:
            signal = "neutral"
            strategy = "iron_condor"  # High volatility, range-bound strategy
            
        return signal, strategy if signal else None

    def get_option_contract(self, symbol, strategy, current_price):
        """Get appropriate option contract based on strategy."""
        try:
            expiration = self.get_datetime() + timedelta(days=30)  # 30-day options
            
            if strategy in ["naked_call", "call_debit_spread"]:
                strike = round(current_price * 1.05, 2)  # 5% OTM for calls
                contract = f"{symbol}{expiration.strftime('%y%m%d')}C{strike}"
            elif strategy in ["covered_call", "put_debit_spread"]:
                strike = round(current_price * 0.95, 2)  # 5% OTM for puts
                contract = f"{symbol}{expiration.strftime('%y%m%d')}P{strike}"
            else:  # iron_condor
                upper_strike = round(current_price * 1.10, 2)  # 10% OTM
                lower_strike = round(current_price * 0.90, 2)  # 10% OTM
                contract = {
                    "upper_call": f"{symbol}{expiration.strftime('%y%m%d')}C{upper_strike}",
                    "lower_put": f"{symbol}{expiration.strftime('%y%m%d')}P{lower_strike}"
                }
                
            return contract
            
        except Exception as e:
            self.logger.error(f"Error getting option contract for {symbol}: {str(e)}")
            return None

    def execute_trade(self, symbol, signal_data):
        """Execute an options trade based on the signal and strategy."""
        try:
            if not signal_data:
                return
                
            signal, strategy = signal_data
            cash, current_price, _ = self.position_sizing(symbol, strategy_type="option")
            
            contract = self.get_option_contract(symbol, strategy, current_price)
            if not contract:
                return
                
            if strategy == "naked_call":
                self._execute_naked_call(symbol, contract, cash)
            elif strategy == "covered_call":
                self._execute_covered_call(symbol, contract, cash)
            elif strategy == "call_debit_spread":
                self._execute_call_spread(symbol, contract, cash)
            elif strategy == "put_debit_spread":
                self._execute_put_spread(symbol, contract, cash)
            elif strategy == "iron_condor":
                self._execute_iron_condor(symbol, contract, cash)
                
        except Exception as e:
            self.logger.error(f"Error executing options trade for {symbol}: {str(e)}")

    def _execute_naked_call(self, symbol, contract, cash):
        """Execute a naked call option trade."""
        quantity = round(cash * 0.1 / self.get_last_price(contract))  # Use 10% of available cash
        order = self.create_order(
            contract,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(contract)
        )
        self.submit_order(order)
        self.logger.info(f"Executed naked call for {symbol} - Contract: {contract}, Quantity: {quantity}")

    def _execute_covered_call(self, symbol, contract, cash):
        """Execute a covered call option trade."""
        # Buy the underlying stock
        stock_price = self.get_last_price(symbol)
        stock_quantity = round(cash * 0.5 / stock_price)
        self.create_order(symbol, stock_quantity, "buy", type="market")
        
        # Sell calls against the stock position
        option_quantity = stock_quantity // 100  # 1 contract per 100 shares
        if option_quantity > 0:
            order = self.create_order(
                contract,
                option_quantity,
                "sell",
                type="limit",
                limit_price=self.get_last_price(contract)
            )
            self.submit_order(order)
            self.logger.info(f"Executed covered call for {symbol} - Contract: {contract}, Quantity: {option_quantity}")

    def _execute_call_spread(self, symbol, contract, cash):
        """Execute a call debit spread."""
        # Buy lower strike call
        quantity = round(cash * 0.1 / self.get_last_price(contract))
        buy_order = self.create_order(
            contract,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(contract)
        )
        
        # Sell higher strike call
        higher_strike = float(contract.split('C')[1]) + 5  # 5 points higher
        sell_contract = f"{contract.split('C')[0]}C{higher_strike}"
        sell_order = self.create_order(
            sell_contract,
            quantity,
            "sell",
            type="limit",
            limit_price=self.get_last_price(sell_contract)
        )
        
        self.submit_order(buy_order)
        self.submit_order(sell_order)
        self.logger.info(f"Executed call spread for {symbol} - Quantity: {quantity}")

    def _execute_iron_condor(self, symbol, contracts, cash):
        """Execute an iron condor strategy."""
        quantity = round(cash * 0.1 / self.get_last_price(contracts['upper_call']))
        
        # Sell OTM put
        put_credit_order = self.create_order(
            contracts['lower_put'],
            quantity,
            "sell",
            type="limit",
            limit_price=self.get_last_price(contracts['lower_put'])
        )
        
        # Buy further OTM put
        lower_put_strike = float(contracts['lower_put'].split('P')[1]) - 5
        lower_put = f"{contracts['lower_put'].split('P')[0]}P{lower_put_strike}"
        put_debit_order = self.create_order(
            lower_put,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(lower_put)
        )
        
        # Sell OTM call
        call_credit_order = self.create_order(
            contracts['upper_call'],
            quantity,
            "sell",
            type="limit",
            limit_price=self.get_last_price(contracts['upper_call'])
        )
        
        # Buy further OTM call
        higher_call_strike = float(contracts['upper_call'].split('C')[1]) + 5
        higher_call = f"{contracts['upper_call'].split('C')[0]}C{higher_call_strike}"
        call_debit_order = self.create_order(
            higher_call,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(higher_call)
        )
        
        self.submit_order(put_credit_order)
        self.submit_order(put_debit_order)
        self.submit_order(call_credit_order)
        self.submit_order(call_debit_order)
        self.logger.info(f"Executed iron condor for {symbol} - Quantity: {quantity}")

    def on_trading_iteration(self):
        """Execute trading iteration for all symbols."""
        if not self.check_risk_limits():
            self.logger.warning("Risk limits breached, skipping trading iteration")
            return

        for symbol in self.symbols:
            try:
                signal_data = self.analyze_symbol(symbol)
                self.execute_trade(symbol, signal_data)
            except Exception as e:
                self.logger.error(f"Error in trading iteration for {symbol}: {str(e)}")
</content>
import pandas as pd
import numpy as np
from datetime import timedelta
from alpaca_trade_api import REST
from strategies.base_strategy import BaseStrategy
import os
import talib

class OptionsStrategy(BaseStrategy):
    def initialize(self, symbols=None, rsi_period=14, rsi_overbought=70, 
                  rsi_oversold=30, volatility_window=20, **kwargs):
        super().initialize(symbols, **kwargs)
        self.api = REST(
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY")
        )
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volatility_window = volatility_window
        self.last_trade_dict = {symbol: None for symbol in self.symbols}
        self.option_positions = {}

    def get_technical_indicators(self, symbol):
        """Calculate technical indicators for the symbol."""
        try:
            # Get historical data
            end = self.get_datetime()
            start = end - timedelta(days=100)
            historical_data = self.get_historical_prices(symbol, 100)
            df = pd.DataFrame(historical_data, columns=['close'])
            
            # Calculate RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            
            # Calculate Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            
            # Calculate MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Calculate Implied Volatility (simplified)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
            
            return df.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
            return None

    def analyze_symbol(self, symbol):
        """Analyze a symbol using technical indicators for options trading."""
        indicators = self.get_technical_indicators(symbol)
        if indicators is None:
            return None
            
        signal = None
        strategy = None
        
        # Determine market conditions
        is_overbought = indicators['rsi'] > self.rsi_overbought
        is_oversold = indicators['rsi'] < self.rsi_oversold
        is_high_volatility = indicators['volatility'] > indicators['volatility'].mean()
        price_above_upper_bb = indicators['close'] > indicators['bb_upper']
        price_below_lower_bb = indicators['close'] < indicators['bb_lower']
        
        # Strategy selection based on market conditions
        if is_overbought and price_above_upper_bb:
            signal = "sell"
            if is_high_volatility:
                strategy = "put_debit_spread"  # Bearish with defined risk
            else:
                strategy = "covered_call"  # Bearish with income
                
        elif is_oversold and price_below_lower_bb:
            signal = "buy"
            if is_high_volatility:
                strategy = "call_debit_spread"  # Bullish with defined risk
            else:
                strategy = "naked_call"  # Bullish with leverage
                
        elif indicators['volatility'] > indicators['volatility'].mean() * 1.5:
            signal = "neutral"
            strategy = "iron_condor"  # High volatility, range-bound strategy
            
        return signal, strategy if signal else None

    def get_option_contract(self, symbol, strategy, current_price):
        """Get appropriate option contract based on strategy."""
        try:
            expiration = self.get_datetime() + timedelta(days=30)  # 30-day options
            
            if strategy in ["naked_call", "call_debit_spread"]:
                strike = round(current_price * 1.05, 2)  # 5% OTM for calls
                contract = f"{symbol}{expiration.strftime('%y%m%d')}C{strike}"
            elif strategy in ["covered_call", "put_debit_spread"]:
                strike = round(current_price * 0.95, 2)  # 5% OTM for puts
                contract = f"{symbol}{expiration.strftime('%y%m%d')}P{strike}"
            else:  # iron_condor
                upper_strike = round(current_price * 1.10, 2)  # 10% OTM
                lower_strike = round(current_price * 0.90, 2)  # 10% OTM
                contract = {
                    "upper_call": f"{symbol}{expiration.strftime('%y%m%d')}C{upper_strike}",
                    "lower_put": f"{symbol}{expiration.strftime('%y%m%d')}P{lower_strike}"
                }
                
            return contract
            
        except Exception as e:
            self.logger.error(f"Error getting option contract for {symbol}: {str(e)}")
            return None

    def execute_trade(self, symbol, signal_data):
        """Execute an options trade based on the signal and strategy."""
        try:
            if not signal_data:
                return
                
            signal, strategy = signal_data
            cash, current_price, _ = self.position_sizing(symbol, strategy_type="option")
            
            contract = self.get_option_contract(symbol, strategy, current_price)
            if not contract:
                return
                
            if strategy == "naked_call":
                self._execute_naked_call(symbol, contract, cash)
            elif strategy == "covered_call":
                self._execute_covered_call(symbol, contract, cash)
            elif strategy == "call_debit_spread":
                self._execute_call_spread(symbol, contract, cash)
            elif strategy == "put_debit_spread":
                self._execute_put_spread(symbol, contract, cash)
            elif strategy == "iron_condor":
                self._execute_iron_condor(symbol, contract, cash)
                
        except Exception as e:
            self.logger.error(f"Error executing options trade for {symbol}: {str(e)}")

    def _execute_naked_call(self, symbol, contract, cash):
        """Execute a naked call option trade."""
        quantity = round(cash * 0.1 / self.get_last_price(contract))  # Use 10% of available cash
        order = self.create_order(
            contract,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(contract)
        )
        self.submit_order(order)
        self.logger.info(f"Executed naked call for {symbol} - Contract: {contract}, Quantity: {quantity}")

    def _execute_covered_call(self, symbol, contract, cash):
        """Execute a covered call option trade."""
        # Buy the underlying stock
        stock_price = self.get_last_price(symbol)
        stock_quantity = round(cash * 0.5 / stock_price)
        self.create_order(symbol, stock_quantity, "buy", type="market")
        
        # Sell calls against the stock position
        option_quantity = stock_quantity // 100  # 1 contract per 100 shares
        if option_quantity > 0:
            order = self.create_order(
                contract,
                option_quantity,
                "sell",
                type="limit",
                limit_price=self.get_last_price(contract)
            )
            self.submit_order(order)
            self.logger.info(f"Executed covered call for {symbol} - Contract: {contract}, Quantity: {option_quantity}")

    def _execute_call_spread(self, symbol, contract, cash):
        """Execute a call debit spread."""
        # Buy lower strike call
        quantity = round(cash * 0.1 / self.get_last_price(contract))
        buy_order = self.create_order(
            contract,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(contract)
        )
        
        # Sell higher strike call
        higher_strike = float(contract.split('C')[1]) + 5  # 5 points higher
        sell_contract = f"{contract.split('C')[0]}C{higher_strike}"
        sell_order = self.create_order(
            sell_contract,
            quantity,
            "sell",
            type="limit",
            limit_price=self.get_last_price(sell_contract)
        )
        
        self.submit_order(buy_order)
        self.submit_order(sell_order)
        self.logger.info(f"Executed call spread for {symbol} - Quantity: {quantity}")

    def _execute_iron_condor(self, symbol, contracts, cash):
        """Execute an iron condor strategy."""
        quantity = round(cash * 0.1 / self.get_last_price(contracts['upper_call']))
        
        # Sell OTM put
        put_credit_order = self.create_order(
            contracts['lower_put'],
            quantity,
            "sell",
            type="limit",
            limit_price=self.get_last_price(contracts['lower_put'])
        )
        
        # Buy further OTM put
        lower_put_strike = float(contracts['lower_put'].split('P')[1]) - 5
        lower_put = f"{contracts['lower_put'].split('P')[0]}P{lower_put_strike}"
        put_debit_order = self.create_order(
            lower_put,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(lower_put)
        )
        
        # Sell OTM call
        call_credit_order = self.create_order(
            contracts['upper_call'],
            quantity,
            "sell",
            type="limit",
            limit_price=self.get_last_price(contracts['upper_call'])
        )
        
        # Buy further OTM call
        higher_call_strike = float(contracts['upper_call'].split('C')[1]) + 5
        higher_call = f"{contracts['upper_call'].split('C')[0]}C{higher_call_strike}"
        call_debit_order = self.create_order(
            higher_call,
            quantity,
            "buy",
            type="limit",
            limit_price=self.get_last_price(higher_call)
        )
        
        self.submit_order(put_credit_order)
        self.submit_order(put_debit_order)
        self.submit_order(call_credit_order)
        self.submit_order(call_debit_order)
        self.logger.info(f"Executed iron condor for {symbol} - Quantity: {quantity}")

    def on_trading_iteration(self):
        """Execute trading iteration for all symbols."""
        if not self.check_risk_limits():
            self.logger.warning("Risk limits breached, skipping trading iteration")
            return

        for symbol in self.symbols:
            try:
                signal_data = self.analyze_symbol(symbol)
                self.execute_trade(symbol, signal_data)
            except Exception as e:
                self.logger.error(f"Error in trading iteration for {symbol}: {str(e)}")
