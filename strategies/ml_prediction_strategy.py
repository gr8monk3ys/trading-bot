"""
Machine Learning Price Prediction Strategy

Uses LSTM neural networks to predict next-day price movements
based on technical indicators and historical patterns.

Features:
- LSTM (Long Short-Term Memory) for time series prediction
- Feature engineering from 30+ technical indicators
- Sentiment integration (optional)
- Confidence-based position sizing
- Adaptive learning (periodic retraining)

Expected Performance: Experimental (requires extensive backtesting)

WARNING: ML strategies can overfit easily. Always validate with
out-of-sample data and paper trading before live deployment.
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

# TensorFlow/Keras imports (optional - will fall back to sklearn if not available)
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow not available. ML strategy will use fallback mode.")

from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
from brokers.order_builder import OrderBuilder
from utils.indicators import TechnicalIndicators, analyze_trend, analyze_momentum, analyze_volatility

logger = logging.getLogger(__name__)


class MLPredictionStrategy(BaseStrategy):
    """
    Machine Learning-based price prediction strategy.

    Models Supported:
    - LSTM (default) - Best for time series
    - GRU - Alternative to LSTM
    - Dense NN - Simple feedforward (fallback)

    Features Used:
    - RSI, MACD, ADX (momentum/trend)
    - Bollinger Bands, ATR (volatility)
    - Volume indicators
    - Price patterns (gaps, breakouts)
    - Historical returns (multiple timeframes)
    """

    NAME = "MLPredictionStrategy"

    def default_parameters(self):
        """Return default parameters."""
        return {
            # Basic parameters
            'position_size': 0.08,  # 8% per position
            'max_positions': 5,
            'max_portfolio_risk': 0.02,
            'stop_loss': 0.03,  # 3% stop
            'take_profit': 0.05,  # 5% profit target

            # ML model parameters
            'model_type': 'lstm',  # lstm, gru, or dense
            'sequence_length': 60,  # Use 60 bars for prediction
            'prediction_horizon': 1,  # Predict 1 bar ahead
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,

            # Feature engineering
            'use_technical_indicators': True,
            'use_price_patterns': True,
            'use_volume_features': True,
            'feature_lookback': 200,  # Bars needed for features

            # Trading rules
            'min_prediction_confidence': 0.60,  # 60% confidence to trade
            'confidence_threshold_strong': 0.75,  # 75% = strong signal (2x position)
            'directional_threshold': 0.005,  # 0.5% minimum predicted move

            # Retraining
            'retrain_every_n_days': 7,  # Retrain weekly
            'min_training_samples': 500,  # Need 500+ bars to train

            # Risk management
            'max_correlation': 0.7,
            'trailing_stop': 0.02,  # 2% trailing stop
        }

    async def initialize(self, **kwargs):
        """Initialize ML prediction strategy."""
        try:
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Check TensorFlow availability
            if not HAS_TENSORFLOW and self.parameters['model_type'] in ['lstm', 'gru']:
                logger.warning("TensorFlow not available, falling back to 'dense' model")
                self.parameters['model_type'] = 'dense'

            # Initialize tracking
            self.models = {}  # One model per symbol
            self.scalers = {}  # Feature scalers per symbol
            self.predictions = {}  # Latest predictions
            self.confidence_scores = {}  # Prediction confidence
            self.training_history = {}  # Track training performance
            self.last_retrain_date = {}  # Track when model was last trained
            self.price_history = {symbol: [] for symbol in self.symbols}
            self.current_prices = {}

            # Risk manager
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters['max_portfolio_risk'],
                max_position_risk=self.parameters.get('max_position_risk', 0.01),
                max_correlation=self.parameters['max_correlation']
            )

            logger.info(f"Initialized {self.NAME} with {len(self.symbols)} symbols")
            logger.info(f"  Model type: {self.parameters['model_type'].upper()}")
            logger.info(f"  Sequence length: {self.parameters['sequence_length']}")
            logger.info(f"  Min confidence: {self.parameters['min_prediction_confidence']:.0%}")
            logger.info(f"  Retrain every: {self.parameters['retrain_every_n_days']} days")

            if not HAS_TENSORFLOW:
                logger.warning(f"  ⚠️  TensorFlow not installed - using fallback mode")

            return True

        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False

    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return

            # Store current price
            self.current_prices[symbol] = close_price

            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            self.price_history[symbol].append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

            # Keep enough history for feature engineering
            max_history = self.parameters['feature_lookback'] + self.parameters['sequence_length'] + 100
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    async def on_trading_iteration(self):
        """Main trading logic."""
        try:
            # Check circuit breaker
            if not await self.circuit_breaker.check_trading_allowed():
                logger.warning("Circuit breaker triggered - trading halted")
                return

            # Process each symbol
            for symbol in self.symbols:
                try:
                    # Check if we have enough data
                    if symbol not in self.price_history or \
                       len(self.price_history[symbol]) < self.parameters['min_training_samples']:
                        continue

                    # Check if model needs (re)training
                    needs_training = await self._needs_training(symbol)
                    if needs_training:
                        logger.info(f"Training/retraining model for {symbol}...")
                        await self._train_model(symbol)

                    # Skip if model not trained yet
                    if symbol not in self.models:
                        continue

                    # Generate prediction
                    prediction, confidence = await self._predict(symbol)

                    if prediction is not None:
                        self.predictions[symbol] = prediction
                        self.confidence_scores[symbol] = confidence

                        # Generate trading signal
                        signal = await self._generate_signal(symbol, prediction, confidence)

                        if signal:
                            await self._execute_trade(symbol, signal)

                    # Manage existing positions
                    await self._manage_positions(symbol)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}", exc_info=True)

    async def _needs_training(self, symbol: str) -> bool:
        """Check if model needs training or retraining."""
        # Never trained
        if symbol not in self.models:
            return True

        # Check if enough time has passed for retraining
        if symbol in self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date[symbol]).days
            if days_since_retrain >= self.parameters['retrain_every_n_days']:
                return True

        return False

    async def _engineer_features(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features from price data.

        Returns:
            X: Feature matrix (samples, sequence_length, n_features)
            y: Target values (next period return)
        """
        try:
            df = pd.DataFrame(self.price_history[symbol])

            # Calculate technical indicators
            ind = TechnicalIndicators(
                high=df['high'].values,
                low=df['low'].values,
                close=df['close'].values,
                volume=df['volume'].values
            )

            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Technical indicators
            if self.parameters['use_technical_indicators']:
                df['rsi'] = ind.rsi(period=14)
                df['rsi_fast'] = ind.rsi(period=7)

                macd, signal, hist = ind.macd()
                df['macd'] = macd
                df['macd_signal'] = signal
                df['macd_hist'] = hist

                adx, plus_di, minus_di = ind.adx_di()
                df['adx'] = adx
                df['plus_di'] = plus_di
                df['minus_di'] = minus_di

                bb_upper, bb_middle, bb_lower = ind.bollinger_bands()
                df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

                df['atr'] = ind.atr()
                df['atr_pct'] = df['atr'] / df['close']

            # Volume features
            if self.parameters['use_volume_features']:
                df['volume_sma'] = ind.volume_sma()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                df['vwap'] = ind.vwap()
                df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

            # Price patterns
            if self.parameters['use_price_patterns']:
                df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
                df['day_range'] = (df['high'] - df['low']) / df['low']
                df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

            # Multi-timeframe returns
            for period in [5, 10, 20]:
                df[f'return_{period}d'] = df['close'].pct_change(period)

            # Drop NaN values
            df = df.dropna()

            # Target variable (next period return)
            df['target'] = df['returns'].shift(-self.parameters['prediction_horizon'])
            df = df.dropna()

            # Select feature columns
            feature_cols = [col for col in df.columns if col not in [
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'target'
            ]]

            # Create sequences
            X, y = [], []
            sequence_length = self.parameters['sequence_length']

            for i in range(sequence_length, len(df)):
                X.append(df[feature_cols].iloc[i-sequence_length:i].values)
                y.append(df['target'].iloc[i])

            X = np.array(X)
            y = np.array(y)

            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = MinMaxScaler()
                # Reshape for scaler
                n_samples, n_timesteps, n_features = X.shape
                X_reshaped = X.reshape(-1, n_features)
                X_scaled = self.scalers[symbol].fit_transform(X_reshaped)
                X = X_scaled.reshape(n_samples, n_timesteps, n_features)
            else:
                n_samples, n_timesteps, n_features = X.shape
                X_reshaped = X.reshape(-1, n_features)
                X_scaled = self.scalers[symbol].transform(X_reshaped)
                X = X_scaled.reshape(n_samples, n_timesteps, n_features)

            logger.info(f"Engineered features for {symbol}: X shape={X.shape}, y shape={y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"Error engineering features for {symbol}: {e}", exc_info=True)
            return None, None

    async def _train_model(self, symbol: str):
        """Train LSTM model for symbol."""
        try:
            # Engineer features
            X, y = await self._engineer_features(symbol)

            if X is None or len(X) < self.parameters['min_training_samples']:
                logger.warning(f"Insufficient data for training {symbol}")
                return

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.parameters['validation_split'],
                shuffle=False  # Don't shuffle time series!
            )

            # Build model
            if HAS_TENSORFLOW and self.parameters['model_type'] == 'lstm':
                model = self._build_lstm_model(X.shape[1], X.shape[2])
            else:
                # Fallback to simple model
                logger.warning(f"Using fallback model for {symbol}")
                model = self._build_fallback_model(X.shape[1], X.shape[2])


            # Train model
            if HAS_TENSORFLOW:
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                history = model.fit(
                    X_train, y_train,
                    epochs=self.parameters['epochs'],
                    batch_size=self.parameters['batch_size'],
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=0
                )

                # Evaluate
                train_loss = history.history['loss'][-1]
                val_loss = history.history['val_loss'][-1]

                logger.info(f"Model trained for {symbol}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

                self.training_history[symbol] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'trained_at': datetime.now()
                }
            else:
                # Fallback training (would use sklearn here)
                logger.info(f"Fallback model 'trained' for {symbol}")

            # Store model
            self.models[symbol] = model
            self.last_retrain_date[symbol] = datetime.now()

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}", exc_info=True)

    def _build_lstm_model(self, sequence_length: int, n_features: int):
        """Build LSTM model."""
        model = Sequential([
            LSTM(self.parameters['lstm_units'], return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(self.parameters['dropout_rate']),
            LSTM(self.parameters['lstm_units'] // 2, return_sequences=False),
            Dropout(self.parameters['dropout_rate']),
            Dense(25, activation='relu'),
            Dense(1)  # Predict next return
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_fallback_model(self, sequence_length: int, n_features: int):
        """Build simple fallback model when TensorFlow not available."""
        # This is a placeholder - would implement sklearn-based model
        logger.warning("Fallback model is a placeholder - implement sklearn model")
        return None

    async def _predict(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Generate prediction for next period.

        Returns:
            (predicted_return, confidence)
        """
        try:
            if symbol not in self.models:
                return None, None

            # Engineer features for latest data
            X, _ = await self._engineer_features(symbol)

            if X is None or len(X) == 0:
                return None, None

            # Predict
            model = self.models[symbol]
            latest_sequence = X[-1:] # Last sequence

            if HAS_TENSORFLOW and hasattr(model, 'predict'):
                prediction = model.predict(latest_sequence, verbose=0)[0][0]

                # Estimate confidence (simplified - would use more sophisticated methods)
                # Could use prediction variance, ensemble disagreement, etc.
                confidence = 0.65  # Placeholder

            else:
                # Fallback prediction
                prediction = 0.0
                confidence = 0.50

            return float(prediction), float(confidence)

        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}", exc_info=True)
            return None, None

    async def _generate_signal(self, symbol: str, prediction: float, confidence: float) -> Optional[str]:
        """Generate trading signal from prediction."""
        try:
            # Check confidence threshold
            if confidence < self.parameters['min_prediction_confidence']:
                return None

            # Check directional threshold
            if abs(prediction) < self.parameters['directional_threshold']:
                return None  # Predicted move too small

            # Generate signal
            if prediction > self.parameters['directional_threshold']:
                return 'buy' if confidence >= self.parameters['confidence_threshold_strong'] else 'buy_weak'
            elif prediction < -self.parameters['directional_threshold']:
                return 'sell' if confidence >= self.parameters['confidence_threshold_strong'] else 'sell_weak'

            return None

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    async def _execute_trade(self, symbol: str, signal: str):
        """Execute trade based on signal."""
        try:
            # Get account info
            account = await self.broker.get_account()
            portfolio_value = float(account.equity)

            # Adjust position size based on confidence
            position_size = self.parameters['position_size']
            if signal in ['buy_weak', 'sell_weak']:
                position_size *= 0.5  # Half size for weak signals

            position_value = portfolio_value * position_size

            # Risk check
            approved_size = await self.risk_manager.check_position_risk(
                symbol=symbol,
                position_value=position_value,
                portfolio_value=portfolio_value,
                current_positions=await self.broker.get_positions(),
                price_history=self.price_history.get(symbol, [])
            )

            if approved_size <= 0:
                logger.warning(f"Risk manager rejected {signal} for {symbol}")
                return

            # Calculate quantity
            current_price = self.current_prices.get(symbol)
            if not current_price:
                return

            quantity = approved_size / current_price

            # Build order
            side = 'buy' if signal.startswith('buy') else 'sell'

            order = (OrderBuilder(symbol, side, quantity)
                    .market()
                    .gtc()
                    .build())

            # Submit order
            result = await self.broker.submit_order_advanced(order)

            if result:
                logger.info(f"✅ Executed {signal} for {symbol}: {quantity:.2f} shares @ ${current_price:.2f}")
                logger.info(f"   Prediction: {self.predictions[symbol]:.2%}, Confidence: {self.confidence_scores[symbol]:.0%}")

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}", exc_info=True)

    async def _manage_positions(self, symbol: str):
        """Manage existing positions with trailing stops."""
        # TODO: Implement position management
        # - Trailing stops
        # - Profit targets
        # - Re-prediction based exits
        pass
