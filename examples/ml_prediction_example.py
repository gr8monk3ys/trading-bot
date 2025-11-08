"""
Machine Learning Price Prediction Strategy Example

Demonstrates LSTM-based price prediction using technical indicators.

WARNING: ML strategies are experimental and can easily overfit!
Always validate with extensive out-of-sample testing and paper trading.

Requirements:
- tensorflow (pip install tensorflow)
- scikit-learn (pip install scikit-learn)

This example shows:
1. Basic ML strategy setup
2. Feature engineering configuration
3. Training and prediction workflow
4. Confidence-based trading
"""

import asyncio
import logging
from datetime import datetime

from brokers.alpaca_broker import AlpacaBroker
from strategies.ml_prediction_strategy import MLPredictionStrategy, HAS_TENSORFLOW

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run ML prediction strategy example."""

    logger.info("=" * 80)
    logger.info("MACHINE LEARNING PRICE PREDICTION EXAMPLE")
    logger.info("=" * 80)
    logger.info("")

    # Check TensorFlow availability
    if HAS_TENSORFLOW:
        logger.info("✅ TensorFlow is installed - LSTM models available")
    else:
        logger.warning("⚠️  TensorFlow not installed - using fallback mode")
        logger.info("   Install with: pip install tensorflow")
    logger.info("")

    # Initialize broker
    broker = AlpacaBroker(paper=True)
    logger.info("✅ Initialized Alpaca broker (PAPER TRADING)")
    logger.info("")

    # Strategy 1: Conservative ML (High Confidence Only)
    logger.info("=" * 80)
    logger.info("STRATEGY 1: Conservative ML Trading")
    logger.info("=" * 80)
    logger.info("")

    conservative_params = {
        'position_size': 0.05,  # Small 5% positions
        'max_positions': 3,

        # Model configuration
        'model_type': 'lstm',
        'sequence_length': 60,  # Use 60 bars (1 hour if 1-min bars)
        'lstm_units': 50,
        'epochs': 30,  # Fewer epochs to avoid overfitting

        # Trading rules - VERY conservative
        'min_prediction_confidence': 0.70,  # Need 70% confidence
        'confidence_threshold_strong': 0.80,  # 80% for full position
        'directional_threshold': 0.01,  # Need 1% predicted move

        # Features
        'use_technical_indicators': True,
        'use_price_patterns': True,
        'use_volume_features': True,

        # Retraining
        'retrain_every_n_days': 7,  # Retrain weekly

        # Risk management
        'stop_loss': 0.02,  # Tight 2% stop
        'take_profit': 0.04,  # 4% profit target
    }

    strategy_conservative = MLPredictionStrategy(
        broker=broker,
        symbols=['SPY', 'QQQ'],  # Liquid ETFs for testing
        parameters=conservative_params
    )

    await strategy_conservative.initialize()

    logger.info("Configuration:")
    logger.info(f"  Model: LSTM with {conservative_params['lstm_units']} units")
    logger.info(f"  Sequence length: {conservative_params['sequence_length']} bars")
    logger.info(f"  Min confidence: {conservative_params['min_prediction_confidence']:.0%}")
    logger.info(f"  Min predicted move: {conservative_params['directional_threshold']:.1%}")
    logger.info(f"  Risk: CONSERVATIVE")
    logger.info("")

    # Strategy 2: Moderate ML (Balanced Approach)
    logger.info("=" * 80)
    logger.info("STRATEGY 2: Moderate ML Trading")
    logger.info("=" * 80)
    logger.info("")

    moderate_params = {
        'position_size': 0.08,
        'max_positions': 5,

        # Model configuration
        'model_type': 'lstm',
        'sequence_length': 90,  # Longer sequence for more context
        'lstm_units': 100,  # Larger model
        'dropout_rate': 0.3,  # More dropout to prevent overfitting
        'epochs': 50,

        # Trading rules - Moderate
        'min_prediction_confidence': 0.60,  # 60% confidence threshold
        'confidence_threshold_strong': 0.75,
        'directional_threshold': 0.005,  # 0.5% minimum move

        # Features
        'use_technical_indicators': True,
        'use_price_patterns': True,
        'use_volume_features': True,

        # Retraining
        'retrain_every_n_days': 5,  # Retrain twice a week

        # Risk management
        'stop_loss': 0.03,
        'take_profit': 0.05,
        'trailing_stop': 0.02,
    }

    strategy_moderate = MLPredictionStrategy(
        broker=broker,
        symbols=['AAPL', 'MSFT', 'NVDA', 'TSLA'],
        parameters=moderate_params
    )

    await strategy_moderate.initialize()

    logger.info("Configuration:")
    logger.info(f"  Model: LSTM with {moderate_params['lstm_units']} units")
    logger.info(f"  Sequence length: {moderate_params['sequence_length']} bars")
    logger.info(f"  Min confidence: {moderate_params['min_prediction_confidence']:.0%}")
    logger.info(f"  Retrain every: {moderate_params['retrain_every_n_days']} days")
    logger.info(f"  Risk: MODERATE")
    logger.info("")

    # Show account info
    logger.info("=" * 80)
    logger.info("ACCOUNT INFORMATION")
    logger.info("=" * 80)
    logger.info("")

    account = await broker.get_account()
    logger.info(f"Account Value: ${float(account.equity):,.2f}")
    logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
    logger.info(f"Cash: ${float(account.cash):,.2f}")
    logger.info("")

    # Important notes
    logger.info("=" * 80)
    logger.info("IMPORTANT NOTES FOR ML TRADING")
    logger.info("=" * 80)
    logger.info("")
    logger.info("1. OVERFITTING IS THE BIGGEST RISK:")
    logger.info("   - Models can memorize training data instead of learning patterns")
    logger.info("   - Always validate on out-of-sample data")
    logger.info("   - Paper trade for AT LEAST 3 months before live")
    logger.info("")
    logger.info("2. DATA REQUIREMENTS:")
    logger.info("   - Need 500+ bars minimum for training")
    logger.info("   - More data = better (aim for 1000+ bars)")
    logger.info("   - Quality > Quantity (clean data is critical)")
    logger.info("")
    logger.info("3. FEATURE ENGINEERING:")
    logger.info("   - Uses 30+ technical indicators as features")
    logger.info("   - Price patterns (gaps, ranges, positions)")
    logger.info("   - Volume indicators (VWAP, volume ratio)")
    logger.info("   - Multi-timeframe returns")
    logger.info("")
    logger.info("4. MODEL ARCHITECTURE:")
    logger.info("   - LSTM: Best for time series, captures temporal dependencies")
    logger.info("   - Dropout: Prevents overfitting")
    logger.info("   - Early stopping: Stops when validation loss stops improving")
    logger.info("")
    logger.info("5. RETRAINING:")
    logger.info("   - Markets change - models must adapt")
    logger.info("   - Retrain periodically (weekly recommended)")
    logger.info("   - Monitor performance - if degrades, investigate")
    logger.info("")
    logger.info("6. BACKTESTING RECOMMENDATIONS:")
    logger.info("   - Use walk-forward analysis")
    logger.info("   - Test on multiple time periods")
    logger.info("   - Check performance in different market regimes")
    logger.info("   - Beware of look-ahead bias")
    logger.info("")
    logger.info("7. PERFORMANCE EXPECTATIONS:")
    logger.info("   - Experimental - no guaranteed Sharpe ratio")
    logger.info("   - Highly dependent on:")
    logger.info("     * Data quality")
    logger.info("     * Feature engineering")
    logger.info("     * Market regime")
    logger.info("     * Retraining frequency")
    logger.info("")
    logger.info("8. WHEN TO USE ML:")
    logger.info("   ✅ Have lots of clean historical data")
    logger.info("   ✅ Can afford extensive backtesting")
    logger.info("   ✅ Understand ML/DL concepts")
    logger.info("   ✅ Can monitor and maintain models")
    logger.info("")
    logger.info("   ❌ Limited data available")
    logger.info("   ❌ Need immediate production deployment")
    logger.info("   ❌ Can't validate thoroughly")
    logger.info("   ❌ Don't understand ML pitfalls")
    logger.info("")

    # Workflow demonstration
    logger.info("=" * 80)
    logger.info("ML TRADING WORKFLOW")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Step 1: Data Collection")
    logger.info("  - Collect price history (on_bar method)")
    logger.info("  - Build up 500+ bars per symbol")
    logger.info("")
    logger.info("Step 2: Feature Engineering")
    logger.info("  - Calculate technical indicators")
    logger.info("  - Create price patterns")
    logger.info("  - Generate volume features")
    logger.info("  - Create sequences for LSTM")
    logger.info("")
    logger.info("Step 3: Model Training")
    logger.info("  - Split train/validation (80/20)")
    logger.info("  - Train LSTM model")
    logger.info("  - Use early stopping")
    logger.info("  - Save best model")
    logger.info("")
    logger.info("Step 4: Prediction")
    logger.info("  - Engineer features from latest data")
    logger.info("  - Generate prediction")
    logger.info("  - Estimate confidence")
    logger.info("")
    logger.info("Step 5: Signal Generation")
    logger.info("  - Check confidence threshold")
    logger.info("  - Check directional threshold")
    logger.info("  - Adjust position size by confidence")
    logger.info("")
    logger.info("Step 6: Execution")
    logger.info("  - Risk management check")
    logger.info("  - Submit order")
    logger.info("  - Track position")
    logger.info("")
    logger.info("Step 7: Monitoring & Retraining")
    logger.info("  - Monitor model performance")
    logger.info("  - Retrain periodically (weekly)")
    logger.info("  - Track training/validation loss")
    logger.info("")

    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("1. Install TensorFlow:")
    logger.info("   pip install tensorflow scikit-learn")
    logger.info("")
    logger.info("2. Collect sufficient data:")
    logger.info("   - Run data collection for several days")
    logger.info("   - Need 500+ bars minimum")
    logger.info("")
    logger.info("3. Backtest extensively:")
    logger.info("   - Use walk-forward analysis")
    logger.info("   - Test on out-of-sample data")
    logger.info("   - Verify no overfitting")
    logger.info("")
    logger.info("4. Paper trade:")
    logger.info("   - Test in live market conditions")
    logger.info("   - Monitor for 3+ months")
    logger.info("   - Track vs. baseline strategies")
    logger.info("")
    logger.info("5. Monitor and iterate:")
    logger.info("   - Track model performance")
    logger.info("   - Adjust hyperparameters")
    logger.info("   - Try different features")
    logger.info("   - Experiment with model architectures")
    logger.info("")

    logger.info("=" * 80)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
