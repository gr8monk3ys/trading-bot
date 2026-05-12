# Trading Strategy Guide

## Available Strategies

### Momentum Strategy

The `MomentumStrategy` uses technical indicators to identify trend strength and momentum for trading decisions:

- **MACD**: Identifies trend changes and momentum
- **RSI**: Indicates overbought or oversold conditions
- **ADX**: Measures trend strength

#### Configuration

```python
{
    'position_size': 0.1,       # Position size as percentage of portfolio
    'max_positions': 5,         # Maximum number of concurrent positions
    'max_portfolio_risk': 0.02, # Maximum portfolio risk
    'stop_loss': 0.03,          # Stop loss percentage
    'take_profit': 0.06,        # Take profit percentage
    'rsi_period': 14,           # RSI calculation period
    'rsi_overbought': 70,       # RSI overbought threshold
    'rsi_oversold': 30,         # RSI oversold threshold
    'macd_fast': 12,            # MACD fast period
    'macd_slow': 26,            # MACD slow period
    'macd_signal': 9,           # MACD signal period
    'adx_period': 14,           # ADX period
    'adx_threshold': 25,        # ADX threshold for strong trend
}
```

### Mean Reversion Strategy

The `MeanReversionStrategy` identifies overbought/oversold conditions and trades on the expectation that prices will revert to the mean.

- **Bollinger Bands**: Identifies deviation from mean
- **RSI**: Confirms overbought/oversold conditions

#### Configuration

```python
{
    'position_size': 0.1,       # Position size as percentage of portfolio
    'max_positions': 5,         # Maximum number of concurrent positions
    'max_portfolio_risk': 0.02, # Maximum portfolio risk
    'stop_loss': 0.03,          # Stop loss percentage
    'take_profit': 0.05,        # Take profit percentage
    'bb_period': 20,            # Bollinger Bands period
    'bb_std_dev': 2,            # Bollinger Bands standard deviation
    'rsi_period': 14,           # RSI period
    'rsi_overbought': 70,       # RSI overbought threshold
    'rsi_oversold': 30,         # RSI oversold threshold
    'mean_reversion_threshold': 1.5, # Z-score threshold for entry
}
```

### Sentiment Strategy

The `SentimentStrategy` uses news sentiment analysis to make trading decisions:

- Analyzes financial news sentiment using FinBERT
- Combines news sentiment with technical indicators
- Adjusts position sizing based on sentiment strength

#### Configuration

```python
{
    'position_size': 0.1,       # Position size as percentage of portfolio
    'max_positions': 5,         # Maximum number of concurrent positions
    'max_portfolio_risk': 0.02, # Maximum portfolio risk
    'stop_loss': 0.03,          # Stop loss percentage
    'take_profit': 0.06,        # Take profit percentage
    'sentiment_threshold': 0.6, # Minimum sentiment score for entry
    'sentiment_lookback': 3,    # Number of days to look back for sentiment
}
```

## Risk Management

The `RiskManager` component handles risk assessment and management:

- Individual position risk calculation
- Portfolio-wide risk evaluation
- Value at Risk (VaR) calculation
- Maximum drawdown control
- Position correlation analysis

## Performance Evaluation

Strategies are evaluated based on:

- Sharpe Ratio (risk-adjusted return)
- Total return
- Maximum drawdown
- Win rate
- Profit factor
- Sortino ratio
- Calmar ratio
