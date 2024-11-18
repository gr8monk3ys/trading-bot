# AI-Powered Trading Bot 🤖

An advanced AI-driven stock trading bot leveraging sentiment analysis, technical indicators, and sophisticated risk management to automate intelligent trading strategies.

## Features 🌟

### Core Capabilities
- **Sentiment Analysis**: Real-time analysis of market sentiment using FinBERT
- **Technical Analysis**: Integration of key technical indicators (SMA, RSI)
- **Risk Management**: Comprehensive position sizing and portfolio risk controls
- **Broker Integration**: Seamless integration with Alpaca Trading API
- **Paper Trading**: Safe testing environment before live deployment

### Risk Management
- Portfolio-wide Risk Limit: 2%
- Individual Position Risk Limit: 1%
- Maximum Position Correlation: 0.7
- Value at Risk (VaR) Confidence: 95%
- Dynamic stop-loss and take-profit mechanisms

### Trading Parameters
- Sentiment Threshold: 0.6
- Position Size: 10% of portfolio
- Maximum Position Size: 25%
- Stop Loss: 2%
- Take Profit: 5%

## Installation 🚀

1. **Create Virtual Environment**
   ```bash
   conda create -n trader python=3.10
   conda activate trader
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   - Create a `.env` file in the root directory
   - Add your Alpaca credentials:
     ```
     ALPACA_API_KEY=your_api_key
     ALPACA_API_SECRET=your_api_secret
     ```

4. **SSL Certificates (if needed)**
   If you encounter SSL errors with Alpaca API:
   1. Download certificates:
      - [Let's Encrypt R3](https://letsencrypt.org/certs/lets-encrypt-r3.pem)
      - [ISRG Root X1](https://letsencrypt.org/certs/isrg-root-x1-cross-signed.pem)
   2. Change extensions to `.cer`
   3. Install certificates using system defaults

## Usage 💻

1. **Start the Bot**
   ```bash
   python trading_bot.py
   ```

2. **Monitor Trading**
   - Check logs for trading activities
   - Monitor positions through Alpaca dashboard
   - Review performance metrics in logging output

## Project Structure 📁

```
trading-bot/
├── brokers/
│   └── alpaca_broker.py      # Alpaca broker integration
├── strategies/
│   ├── base_strategy.py      # Base strategy framework
│   ├── sentiment_stock_strategy.py  # Main trading strategy
│   └── risk_manager.py       # Risk management system
├── trading_bot.py            # Main bot implementation
├── config.py                 # Configuration settings
└── requirements.txt          # Project dependencies
```

## Trading Strategy 📈

The bot implements a sophisticated trading strategy combining:
- Sentiment analysis of market news
- Technical indicators for trend confirmation
- Risk management rules for position sizing
- Portfolio correlation analysis
- Dynamic stop-loss and take-profit levels

## Dependencies 📦

Key dependencies include:
- lumibot>=3.0.0: Trading framework
- alpaca-trade-api==3.0.0: Broker integration
- transformers>=4.30.0: Sentiment analysis
- torch>=2.0.0: Machine learning support
- pandas>=2.0.0: Data manipulation
- numpy>=1.24.3: Numerical computations
- ta-lib: Technical analysis

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📜

This project is licensed under the MIT License.

## Author 👨‍💻

Lorenzo Scaturchio

## Disclaimer ⚠️

This trading bot is for educational purposes only. Always understand the risks involved with algorithmic trading and never trade with money you cannot afford to lose.
