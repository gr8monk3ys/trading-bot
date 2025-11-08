# Installation Guide

## Prerequisites

- Python 3.8+ 
- pip package manager
- TA-Lib (see special installation notes below)

## Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## TA-Lib Installation

TA-Lib is a technical analysis library that requires special installation steps:

### macOS

Using Homebrew:
```bash
brew install ta-lib
pip install ta-lib
```

### Windows

Download prebuilt binaries from [here](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip), then:
```bash
pip install ta-lib
```

### Linux

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

## Environment Setup

1. Create a `.env` file in the root directory:
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
PAPER=True  # Set to False for live trading
```

2. Test your connection:
```bash
python tests/test_connection.py
```

## Running the Bot

Basic usage:
```bash
python main.py
```

With specific options:
```bash
python main.py --strategies momentum,mean_reversion --backtest --days 30
```
