# Trading Bot Setup Guide

**Complete step-by-step setup instructions for the AI-Powered Trading Bot**

---

## Before You Begin

### What You Need

- [ ] Computer running macOS, Linux, or Windows
- [ ] Internet connection
- [ ] 30-60 minutes for complete setup
- [ ] Alpaca trading account (free paper trading)

### Important Notes

- This guide uses **paper trading only** (no real money)
- You will need to create a free Alpaca account
- All commands assume you're in the project root directory
- If you encounter issues, see [Troubleshooting](#troubleshooting) section

---

## Step 1: Install Prerequisites

### 1.1 Install Conda/Anaconda

**Why:** Conda manages Python environments and dependencies cleanly.

**Check if already installed:**
```bash
conda --version
```

**If not installed:**

- **macOS/Linux:** Download from [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Windows:** Download from [Anaconda](https://www.anaconda.com/download)

After installation, close and reopen your terminal.

### 1.2 Install Homebrew (macOS only)

**Why:** Needed for TA-Lib installation on macOS.

```bash
# Check if installed
brew --version

# If not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.3 Install TA-Lib (System Library)

**macOS:**
```bash
brew install ta-lib
```

**Linux (Ubuntu/Debian):**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

**Windows:**
- Download from [TA-Lib Windows](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- Follow installation instructions for your Python version

---

## Step 2: Get Alpaca API Credentials

### 2.1 Create Alpaca Account

1. Go to [https://alpaca.markets](https://alpaca.markets)
2. Click "Sign Up" (top right)
3. Fill out registration form
4. Verify your email address
5. Complete account setup

**Note:** You do NOT need to deposit any money. Paper trading is completely free.

### 2.2 Get API Keys

1. Log in to [https://app.alpaca.markets](https://app.alpaca.markets)
2. Navigate to: **Paper Trading** (in left sidebar)
3. Go to: **API Keys** section
4. Click: **Generate New Key**
5. **IMPORTANT:** Copy both:
   - API Key ID
   - Secret Key
6. Store them securely (you'll need them in Step 4)

**Security Warning:** Never share your API keys or commit them to git!

---

## Step 3: Set Up Python Environment

### 3.1 Navigate to Project Directory

```bash
cd /Users/gr8monk3ys/code/trading-bot
```

*Adjust path to wherever you cloned/downloaded the repository.*

### 3.2 Create Conda Environment

```bash
# Create new environment with Python 3.10
conda create -n trader python=3.10

# Activate the environment
conda activate trader
```

**Verify:**
```bash
python --version
# Should show: Python 3.10.x
```

### 3.3 Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**This will install:**
- PyTorch (ML framework)
- Transformers (for sentiment analysis)
- Lumibot (trading framework)
- Alpaca-py (broker API)
- TA-Lib (technical analysis)
- Pandas, NumPy (data processing)
- And more...

**Expected time:** 5-10 minutes depending on your internet speed.

**If TA-Lib fails:**
```bash
# Make sure system TA-Lib is installed (Step 1.3)
# Then try again:
pip install ta-lib
```

---

## Step 4: Configure Environment Variables

### 4.1 Create .env File

```bash
# Copy the example file
cp .env.example .env
```

### 4.2 Edit .env File

Open `.env` in your favorite text editor:

```bash
# Use nano (beginner-friendly):
nano .env

# Or use vim:
vim .env

# Or use VS Code:
code .env
```

### 4.3 Add Your Credentials

Replace the placeholder values with your actual Alpaca API credentials:

```bash
ALPACA_API_KEY=your_actual_api_key_here
ALPACA_SECRET_KEY=your_actual_secret_key_here
PAPER=True
```

**Example:**
```bash
ALPACA_API_KEY=PKABCDEF12345678
ALPACA_SECRET_KEY=abcdefghijklmnopqrstuvwxyz1234567890
PAPER=True
```

**Save the file:**
- In nano: Press `Ctrl+X`, then `Y`, then `Enter`
- In vim: Press `Esc`, type `:wq`, press `Enter`
- In VS Code: Press `Cmd+S` (Mac) or `Ctrl+S` (Windows/Linux)

### 4.4 Verify .env File

```bash
# Check file exists
ls -la .env

# Check it has content (without showing secrets)
cat .env | grep "ALPACA_API_KEY"
# Should show: ALPACA_API_KEY=PK...
```

---

## Step 5: Verify Installation

### 5.1 Test Python Imports

```bash
# Test core imports
python -c "from brokers.alpaca_broker import AlpacaBroker; print('✓ AlpacaBroker import OK')"

python -c "from strategies.base_strategy import BaseStrategy; print('✓ BaseStrategy import OK')"

python -c "from engine.strategy_manager import StrategyManager; print('✓ StrategyManager import OK')"
```

**Expected output:**
```
✓ AlpacaBroker import OK
✓ BaseStrategy import OK
✓ StrategyManager import OK
```

**If you see errors:** See [Troubleshooting](#troubleshooting) section below.

### 5.2 Test Alpaca Connection

```bash
# Run connection test
python tests/test_connection.py
```

**Expected output:**
```
Testing Alpaca connection...
✓ Connection successful!
Account ID: ...
Buying Power: $100000.00
Status: ACTIVE
```

**If connection fails:**
- Verify your API keys in `.env`
- Make sure `PAPER=True`
- Check your internet connection
- See [Troubleshooting](#troubleshooting) section

### 5.3 Test Environment Variables

```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('API Key loaded:', bool(os.getenv('ALPACA_API_KEY')))
print('Secret Key loaded:', bool(os.getenv('ALPACA_SECRET_KEY')))
print('Paper mode:', os.getenv('PAPER'))
"
```

**Expected output:**
```
API Key loaded: True
Secret Key loaded: True
Paper mode: True
```

---

## Step 6: Run Your First Test

### 6.1 List Available Strategies

```bash
python -c "
from engine.strategy_manager import StrategyManager
import asyncio

async def list_strategies():
    manager = StrategyManager(None)
    strategies = manager.get_available_strategy_names()
    print('Available strategies:')
    for s in strategies:
        print(f'  - {s}')

asyncio.run(list_strategies())
"
```

### 6.2 Run a Simple Backtest (Optional)

```bash
# Backtest momentum strategy
python main.py backtest --strategy MomentumStrategy --start-date 2024-01-01 --end-date 2024-02-01
```

This will:
- Simulate the strategy on historical data
- Show performance metrics
- NOT execute any real trades
- Take a few minutes to complete

---

## Step 7: Understanding Paper vs Live Trading

### Paper Trading (Safe)

**What it is:**
- Simulated trading with fake money
- Uses real market data
- No financial risk
- Free to use
- Perfect for testing

**Your setup:**
```bash
# In .env:
PAPER=True
```

**Running paper trading:**
```bash
python main.py live --strategy MomentumStrategy
```

### Live Trading (Real Money)

**What it is:**
- Actual trading with real money
- Real financial risk
- Requires funded account

**NOT RECOMMENDED until:**
- [ ] You've run paper trading for at least 30 days
- [ ] You understand all strategies being used
- [ ] You've tested thoroughly
- [ ] You're comfortable with potential losses
- [ ] You've reviewed all code

**To switch to live (NOT RECOMMENDED NOW):**
```bash
# In .env:
PAPER=False

# When running:
python main.py live --strategy MomentumStrategy --real
```

---

## Next Steps

### Recommended Learning Path

1. **Explore the Code** (1-2 hours)
   - Read `strategies/momentum_strategy.py`
   - Read `strategies/base_strategy.py`
   - Understand how strategies work

2. **Run Backtests** (1-2 hours)
   ```bash
   # Test different strategies
   python main.py backtest --strategy all --start-date 2024-01-01
   ```

3. **Paper Trade** (1-2 weeks)
   ```bash
   # Start with one strategy
   python main.py live --strategy MomentumStrategy --force

   # Monitor daily
   # Check Alpaca dashboard: https://app.alpaca.markets/paper/dashboard
   ```

4. **Learn Advanced Features** (1 week)
   - Read `docs/advanced_orders_guide.md`
   - Study `strategies/bracket_momentum_strategy.py`
   - Understand bracket orders and risk management

5. **Monitor and Optimize** (Ongoing)
   - Review logs daily
   - Analyze performance
   - Adjust parameters
   - Test improvements in backtest first

### Important Files to Read

- [README.md](README.md) - Overview and features
- [CLAUDE.md](CLAUDE.md) - Architecture and development guide
- [TODO.md](TODO.md) - Known issues and roadmap
- [docs/advanced_orders_guide.md](docs/advanced_orders_guide.md) - Advanced trading features

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
# Make sure you're in the right environment
conda activate trader

# Reinstall dependencies
pip install -r requirements.txt

# For specific module:
pip install module-name
```

### Circular Import Errors

**Problem:** `ImportError` when importing `AlpacaBroker`

**Solution:**
See [CLAUDE.md Troubleshooting section](CLAUDE.md#troubleshooting) for detailed fix.

Quick test:
```bash
python -c "from brokers.alpaca_broker import AlpacaBroker; print('OK')"
```

### Connection Failures

**Problem:** Can't connect to Alpaca API

**Checklist:**
- [ ] Is your internet working?
- [ ] Are your API keys correct in `.env`?
- [ ] Is `PAPER=True` for paper trading keys?
- [ ] Did you save the `.env` file?
- [ ] Are you using paper trading endpoint?

**Test:**
```bash
python tests/test_connection.py
```

### TA-Lib Installation Issues

**macOS:**
```bash
# Install system library first
brew install ta-lib

# Then Python package
pip install ta-lib
```

**Linux:**
```bash
# Follow Step 1.3 for system library installation
# Then:
pip install ta-lib
```

**Windows:**
- Use prebuilt wheel from [Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

### SSL Certificate Errors

**Problem:** SSL errors when connecting to Alpaca

**Solution:**
1. Download certificates:
   - [Let's Encrypt R3](https://letsencrypt.org/certs/lets-encrypt-r3.pem)
   - [ISRG Root X1](https://letsencrypt.org/certs/isrg-root-x1-cross-signed.pem)
2. Change extensions to `.cer`
3. Install using system certificate manager

### Market Closed Errors

**Problem:** "Market is closed" when trying to run

**Solution:**
```bash
# Use --force flag for testing
python main.py live --strategy MomentumStrategy --force
```

**Note:** Market hours are:
- Monday-Friday, 9:30 AM - 4:00 PM Eastern Time
- Pre-market: 4:00 AM - 9:30 AM
- After-hours: 4:00 PM - 8:00 PM

---

## Getting Help

### Check Documentation

1. [README.md](README.md) - General overview
2. [CLAUDE.md](CLAUDE.md) - Development guide and troubleshooting
3. [TODO.md](TODO.md) - Known issues
4. [docs/](docs/) - Additional guides

### Run Diagnostics

```bash
# Test all imports
python -c "from brokers.alpaca_broker import AlpacaBroker; print('✓ AlpacaBroker')"
python -c "from brokers.order_builder import OrderBuilder; print('✓ OrderBuilder')"
python -c "from strategies.base_strategy import BaseStrategy; print('✓ BaseStrategy')"

# Test connection
python tests/test_connection.py

# Test environment
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key exists:', bool(os.getenv('ALPACA_API_KEY')))"
```

### Common Solutions

1. **When in doubt, restart:**
   ```bash
   # Deactivate environment
   conda deactivate

   # Reactivate
   conda activate trader
   ```

2. **Clean Python cache:**
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

3. **Reinstall environment:**
   ```bash
   # Remove old environment
   conda deactivate
   conda env remove -n trader

   # Create new one
   conda create -n trader python=3.10
   conda activate trader
   pip install -r requirements.txt
   ```

---

## Security Best Practices

### Protect Your API Keys

- [ ] Never commit `.env` to git
- [ ] Never share your API keys
- [ ] Use paper trading keys for testing
- [ ] Rotate keys periodically
- [ ] Keep separate keys for paper vs live

### Verify .gitignore

```bash
# Check .env is ignored
cat .gitignore | grep ".env"

# Should show:
# .env
```

### Safe Development

- [ ] Always use paper trading for development
- [ ] Test changes in backtest first
- [ ] Review code before running
- [ ] Monitor bot closely during first runs
- [ ] Start with small position sizes

---

## Checklist: Setup Complete

Before running the bot, verify:

### Environment
- [ ] Conda environment created and activated
- [ ] Python 3.10.x installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] TA-Lib installed (both system and Python package)

### Configuration
- [ ] Alpaca account created (paper trading)
- [ ] API keys obtained
- [ ] `.env` file created from `.env.example`
- [ ] API keys added to `.env`
- [ ] `PAPER=True` set in `.env`

### Testing
- [ ] All imports work without errors
- [ ] `python tests/test_connection.py` passes
- [ ] Can list available strategies
- [ ] Backtest runs successfully (optional)

### Understanding
- [ ] Read README.md
- [ ] Understand paper vs live trading
- [ ] Know how to check Alpaca dashboard
- [ ] Know where to find logs
- [ ] Reviewed known issues in TODO.md

### Next Actions
- [ ] Run first backtest
- [ ] Start paper trading (monitor closely)
- [ ] Review results daily
- [ ] Learn advanced features
- [ ] Never use real money until thoroughly tested

---

## Quick Reference

### Activate Environment
```bash
conda activate trader
```

### Run Paper Trading
```bash
python main.py live --strategy MomentumStrategy
```

### Run Backtest
```bash
python main.py backtest --strategy MomentumStrategy --start-date 2024-01-01
```

### Test Connection
```bash
python tests/test_connection.py
```

### Check Account
```bash
# View at: https://app.alpaca.markets/paper/dashboard
```

### Get Help
```bash
python main.py --help
```

---

**Setup complete! You're ready to start trading (in paper mode).**

Remember:
- Start with paper trading
- Test thoroughly
- Monitor closely
- Never risk money you can't afford to lose
- Review TODO.md for known issues

**Happy trading!**
