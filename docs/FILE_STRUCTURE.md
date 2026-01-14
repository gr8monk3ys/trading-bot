# Repository File Structure

**Last Updated:** 2025-11-08

## 📁 Directory Organization

```
trading-bot/
├── 📄 Main Entry Points
│   ├── main.py              # Primary CLI (live, backtest, optimize modes)
│   ├── live_trader.py       # Simplified live trading launcher
│   ├── config.py            # Configuration parameters
│   └── requirements.txt     # Python dependencies
│
├── 📚 Documentation
│   ├── README.md            # Project overview and quick start
│   ├── CLAUDE.md            # Claude Code integration guide
│   ├── TODO.md              # Development roadmap and task tracking
│   └── docs/                # Additional documentation
│       ├── SETUP.md         # Detailed setup instructions
│       ├── TESTING.md       # Testing guide
│       ├── STATUS.md        # Project status
│       ├── ADVANCED_FEATURES.md
│       ├── IMPLEMENTATION_SUMMARY.md
│       └── ...
│
├── 🤖 Core Modules
│   ├── strategies/          # Trading strategies
│   │   ├── base_strategy.py
│   │   ├── momentum_strategy.py
│   │   ├── mean_reversion_strategy.py
│   │   ├── bracket_momentum_strategy.py
│   │   ├── ensemble_strategy.py
│   │   ├── pairs_trading_strategy.py
│   │   ├── ml_prediction_strategy.py
│   │   ├── sentiment_stock_strategy.py (DISABLED)
│   │   ├── options_strategy.py (EXPERIMENTAL)
│   │   └── risk_manager.py
│   │
│   ├── brokers/             # Broker integrations
│   │   ├── alpaca_broker.py
│   │   ├── backtest_broker.py
│   │   ├── order_builder.py
│   │   └── __init__.py
│   │
│   ├── engine/              # Trading engine
│   │   ├── strategy_manager.py
│   │   ├── backtest_engine.py
│   │   ├── performance_metrics.py
│   │   ├── strategy_evaluator.py
│   │   └── ...
│   │
│   └── utils/               # Utility modules
│       ├── circuit_breaker.py
│       ├── multi_timeframe.py
│       ├── extended_hours.py
│       ├── kelly_criterion.py
│       ├── indicators.py
│       ├── portfolio_rebalancer.py
│       ├── performance_tracker.py
│       ├── notifier.py
│       ├── sentiment_analysis.py
│       └── stock_scanner.py
│
├── 🛠️ Scripts & Utilities
│   └── scripts/             # Runner scripts and utilities
│       ├── README.md        # Scripts documentation
│       ├── dashboard.py     # Real-time monitoring
│       ├── quickstart.py    # Interactive setup
│       ├── simple_backtest.py
│       ├── smart_backtest.py
│       ├── simple_trader.py
│       ├── run.py
│       ├── run_now.py
│       ├── mock_strategies.py
│       ├── mcp_server.py
│       └── mcp.json
│
├── 📊 Data & Results
│   ├── data/                # Data storage
│   │   ├── historical/      # Historical price data
│   │   ├── logs/            # Data logs
│   │   ├── models/          # ML models
│   │   └── results/         # Backtest results
│   │
│   ├── results/             # Backtest output
│   ├── logs/                # Application logs
│   └── .env                 # Environment variables (API keys)
│
├── 🧪 Testing
│   ├── tests/               # Test suite
│   │   ├── test_*.py        # Unit tests
│   │   ├── conftest.py      # Pytest configuration
│   │   └── ...
│   │
│   └── examples/            # Example scripts
│       ├── short_selling_strategy_example.py
│       ├── extended_hours_trading_example.py
│       ├── kelly_criterion_example.py
│       ├── multi_timeframe_strategy_example.py
│       └── ...
│
└── ⚙️ Configuration
    ├── .env                 # API keys and secrets
    ├── .env.example         # Environment template
    ├── pyproject.toml       # Project metadata
    ├── .gitignore           # Git ignore rules
    └── .windsurfrules       # Windsurf AI rules
```

## 🎯 Key Features by Directory

### strategies/
- **Production Ready:** Momentum, MeanReversion, BracketMomentum, Ensemble, PairsTrading
- **Experimental:** MLPrediction, Options
- **Disabled:** SentimentStock (fake news data)
- **All strategies support:**
  - ✅ Multi-timeframe filtering
  - ✅ Short selling
  - ✅ Bracket orders with auto TP/SL
  - ✅ Risk management via RiskManager
  - ✅ Correlation enforcement

### utils/
- **Safety:** circuit_breaker.py (daily loss limits)
- **Analysis:** multi_timeframe.py, indicators.py
- **Position Sizing:** kelly_criterion.py
- **Extended Hours:** extended_hours.py (pre/post market)
- **Rebalancing:** portfolio_rebalancer.py
- **Tracking:** performance_tracker.py, notifier.py

### engine/
- **StrategyManager:** Multi-strategy orchestration
- **BacktestEngine:** Historical simulation with slippage
- **PerformanceMetrics:** Sharpe, drawdown, win rate, etc.
- **StrategyEvaluator:** Automatic strategy selection

## 📝 File Organization Changes (2025-11-08)

**Reorganized for cleaner structure:**
- Moved runner scripts: `scripts/` directory
- Moved documentation: `docs/` directory
- Kept main entry points at root: `main.py`, `live_trader.py`
- Organized by function instead of dumping everything at root

## 🚀 Quick Navigation

**To start trading:**
```bash
python main.py live --strategy auto
```

**To run dashboard:**
```bash
python scripts/dashboard.py
```

**To backtest:**
```bash
python main.py backtest --strategy all --start-date 2024-01-01
```

**To run tests:**
```bash
pytest tests/
```

## 📚 Documentation Hierarchy

1. **README.md** - Start here (overview & quick start)
2. **CLAUDE.md** - Developer guide for Claude Code integration
3. **TODO.md** - Development roadmap and tasks
4. **docs/SETUP.md** - Detailed installation & configuration
5. **docs/TESTING.md** - Testing strategies
6. **docs/ADVANCED_FEATURES.md** - Advanced usage
