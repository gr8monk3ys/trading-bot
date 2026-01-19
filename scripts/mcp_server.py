#!/usr/bin/env python3
"""
MCP Server for Trading Bot

Provides Model Context Protocol interface for accessing trading bot functionality
through Claude or other LLM integrations.

Usage:
    python -m mcp_server

Or via Claude Desktop MCP configuration:
    {
      "mcpServers": {
        "trading-bot": {
          "command": "python",
          "args": ["-m", "mcp_server"],
          "env": {"PYTHONPATH": "/path/to/trading-bot"}
        }
      }
    }
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict

# Trading bot imports
from brokers.alpaca_broker import AlpacaBroker
from strategies.ensemble_strategy import EnsembleStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.pairs_trading_strategy import PairsTradingStrategy
from utils.indicators import (
    TechnicalIndicators,
    analyze_momentum,
    analyze_trend,
    analyze_volatility,
)


class TradingBotMCPServer:
    """MCP Server for Trading Bot integration."""

    def __init__(self):
        self.broker = None
        self.strategies = {}

    async def initialize(self):
        """Initialize broker and strategies."""
        self.broker = AlpacaBroker(paper=True)

        # Initialize available strategies
        self.strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "ensemble": EnsembleStrategy,
            "pairs_trading": PairsTradingStrategy,
        }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        method = request.get("method")
        params = request.get("params", {})

        if method == "backtest_strategy":
            return await self._backtest_strategy(params)
        elif method == "analyze_symbol":
            return await self._analyze_symbol(params)
        elif method == "get_positions":
            return await self._get_positions()
        elif method == "get_account":
            return await self._get_account()
        elif method == "calculate_indicators":
            return await self._calculate_indicators(params)
        elif method == "check_cointegration":
            return await self._check_cointegration(params)
        elif method == "detect_market_regime":
            return await self._detect_market_regime(params)
        elif method == "list_resources":
            return self._list_resources()
        elif method == "get_resource":
            return await self._get_resource(params)
        else:
            return {"error": f"Unknown method: {method}"}

    async def _backtest_strategy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest on strategy."""
        strategy_name = params.get("strategy")
        symbols = params.get("symbols", [])
        start_date = params.get("start_date")
        end_date = params.get("end_date", datetime.now().strftime("%Y-%m-%d"))

        if strategy_name not in self.strategies:
            return {"error": f"Unknown strategy: {strategy_name}"}

        # Initialize strategy
        StrategyClass = self.strategies[strategy_name]
        strategy = StrategyClass(broker=self.broker, symbols=symbols)

        # Run backtest (simplified - full implementation would use BacktestEngine)
        return {
            "strategy": strategy_name,
            "symbols": symbols,
            "period": f"{start_date} to {end_date}",
            "status": "Backtest functionality - integrate with BacktestEngine for full results",
        }

    async def _analyze_symbol(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze symbol with technical indicators."""
        symbol = params.get("symbol")
        timeframe = params.get("timeframe", "1Day")

        # Get historical data
        bars = await self.broker.get_bars(symbol, timeframe, limit=200)

        if not bars or len(bars) == 0:
            return {"error": f"No data available for {symbol}"}

        # Extract price arrays
        closes = [bar.close for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        volumes = [bar.volume for bar in bars]

        # Analyze
        trend = analyze_trend(closes, highs, lows)
        momentum = analyze_momentum(closes, highs, lows)
        volatility = analyze_volatility(closes, highs, lows)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": closes[-1],
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
            "timestamp": datetime.now().isoformat(),
        }

    async def _get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        positions = await self.broker.get_positions()

        return {
            "positions": [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                }
                for pos in positions
            ],
            "count": len(positions),
            "timestamp": datetime.now().isoformat(),
        }

    async def _get_account(self) -> Dict[str, Any]:
        """Get account information."""
        account = await self.broker.get_account()

        return {
            "account_id": account.id,
            "status": account.status,
            "currency": account.currency,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.equity),
            "pattern_day_trader": account.pattern_day_trader,
            "timestamp": datetime.now().isoformat(),
        }

    async def _calculate_indicators(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators."""
        symbol = params.get("symbol")
        indicators_list = params.get("indicators", ["RSI", "MACD", "ADX"])

        # Get historical data
        bars = await self.broker.get_bars(symbol, "1Day", limit=200)

        if not bars:
            return {"error": f"No data for {symbol}"}

        # Extract arrays
        closes = [bar.close for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        volumes = [bar.volume for bar in bars]

        # Calculate indicators
        ind = TechnicalIndicators(high=highs, low=lows, close=closes, volume=volumes)

        results = {}
        for indicator in indicators_list:
            if indicator.upper() == "RSI":
                rsi = ind.rsi()
                results["RSI"] = float(rsi[-1]) if len(rsi) > 0 else None
            elif indicator.upper() == "MACD":
                macd, signal, hist = ind.macd()
                results["MACD"] = {
                    "macd": float(macd[-1]) if len(macd) > 0 else None,
                    "signal": float(signal[-1]) if len(signal) > 0 else None,
                    "histogram": float(hist[-1]) if len(hist) > 0 else None,
                }
            elif indicator.upper() == "ADX":
                adx, plus_di, minus_di = ind.adx_di()
                results["ADX"] = {
                    "adx": float(adx[-1]) if len(adx) > 0 else None,
                    "plus_di": float(plus_di[-1]) if len(plus_di) > 0 else None,
                    "minus_di": float(minus_di[-1]) if len(minus_di) > 0 else None,
                }
            elif indicator.upper() == "VWAP":
                vwap = ind.vwap()
                results["VWAP"] = float(vwap[-1]) if len(vwap) > 0 else None

        return {"symbol": symbol, "indicators": results, "timestamp": datetime.now().isoformat()}

    async def _check_cointegration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if two symbols are cointegrated."""
        symbol1 = params.get("symbol1")
        symbol2 = params.get("symbol2")
        lookback_days = params.get("lookback_days", 60)

        # This would use the pairs trading strategy's cointegration test
        return {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "lookback_days": lookback_days,
            "status": "Use PairsTradingStrategy for full cointegration testing",
        }

    async def _detect_market_regime(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market regime."""
        symbol = params.get("symbol", "SPY")

        # Get data and analyze
        bars = await self.broker.get_bars(symbol, "1Day", limit=200)

        if not bars:
            return {"error": f"No data for {symbol}"}

        closes = [bar.close for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]

        # Analyze trend
        trend = analyze_trend(closes, highs, lows)
        volatility = analyze_volatility(closes, highs, lows)

        return {
            "symbol": symbol,
            "regime": {
                "trend_direction": trend.get("direction"),
                "trend_strength": trend.get("strength"),
                "volatility_state": volatility.get("state"),
                "adx": trend.get("adx"),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _list_resources(self) -> Dict[str, Any]:
        """List available resources."""
        return {
            "resources": [
                {
                    "uri": "trading://strategies",
                    "name": "Trading Strategies",
                    "description": "Available trading strategies",
                },
                {
                    "uri": "trading://positions",
                    "name": "Current Positions",
                    "description": "Real-time portfolio positions",
                },
                {
                    "uri": "trading://market-data",
                    "name": "Market Data",
                    "description": "Real-time market data",
                },
                {
                    "uri": "trading://performance",
                    "name": "Performance Metrics",
                    "description": "Strategy performance metrics",
                },
            ]
        }

    async def _get_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get specific resource."""
        uri = params.get("uri")

        if uri == "trading://strategies":
            return {"strategies": list(self.strategies.keys()), "count": len(self.strategies)}
        elif uri == "trading://positions":
            return await self._get_positions()
        else:
            return {"error": f"Unknown resource: {uri}"}


async def main():
    """Main MCP server loop."""
    server = TradingBotMCPServer()
    await server.initialize()

    print("Trading Bot MCP Server initialized", file=sys.stderr)
    print("Listening for requests...", file=sys.stderr)

    # Read requests from stdin, write responses to stdout
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            response = await server.handle_request(request)

            # Write response to stdout
            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"Server error: {e}"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
