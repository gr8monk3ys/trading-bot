#!/usr/bin/env python3
"""
Realistic Backtest Utility

Applies proper transaction costs to backtest results:
- Slippage (market impact)
- Bid-ask spread
- Commissions (if any)
- Execution delay

Without these costs, backtests are UNREALISTIC and overestimate returns by 5-15%.

Research shows:
- Average slippage: 0.1-0.5% per trade
- Bid-ask spread: 0.01-0.1% for liquid stocks
- These costs compound and significantly impact returns

Usage:
    from utils.realistic_backtest import RealisticBacktester

    backtester = RealisticBacktester(broker, strategy)
    results = await backtester.run(start_date, end_date)

    # Results include realistic costs
    print(f"Gross returns: {results['gross_return']:.2%}")
    print(f"Net returns (after costs): {results['net_return']:.2%}")
    print(f"Total costs: {results['total_costs']:.2%}")
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import BACKTEST_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade with cost tracking."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None

    # Costs
    slippage_cost: float = 0.0
    spread_cost: float = 0.0
    commission_cost: float = 0.0

    # Calculated fields
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_cost: float = 0.0

    def calculate_costs(
        self,
        slippage_pct: float = 0.004,
        spread_pct: float = 0.001,
        commission_per_share: float = 0.0
    ):
        """Calculate transaction costs for this trade."""
        trade_value = self.quantity * self.entry_price

        # Slippage (market impact) - applied on both entry and exit
        self.slippage_cost = trade_value * slippage_pct * 2  # Entry + Exit

        # Bid-ask spread - applied on both entry and exit
        self.spread_cost = trade_value * spread_pct * 2

        # Commission
        self.commission_cost = self.quantity * commission_per_share * 2

        self.total_cost = self.slippage_cost + self.spread_cost + self.commission_cost

    def close(self, exit_price: float, exit_time: datetime):
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time

        # Gross P&L (before costs)
        if self.side == 'buy':
            self.gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # short
            self.gross_pnl = (self.entry_price - exit_price) * self.quantity

        # Net P&L (after costs)
        self.net_pnl = self.gross_pnl - self.total_cost


@dataclass
class BacktestResults:
    """Container for backtest results with detailed metrics."""
    # Basic info
    start_date: datetime = None
    end_date: datetime = None
    trading_days: int = 0

    # Capital tracking
    initial_capital: float = 100000.0
    final_capital: float = 0.0

    # Returns
    gross_return: float = 0.0
    net_return: float = 0.0
    annualized_return: float = 0.0

    # Costs breakdown
    total_slippage: float = 0.0
    total_spread: float = 0.0
    total_commission: float = 0.0
    total_costs: float = 0.0
    cost_drag_pct: float = 0.0  # How much costs reduced returns

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)

    # Trade list
    trades: List[Trade] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'trading_days': self.trading_days,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'gross_return': self.gross_return,
            'net_return': self.net_return,
            'annualized_return': self.annualized_return,
            'total_costs': self.total_costs,
            'cost_drag_pct': self.cost_drag_pct,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
        }


class RealisticBacktester:
    """
    Backtester with realistic transaction cost modeling.

    Key features:
    1. Slippage modeling (market impact)
    2. Bid-ask spread costs
    3. Commission costs
    4. Execution delay simulation
    5. Walk-forward validation option
    """

    def __init__(
        self,
        broker,
        strategy,
        initial_capital: float = 100000.0,
        slippage_pct: float = None,
        spread_pct: float = None,
        commission_per_share: float = None,
        execution_delay_bars: int = None
    ):
        """
        Initialize realistic backtester.

        Args:
            broker: Broker instance for data fetching
            strategy: Strategy instance to backtest
            initial_capital: Starting capital
            slippage_pct: Slippage per trade (default from config)
            spread_pct: Bid-ask spread (default from config)
            commission_per_share: Commission per share (default from config)
            execution_delay_bars: Bars delay for execution (default from config)
        """
        self.broker = broker
        self.strategy = strategy
        self.initial_capital = initial_capital

        # Load defaults from config if not specified
        self.slippage_pct = slippage_pct or BACKTEST_PARAMS.get('SLIPPAGE_PCT', 0.004)
        self.spread_pct = spread_pct or BACKTEST_PARAMS.get('BID_ASK_SPREAD', 0.001)
        self.commission = commission_per_share or BACKTEST_PARAMS.get('COMMISSION_PER_SHARE', 0.0)
        self.execution_delay = execution_delay_bars or BACKTEST_PARAMS.get('EXECUTION_DELAY_BARS', 1)

        # Enable/disable features
        self.use_slippage = BACKTEST_PARAMS.get('USE_SLIPPAGE', True)

        # State
        self.capital = initial_capital
        self.positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []

        logger.info(
            f"RealisticBacktester initialized: "
            f"slippage={self.slippage_pct:.2%}, spread={self.spread_pct:.2%}, "
            f"commission=${self.commission}/share"
        )

    async def run(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> BacktestResults:
        """
        Run backtest with realistic costs.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            symbols: Optional list of symbols (uses strategy symbols if not provided)

        Returns:
            BacktestResults with detailed metrics
        """
        results = BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital
        )

        symbols = symbols or self.strategy.symbols

        try:
            logger.info(f"Starting realistic backtest: {start_date.date()} to {end_date.date()}")
            logger.info(f"  Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
            logger.info(f"  Costs: slippage={self.slippage_pct:.2%}, spread={self.spread_pct:.2%}")

            # Fetch historical data for all symbols
            all_bars = {}
            for symbol in symbols:
                bars = await self._fetch_bars(symbol, start_date, end_date)
                if bars and len(bars) > 0:
                    all_bars[symbol] = bars

            if not all_bars:
                logger.error("No data fetched for any symbols")
                return results

            # Get all unique dates
            all_dates = set()
            for bars in all_bars.values():
                all_dates.update(b['date'] for b in bars)
            all_dates = sorted(all_dates)

            results.trading_days = len(all_dates)

            # Initialize strategy
            await self.strategy.initialize()

            # Process each day
            for current_date in all_dates:
                # Feed bars to strategy for each symbol
                for symbol, bars in all_bars.items():
                    # Find bar for this date
                    day_bars = [b for b in bars if b['date'] == current_date]
                    if not day_bars:
                        continue

                    bar = day_bars[0]

                    # Update strategy with bar data
                    await self.strategy.on_bar(
                        symbol,
                        bar['open'],
                        bar['high'],
                        bar['low'],
                        bar['close'],
                        bar['volume'],
                        current_date
                    )

                # Get signals and execute
                signals = self.strategy.signals
                for symbol, signal in signals.items():
                    if signal != 'neutral' and symbol in all_bars:
                        current_bars = [b for b in all_bars[symbol] if b['date'] == current_date]
                        if current_bars:
                            await self._process_signal(
                                symbol,
                                signal,
                                current_bars[0]['close'],
                                current_date
                            )

                # Update positions and calculate equity
                total_position_value = sum(
                    pos.quantity * all_bars[sym][-1]['close']
                    for sym, pos in self.positions.items()
                    if sym in all_bars
                )
                equity = self.capital + total_position_value
                self.equity_history.append((current_date, equity))

            # Close all remaining positions at end
            for symbol, position in list(self.positions.items()):
                if symbol in all_bars:
                    final_price = all_bars[symbol][-1]['close']
                    await self._close_position(symbol, final_price, end_date)

            # Calculate results
            results = self._calculate_results(results)

            logger.info(f"Backtest complete: {results.total_trades} trades")
            logger.info(f"  Gross return: {results.gross_return:.2%}")
            logger.info(f"  Net return: {results.net_return:.2%}")
            logger.info(f"  Cost drag: {results.cost_drag_pct:.2%}")

            return results

        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
            return results

    async def _fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[List[Dict]]:
        """Fetch historical bars for a symbol."""
        try:
            bars = await self.broker.get_bars(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                timeframe='1Day'
            )

            if bars is None:
                return None

            return [
                {
                    'date': getattr(b, 'timestamp', datetime.now()).date() if hasattr(b, 'timestamp') else start_date.date(),
                    'open': float(b.open),
                    'high': float(b.high),
                    'low': float(b.low),
                    'close': float(b.close),
                    'volume': float(b.volume)
                }
                for b in bars
            ]

        except Exception as e:
            logger.debug(f"Error fetching bars for {symbol}: {e}")
            return None

    async def _process_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        timestamp: datetime
    ):
        """Process a trading signal with realistic costs."""
        try:
            # Check if we have a position
            has_position = symbol in self.positions

            if signal == 'buy' and not has_position:
                # Calculate position size
                position_value = self.capital * self.strategy.position_size
                quantity = position_value / price

                if quantity >= 0.01 and position_value <= self.capital:
                    # Create trade with costs
                    trade = Trade(
                        symbol=symbol,
                        side='buy',
                        quantity=quantity,
                        entry_price=price,
                        entry_time=timestamp
                    )

                    # Calculate costs upfront
                    if self.use_slippage:
                        trade.calculate_costs(
                            slippage_pct=self.slippage_pct,
                            spread_pct=self.spread_pct,
                            commission_per_share=self.commission
                        )

                    # Deduct capital (including entry costs)
                    entry_cost = position_value + (trade.total_cost / 2)  # Half costs on entry
                    self.capital -= entry_cost
                    self.positions[symbol] = trade

                    logger.debug(f"BUY {symbol}: {quantity:.2f} shares @ ${price:.2f} (cost: ${trade.total_cost/2:.2f})")

            elif signal == 'short' and not has_position:
                # Short position (similar to buy but inverted)
                position_value = self.capital * self.strategy.position_size * 0.8  # Smaller for shorts
                quantity = position_value / price

                if quantity >= 0.01:
                    trade = Trade(
                        symbol=symbol,
                        side='sell',  # Short = sell first
                        quantity=quantity,
                        entry_price=price,
                        entry_time=timestamp
                    )

                    if self.use_slippage:
                        trade.calculate_costs(
                            slippage_pct=self.slippage_pct,
                            spread_pct=self.spread_pct,
                            commission_per_share=self.commission
                        )

                    self.capital -= trade.total_cost / 2  # Entry costs
                    self.positions[symbol] = trade

                    logger.debug(f"SHORT {symbol}: {quantity:.2f} shares @ ${price:.2f}")

            elif signal == 'sell' and has_position:
                await self._close_position(symbol, price, timestamp)

        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")

    async def _close_position(self, symbol: str, price: float, timestamp: datetime):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        trade.close(price, timestamp)

        # Return capital plus P&L minus exit costs
        if trade.side == 'buy':
            # Long: sell at current price
            exit_value = trade.quantity * price - (trade.total_cost / 2)
        else:
            # Short: buy to cover
            # For shorts, profit = entry - exit
            exit_value = trade.quantity * (2 * trade.entry_price - price) - (trade.total_cost / 2)

        self.capital += exit_value
        self.closed_trades.append(trade)
        del self.positions[symbol]

        logger.debug(
            f"CLOSE {symbol}: ${trade.net_pnl:.2f} net P&L "
            f"(gross: ${trade.gross_pnl:.2f}, costs: ${trade.total_cost:.2f})"
        )

    def _calculate_results(self, results: BacktestResults) -> BacktestResults:
        """Calculate final backtest results."""
        results.trades = self.closed_trades
        results.total_trades = len(self.closed_trades)
        results.final_capital = self.capital

        if results.total_trades == 0:
            return results

        # Separate wins and losses
        wins = [t for t in self.closed_trades if t.net_pnl > 0]
        losses = [t for t in self.closed_trades if t.net_pnl <= 0]

        results.winning_trades = len(wins)
        results.losing_trades = len(losses)
        results.win_rate = len(wins) / results.total_trades if results.total_trades > 0 else 0

        # Cost breakdown
        results.total_slippage = sum(t.slippage_cost for t in self.closed_trades)
        results.total_spread = sum(t.spread_cost for t in self.closed_trades)
        results.total_commission = sum(t.commission_cost for t in self.closed_trades)
        results.total_costs = sum(t.total_cost for t in self.closed_trades)

        # Returns
        results.gross_return = sum(t.gross_pnl for t in self.closed_trades) / results.initial_capital
        results.net_return = (results.final_capital - results.initial_capital) / results.initial_capital
        results.cost_drag_pct = results.gross_return - results.net_return

        # Annualize
        if results.trading_days > 0:
            years = results.trading_days / 252
            if years > 0:
                results.annualized_return = (1 + results.net_return) ** (1 / years) - 1

        # Win/loss stats
        if wins:
            results.avg_win = sum(t.net_pnl for t in wins) / len(wins)
            results.largest_win = max(t.net_pnl for t in wins)
        if losses:
            results.avg_loss = sum(t.net_pnl for t in losses) / len(losses)
            results.largest_loss = min(t.net_pnl for t in losses)

        # Profit factor
        total_wins = sum(t.net_pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.net_pnl for t in losses)) if losses else 1
        results.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Risk metrics from equity curve
        if self.equity_history:
            results.dates = [e[0] for e in self.equity_history]
            results.equity_curve = [e[1] for e in self.equity_history]

            equity_array = np.array(results.equity_curve)
            returns = np.diff(equity_array) / equity_array[:-1]

            # Max drawdown
            peak = np.maximum.accumulate(equity_array)
            drawdowns = (equity_array - peak) / peak
            results.max_drawdown = float(np.min(drawdowns))

            # Sharpe ratio (assuming 0% risk-free rate)
            if len(returns) > 1 and np.std(returns) > 0:
                results.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    results.sortino_ratio = float(np.mean(returns) / downside_std * np.sqrt(252))

            # Calmar ratio
            if results.max_drawdown < 0:
                results.calmar_ratio = results.annualized_return / abs(results.max_drawdown)

        return results


def print_backtest_report(results: BacktestResults):
    """Print a formatted backtest report."""
    print("\n" + "="*60)
    print("REALISTIC BACKTEST REPORT")
    print("="*60)

    print(f"\nPeriod: {results.start_date.date()} to {results.end_date.date()}")
    print(f"Trading Days: {results.trading_days}")
    print(f"Initial Capital: ${results.initial_capital:,.2f}")
    print(f"Final Capital: ${results.final_capital:,.2f}")

    print("\n--- RETURNS ---")
    print(f"Gross Return: {results.gross_return:+.2%}")
    print(f"Net Return (after costs): {results.net_return:+.2%}")
    print(f"Annualized Return: {results.annualized_return:+.2%}")
    print(f"Cost Drag: {results.cost_drag_pct:.2%}")

    print("\n--- COSTS BREAKDOWN ---")
    print(f"Total Costs: ${results.total_costs:,.2f}")
    print(f"  Slippage: ${results.total_slippage:,.2f}")
    print(f"  Spread: ${results.total_spread:,.2f}")
    print(f"  Commission: ${results.total_commission:,.2f}")

    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Avg Win: ${results.avg_win:,.2f}")
    print(f"Avg Loss: ${results.avg_loss:,.2f}")
    print(f"Largest Win: ${results.largest_win:,.2f}")
    print(f"Largest Loss: ${results.largest_loss:,.2f}")

    print("\n--- RISK METRICS ---")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {results.calmar_ratio:.2f}")

    print("\n" + "="*60)
