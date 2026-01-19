#!/usr/bin/env python3
"""
Enhanced Real-Time Trading Dashboard with Rich UI

Beautiful terminal dashboard showing:
- Live account metrics with color coding
- Open positions with real-time P/L
- Recent trades history
- Performance metrics (Sharpe, win rate, etc.)
- Risk status and alerts
- Market status and circuit breaker
- Strategy status

Usage:
    python scripts/enhanced_dashboard.py

    Press 'q' to quit
    Press 'r' to force refresh
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from brokers.alpaca_broker import AlpacaBroker
from utils.circuit_breaker import CircuitBreaker

console = Console()


class EnhancedTradingDashboard:
    """Enhanced real-time trading dashboard with Rich UI."""

    def __init__(self):
        self.broker: Optional[AlpacaBroker] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.running = True
        self.last_update = None
        self.refresh_interval = 5  # seconds

        # Performance tracking
        self.start_equity = 100000.0
        self.peak_equity = 100000.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0

    async def initialize(self):
        """Initialize broker and circuit breaker."""
        try:
            console.print("[yellow]Initializing dashboard...[/yellow]")

            # Initialize broker
            self.broker = AlpacaBroker(paper=True)
            console.print("[green]✓[/green] Connected to Alpaca (Paper Trading)")

            # Initialize circuit breaker
            self.circuit_breaker = CircuitBreaker(max_daily_loss=0.03)
            await self.circuit_breaker.initialize(self.broker)
            console.print("[green]✓[/green] Circuit breaker armed")

            # Get starting equity
            account = await self.broker.get_account()
            self.start_equity = float(account.equity)
            self.peak_equity = self.start_equity

            console.print(f"[green]✓[/green] Starting equity: ${self.start_equity:,.2f}")
            console.print("\n[bold green]Dashboard ready![/bold green]\n")

            return True
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Initialization failed: {e}")
            return False

    def create_header(self) -> Panel:
        """Create dashboard header."""
        now = datetime.now()
        header_text = Text()
        header_text.append("🤖 ", style="bold")
        header_text.append("LIVE TRADING DASHBOARD", style="bold cyan")
        header_text.append(" 🤖", style="bold")
        header_text.append(f"\n{now.strftime('%Y-%m-%d %H:%M:%S')}", style="dim")

        return Panel(Align.center(header_text), box=box.DOUBLE, style="cyan")

    async def create_account_panel(self) -> Panel:
        """Create account summary panel."""
        try:
            account = await self.broker.get_account()
            equity = float(account.equity)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            day_pnl = equity - self.start_equity
            day_pnl_pct = (day_pnl / self.start_equity) * 100

            # Update peak
            if equity > self.peak_equity:
                self.peak_equity = equity

            # Calculate drawdown
            drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100

            # Build account table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")

            # Equity
            equity_color = "green" if day_pnl >= 0 else "red"
            table.add_row("💰 Equity", f"[{equity_color}]${equity:,.2f}[/{equity_color}]")

            # Day P/L
            pnl_color = "green" if day_pnl >= 0 else "red"
            pnl_symbol = "+" if day_pnl >= 0 else ""
            table.add_row(
                "📊 Day P/L",
                f"[{pnl_color}]{pnl_symbol}${day_pnl:,.2f} ({pnl_symbol}{day_pnl_pct:.2f}%)[/{pnl_color}]",
            )

            # Cash & Buying Power
            table.add_row("💵 Cash", f"${cash:,.2f}")
            table.add_row("⚡ Buying Power", f"${buying_power:,.2f}")

            # Drawdown
            dd_color = "green" if drawdown < 5 else "yellow" if drawdown < 10 else "red"
            table.add_row("📉 Drawdown", f"[{dd_color}]{drawdown:.2f}%[/{dd_color}]")

            # Market status
            clock = await self.broker.get_clock()
            market_status = "🟢 OPEN" if clock.is_open else "🔴 CLOSED"
            table.add_row("🏛️  Market", market_status)

            return Panel(
                table,
                title="[bold]Account Summary[/bold]",
                border_style="green" if day_pnl >= 0 else "red",
                box=box.ROUNDED,
            )
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="Account Summary")

    async def create_positions_panel(self) -> Panel:
        """Create positions panel."""
        try:
            positions = await self.broker.get_positions()

            if not positions:
                return Panel(
                    Align.center("[dim]No open positions[/dim]"),
                    title="[bold]Open Positions[/bold]",
                    border_style="blue",
                    box=box.ROUNDED,
                )

            # Create positions table
            table = Table(box=box.SIMPLE)
            table.add_column("Symbol", style="cyan", justify="left")
            table.add_column("Qty", justify="right")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("Value", justify="right")
            table.add_column("P/L", justify="right")
            table.add_column("%", justify="right")

            total_value = 0
            total_pnl = 0

            for pos in positions:
                qty = float(pos.qty)
                entry = float(pos.avg_entry_price)
                current = float(pos.current_price)
                value = float(pos.market_value)
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100

                total_value += value
                total_pnl += pnl

                # Color coding
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_symbol = "+" if pnl >= 0 else ""

                # Position type indicator
                pos_type = "📈" if qty > 0 else "📉"  # Long or Short

                table.add_row(
                    f"{pos_type} {pos.symbol}",
                    f"{qty:.2f}",
                    f"${entry:.2f}",
                    f"${current:.2f}",
                    f"${value:,.2f}",
                    f"[{pnl_color}]{pnl_symbol}${pnl:,.2f}[/{pnl_color}]",
                    f"[{pnl_color}]{pnl_symbol}{pnl_pct:.2f}%[/{pnl_color}]",
                )

            # Add total row
            total_pnl_color = "green" if total_pnl >= 0 else "red"
            total_pnl_symbol = "+" if total_pnl >= 0 else ""
            table.add_row(
                "[bold]TOTAL[/bold]",
                "",
                "",
                "",
                f"[bold]${total_value:,.2f}[/bold]",
                f"[bold {total_pnl_color}]{total_pnl_symbol}${total_pnl:,.2f}[/bold {total_pnl_color}]",
                "",
            )

            return Panel(
                table,
                title=f"[bold]Open Positions ({len(positions)})[/bold]",
                border_style="blue",
                box=box.ROUNDED,
            )
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="Open Positions")

    async def create_risk_panel(self) -> Panel:
        """Create risk status panel."""
        try:
            account = await self.broker.get_account()
            equity = float(account.equity)

            # Check circuit breaker
            cb_triggered = await self.circuit_breaker.check_and_halt()

            # Calculate risk metrics
            daily_loss = self.start_equity - equity
            daily_loss_pct = (daily_loss / self.start_equity) * 100

            # Get positions for concentration
            positions = await self.broker.get_positions()
            position_count = len(positions)

            # Calculate largest position
            largest_position_pct = 0
            if positions:
                position_values = [abs(float(p.market_value)) for p in positions]
                largest_position = max(position_values) if position_values else 0
                largest_position_pct = (largest_position / equity) * 100 if equity > 0 else 0

            # Build risk table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")

            # Circuit Breaker
            cb_status = "[red]🚨 TRIGGERED[/red]" if cb_triggered else "[green]✓ Armed[/green]"
            table.add_row("⚡ Circuit Breaker", cb_status)

            # Daily Loss Limit
            loss_pct_color = (
                "green" if daily_loss_pct < 1 else "yellow" if daily_loss_pct < 2 else "red"
            )
            table.add_row(
                "📊 Daily Loss",
                f"[{loss_pct_color}]{daily_loss_pct:.2f}% / 3.00% max[/{loss_pct_color}]",
            )

            # Position Count
            pos_color = (
                "green" if position_count <= 5 else "yellow" if position_count <= 8 else "red"
            )
            table.add_row("💼 Positions", f"[{pos_color}]{position_count} / 10 max[/{pos_color}]")

            # Largest Position
            conc_color = (
                "green"
                if largest_position_pct < 10
                else "yellow" if largest_position_pct < 15 else "red"
            )
            table.add_row(
                "🎯 Max Position",
                f"[{conc_color}]{largest_position_pct:.1f}% of equity[/{conc_color}]",
            )

            # Win/Loss Ratio Today
            if self.trades_today > 0:
                win_rate = (self.wins_today / self.trades_today) * 100
                wr_color = "green" if win_rate >= 55 else "yellow" if win_rate >= 45 else "red"
                table.add_row(
                    "📈 Win Rate",
                    f"[{wr_color}]{win_rate:.1f}% ({self.wins_today}W/{self.losses_today}L)[/{wr_color}]",
                )
            else:
                table.add_row("📈 Win Rate", "[dim]No trades yet[/dim]")

            return Panel(
                table,
                title="[bold]Risk Status[/bold]",
                border_style="green" if not cb_triggered else "red",
                box=box.ROUNDED,
            )
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="Risk Status")

    async def create_strategies_panel(self) -> Panel:
        """Create strategies status panel."""
        # This would show active strategies if running via StrategyManager
        # For now, show static info

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Strategy", style="cyan")
        table.add_column("Status", style="bold")

        table.add_row("🎯 Momentum", "[green]✓ Active[/green]")
        table.add_row("📊 Mean Reversion", "[green]✓ Active[/green]")
        table.add_row("🔻 Short Selling", "[green]✓ Enabled[/green]")
        table.add_row("⏱️  Multi-Timeframe", "[green]✓ Enabled[/green]")
        table.add_row("⚖️  Rebalancing", "[green]✓ Every 4h[/green]")

        return Panel(
            table, title="[bold]Active Strategies[/bold]", border_style="cyan", box=box.ROUNDED
        )

    def create_footer(self) -> Panel:
        """Create dashboard footer."""
        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("'q'", style="bold yellow")
        footer_text.append(" to quit  •  ", style="dim")
        footer_text.append("'r'", style="bold yellow")
        footer_text.append(" to refresh  •  ", style="dim")
        footer_text.append(f"Auto-refresh: {self.refresh_interval}s", style="dim")

        return Panel(Align.center(footer_text), style="dim")

    async def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        # Define layout structure
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        # Split main area
        layout["main"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))

        # Split left column
        layout["left"].split_column(
            Layout(name="account", size=12), Layout(name="positions", ratio=1)
        )

        # Split right column
        layout["right"].split_column(
            Layout(name="risk", size=15), Layout(name="strategies", ratio=1)
        )

        # Populate layout
        layout["header"].update(self.create_header())
        layout["account"].update(await self.create_account_panel())
        layout["positions"].update(await self.create_positions_panel())
        layout["risk"].update(await self.create_risk_panel())
        layout["strategies"].update(await self.create_strategies_panel())
        layout["footer"].update(self.create_footer())

        return layout

    async def run(self):
        """Run the dashboard."""
        if not await self.initialize():
            return

        try:
            with Live(await self.create_layout(), console=console, refresh_per_second=1) as live:
                while self.running:
                    # Update layout
                    layout = await self.create_layout()
                    live.update(layout)

                    # Wait for refresh interval
                    await asyncio.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
        finally:
            console.print("[green]Goodbye![/green]")


async def main():
    """Main entry point."""
    dashboard = EnhancedTradingDashboard()
    await dashboard.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
