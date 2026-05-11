"""
AlpacaBroker portfolio history / analytics mixin.

Contains:
    - get_portfolio_history (raw Alpaca portfolio history API)
    - get_equity_curve (timestamp/equity tuples)
    - get_performance_summary (return + drawdown over a period)
    - _calculate_max_drawdown (helper)
    - get_intraday_equity (today's intraday history)
    - get_historical_performance (custom date range)
"""

import logging
from datetime import datetime
from typing import Optional, cast

from brokers.alpaca._retry import DEBUG_MODE, retry_with_backoff

logger = logging.getLogger(__name__)


class AlpacaPortfolioMixin:
    """Portfolio history queries and equity-curve analytics."""

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_portfolio_history(
        self,
        period: str = "1M",
        timeframe: str = "1D",
        extended_hours: bool = True,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> Optional[dict]:
        """
        Get portfolio history from Alpaca.

        This method retrieves historical portfolio data including equity values,
        profit/loss, and timestamps for performance tracking and analysis.

        Args:
            period: Time period - "1D", "1W", "1M", "3M", "6M", "1A", "all"
                   Ignored if date_start is provided.
            timeframe: Resolution - "1Min", "5Min", "15Min", "1H", "1D"
            extended_hours: Include extended hours data in the results
            date_start: Custom start date (overrides period if provided)
            date_end: Custom end date (defaults to now)

        Returns:
            Dict with:
            - timestamp: List of Unix timestamps
            - equity: List of equity values
            - profit_loss: List of daily P&L values
            - profit_loss_pct: List of daily P&L percentages
            - base_value: Starting portfolio value
            - timeframe: Timeframe used
            Returns None on error.

        Example:
            history = await broker.get_portfolio_history(period="1M", timeframe="1D")
            if history:
                for ts, eq in zip(history["timestamp"], history["equity"]):
                    print(f"{datetime.fromtimestamp(ts)}: ${eq:,.2f}")
        """
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            # Use date_start/date_end if provided, otherwise use period
            if date_start is not None:
                request = GetPortfolioHistoryRequest(
                    timeframe=timeframe,
                    extended_hours=extended_hours,
                    start=date_start,
                    end=date_end,
                )
            else:
                request = GetPortfolioHistoryRequest(
                    timeframe=timeframe,
                    extended_hours=extended_hours,
                    period=period,
                )

            # Execute API call with timeout protection (data-heavy operation)
            history = await self._async_call_with_timeout(
                self.trading_client.get_portfolio_history,
                request,
                timeout=self.DATA_API_TIMEOUT,
                operation_name="get_portfolio_history",
            )

            # Convert to serializable dict
            # Handle potential None values in the response
            return {
                "timestamp": list(history.timestamp) if history.timestamp else [],
                "equity": list(history.equity) if history.equity else [],
                "profit_loss": list(history.profit_loss) if history.profit_loss else [],
                "profit_loss_pct": (
                    list(history.profit_loss_pct) if history.profit_loss_pct else []
                ),
                "base_value": history.base_value,
                "timeframe": str(history.timeframe) if history.timeframe else timeframe,
            }

        except ImportError as e:
            logger.error(
                f"Portfolio History API import error: {e}. " "Ensure alpaca-py is up to date."
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching portfolio history: {e}", exc_info=DEBUG_MODE)
            return None

    async def get_equity_curve(self, days: int = 30) -> list:
        """
        Get equity curve for the last N days.

        Convenience method that returns a list of (timestamp, equity) tuples
        for easy plotting and analysis.

        Args:
            days: Number of days of history to retrieve (default: 30)

        Returns:
            List of (timestamp, equity) tuples sorted chronologically.
            Returns empty list on error.

        Example:
            curve = await broker.get_equity_curve(days=30)
            for timestamp, equity in curve:
                date = datetime.fromtimestamp(timestamp)
                print(f"{date.strftime('%Y-%m-%d')}: ${equity:,.2f}")
        """
        # Map days to Alpaca period strings
        period_map = {
            1: "1D",
            7: "1W",
            30: "1M",
            90: "3M",
            180: "6M",
            365: "1A",
        }

        # Find the smallest period that covers the requested days
        period = "all"
        for threshold, period_str in sorted(period_map.items()):
            if days <= threshold:
                period = period_str
                break

        history = await self.get_portfolio_history(period=period, timeframe="1D")
        if not history or not history.get("equity"):
            return []

        # Combine timestamps and equity into tuples
        timestamps = history.get("timestamp", [])
        equity_values = history.get("equity", [])

        # Ensure both lists have the same length
        min_len = min(len(timestamps), len(equity_values))
        return list(zip(timestamps[:min_len], equity_values[:min_len], strict=False))

    async def get_performance_summary(self, period: str = "1M") -> Optional[dict]:
        """
        Get performance summary for a period.

        Calculates key performance metrics from portfolio history including
        total return, max drawdown, and equity range.

        Args:
            period: Time period - "1D", "1W", "1M", "3M", "6M", "1A", "all"

        Returns:
            Dict with:
            - period: The requested period
            - start_equity: Equity at start of period
            - end_equity: Equity at end of period
            - total_return: Absolute return in dollars
            - total_return_pct: Return as percentage
            - max_equity: Highest equity value in period
            - min_equity: Lowest equity value in period
            - max_drawdown: Maximum drawdown as percentage
            - data_points: Number of data points in the period
            Returns None on error or if no data available.

        Example:
            summary = await broker.get_performance_summary(period="1M")
            if summary:
                print(f"1-Month Return: {summary['total_return_pct']:.2f}%")
                print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
        """
        history = await self.get_portfolio_history(period=period)
        if not history or not history.get("equity"):
            return None

        equity = history["equity"]
        pnl = history.get("profit_loss", [])

        # Filter out None values from equity list
        equity = [e for e in equity if e is not None]
        if not equity:
            return None

        start_equity = equity[0]
        end_equity = equity[-1]

        # Calculate total return
        if pnl:
            # Filter None values
            pnl = [p for p in pnl if p is not None]
            total_return = sum(pnl) if pnl else 0
        else:
            total_return = end_equity - start_equity

        # Calculate percentage return
        if start_equity > 0:
            total_return_pct = ((end_equity / start_equity) - 1) * 100
        else:
            total_return_pct = 0.0

        return {
            "period": period,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_equity": max(equity),
            "min_equity": min(equity),
            "max_drawdown": self._calculate_max_drawdown(equity),
            "data_points": len(equity),
        }

    def _calculate_max_drawdown(self, equity: list) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Maximum drawdown is the largest peak-to-trough decline in portfolio
        value, expressed as a percentage.

        Args:
            equity: List of equity values (chronological order)

        Returns:
            Maximum drawdown as a percentage (e.g., 5.5 for 5.5% drawdown).
            Returns 0.0 if equity list is empty or all values are None.
        """
        if not equity:
            return 0.0

        # Filter out None values
        equity = [e for e in equity if e is not None]
        if not equity:
            return 0.0

        peak = equity[0]
        max_dd = 0.0

        for value in equity:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)

        return max_dd * 100  # Return as percentage

    async def get_intraday_equity(self, timeframe: str = "1H") -> Optional[dict]:
        """
        Get intraday equity data for today.

        Useful for monitoring real-time portfolio performance throughout
        the trading day.

        Args:
            timeframe: Resolution - "1Min", "5Min", "15Min", "1H"

        Returns:
            Dict with timestamp, equity, profit_loss, profit_loss_pct
            for today's trading session. Returns None on error.

        Example:
            intraday = await broker.get_intraday_equity(timeframe="15Min")
            if intraday:
                print(f"Current P&L: ${intraday['profit_loss'][-1]:,.2f}")
        """
        history = await self.get_portfolio_history(
            period="1D",
            timeframe=timeframe,
            extended_hours=True,
        )
        return cast(Optional[dict], history)

    async def get_historical_performance(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = "1D",
    ) -> Optional[dict]:
        """
        Get historical portfolio performance for a custom date range.

        Args:
            start_date: Start of the date range
            end_date: End of the date range (defaults to now)
            timeframe: Resolution - "1Min", "5Min", "15Min", "1H", "1D"

        Returns:
            Dict with portfolio history data for the specified range.
            Returns None on error.

        Example:
            from datetime import datetime
            start = datetime(2024, 1, 1)
            end = datetime(2024, 6, 30)
            history = await broker.get_historical_performance(start, end)
            if history:
                print(f"6-month equity data: {len(history['equity'])} points")
        """
        history = await self.get_portfolio_history(
            timeframe=timeframe,
            extended_hours=True,
            date_start=start_date,
            date_end=end_date or datetime.now(),
        )
        return cast(Optional[dict], history)
