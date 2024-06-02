"""
Gap-risk modeling for BacktestBroker.

Provides:
- :class:`GapEvent` and :class:`GapStatistics` dataclasses.
- :class:`BacktestBrokerGapsMixin` — overnight + start-of-day gap simulation,
  stop-order registration/clearing, and gap statistics aggregation.

The mixin assumes the host class provides the attributes/methods initialized
in :class:`brokers.backtest.core.BacktestBrokerCore`:
- ``self.positions``, ``self.price_data``, ``self._current_date``
- ``self._gap_events``, ``self._stop_orders``, ``self._prev_day_close``
- ``self.get_price(...)`` (price retrieval)
- ``self.place_order(...)`` (provided by the execution mixin)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GapEvent:
    """Record of an overnight gap event affecting a position."""

    symbol: str
    date: datetime
    prev_close: float
    open_price: float
    gap_pct: float
    position_side: str  # 'long' or 'short'
    position_qty: int
    stop_price: Optional[float]
    stop_triggered: bool
    slippage_from_stop: float  # Difference between stop and actual fill


@dataclass
class GapStatistics:
    """Aggregate gap statistics for analysis."""

    total_gaps: int
    gaps_exceeding_2pct: int
    stops_gapped_through: int
    total_gap_slippage: float
    largest_gap_pct: float
    average_gap_pct: float


class BacktestBrokerGapsMixin:
    """Mixin providing gap-risk modeling for :class:`BacktestBroker`."""

    # =========================================================================
    # GAP RISK MODELING - INSTITUTIONAL GRADE
    # =========================================================================

    def set_stop_order(
        self, symbol: str, stop_price: float, quantity: int, side: str = "sell"
    ) -> None:
        """
        Register a stop order for gap risk tracking.

        Args:
            symbol: Stock symbol
            stop_price: Stop trigger price
            quantity: Quantity to sell/cover
            side: 'sell' for long positions, 'buy' for short covers
        """
        self._stop_orders[symbol] = {
            "stop_price": stop_price,
            "quantity": quantity,
            "side": side,
            "created_at": self._current_date,
        }
        logger.debug(f"Stop order registered: {symbol} @ ${stop_price:.2f}")

    def clear_stop_order(self, symbol: str) -> None:
        """Clear stop order for a symbol (e.g., when position closed)."""
        if symbol in self._stop_orders:
            del self._stop_orders[symbol]

    def update_prev_day_close(self, symbol: str, close_price: float) -> None:
        """Update previous day's closing price for gap calculation."""
        self._prev_day_close[symbol] = close_price

    def simulate_overnight_gap(
        self,
        symbol: str,
        open_price: float,
        date: datetime,
    ) -> Optional[GapEvent]:
        """
        Simulate overnight gap impact on positions.

        This is the core gap risk modeling function. It:
        1. Calculates gap percentage from previous close
        2. Checks if any stop orders were gapped through
        3. If stop was gapped through, fills at OPEN price (not stop price)
        4. Records the gap event for analysis

        Args:
            symbol: Stock symbol
            open_price: Opening price for the day
            date: Current trading date

        Returns:
            GapEvent if a significant gap occurred, None otherwise
        """
        # Need previous close to calculate gap
        if symbol not in self._prev_day_close:
            return None

        prev_close = self._prev_day_close[symbol]
        gap_pct = (open_price - prev_close) / prev_close

        # Only track gaps > 0.5% (material gaps)
        if abs(gap_pct) < 0.005:
            return None

        # Check if we have a position in this symbol
        position = self.positions.get(symbol)
        if position is None:
            # No position, no risk impact (but still record for statistics)
            return None

        position_qty = position["quantity"]
        position_side = "long" if position_qty > 0 else "short"
        stop_order = self._stop_orders.get(symbol)
        stop_price = stop_order["stop_price"] if stop_order else None

        # Check if stop was gapped through
        stop_triggered = False
        slippage_from_stop = 0.0

        if stop_order:
            stop_price = stop_order["stop_price"]

            if position_side == "long":
                # Long position: stop triggers if open < stop_price (gap down)
                if open_price < stop_price:
                    stop_triggered = True
                    slippage_from_stop = stop_price - open_price
                    logger.warning(
                        f"GAP RISK: {symbol} stop GAPPED THROUGH! "
                        f"Stop: ${stop_price:.2f}, Open: ${open_price:.2f}, "
                        f"Slippage: ${slippage_from_stop:.2f} ({slippage_from_stop/stop_price*100:.2f}%)"
                    )
            else:
                # Short position: stop triggers if open > stop_price (gap up)
                if open_price > stop_price:
                    stop_triggered = True
                    slippage_from_stop = open_price - stop_price
                    logger.warning(
                        f"GAP RISK: {symbol} short cover GAPPED THROUGH! "
                        f"Stop: ${stop_price:.2f}, Open: ${open_price:.2f}, "
                        f"Slippage: ${slippage_from_stop:.2f}"
                    )

        # Record gap event
        gap_event = GapEvent(
            symbol=symbol,
            date=date,
            prev_close=prev_close,
            open_price=open_price,
            gap_pct=gap_pct,
            position_side=position_side,
            position_qty=abs(position_qty),
            stop_price=stop_price,
            stop_triggered=stop_triggered,
            slippage_from_stop=slippage_from_stop,
        )
        self._gap_events.append(gap_event)

        # If stop was gapped through, execute at OPEN price (not stop price)
        if stop_triggered:
            self._execute_gap_stop(symbol, open_price, stop_order)

        return gap_event

    def _execute_gap_stop(self, symbol: str, fill_price: float, stop_order: Dict) -> None:
        """
        Execute a stop order that was gapped through at the open price.

        This is institutional-grade behavior: stops that gap through fill at
        the market (open) price, not the stop price. This prevents the
        unrealistic assumption that stops always fill at their trigger price.

        Args:
            symbol: Stock symbol
            fill_price: Actual fill price (the open)
            stop_order: Stop order details
        """
        quantity = stop_order["quantity"]
        side = stop_order["side"]

        logger.info(
            f"Executing gapped stop for {symbol}: {side} {quantity} @ ${fill_price:.2f} "
            f"(stop was ${stop_order['stop_price']:.2f})"
        )

        # Place the order at the fill price (the open)
        self.place_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=fill_price,
            order_type="market",  # Gapped stops fill at market
        )

        # Clear the stop order
        self.clear_stop_order(symbol)

    def process_day_start_gaps(self, date: datetime) -> List[GapEvent]:
        """
        Process overnight gaps for all positions at the start of trading day.

        Call this method at the beginning of each trading day in backtest.

        Args:
            date: Current trading date

        Returns:
            List of GapEvent objects for positions that experienced gaps
        """
        gap_events = []

        for symbol in list(self.positions.keys()):
            if symbol not in self.price_data:
                continue

            # Get today's open price
            try:
                open_price = self._get_open_price(symbol, date)
                if open_price is None:
                    continue

                gap_event = self.simulate_overnight_gap(symbol, open_price, date)
                if gap_event:
                    gap_events.append(gap_event)

            except Exception as e:
                logger.debug(f"Error processing gap for {symbol}: {e}")
                continue

        return gap_events

    def _get_open_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get the opening price for a symbol on a given date."""
        if symbol not in self.price_data:
            return None

        df = self.price_data[symbol]

        try:
            import pytz

            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            try:
                # Try exact match
                if date in df.index:
                    return float(df.loc[date, "open"])
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, "tzinfo") else date
                if date_naive in df_naive.index:
                    return float(df_naive.loc[date_naive, "open"])

            # Get closest date using indexer
            try:
                idx = df.index.get_indexer([date], method="pad")[0]
                if idx >= 0 and idx < len(df):
                    return float(df.iloc[idx]["open"])
            except TypeError:
                df_naive = df.copy()
                df_naive.index = df_naive.index.tz_localize(None)
                date_naive = date.replace(tzinfo=None) if hasattr(date, "tzinfo") else date
                idx = df_naive.index.get_indexer([date_naive], method="pad")[0]
                if idx >= 0 and idx < len(df):
                    return float(df_naive.iloc[idx]["open"])

        except Exception as e:
            logger.debug(f"Error getting open price for {symbol}: {e}")

        return None

    def update_prev_day_closes(self, date: datetime) -> None:
        """
        Update previous day closes for all held symbols at end of trading day.

        Call this method at the end of each trading day in backtest.

        Args:
            date: Current trading date (end of day)
        """
        for symbol in self.positions.keys():
            try:
                close_price = self.get_price(symbol, date)
                self._prev_day_close[symbol] = close_price
            except Exception as e:
                logger.debug(f"Error updating prev close for {symbol}: {e}")

    def get_gap_events(self) -> List[GapEvent]:
        """Get all recorded gap events."""
        return self._gap_events.copy()

    def get_gap_statistics(self) -> GapStatistics:
        """
        Calculate aggregate gap statistics for analysis.

        Returns:
            GapStatistics with summary metrics
        """
        if not self._gap_events:
            return GapStatistics(
                total_gaps=0,
                gaps_exceeding_2pct=0,
                stops_gapped_through=0,
                total_gap_slippage=0.0,
                largest_gap_pct=0.0,
                average_gap_pct=0.0,
            )

        gap_pcts = [abs(e.gap_pct) for e in self._gap_events]
        stops_gapped = [e for e in self._gap_events if e.stop_triggered]

        return GapStatistics(
            total_gaps=len(self._gap_events),
            gaps_exceeding_2pct=sum(1 for g in gap_pcts if g > 0.02),
            stops_gapped_through=len(stops_gapped),
            total_gap_slippage=sum(e.slippage_from_stop for e in stops_gapped),
            largest_gap_pct=max(gap_pcts) if gap_pcts else 0.0,
            average_gap_pct=sum(gap_pcts) / len(gap_pcts) if gap_pcts else 0.0,
        )

    def clear_gap_tracking(self) -> None:
        """Clear all gap tracking data (for new backtest run)."""
        self._gap_events.clear()
        self._stop_orders.clear()
        self._prev_day_close.clear()
