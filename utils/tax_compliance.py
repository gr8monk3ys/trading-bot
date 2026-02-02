"""
Tax Compliance Module - Wash-Sale Tracking for Institutional Trading.

This module tracks potential wash sales and provides tax-loss harvesting
compliance features. Required for accounts over certain thresholds and
for institutional trading operations.

Key Features:
- Wash-sale detection (30-day window)
- Tax-lot tracking with cost basis methods
- Tax-loss harvesting opportunity identification
- Disallowed loss tracking for tax reporting
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CostBasisMethod(Enum):
    """Cost basis calculation methods."""

    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    HIFO = "hifo"  # Highest In, First Out
    SPECIFIC_ID = "specific_id"  # Specific lot identification
    AVERAGE = "average"  # Average cost


@dataclass
class TaxLot:
    """A single tax lot (purchase of shares)."""

    lot_id: str
    symbol: str
    quantity: float
    cost_basis: float  # Per share
    purchase_date: datetime
    remaining_quantity: float = 0
    disallowed_loss: float = 0  # Wash sale adjustment

    def __post_init__(self):
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

    @property
    def total_cost_basis(self) -> float:
        """Total cost basis for remaining shares."""
        return self.remaining_quantity * self.cost_basis

    @property
    def holding_period_days(self) -> int:
        """Days since purchase."""
        return (datetime.now() - self.purchase_date).days

    @property
    def is_long_term(self) -> bool:
        """True if held for more than 1 year."""
        return self.holding_period_days > 365


@dataclass
class SaleRecord:
    """Record of a sale for wash-sale tracking."""

    sale_id: str
    symbol: str
    quantity: float
    sale_price: float  # Per share
    sale_date: datetime
    cost_basis: float  # Per share (from tax lot)
    lot_id: str
    realized_gain_loss: float
    is_wash_sale: bool = False
    disallowed_loss: float = 0


@dataclass
class WashSaleEvent:
    """Record of a wash sale event."""

    symbol: str
    sale_date: datetime
    repurchase_date: datetime
    sale_record: SaleRecord
    replacement_lot: TaxLot
    disallowed_loss: float
    reason: str


class WashSaleTracker:
    """
    Tracks wash sales and manages tax-lot accounting.

    A wash sale occurs when you sell a security at a loss and buy
    substantially identical securities within 30 days before or after
    the sale. The loss is disallowed for tax purposes and added to
    the cost basis of the replacement shares.

    Usage:
        tracker = WashSaleTracker()

        # Record purchase
        tracker.add_tax_lot("AAPL", 100, 150.0, datetime(2024, 1, 1))

        # Check before selling
        wash_risk = tracker.check_wash_sale_risk("AAPL", datetime(2024, 1, 20))

        # Record sale
        tracker.record_sale("AAPL", 50, 140.0, datetime(2024, 1, 15))
    """

    WASH_SALE_WINDOW_DAYS = 30

    def __init__(self, cost_basis_method: CostBasisMethod = CostBasisMethod.FIFO):
        """
        Initialize the wash sale tracker.

        Args:
            cost_basis_method: Method for selecting lots when selling
        """
        self.cost_basis_method = cost_basis_method

        # Tax lots by symbol
        self._tax_lots: Dict[str, List[TaxLot]] = {}

        # Sales history for wash sale detection
        self._sales: List[SaleRecord] = []

        # Wash sale events
        self._wash_sales: List[WashSaleEvent] = []

        # Next lot ID counter
        self._lot_counter = 0

    def _generate_lot_id(self) -> str:
        """Generate unique lot ID."""
        self._lot_counter += 1
        return f"LOT-{self._lot_counter:06d}"

    def add_tax_lot(
        self,
        symbol: str,
        quantity: float,
        cost_basis: float,
        purchase_date: datetime,
    ) -> TaxLot:
        """
        Add a new tax lot (record a purchase).

        Also checks for wash sale conditions from prior sales.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            cost_basis: Cost per share
            purchase_date: Date of purchase

        Returns:
            The created TaxLot
        """
        lot = TaxLot(
            lot_id=self._generate_lot_id(),
            symbol=symbol,
            quantity=quantity,
            cost_basis=cost_basis,
            purchase_date=purchase_date,
        )

        # Check if this purchase triggers wash sale from prior loss
        wash_sale = self._check_purchase_triggers_wash_sale(lot)
        if wash_sale:
            # Adjust cost basis by adding disallowed loss
            lot.cost_basis += wash_sale.disallowed_loss / quantity
            lot.disallowed_loss = wash_sale.disallowed_loss
            self._wash_sales.append(wash_sale)

            logger.warning(
                f"WASH SALE TRIGGERED: {symbol} purchase on {purchase_date.date()} "
                f"within 30 days of loss sale. Disallowed loss: ${wash_sale.disallowed_loss:.2f}"
            )

        # Add to lots
        if symbol not in self._tax_lots:
            self._tax_lots[symbol] = []
        self._tax_lots[symbol].append(lot)

        return lot

    def _check_purchase_triggers_wash_sale(self, lot: TaxLot) -> Optional[WashSaleEvent]:
        """Check if a purchase triggers wash sale from prior loss."""
        symbol = lot.symbol
        purchase_date = lot.purchase_date

        # Look for sales at loss within 30 days before this purchase
        window_start = purchase_date - timedelta(days=self.WASH_SALE_WINDOW_DAYS)

        for sale in self._sales:
            if sale.symbol != symbol:
                continue

            if sale.realized_gain_loss >= 0:
                continue  # Only losses trigger wash sales

            if not (window_start <= sale.sale_date <= purchase_date):
                continue

            # Already marked as wash sale
            if sale.is_wash_sale:
                continue

            # This is a wash sale!
            disallowed_loss = abs(sale.realized_gain_loss)

            # Mark the sale as wash sale
            sale.is_wash_sale = True
            sale.disallowed_loss = disallowed_loss

            return WashSaleEvent(
                symbol=symbol,
                sale_date=sale.sale_date,
                repurchase_date=purchase_date,
                sale_record=sale,
                replacement_lot=lot,
                disallowed_loss=disallowed_loss,
                reason=f"Repurchase within {(purchase_date - sale.sale_date).days} days of loss sale",
            )

        return None

    def record_sale(
        self,
        symbol: str,
        quantity: float,
        sale_price: float,
        sale_date: datetime,
        lot_id: Optional[str] = None,
    ) -> List[SaleRecord]:
        """
        Record a sale and calculate gain/loss.

        Args:
            symbol: Stock symbol
            quantity: Number of shares sold
            sale_price: Sale price per share
            sale_date: Date of sale
            lot_id: Specific lot to sell from (for SPECIFIC_ID method)

        Returns:
            List of SaleRecord objects (one per lot used)
        """
        if symbol not in self._tax_lots:
            logger.warning(f"No tax lots found for {symbol}")
            return []

        lots = self._tax_lots[symbol]
        if not lots:
            logger.warning(f"Empty tax lots for {symbol}")
            return []

        # Sort lots based on cost basis method
        available_lots = [lot for lot in lots if lot.remaining_quantity > 0]

        if not available_lots:
            logger.warning(f"No available lots for {symbol}")
            return []

        # Select lots based on method
        if lot_id:
            # Specific identification
            selected_lots = [lot for lot in available_lots if lot.lot_id == lot_id]
        else:
            selected_lots = self._select_lots_by_method(available_lots)

        # Process sale
        sale_records = []
        remaining_to_sell = quantity

        for lot in selected_lots:
            if remaining_to_sell <= 0:
                break

            qty_from_lot = min(remaining_to_sell, lot.remaining_quantity)
            lot.remaining_quantity -= qty_from_lot
            remaining_to_sell -= qty_from_lot

            # Calculate gain/loss
            realized_gain_loss = (sale_price - lot.cost_basis) * qty_from_lot

            record = SaleRecord(
                sale_id=f"SALE-{len(self._sales) + 1:06d}",
                symbol=symbol,
                quantity=qty_from_lot,
                sale_price=sale_price,
                sale_date=sale_date,
                cost_basis=lot.cost_basis,
                lot_id=lot.lot_id,
                realized_gain_loss=realized_gain_loss,
            )

            # Check if this sale triggers wash sale (purchase within 30 days AFTER)
            wash_check = self._check_sale_triggers_wash_sale(record)
            if wash_check:
                record.is_wash_sale = True
                record.disallowed_loss = abs(realized_gain_loss)
                self._wash_sales.append(wash_check)

            self._sales.append(record)
            sale_records.append(record)

        if remaining_to_sell > 0:
            logger.warning(
                f"Could not sell full quantity for {symbol}: "
                f"{quantity - remaining_to_sell}/{quantity} sold"
            )

        return sale_records

    def _select_lots_by_method(self, lots: List[TaxLot]) -> List[TaxLot]:
        """Select and order lots based on cost basis method."""
        if self.cost_basis_method == CostBasisMethod.FIFO:
            return sorted(lots, key=lambda x: x.purchase_date)
        elif self.cost_basis_method == CostBasisMethod.LIFO:
            return sorted(lots, key=lambda x: x.purchase_date, reverse=True)
        elif self.cost_basis_method == CostBasisMethod.HIFO:
            return sorted(lots, key=lambda x: x.cost_basis, reverse=True)
        elif self.cost_basis_method == CostBasisMethod.AVERAGE:
            # For average, we still need to select lots, but basis is averaged
            return lots
        else:
            return lots

    def _check_sale_triggers_wash_sale(self, sale: SaleRecord) -> Optional[WashSaleEvent]:
        """Check if a loss sale is already in wash sale territory (future purchase exists)."""
        if sale.realized_gain_loss >= 0:
            return None  # Not a loss

        symbol = sale.symbol
        sale_date = sale.sale_date
        window_end = sale_date + timedelta(days=self.WASH_SALE_WINDOW_DAYS)

        # Check for purchases after this sale
        if symbol not in self._tax_lots:
            return None

        for lot in self._tax_lots[symbol]:
            if sale_date < lot.purchase_date <= window_end:
                # Purchase after sale within window = wash sale
                disallowed_loss = abs(sale.realized_gain_loss)

                return WashSaleEvent(
                    symbol=symbol,
                    sale_date=sale_date,
                    repurchase_date=lot.purchase_date,
                    sale_record=sale,
                    replacement_lot=lot,
                    disallowed_loss=disallowed_loss,
                    reason=f"Purchase {(lot.purchase_date - sale_date).days} days after loss sale",
                )

        return None

    def check_wash_sale_risk(
        self,
        symbol: str,
        intended_sale_date: datetime,
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Check wash sale risk before selling.

        Use this to warn users before they sell at a loss in a wash sale window.

        Args:
            symbol: Stock symbol to check
            intended_sale_date: When user intends to sell
            current_price: Current market price

        Returns:
            Dict with risk assessment
        """
        if symbol not in self._tax_lots:
            return {"risk": "NONE", "message": "No tax lots found"}

        lots = self._tax_lots[symbol]
        window_start = intended_sale_date - timedelta(days=self.WASH_SALE_WINDOW_DAYS)
        window_end = intended_sale_date + timedelta(days=self.WASH_SALE_WINDOW_DAYS)

        # Check for recent purchases (before intended sale)
        recent_purchases = [
            lot
            for lot in lots
            if window_start <= lot.purchase_date <= intended_sale_date
        ]

        # Check if sale would be at a loss
        losses_at_risk = []
        for lot in lots:
            if lot.remaining_quantity > 0 and current_price < lot.cost_basis:
                potential_loss = (lot.cost_basis - current_price) * lot.remaining_quantity
                losses_at_risk.append(
                    {
                        "lot_id": lot.lot_id,
                        "quantity": lot.remaining_quantity,
                        "cost_basis": lot.cost_basis,
                        "potential_loss": potential_loss,
                        "purchase_date": lot.purchase_date.isoformat(),
                    }
                )

        if not losses_at_risk:
            return {
                "risk": "NONE",
                "message": "No loss at current price",
                "current_price": current_price,
            }

        if recent_purchases:
            total_at_risk = sum(l["potential_loss"] for l in losses_at_risk)
            return {
                "risk": "HIGH",
                "message": f"Sale would trigger wash sale - recent purchases within 30 days",
                "recent_purchases": len(recent_purchases),
                "losses_at_risk": losses_at_risk,
                "total_at_risk": total_at_risk,
                "window_end": window_end.isoformat(),
                "recommendation": f"Wait until {window_end.date()} to sell and avoid wash sale",
            }

        return {
            "risk": "LOW",
            "message": "No recent purchases, but avoid repurchasing within 30 days",
            "losses_at_risk": losses_at_risk,
            "window_end": window_end.isoformat(),
        }

    def block_wash_sale_purchase(
        self,
        symbol: str,
        intended_purchase_date: datetime,
    ) -> Tuple[bool, str]:
        """
        Check if a purchase should be blocked due to wash sale.

        Use this in OrderGateway to prevent unintentional wash sales.

        Args:
            symbol: Symbol to purchase
            intended_purchase_date: When user intends to purchase

        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        window_start = intended_purchase_date - timedelta(days=self.WASH_SALE_WINDOW_DAYS)

        # Check for loss sales within window
        loss_sales = [
            sale
            for sale in self._sales
            if sale.symbol == symbol
            and sale.realized_gain_loss < 0
            and not sale.is_wash_sale
            and window_start <= sale.sale_date <= intended_purchase_date
        ]

        if loss_sales:
            total_loss = sum(abs(s.realized_gain_loss) for s in loss_sales)
            most_recent = max(s.sale_date for s in loss_sales)
            safe_date = most_recent + timedelta(days=self.WASH_SALE_WINDOW_DAYS + 1)

            return (
                True,
                f"WASH SALE WARNING: {symbol} sold at loss within 30 days. "
                f"Purchasing now would disallow ${total_loss:.2f} in losses. "
                f"Safe to purchase after {safe_date.date()}",
            )

        return (False, "No wash sale risk")

    def get_wash_sale_summary(self) -> Dict[str, Any]:
        """Get summary of all wash sale events."""
        return {
            "total_wash_sales": len(self._wash_sales),
            "total_disallowed_losses": sum(ws.disallowed_loss for ws in self._wash_sales),
            "events_by_symbol": self._group_wash_sales_by_symbol(),
        }

    def _group_wash_sales_by_symbol(self) -> Dict[str, List[Dict]]:
        """Group wash sales by symbol."""
        by_symbol = {}
        for ws in self._wash_sales:
            if ws.symbol not in by_symbol:
                by_symbol[ws.symbol] = []
            by_symbol[ws.symbol].append(
                {
                    "sale_date": ws.sale_date.isoformat(),
                    "repurchase_date": ws.repurchase_date.isoformat(),
                    "disallowed_loss": ws.disallowed_loss,
                    "reason": ws.reason,
                }
            )
        return by_symbol

    def get_tax_lot_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of tax lots."""
        if symbol:
            lots = self._tax_lots.get(symbol, [])
            return self._summarize_lots(symbol, lots)

        return {
            symbol: self._summarize_lots(symbol, lots)
            for symbol, lots in self._tax_lots.items()
        }

    def _summarize_lots(self, symbol: str, lots: List[TaxLot]) -> Dict[str, Any]:
        """Summarize lots for a symbol."""
        active_lots = [lot for lot in lots if lot.remaining_quantity > 0]

        return {
            "symbol": symbol,
            "total_lots": len(lots),
            "active_lots": len(active_lots),
            "total_shares": sum(lot.remaining_quantity for lot in active_lots),
            "total_cost_basis": sum(lot.total_cost_basis for lot in active_lots),
            "avg_cost_basis": (
                sum(lot.total_cost_basis for lot in active_lots)
                / sum(lot.remaining_quantity for lot in active_lots)
                if active_lots
                else 0
            ),
            "lots": [
                {
                    "lot_id": lot.lot_id,
                    "quantity": lot.remaining_quantity,
                    "cost_basis": lot.cost_basis,
                    "purchase_date": lot.purchase_date.isoformat(),
                    "is_long_term": lot.is_long_term,
                    "disallowed_loss": lot.disallowed_loss,
                }
                for lot in active_lots
            ],
        }

    def identify_tax_loss_opportunities(
        self,
        current_prices: Dict[str, float],
        min_loss_threshold: float = 100,
    ) -> List[Dict[str, Any]]:
        """
        Identify tax-loss harvesting opportunities.

        Args:
            current_prices: Dict of {symbol: current_price}
            min_loss_threshold: Minimum loss to consider

        Returns:
            List of opportunities sorted by potential loss
        """
        opportunities = []

        for symbol, lots in self._tax_lots.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            for lot in lots:
                if lot.remaining_quantity <= 0:
                    continue

                potential_loss = (lot.cost_basis - current_price) * lot.remaining_quantity

                if potential_loss > min_loss_threshold:
                    # Check wash sale risk
                    wash_risk = self.check_wash_sale_risk(
                        symbol, datetime.now(), current_price
                    )

                    opportunities.append(
                        {
                            "symbol": symbol,
                            "lot_id": lot.lot_id,
                            "quantity": lot.remaining_quantity,
                            "cost_basis": lot.cost_basis,
                            "current_price": current_price,
                            "potential_loss": potential_loss,
                            "is_long_term": lot.is_long_term,
                            "holding_period_days": lot.holding_period_days,
                            "wash_sale_risk": wash_risk["risk"],
                            "wash_sale_message": wash_risk.get("message"),
                        }
                    )

        # Sort by potential loss (largest first)
        return sorted(opportunities, key=lambda x: x["potential_loss"], reverse=True)

    def clear(self) -> None:
        """Clear all tracking data (for new year or reset)."""
        self._tax_lots.clear()
        self._sales.clear()
        self._wash_sales.clear()
        self._lot_counter = 0
