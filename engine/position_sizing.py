"""Position sizing: the one decision of how large an intent should be.

base fraction (or Kelly) → regime multiplier → risk-manager correlation
haircut → hard equity cap. A strategy asks once and gets a Sizing back.
Kelly is off unless the strategy enables it (ADR 0006); when on, it is fed
by the TradeRecorder so it sizes from real fills, never an empty history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sizing:
    value: float  # dollars
    qty: float  # shares (fractional)
    fraction: float  # of equity
    steps: tuple = ()  # what shaped the number, for the log

    @property
    def tradeable(self) -> bool:
        return self.qty >= 0.01


class PositionSizer:
    def __init__(
        self,
        *,
        base_fraction: float = 0.10,
        short_fraction: float = 0.08,
        max_position_fraction: float = 0.05,
        kelly: Any = None,
        risk_manager: Any = None,
        regime_multiplier: Optional[Callable[[str], float]] = None,
    ):
        self.base_fraction = float(base_fraction)
        self.short_fraction = float(short_fraction)
        self.max_position_fraction = float(max_position_fraction)
        self.kelly = kelly
        self.risk_manager = risk_manager
        self.regime_multiplier = regime_multiplier

    def size(
        self,
        symbol: str,
        price: float,
        view: Any,
        closes: Optional[List[float]] = None,
        *,
        is_short: bool = False,
        held_closes: Optional[Dict[str, List[float]]] = None,
    ) -> Sizing:
        if not price or price <= 0:
            return Sizing(0.0, 0.0, 0.0, ("invalid price",))
        equity = float(view.equity) or float(view.cash)
        steps = []

        if self.kelly is not None:
            value, fraction = self.kelly.calculate_position_size(
                current_capital=equity, current_price=price
            )
            if is_short:
                value *= 0.8
            steps.append(f"kelly {fraction:.1%}" + (" x0.8 short" if is_short else ""))
        else:
            fraction = self.short_fraction if is_short else self.base_fraction
            value = float(view.buying_power) * fraction
            steps.append(f"fixed {fraction:.1%} of buying power")

        if self.regime_multiplier is not None:
            mult = float(self.regime_multiplier(symbol))
            value *= mult
            steps.append(f"regime x{mult:.2f}")

        if self.risk_manager is not None and closes and len(closes) > 20:
            held = {}
            for held_symbol, pos in view.positions.items():
                history = (held_closes or {}).get(held_symbol)
                if history:
                    held[held_symbol] = {
                        "value": abs(float(pos.market_value)),
                        "price_history": history,
                        "risk": None,
                    }
            adjusted = float(self.risk_manager.adjust_position_size(symbol, value, closes, held))
            if adjusted != value:
                steps.append(f"risk haircut {value:,.0f} -> {adjusted:,.0f}")
            value = adjusted
            if value <= 0:
                return Sizing(0.0, 0.0, 0.0, tuple(steps + ["risk manager rejected"]))

        cap = equity * self.max_position_fraction
        if value > cap:
            steps.append(f"capped at {self.max_position_fraction:.1%} of equity")
            value = cap

        return Sizing(value, value / price, (value / equity) if equity else 0.0, tuple(steps))
