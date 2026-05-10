"""Position sizing — risk-based with notional cap."""
from __future__ import annotations

from dataclasses import dataclass

from trading.backtester.types import PnLMode


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    quantity: int
    notional: float
    risk_rupees: float


class PositionSizer:
    """Compute position size from risk parameters.

    For RUPEES mode: qty = max_risk / risk_per_share, capped by max_notional.
    For POINTS mode: qty = 1 (no sizing, just track points).
    """

    def __init__(
        self,
        capital: float,
        max_risk_pct: float = 1.0,
        max_position_pct: float = 15.0,
        pnl_mode: PnLMode = PnLMode.RUPEES,
    ):
        self.capital = capital
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct
        self.pnl_mode = pnl_mode

    def compute(self, entry_price: float, risk_points: float) -> PositionSize:
        """Compute quantity respecting both risk and notional limits."""
        if self.pnl_mode == PnLMode.POINTS:
            return PositionSize(quantity=1, notional=0, risk_rupees=0)

        if risk_points <= 0 or entry_price <= 0:
            return PositionSize(quantity=0, notional=0, risk_rupees=0)

        max_risk = self.capital * self.max_risk_pct / 100
        qty = int(max_risk / risk_points)

        if qty <= 0:
            return PositionSize(quantity=0, notional=0, risk_rupees=0)

        # Cap notional
        notional = qty * entry_price
        max_notional = self.capital * self.max_position_pct / 100
        if notional > max_notional:
            qty = int(max_notional / entry_price)
            notional = qty * entry_price

        if qty <= 0:
            return PositionSize(quantity=0, notional=0, risk_rupees=0)

        return PositionSize(
            quantity=qty,
            notional=round(notional, 2),
            risk_rupees=round(risk_points * qty, 2),
        )

    def update_capital(self, pnl: float):
        """Update running capital after a trade closes."""
        self.capital += pnl
