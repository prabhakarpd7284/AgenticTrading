"""Trade — event-driven lifecycle, live-ready."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading.backtester.types import Bar, EntrySignal, PnLMode, TradeSide, TradeState


@dataclass
class Trade:
    """Single trade with full lifecycle managed by events.

    Lifecycle:
        on_entry(signal, qty, slippage) → OPEN
        on_bar(bar)                     → updates excursions, bars_held
        on_partial(price, qty, reason)  → PARTIAL (books some P&L)
        on_exit(price, reason)          → CLOSED (computes final P&L)
    """

    # ── Identity ──
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    state: TradeState = TradeState.PENDING

    # ── Entry ──
    entry_price: float = 0.0
    entry_time: str = ""
    quantity: int = 1
    notional: float = 0.0

    # ── Current levels (mutable — exit checks modify these) ──
    stoploss: float = 0.0
    target: float = 0.0

    # ── Risk ──
    risk_points: float = 0.0
    atr: float = 0.0

    # ── Exit ──
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""

    # ── Tracking ──
    bars_held: int = 0
    max_favorable: float = 0.0    # Best unrealized P&L per share
    max_adverse: float = 0.0      # Worst unrealized drawdown per share

    # ── P&L ──
    pnl: float = 0.0              # Total P&L (points or rupees)
    pnl_pct: float = 0.0
    rr_achieved: float = 0.0

    # ── Partial exit ──
    remaining_qty: int = 0
    partial_pnl: float = 0.0     # Realized P&L from partial exits
    partial_exits: int = 0

    # ── Flags (for exit management) ──
    breakeven_done: bool = False
    trail_active: bool = False

    # ── Strategy metadata ──
    metadata: dict = field(default_factory=dict)

    # ── Internals ──
    _pnl_mode: PnLMode = PnLMode.RUPEES

    # ──────────────────────────────────────
    # Properties
    # ──────────────────────────────────────

    @property
    def won(self) -> bool:
        return self.pnl > 0

    @property
    def trade_type(self) -> str:
        return self.metadata.get("trade_type", "")

    @property
    def current_r(self) -> float:
        """Current R-multiple based on max favorable excursion."""
        return self.max_favorable / self.risk_points if self.risk_points > 0 else 0.0

    # ──────────────────────────────────────
    # Lifecycle Events
    # ──────────────────────────────────────

    def on_entry(
        self,
        signal: EntrySignal,
        quantity: int,
        slippage_pct: float = 0.0,
        pnl_mode: PnLMode = PnLMode.RUPEES,
    ):
        """Transition from PENDING to OPEN."""
        self.symbol = signal.symbol
        self.side = signal.side
        self.state = TradeState.OPEN
        self._pnl_mode = pnl_mode

        # Apply slippage
        slip = slippage_pct / 100
        if signal.side == TradeSide.BUY:
            self.entry_price = round(signal.entry_price * (1 + slip), 2)
        else:
            self.entry_price = round(signal.entry_price * (1 - slip), 2)

        self.entry_time = signal.timestamp
        self.quantity = quantity
        self.remaining_qty = quantity
        self.notional = round(self.entry_price * quantity, 2)
        self.stoploss = signal.stoploss
        self.target = signal.target
        self.risk_points = signal.risk_points
        self.atr = signal.atr
        self.metadata = dict(signal.metadata)

    def on_bar(self, bar: Bar):
        """Process a new bar — update excursions and bar count."""
        if self.state not in (TradeState.OPEN, TradeState.PARTIAL):
            return

        self.bars_held += 1

        if self.side == TradeSide.BUY:
            self.max_favorable = max(self.max_favorable, bar.high - self.entry_price)
            self.max_adverse = max(self.max_adverse, self.entry_price - bar.low)
        else:
            self.max_favorable = max(self.max_favorable, self.entry_price - bar.low)
            self.max_adverse = max(self.max_adverse, bar.high - self.entry_price)

    def on_partial(self, price: float, fraction: float, reason: str):
        """Book a partial exit — reduce remaining_qty, accumulate partial_pnl."""
        if self.state == TradeState.CLOSED:
            return

        exit_qty = int(self.remaining_qty * fraction)
        if exit_qty <= 0:
            return

        pnl_per = self._pnl_per_share(price)

        if self._pnl_mode == PnLMode.RUPEES:
            self.partial_pnl += pnl_per * exit_qty
        else:
            self.partial_pnl += pnl_per

        self.remaining_qty -= exit_qty
        self.partial_exits += 1
        self.state = TradeState.PARTIAL

    def on_exit(self, price: float, time: str, reason: str):
        """Close the trade — compute final P&L including any partials."""
        self.exit_price = round(price, 2)
        self.exit_time = time
        self.exit_reason = reason
        self.state = TradeState.CLOSED

        pnl_per = self._pnl_per_share(price)

        if self._pnl_mode == PnLMode.RUPEES:
            self.pnl = round(pnl_per * self.remaining_qty + self.partial_pnl, 2)
        else:
            self.pnl = round(pnl_per + self.partial_pnl, 2)

        if self.entry_price > 0:
            self.pnl_pct = round(pnl_per / self.entry_price * 100, 2)
        if self.risk_points > 0 and self.quantity > 0:
            if self._pnl_mode == PnLMode.RUPEES:
                self.rr_achieved = round(self.pnl / (self.risk_points * self.quantity), 2)
            else:
                self.rr_achieved = round(pnl_per / self.risk_points, 2)

    # ──────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────

    def _pnl_per_share(self, price: float) -> float:
        if self.side == TradeSide.BUY:
            return price - self.entry_price
        return self.entry_price - price

    def r_at_price(self, price: float) -> float:
        """R-multiple at a given price."""
        if self.risk_points <= 0:
            return 0.0
        return self._pnl_per_share(price) / self.risk_points

    def to_dict(self) -> dict:
        """Serialize for API / JSON output."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry": self.entry_price,
            "entry_time": self.entry_time,
            "exit": self.exit_price,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason,
            "sl": self.stoploss,
            "target": self.target,
            "qty": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "rr": self.rr_achieved,
            "bars_held": self.bars_held,
            "won": self.won,
            "trade_type": self.trade_type,
            **{k: v for k, v in self.metadata.items() if k != "trade_type"},
        }
