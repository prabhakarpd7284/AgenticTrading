"""Basket position manager — trailing SL, breakeven, pyramiding, EOD close.

Equity: trail SL at 5 EMA (recomputed each bar), breakeven at +0.5R.
Options: SL at 30% loss on premium, trail at 50% of gained premium after +1R.
EOD: close everything at 15:15.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from logzero import logger

from plugins.strategy_basket.config import BasketConfig
from plugins.strategy_basket.executor import BasketExecutor, LegExecution
from plugins.strategy_basket.signals import BasketSignal
from trading.utils.indicators import _ema


@dataclass
class LegState:
    """Live state of a basket leg during management loop."""
    leg: LegExecution
    current_sl: float
    current_price: float = 0.0
    bars_since_entry: int = 0
    breakeven_done: bool = False
    trail_active: bool = False
    closed: bool = False
    close_reason: str = ""
    pnl: float = 0.0
    close_price: float = 0.0

    # Price history for 5 EMA trail
    close_prices: List[float] = field(default_factory=list)

    @property
    def risk_points(self) -> float:
        return self.leg.signal.risk_points

    @property
    def entry_price(self) -> float:
        return self.leg.avg_entry if self.leg.avg_entry > 0 else self.leg.signal.entry_price

    @property
    def current_r(self) -> float:
        if self.risk_points <= 0:
            return 0
        if self.leg.signal.side == "BUY":
            return (self.current_price - self.entry_price) / self.risk_points
        return (self.entry_price - self.current_price) / self.risk_points


class BasketPositionManager:
    """Manage active basket legs — trailing SL, breakeven, pyramid, EOD."""

    def __init__(self, cfg: BasketConfig = None, executor: BasketExecutor = None):
        self.cfg = cfg or BasketConfig()
        self.executor = executor
        self.legs: List[LegState] = []

    def add_leg(self, leg_exec: LegExecution):
        """Register a new leg for management."""
        state = LegState(
            leg=leg_exec,
            current_sl=leg_exec.signal.stoploss,
        )
        self.legs.append(state)

    def update(self, prices: Dict[str, float], bar_time: str = "") -> List[dict]:
        """Process one bar update across all active legs.

        Args:
            prices: {symbol: current_price}
            bar_time: current timestamp string (for EOD check)

        Returns:
            List of events [{symbol, event, details}]
        """
        events: List[dict] = []

        for state in self.legs:
            if state.closed:
                continue

            symbol = state.leg.signal.symbol
            price = prices.get(symbol, 0)
            if price <= 0:
                continue

            state.current_price = price
            state.bars_since_entry += 1
            state.close_prices.append(price)

            # ── EOD check ──
            if self._is_eod(bar_time):
                self._close_leg(state, price, "EOD close")
                events.append({"symbol": symbol, "event": "EOD_CLOSE", "price": price, "pnl": state.pnl})
                continue

            # ── SL hit check ──
            if state.leg.signal.side == "BUY" and price <= state.current_sl:
                reason = "Trail SL" if state.trail_active else ("BE stop" if state.breakeven_done else "SL hit")
                self._close_leg(state, state.current_sl, reason)
                events.append({"symbol": symbol, "event": reason, "price": state.current_sl, "pnl": state.pnl})
                continue
            if state.leg.signal.side == "SELL" and price >= state.current_sl:
                reason = "Trail SL" if state.trail_active else ("BE stop" if state.breakeven_done else "SL hit")
                self._close_leg(state, state.current_sl, reason)
                events.append({"symbol": symbol, "event": reason, "price": state.current_sl, "pnl": state.pnl})
                continue

            # ── Breakeven ──
            if not state.breakeven_done and state.current_r >= self.cfg.breakeven_at_r:
                if state.leg.signal.side == "BUY":
                    state.current_sl = max(state.current_sl, state.entry_price + 0.5)
                else:
                    state.current_sl = min(state.current_sl, state.entry_price - 0.5)
                state.breakeven_done = True
                events.append({"symbol": symbol, "event": "BREAKEVEN", "sl": state.current_sl})

            # ── Trailing SL ──
            if state.current_r >= self.cfg.trail_start_r:
                state.trail_active = True

                if state.leg.signal.leg_type == "equity":
                    # Trail at 5 EMA
                    new_sl = self._compute_ema_trail(state)
                else:
                    # Options: trail at 50% of premium gained
                    premium_gain = price - state.entry_price if state.leg.signal.side == "BUY" else state.entry_price - price
                    new_sl = state.entry_price + premium_gain * 0.5 if state.leg.signal.side == "BUY" else state.entry_price - premium_gain * 0.5

                if new_sl and state.leg.signal.side == "BUY":
                    state.current_sl = max(state.current_sl, new_sl)
                elif new_sl and state.leg.signal.side == "SELL":
                    state.current_sl = min(state.current_sl, new_sl)

            # ── T3 confirmation ──
            if state.bars_since_entry == self.cfg.t3_confirm_bars and self.executor:
                # Check if price held above entry for 3 bars
                if state.current_r > 0:
                    filled = self.executor.execute_t3(state.leg)
                    if filled:
                        events.append({"symbol": symbol, "event": "T3_FILLED", "qty": state.leg.filled_qty})

            # ── Pyramid check ──
            if (state.current_r >= self.cfg.pyramid_at_r
                    and state.leg.pyramids < self.cfg.max_pyramids
                    and self.executor):
                result = self.executor.execute_pyramid(state.leg, price)
                if result and result.status == "FILLED":
                    # Update SL for pyramid to current 5 EMA
                    events.append({
                        "symbol": symbol, "event": "PYRAMID",
                        "pyramid_num": state.leg.pyramids,
                        "add_qty": result.fill_qty,
                    })

        return events

    def close_all(self, prices: Dict[str, float], reason: str = "Manual close") -> List[dict]:
        """Close all open legs."""
        events = []
        for state in self.legs:
            if state.closed:
                continue
            price = prices.get(state.leg.signal.symbol, state.current_price)
            if price > 0:
                self._close_leg(state, price, reason)
                events.append({"symbol": state.leg.signal.symbol, "event": reason, "pnl": state.pnl})
        return events

    def total_pnl(self) -> float:
        return sum(s.pnl for s in self.legs)

    def summary(self) -> dict:
        """Current basket summary."""
        open_legs = [s for s in self.legs if not s.closed]
        closed_legs = [s for s in self.legs if s.closed]
        return {
            "open_legs": len(open_legs),
            "closed_legs": len(closed_legs),
            "total_pnl": round(self.total_pnl(), 2),
            "legs": [
                {
                    "symbol": s.leg.signal.symbol,
                    "side": s.leg.signal.side,
                    "type": s.leg.signal.leg_type,
                    "entry": s.entry_price,
                    "current": s.current_price,
                    "sl": round(s.current_sl, 2),
                    "pnl": round(s.pnl, 2),
                    "bars": s.bars_since_entry,
                    "pyramids": s.leg.pyramids,
                    "filled_qty": s.leg.filled_qty,
                    "closed": s.closed,
                    "close_reason": s.close_reason,
                }
                for s in self.legs
            ],
        }

    def _close_leg(self, state: LegState, price: float, reason: str):
        """Close a leg and compute P&L."""
        state.closed = True
        state.close_reason = reason
        state.close_price = price

        if state.leg.signal.side == "BUY":
            pnl_per = price - state.entry_price
        else:
            pnl_per = state.entry_price - price

        state.pnl = round(pnl_per * state.leg.filled_qty, 2)

        if self.executor:
            self.executor.close_leg(state.leg, price, reason)

    def _compute_ema_trail(self, state: LegState) -> Optional[float]:
        """Compute 5 EMA of recent close prices for trailing."""
        if len(state.close_prices) < self.cfg.ema_period:
            return None
        ema_vals = _ema(state.close_prices, self.cfg.ema_period)
        return round(ema_vals[-1], 2) if ema_vals else None

    def _is_eod(self, bar_time: str) -> bool:
        """Check if we've hit EOD close time."""
        if not bar_time:
            return False
        try:
            clean = bar_time.replace("+05:30", "").replace("T", " ")
            parts = clean.split(" ")
            if len(parts) >= 2:
                time_parts = parts[1].split(":")
                h, m = int(time_parts[0]), int(time_parts[1])
                eod_parts = self.cfg.eod_close_time.split(":")
                eod_h, eod_m = int(eod_parts[0]), int(eod_parts[1])
                return h > eod_h or (h == eod_h and m >= eod_m)
        except (ValueError, IndexError):
            pass
        return False
