"""Composable exit strategies — the core of trade management.

Each ExitCheck is a single responsibility:
  - BreakevenExit: moves SL to entry when +trigger_r reached
  - TrailingSLExit: trails SL using ATR after trigger_r
  - StoplossExit: checks if current SL is hit
  - TargetExit: checks if target is hit
  - EODExit: hard close at end-of-day cutoff
  - MaxHoldExit: close after N bars
  - PartialExit: book fraction at trigger_r

ExitManager composes them with smart SL/target priority resolution.
Same logic works for backtest and live trading.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from trading.backtester.types import Bar, ExitSignal, TradeSide


class ExitCheck(ABC):
    """Base class for a single exit condition."""

    @abstractmethod
    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        """Return ExitSignal if this exit triggers, None otherwise.

        May also MODIFY trade.stoploss (breakeven/trailing) without
        returning an exit signal.
        """
        ...


# ──────────────────────────────────────────────
# SL Modifiers (update SL, don't trigger exit)
# ──────────────────────────────────────────────

class BreakevenExit(ExitCheck):
    """Move SL to entry + buffer when trade reaches trigger_r."""

    def __init__(self, trigger_r: float = 0.5, buffer: float = 0.5):
        self.trigger_r = trigger_r
        self.buffer = buffer

    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        if trade.breakeven_done or trade.risk_points <= 0:
            return None

        # Check if price reached trigger_r
        if trade.side == TradeSide.BUY:
            current_r = (bar.high - trade.entry_price) / trade.risk_points
        else:
            current_r = (trade.entry_price - bar.low) / trade.risk_points

        if current_r >= self.trigger_r:
            if trade.side == TradeSide.BUY:
                trade.stoploss = max(trade.stoploss, trade.entry_price + self.buffer)
            else:
                trade.stoploss = min(trade.stoploss, trade.entry_price - self.buffer)
            trade.breakeven_done = True

        return None  # Never triggers exit directly


class TrailingSLExit(ExitCheck):
    """Trail SL using ATR factor after trade reaches trigger_r.

    Tightens the trail factor at tight_r for locking profits.
    """

    def __init__(
        self,
        trigger_r: float = 1.0,
        atr_factor: float = 0.3,
        tight_r: float = 1.5,
        tight_factor: float = 0.2,
    ):
        self.trigger_r = trigger_r
        self.atr_factor = atr_factor
        self.tight_r = tight_r
        self.tight_factor = tight_factor

    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        if trade.risk_points <= 0:
            return None

        if trade.side == TradeSide.BUY:
            current_r = (bar.high - trade.entry_price) / trade.risk_points
        else:
            current_r = (trade.entry_price - bar.low) / trade.risk_points

        if current_r < self.trigger_r:
            return None

        trade.trail_active = True
        atr_val = trade.atr if trade.atr > 0 else trade.risk_points
        factor = self.tight_factor if current_r >= self.tight_r else self.atr_factor

        if trade.side == TradeSide.BUY:
            trail_sl = bar.high - (atr_val * factor)
            trade.stoploss = max(trade.stoploss, trail_sl)
        else:
            trail_sl = bar.low + (atr_val * factor)
            trade.stoploss = min(trade.stoploss, trail_sl)

        return None


class PartialExit(ExitCheck):
    """Book a fraction of the position at trigger_r."""

    def __init__(self, trigger_r: float = 1.0, fraction: float = 0.5):
        self.trigger_r = trigger_r
        self.fraction = fraction

    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        if trade.partial_exits > 0 or trade.risk_points <= 0:
            return None

        if trade.side == TradeSide.BUY:
            current_r = (bar.high - trade.entry_price) / trade.risk_points
            price = trade.entry_price + (trade.risk_points * self.trigger_r)
        else:
            current_r = (trade.entry_price - bar.low) / trade.risk_points
            price = trade.entry_price - (trade.risk_points * self.trigger_r)

        if current_r >= self.trigger_r:
            return ExitSignal(price=price, reason="Partial", partial=True, partial_fraction=self.fraction)

        return None


# ──────────────────────────────────────────────
# Exit Triggers (check if SL/target/time hit)
# ──────────────────────────────────────────────

class StoplossExit(ExitCheck):
    """Check if current stoploss (possibly moved by breakeven/trail) is hit."""

    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        if trade.side == TradeSide.BUY and bar.low <= trade.stoploss:
            reason = "Trail SL" if trade.trail_active else ("BE stop" if trade.breakeven_done else "SL hit")
            return ExitSignal(price=trade.stoploss, reason=reason)
        if trade.side == TradeSide.SHORT and bar.high >= trade.stoploss:
            reason = "Trail SL" if trade.trail_active else ("BE stop" if trade.breakeven_done else "SL hit")
            return ExitSignal(price=trade.stoploss, reason=reason)
        return None


class TargetExit(ExitCheck):
    """Check if target price is hit."""

    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        if trade.side == TradeSide.BUY and bar.high >= trade.target:
            return ExitSignal(price=trade.target, reason="Target hit")
        if trade.side == TradeSide.SHORT and bar.low <= trade.target:
            return ExitSignal(price=trade.target, reason="Target hit")
        return None


class EODExit(ExitCheck):
    """Hard close at end-of-day cutoff time."""

    def __init__(self, hour: int = 15, minute: int = 20):
        self.hour = hour
        self.minute = minute

    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        try:
            ts = bar.timestamp.replace("+05:30", "").replace("T", " ")
            parts = ts.split(" ")
            if len(parts) >= 2:
                time_parts = parts[1].split(":")
                h, m = int(time_parts[0]), int(time_parts[1])
                if h > self.hour or (h == self.hour and m >= self.minute):
                    return ExitSignal(price=bar.close, reason="EOD close")
        except (ValueError, IndexError):
            pass
        return None


class MaxHoldExit(ExitCheck):
    """Close after N bars held."""

    def __init__(self, max_bars: int = 10):
        self.max_bars = max_bars

    def check(self, trade, bar: Bar) -> Optional[ExitSignal]:
        if trade.bars_held >= self.max_bars:
            return ExitSignal(price=bar.close, reason="Max hold")
        return None


# ──────────────────────────────────────────────
# ExitManager — composes checks with smart priority
# ──────────────────────────────────────────────

class ExitManager:
    """Composes multiple ExitChecks with smart SL/target priority resolution.

    Processing order:
      1. SL modifiers (breakeven, trailing) — may update trade.stoploss
      2. Partial exit — may book fraction
      3. Priority exits (SL vs target) — smart open-direction resolution
      4. Time exits (EOD, max hold) — fallback
    """

    def __init__(self, checks: List[ExitCheck]):
        # Separate by type for ordered processing
        self._modifiers: List[ExitCheck] = []
        self._partials: List[ExitCheck] = []
        self._sl_target: List[ExitCheck] = []
        self._time_exits: List[ExitCheck] = []

        for c in checks:
            if isinstance(c, (BreakevenExit, TrailingSLExit)):
                self._modifiers.append(c)
            elif isinstance(c, PartialExit):
                self._partials.append(c)
            elif isinstance(c, (StoplossExit, TargetExit)):
                self._sl_target.append(c)
            else:
                self._time_exits.append(c)

    def process(self, trade, bar: Bar) -> Optional[ExitSignal]:
        """Run all exit checks. Returns ExitSignal if trade should exit.

        Smart priority: when both SL and target hit in the same bar,
        the one closer to bar.open is assumed to have hit first.
        """
        # 1. Run modifiers (update SL, never return exit)
        for c in self._modifiers:
            c.check(trade, bar)

        # 2. Check partials
        for c in self._partials:
            sig = c.check(trade, bar)
            if sig:
                return sig

        # 3. Check SL and target with smart priority
        sl_signal: Optional[ExitSignal] = None
        tgt_signal: Optional[ExitSignal] = None

        for c in self._sl_target:
            sig = c.check(trade, bar)
            if sig:
                if isinstance(c, StoplossExit):
                    sl_signal = sig
                elif isinstance(c, TargetExit):
                    tgt_signal = sig

        if sl_signal and tgt_signal:
            # Both hit same bar — use distance from open to determine priority
            sl_dist = abs(bar.open - sl_signal.price)
            tgt_dist = abs(bar.open - tgt_signal.price)
            return sl_signal if sl_dist <= tgt_dist else tgt_signal
        if sl_signal:
            return sl_signal
        if tgt_signal:
            return tgt_signal

        # 4. Time-based exits
        for c in self._time_exits:
            sig = c.check(trade, bar)
            if sig:
                return sig

        return None
