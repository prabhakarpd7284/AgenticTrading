"""Owner rules (docs/strategies/TRADING_RULES.md) as research-local, pure code.

These mirror ``apps/trading/services/trailing_stop.py`` (``next_stop``,
``wilder_atr``, ``atr_for_symbol``) and ``apps/trading/services/cooldown.py``
(``evaluate_cooldown``). They are re-implemented here — not imported — because
those modules belong to another agent and are still moving; the unit tests
check equivalence against them whenever they import cleanly.

Trailing (§2)
    open profit ≥ 1R (best price vs entry fill) → stop to breakeven (entry);
    from then on stop = max(entry, best − 1.5 × ATR) for a long (mirror short);
    the stop only ratchets in the trade's favour.
    ATR = ``wilder_atr`` over the last (period + 2) completed bars of the
    trading timeframe — exactly the window ``atr_for_symbol`` reads live
    ((period + 2) × tf one-minute rows) — continuous across sessions.

Cooldown (§1)
    loss day = a day WITH trades whose net P&L < 0; 3 consecutive loss days
    (days without trades are skipped) → no new entries on the next 2 NSE
    trading days → then risk × 0.5 until the first profitable day.
    Days inside the cooldown neither extend the streak nor clear reduced mode;
    a day with net exactly 0 breaks the streak but does not end reduced mode.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

NORMAL, COOLDOWN, REDUCED = "normal", "cooldown", "reduced"


# ── trailing stop ───────────────────────────────────────────────────────────

def trail_next_stop(*, side: int, entry: float, initial_stop: float, best: float,
                    atr: float | None, current_stop: float, breakeven_r: float = 1.0,
                    atr_mult: float = 1.5) -> float:
    """New stop (ratchet only). ``side`` +1 long / −1 short; ``best`` = most
    favourable price since entry; ``atr`` None/≤0 → breakeven move only."""
    risk = abs(entry - initial_stop)
    if risk <= 0:
        return current_stop
    favour = side * (best - entry)
    cand = initial_stop
    if favour >= breakeven_r * risk:
        cand = entry
        if atr and atr > 0:
            trail = best - side * atr_mult * atr
            cand = max(cand, trail) if side > 0 else min(cand, trail)
    return max(current_stop, cand) if side > 0 else min(current_stop, cand)


def wilder_atr(hlc: Sequence[tuple[float, float, float]], period: int = 14) -> float | None:
    """Wilder ATR over (high, low, close) bars, oldest first; None if < period + 1 bars."""
    if len(hlc) < period + 1:
        return None
    trs = []
    for i in range(1, len(hlc)):
        h, lo, _ = hlc[i]
        pc = hlc[i - 1][2]
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def live_window_atr_series(bars_tf: Sequence, period: int = 14) -> list[float | None]:
    """ATR published at the close of each timeframe bar, using the live window
    (the last ``period + 2`` bars). ``bars_tf`` items need ``.h .l .c``."""
    out: list[float | None] = []
    w = period + 2
    for k in range(len(bars_tf)):
        lo = max(0, k + 1 - w)
        out.append(wilder_atr([(b.h, b.l, b.c) for b in bars_tf[lo:k + 1]], period))
    return out


# ── cooldown state machine ──────────────────────────────────────────────────

def nth_trading_day_after(d: date, n: int, is_trading_day: Callable[[date], bool]) -> date:
    cur, seen = d, 0
    while seen < n:
        cur = cur + timedelta(days=1)
        if is_trading_day(cur):
            seen += 1
    return cur


@dataclass
class CooldownTracker:
    """Incremental equivalent of ``cooldown.evaluate_cooldown``.

    Call ``status(day)`` before trading ``day``; call ``close_day(day, net)``
    after a day on which trades were taken (days without trades: don't call).
    """
    is_trading_day: Callable[[date], bool]
    loss_days: int = 3
    cooldown_days: int = 2
    reduced_mult: float = 0.5
    streak: int = 0
    trigger: date | None = None
    until: date | None = None
    reduced: bool = False

    def in_cooldown(self, d: date) -> bool:
        return self.trigger is not None and self.until is not None and self.trigger < d <= self.until

    def status(self, d: date) -> tuple[str, float]:
        if self.in_cooldown(d):
            return COOLDOWN, 0.0
        if self.reduced:
            return REDUCED, self.reduced_mult
        return NORMAL, 1.0

    def close_day(self, d: date, net_pnl: float) -> None:
        if self.in_cooldown(d):
            return
        if net_pnl > 0:
            self.streak = 0
            self.reduced = False
        elif net_pnl < 0:
            self.streak += 1
            if self.streak >= self.loss_days:
                self.trigger = d
                self.until = nth_trading_day_after(d, self.cooldown_days, self.is_trading_day)
                self.reduced = True
                self.streak = 0
        else:
            self.streak = 0
