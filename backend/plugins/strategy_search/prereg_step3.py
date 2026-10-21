"""Step 3 pre-registered rules (docs/strategies/STRATEGY_SEARCH_RESULTS.md, "Step 3").

Frozen parameters — do NOT tune. Every rule here implements exactly the text of
the pre-registration; where the text left a detail open, the choice made is
documented in the docstring and in the Step 4 section of the results doc.

Filter block (re-sizes Step 2 screener candidates, ``candidates.pkl``):
    * ``s0_first_per_day``  — first signal of a strategy per symbol-day.
    * ``f4_vetoes``         — gap-context veto (pure; needs that day's 1m bars).
    FB3/FB4 = S0 applied to the raw signal stream, then the F4 veto on the
    survivors (so FB3 ⊆ FB1, FB4 ⊆ FB2; a vetoed first signal is NOT replaced
    by a later one).

Candidates (signal sources for ``runner``):
    C1 ``PPExtFade``, C2 ``OR30FailFade``, C3/C4 ``ORB30`` (side −1 / +1),
    C5 ``XSReversal1D``.

Shared definitions (pre-reg 3.0): dATR = ``bias.daily_atr(daily_history, 14)``
(day skipped if None); PP = (H+L+C)/3 of ``prev_daily``; open = 09:15 1m open;
OR30 = high/low of 1m bars 09:15–09:44; "entry" (for stop distances) = the
harness fill = open of the first 1m bar with t ≥ signal_t, slipped 0.02 %
against the trade, so 1R is exactly the pre-registered distance. Exits: the
owner trail on 5m (``trail_tf=5``) + flat 15:14 (harness).
"""
from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from datetime import date

from plugins.strategy_pivot_boss.bars import Bar, ist_minute

from .bias import daily_atr
from .costs import slip
from .data import SymbolData, day_of, epoch_at, resample
from .sim import Candidate, Intent

SLIPPAGE_PCT = 0.02          # harness default; used only to place stops off the fill
OR_END_MIN = 9 * 60 + 44     # OR30 = 09:15..09:44 1m bars
M0916 = 9 * 60 + 16


def _m(hh: int, mm: int) -> int:
    return hh * 60 + mm


# ── shared helpers ──────────────────────────────────────────────────────────


def pivot_pp(prev: Bar) -> float:
    return (prev.h + prev.l + prev.c) / 3.0


def or30(bars: Sequence[Bar]) -> tuple[float, float] | None:
    w = [b for b in bars if ist_minute(b.t) <= OR_END_MIN]
    if not w:
        return None
    return max(b.h for b in w), min(b.l for b in w)


def entry_bar(bars: Sequence[Bar], signal_t: int) -> Bar | None:
    i = bisect.bisect_left([b.t for b in bars], signal_t)
    return bars[i] if i < len(bars) else None


def entry_fill(bars: Sequence[Bar], signal_t: int, side: int,
               slippage_pct: float = SLIPPAGE_PCT) -> float | None:
    b = entry_bar(bars, signal_t)
    return None if b is None else slip(b.o, side, slippage_pct, entering=True)


def _datr(s) -> float | None:
    a = daily_atr(s.daily_history, 14)
    return a if a and a > 0 else None


# ── filter block ────────────────────────────────────────────────────────────


def s0_first_per_day(cands: Iterable[Candidate], strategy: str) -> list[Candidate]:
    """Keep only the first (by signal_t, then entry_t) signal of ``strategy`` per
    symbol-day. Other strategies are dropped."""
    best: dict[tuple[str, date], Candidate] = {}
    for c in cands:
        it = c.intent
        if it.strategy != strategy:
            continue
        k = (it.symbol, it.day)
        cur = best.get(k)
        if cur is None or (it.signal_t, c.entry_t) < (cur.intent.signal_t, cur.entry_t):
            best[k] = c
    return sorted(best.values(), key=lambda c: (c.intent.day, c.intent.signal_t, c.intent.symbol))


def f4_vetoes(side: int, signal_t: int, day_open: float, prev_close: float,
              bars: Sequence[Bar]) -> bool:
    """True when F4 drops the signal: 0.25 % ≤ |gap| ≤ 1.0 %, side == sign(gap),
    signal_t ≤ 10:15 IST, and no 1m bar from 09:16 up to the signal bar (bars that
    closed by signal_t, i.e. t + 60 ≤ signal_t) has touched prev close."""
    if not prev_close or prev_close <= 0:
        return False
    gap = (day_open - prev_close) / prev_close
    if not (0.0025 <= abs(gap) <= 0.01):
        return False
    if (1 if side > 0 else -1) != (1 if gap > 0 else -1):
        return False
    if ist_minute(signal_t) > _m(10, 15):
        return False
    for b in bars:
        if b.t + 60 > signal_t:
            break
        if ist_minute(b.t) < M0916:
            continue
        if b.l <= prev_close <= b.h:
            return False
    return True


# ── C1 PP-extension fade ────────────────────────────────────────────────────


class PPExtFade:
    """d = (open − PP)/dATR; 0.5 ≤ |d| ≤ 1.0 → side = −sign(d), entry 09:16 open,
    stop entry − side·1.0·dATR, target PP, priority |d|."""
    name = "C1 pp_ext_fade"

    def generate(self, data: SymbolData) -> list[Intent]:
        out = []
        for s in data.period_sessions:
            if not s.bars or s.prev_daily is None or ist_minute(s.bars[0].t) != _m(9, 15):
                continue
            a = _datr(s)
            if a is None:
                continue
            pp = pivot_pp(s.prev_daily)
            d = (s.bars[0].o - pp) / a
            if not (0.5 <= abs(d) <= 1.0):
                continue
            side = -1 if d > 0 else 1
            st = epoch_at(s.day, 9, 16)
            e = entry_fill(s.bars, st, side)
            if e is None:
                continue
            out.append(Intent(strategy=self.name, symbol=data.symbol, day=s.day, side=side,
                              signal_t=st, stop=e - side * a, target=pp, trail_tf=5,
                              priority=abs(d), meta={"d": round(d, 3)}))
        return out


# ── C2 failed OR30 breakout fade ────────────────────────────────────────────


class OR30FailFade:
    """5m bars from 09:45: excursion = close beyond OR30; failure = a later 5m close
    back inside [OR30 low, OR30 high]. Only the FIRST failure of the day counts (either
    direction); it is traded only if its signal_t ∈ [10:00, 13:30]. Fade the excursion.
    Stop (short) = max(session high through the signal bar + 0.1·dATR, entry + 0.75·dATR);
    target = opposite OR30 extreme; priority = max excursion beyond OR30 / dATR."""
    name = "C2 or30_fail_fade"

    def generate(self, data: SymbolData) -> list[Intent]:
        out = []
        for s in data.period_sessions:
            a = _datr(s)
            r = or30(s.bars)
            if a is None or r is None:
                continue
            it = self._day(data.symbol, s.day, s.bars, r, a)
            if it:
                out.append(it)
        return out

    def _day(self, symbol: str, day: date, bars: Sequence[Bar], r: tuple[float, float],
             a: float) -> Intent | None:
        hi, lo = r
        exc = 0              # +1 above, −1 below, 0 none since last inside
        for b in resample(list(bars), 5):
            if ist_minute(b.t) < _m(9, 45):
                continue
            if b.c > hi:
                exc = 1
                continue
            if b.c < lo:
                exc = -1
                continue
            if exc == 0:
                continue
            # first failure of the day
            st = b.t + 300
            if not (_m(10, 0) <= ist_minute(st) <= _m(13, 30)):
                return None
            side = -exc
            upto = [x for x in bars if x.t < st]
            e = entry_fill(bars, st, side)
            if e is None:
                return None
            if side < 0:
                ext = max(x.h for x in upto)
                stop = max(ext + 0.1 * a, e + 0.75 * a)
                tgt, prio = lo, (ext - hi) / a
            else:
                ext = min(x.l for x in upto)
                stop = min(ext - 0.1 * a, e - 0.75 * a)
                tgt, prio = hi, (lo - ext) / a
            return Intent(strategy=self.name, symbol=symbol, day=day, side=side, signal_t=st,
                          stop=stop, target=tgt, trail_tf=5, priority=prio,
                          meta={"or_h": hi, "or_l": lo})
        return None


# ── C3 / C4 ORB30 one-per-day, wide stop ────────────────────────────────────


class ORB30:
    """side −1 (C3): first 15m bar (aligned 09:15, start ≥ 09:45) closing below OR30
    low with signal_t ≤ 12:00 → short; stop = entry + max(OR30 high − entry, 0.75·dATR);
    no target; priority (OR30 low − close)/dATR. side +1 (C4) is the exact mirror.
    Only the first such bar is considered (a break in the other direction is ignored)."""

    def __init__(self, side: int):
        self.side = 1 if side > 0 else -1
        self.name = "C4 orb30_long" if self.side > 0 else "C3 orb30_short"

    def generate(self, data: SymbolData) -> list[Intent]:
        out = []
        for s in data.period_sessions:
            a = _datr(s)
            r = or30(s.bars)
            if a is None or r is None:
                continue
            it = self._day(data.symbol, s.day, s.bars, r, a)
            if it:
                out.append(it)
        return out

    def _day(self, symbol, day, bars, r, a) -> Intent | None:
        hi, lo = r
        side = self.side
        for b in resample(list(bars), 15):
            if ist_minute(b.t) < _m(9, 45):
                continue
            st = b.t + 900
            if ist_minute(st) > _m(12, 0):
                return None
            if (side < 0 and b.c < lo) or (side > 0 and b.c > hi):
                e = entry_fill(bars, st, side)
                if e is None:
                    return None
                if side < 0:
                    stop = e + max(hi - e, 0.75 * a)
                    prio = (lo - b.c) / a
                else:
                    stop = e - max(e - lo, 0.75 * a)
                    prio = (b.c - hi) / a
                return Intent(strategy=self.name, symbol=symbol, day=day, side=side,
                              signal_t=st, stop=stop, target=None, trail_tf=5, priority=prio)
        return None


# ── C5 prior-day cross-sectional reversal ───────────────────────────────────


def xs_picks(daily_by_sym: dict[str, Sequence[Bar]], days: Iterable[date]
             ) -> dict[date, list[tuple[str, int, float, float]]]:
    """For each day: [(symbol, side, z, dATR)] — short the highest z if z ≥ 1, long
    the lowest z if z ≤ −1. z = (prev close − close two sessions back)/dATR using only
    bhavcopy bars strictly before the day. Ties broken by symbol name."""
    idx = {s: [day_of(b.t) for b in bars] for s, bars in daily_by_sym.items()}
    picks: dict[date, list] = {}
    for d in days:
        zs = []
        for sym, bars in daily_by_sym.items():
            hist = bars[:bisect.bisect_left(idx[sym], d)]
            if len(hist) < 2:
                continue
            a = daily_atr(hist, 14)
            if not a or a <= 0:
                continue
            zs.append(((hist[-1].c - hist[-2].c) / a, sym, a))
        out = []
        if zs:
            top = min(zs, key=lambda x: (-x[0], x[1]))
            bot = min(zs, key=lambda x: (x[0], x[1]))
            if top[0] >= 1.0:
                out.append((top[1], -1, top[0], top[2]))
            if bot[0] <= -1.0:
                out.append((bot[1], 1, bot[0], bot[2]))
        picks[d] = out
    return picks


class XSReversal1D:
    """Pre-computes daily picks over ``symbols`` from 1d bars (≤ the source's
    max_date, IS only) at construction; ``generate`` emits for its own symbol: entry
    09:16 open, stop entry ∓ 1.0·dATR, no target, priority |z|."""
    name = "C5 xs_reversal_1d"

    def __init__(self, picks: dict[date, list[tuple[str, int, float, float]]]):
        self.by_sym: dict[str, dict[date, tuple[int, float, float]]] = {}
        for d, lst in picks.items():
            for sym, side, z, a in lst:
                self.by_sym.setdefault(sym, {})[d] = (side, z, a)

    @classmethod
    def from_source(cls, src, symbols: Sequence[str], days: Sequence[date]) -> XSReversal1D:
        end = max(days)
        daily = {s: src.daily_bars(s, end) for s in symbols}
        return cls(xs_picks(daily, days))

    def generate(self, data: SymbolData) -> list[Intent]:
        mine = self.by_sym.get(data.symbol, {})
        out = []
        for s in data.period_sessions:
            p = mine.get(s.day)
            if p is None:
                continue
            side, z, a = p
            st = epoch_at(s.day, 9, 16)
            e = entry_fill(s.bars, st, side)
            if e is None:
                continue
            out.append(Intent(strategy=self.name, symbol=data.symbol, day=s.day, side=side,
                              signal_t=st, stop=e - side * a, target=None, trail_tf=5,
                              priority=abs(z), meta={"z": round(z, 3)}))
        return out
