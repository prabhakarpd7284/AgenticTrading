"""The book's setups as per-stock-day signal generators (spec §4).

Each generator looks at one session (``DayContext``) and returns at most one
``Signal`` — the first valid one of the day. Operational choices where the
book is qualitative are listed in ``SetupParams`` and in the results doc.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .backtest import Signal
from .bars import Bar, ist_minute
from .levels import CPR, Camarilla, FloorPivots, MoneyZone


@dataclass(frozen=True)
class SetupParams:
    narrow_cpr_pct: float = 0.25  # replace with NSE in-sample percentile
    wide_cpr_pct: float = 0.75
    tick: float = 0.05
    entry_cutoff_min: int = 14 * 60 + 30  # no fresh entries after 14:30 IST
    trail_mult: float = 1.5  # × 15-min ATR(14)
    retest_tol_pct: float = 0.10  # breakaway: first-15m low within 0.10 % of the pivot
    magnet_gap_atr: tuple[float, float] = (0.25, 1.0)  # "moderate" gap, daily-ATR units
    magnet_cpr_near_frac: float = 0.5  # CPR edge within 0.5×gap of the prior close
    magnet_max_hold_min: int = 60  # abandon the fill after 60 min


@dataclass
class DayContext:
    symbol: str
    day: str
    bars1m: Sequence[Bar]  # today's 1m bars ≤ 15:14
    bars5: Sequence[Bar]  # today's 5m bars
    bars15: Sequence[Bar]  # today's 15m bars
    trig15: Sequence[int]  # any-trigger series aligned with bars15 (multi-day history)
    prev_h: float
    prev_l: float
    prev_c: float
    floor: FloorPivots  # today's, from prev H/L/C
    cpr: CPR
    cam: Camarilla
    cpr_prev: CPR | None  # yesterday's CPR (from the session before)
    prev_mz: MoneyZone | None  # yesterday's Money Zone
    atr_daily: float | None
    atr15_at: Sequence[float | None]  # per 1m bar: 15m ATR(14) at 15m bar closes

    def idx_at(self, minute: int) -> int:
        """First 1m index whose IST minute ≥ ``minute`` (len if none)."""
        for i, b in enumerate(self.bars1m):
            if ist_minute(b.t) >= minute:
                return i
        return len(self.bars1m)

    @property
    def open(self) -> float:
        return self.bars1m[0].o


def _entry_after(ctx: DayContext, bar: Bar, minutes: int, p: SetupParams) -> int | None:
    m = ist_minute(bar.t) + minutes
    if m > p.entry_cutoff_min:
        return None
    i = ctx.idx_at(m)
    return i if i < len(ctx.bars1m) else None


# ─────────────────────────────────────────────────────────────────────────────


def floor_breakaway(ctx: DayContext, p: SetupParams, require_narrow: bool = True) -> Signal | None:
    """Gap beyond prior range & value and beyond R1/S1 (not past R2/S2); the
    first 15-min bar retests the surpassed pivot and holds (close beyond it, in
    the favourable half of its range); enter next bar open; stop beyond day
    extreme; target R3/S3 + 1.5 ATR trail; flat 15:14."""
    if not ctx.bars15 or (require_narrow and ctx.cpr.width_pct >= p.narrow_cpr_pct):
        return None
    o, f, b = ctx.open, ctx.floor, ctx.bars15[0]
    v_hi = ctx.prev_mz.vah if ctx.prev_mz else ctx.cpr.tc
    v_lo = ctx.prev_mz.val if ctx.prev_mz else ctx.cpr.bc
    tol = p.retest_tol_pct / 100.0
    mid = (b.h + b.l) / 2
    if o > ctx.prev_h and o > v_hi and f.r1 < o <= f.r2:
        if b.l <= f.r1 * (1 + tol) and b.c > f.r1 and b.c >= mid:
            i = _entry_after(ctx, b, 15, p)
            if i is not None:
                return Signal("floor_breakaway", 1, i, stop=b.l - p.tick, target=f.r3,
                              trail_mult=p.trail_mult, atr_at=ctx.atr15_at)
    if o < ctx.prev_l and o < v_lo and f.s2 <= o < f.s1:
        if b.h >= f.s1 * (1 - tol) and b.c < f.s1 and b.c <= mid:
            i = _entry_after(ctx, b, 15, p)
            if i is not None:
                return Signal("floor_breakaway", -1, i, stop=b.h + p.tick, target=f.s3,
                              trail_mult=p.trail_mult, atr_at=ctx.atr15_at)
    return None


def camarilla_h3l3_reversal(ctx: DayContext, p: SetupParams) -> Signal | None:
    """Open inside the prior range. 5-min bar wicks through H3 and closes back
    below; the next bar confirms (high < H3, close < prior close); enter at the
    following bar's open; stop just beyond H4; target L3. Mirror at L3."""
    if not (ctx.prev_l <= ctx.open <= ctx.prev_h):
        return None
    cam, bs = ctx.cam, ctx.bars5
    for k in range(len(bs) - 1):
        b, c = bs[k], bs[k + 1]
        if b.h > cam.h3 and b.c < cam.h3 and c.h < cam.h3 and c.c < b.c:
            i = _entry_after(ctx, c, 5, p)
            if i is not None:
                return Signal("cam_h3l3_reversal", -1, i, stop=cam.h4 + p.tick, target=cam.l3)
        if b.l < cam.l3 and b.c > cam.l3 and c.l > cam.l3 and c.c > b.c:
            i = _entry_after(ctx, c, 5, p)
            if i is not None:
                return Signal("cam_h3l3_reversal", 1, i, stop=cam.l4 - p.tick, target=cam.h3)
    return None


def camarilla_h4l4_breakout(ctx: DayContext, p: SetupParams) -> Signal | None:
    """First 5-min CLOSE beyond H4 that crosses it (previous close / the open at
    or below H4); enter next bar open; stop H3; target H5. Mirror at L4."""
    cam, bs = ctx.cam, ctx.bars5
    prev_c = ctx.open
    for b in bs:
        if b.c > cam.h4 and prev_c <= cam.h4:
            i = _entry_after(ctx, b, 5, p)
            if i is not None:
                return Signal("cam_h4l4_breakout", 1, i, stop=cam.h3, target=cam.h5)
            return None
        if b.c < cam.l4 and prev_c >= cam.l4:
            i = _entry_after(ctx, b, 5, p)
            if i is not None:
                return Signal("cam_h4l4_breakout", -1, i, stop=cam.l3, target=cam.l5)
            return None
        prev_c = b.c
    return None


def magnet(ctx: DayContext, p: SetupParams) -> Signal | None:
    """Moderate gap (0.25–1.0 daily ATR) away from the prior close, open beyond
    the CPR, CPR edge close to the prior close, no R1/S1 between the open and
    the CPR. First 15-min bar is a reversal candle (opposite colour, close in
    the gap-fill half of its range, or any Appendix-B trigger). Enter next bar
    open; stop beyond the day's extreme; target the near CPR edge; abandon
    after 60 min."""
    if not ctx.bars15 or not ctx.atr_daily:
        return None
    o, pc, cp, f = ctx.open, ctx.prev_c, ctx.cpr, ctx.floor
    gap = o - pc
    g_atr = abs(gap) / ctx.atr_daily
    if not (p.magnet_gap_atr[0] <= g_atr <= p.magnet_gap_atr[1]):
        return None
    b = ctx.bars15[0]
    mid = (b.h + b.l) / 2
    trig = ctx.trig15[0] if ctx.trig15 else 0
    if gap > 0 and o > cp.tc and abs(cp.tc - pc) <= p.magnet_cpr_near_frac * gap \
            and not (cp.tc < f.r1 < o):
        if (b.c < b.o and b.c <= mid) or trig < 0:
            i = _entry_after(ctx, b, 15, p)
            if i is not None:
                ex = ctx.idx_at(ist_minute(ctx.bars1m[i].t) + p.magnet_max_hold_min)
                return Signal("magnet", -1, i, stop=b.h + p.tick, target=cp.tc,
                              time_exit_idx=min(ex, len(ctx.bars1m) - 1))
    if gap < 0 and o < cp.bc and abs(pc - cp.bc) <= p.magnet_cpr_near_frac * -gap \
            and not (o < f.s1 < cp.bc):
        if (b.c > b.o and b.c >= mid) or trig > 0:
            i = _entry_after(ctx, b, 15, p)
            if i is not None:
                ex = ctx.idx_at(ist_minute(ctx.bars1m[i].t) + p.magnet_max_hold_min)
                return Signal("magnet", 1, i, stop=b.l - p.tick, target=cp.bc,
                              time_exit_idx=min(ex, len(ctx.bars1m) - 1))
    return None


def golden_pivot_zone(ctx: DayContext, p: SetupParams) -> Signal | None:
    """Zone = [BC, TC] containing ≥1 of H3/L3/prior VAH/VAL/POC/VPOC. Uptrend =
    yesterday closed above its own CPR and today's prior close and open are both
    above today's TC. First 15-min bar that pulls back into the zone (low ≤ TC)
    and fires any Appendix-B long trigger while closing above BC → enter next
    bar open; stop below the zone/bar low; target nearest of R1/H3 above entry.
    Mirror for downtrend."""
    if ctx.cpr_prev is None or not ctx.bars15:
        return None
    cp, cam, f = ctx.cpr, ctx.cam, ctx.floor
    cands = [cam.h3, cam.l3]
    if ctx.prev_mz:
        cands += [ctx.prev_mz.vah, ctx.prev_mz.val, ctx.prev_mz.poc, ctx.prev_mz.vpoc]
    if not any(cp.bc <= x <= cp.tc for x in cands):
        return None
    o, pc = ctx.open, ctx.prev_c
    up = pc > cp.tc and o > cp.tc and pc > ctx.cpr_prev.tc
    dn = pc < cp.bc and o < cp.bc and pc < ctx.cpr_prev.bc
    if not (up or dn):
        return None
    for k, b in enumerate(ctx.bars15):
        t = ctx.trig15[k] if k < len(ctx.trig15) else 0
        if up and b.l <= cp.tc and b.c > cp.bc and t > 0:
            i = _entry_after(ctx, b, 15, p)
            if i is None:
                return None
            entry = ctx.bars1m[i].o
            tgts = [x for x in (f.r1, cam.h3) if x > entry]
            if not tgts:
                return None
            return Signal("golden_pivot_zone", 1, i, stop=min(cp.bc, b.l) - p.tick,
                          target=min(tgts))
        if dn and b.h >= cp.bc and b.c < cp.tc and t < 0:
            i = _entry_after(ctx, b, 15, p)
            if i is None:
                return None
            entry = ctx.bars1m[i].o
            tgts = [x for x in (f.s1, cam.l3) if x < entry]
            if not tgts:
                return None
            return Signal("golden_pivot_zone", -1, i, stop=max(cp.tc, b.h) + p.tick,
                          target=max(tgts))
    return None


SETUPS = {
    "floor_breakaway": floor_breakaway,
    "floor_breakaway_any_cpr": lambda ctx, p: _rename(floor_breakaway(ctx, p, require_narrow=False),
                                                      "floor_breakaway_any_cpr"),
    "cam_h3l3_reversal": camarilla_h3l3_reversal,
    "cam_h4l4_breakout": camarilla_h4l4_breakout,
    "magnet": magnet,
    "golden_pivot_zone": golden_pivot_zone,
}


def _rename(sig: Signal | None, name: str) -> Signal | None:
    if sig is not None:
        sig.setup = name
    return sig
