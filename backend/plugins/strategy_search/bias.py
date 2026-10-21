"""Step 1 — bias verification (IN-SAMPLE only).

Question: are the Pivot Boss "survivors" (open-vs-PP side persistence, PP
untouched → directional day, H3/L3 first touch = continuation, small-gap fill)
tradeable edges after NSE costs, or mechanical artefacts?

One pass over the universe, one symbol at a time:
* ``day_features`` → one row per stock-day (levels from the PRIOR bhavcopy, the
  day's 1m path, and — for outcome measurement only — the day's own bhavcopy
  close, which is never used to form a signal).
* ``BiasSources.generate`` → harness ``Intent``s for each tradeable variant
  (entry at the next 1m open after the signal is known, harness exits).
* ``bias_tables`` → markdown tables (descriptive + hypothetical open→15:14
  trades after costs, day-clustered t-stats).

Mechanical null for "close stays on the open's side of PP": for every day with
open > PP, the probability that open·(1+r) > PP when r is drawn from the SAME
stock's IS open→close returns (all days, unconditional). If the observed rate
≈ the null, persistence is just "the open is already on that side".
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date

from plugins.strategy_pivot_boss.bars import Bar, ist_minute
from plugins.strategy_pivot_boss.levels import camarilla, cpr, floor_pivots

from .costs import CostModel
from .data import DaySession, SymbolData, day_of
from .sim import Intent

M = lambda hh, mm: hh * 60 + mm  # noqa: E731
SLIP_PCT = 0.02                     # per side, % (harness default)
NOTIONALS = (50_000.0, 500_000.0)   # 10 % RiskEngine cap / 100 % (MIS sensitivity)
IS_HALF = date(2026, 7, 1)          # stability split inside IS


# ── per-day features ────────────────────────────────────────────────────────

def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def daily_atr(hist: Sequence[Bar], n: int = 14) -> float | None:
    """Simple mean true range over the last ``n`` sessions before the day
    (needs n + 1 bars, as in the Pivot Boss study)."""
    if len(hist) < n + 1:
        return None
    w = hist[-(n + 1):]
    trs = [max(w[i].h - w[i].l, abs(w[i].h - w[i - 1].c), abs(w[i].l - w[i - 1].c))
           for i in range(1, len(w))]
    return sum(trs) / n


def _first_at(bars: Sequence[Bar], minute: int) -> Bar | None:
    for b in bars:
        if ist_minute(b.t) >= minute:
            return b
    return None


def _last_upto(bars: Sequence[Bar], minute: int) -> Bar | None:
    out = None
    for b in bars:
        if ist_minute(b.t) <= minute:
            out = b
        else:
            break
    return out


def _touched(bars: Sequence[Bar], level: float, upto_min: int, from_min: int = 0) -> bool:
    return any(b.l <= level <= b.h for b in bars
               if from_min <= ist_minute(b.t) <= upto_min)


def day_features(sym: str, s: DaySession, bhav_close: float | None) -> dict | None:
    pdl, bars = s.prev_daily, s.bars
    if pdl is None or not bars or ist_minute(bars[0].t) != M(9, 15):
        return None
    if ist_minute(bars[-1].t) < M(15, 0):
        return None
    fp = floor_pivots(pdl.h, pdl.l, pdl.c)
    cp = cpr(pdl.h, pdl.l, pdl.c)
    cam = camarilla(pdl.h, pdl.l, pdl.c)
    atr = daily_atr(s.daily_history)
    pp, pc = fp.pp, pdl.c
    o = bars[0].o
    b916, b930, b945, b1015 = (_first_at(bars, M(9, 16)), _first_at(bars, M(9, 30)),
                               _first_at(bars, M(9, 45)), _first_at(bars, M(10, 15)))
    c15, c30 = _last_upto(bars, M(9, 29)), _last_upto(bars, M(9, 44))
    c1514 = bars[-1].c
    r = lambda a, b: (b / a - 1) * 100 if a and b else None  # noqa: E731
    gap_dir = _sign(o - pc)
    row = {
        "symbol": sym, "day": s.day, "open": o, "pp": pp, "tc": cp.tc, "bc": cp.bc,
        "r1": fp.r1, "s1": fp.s1, "h3": cam.h3, "l3": cam.l3, "h4": cam.h4, "l4": cam.l4,
        "prev_close": pc, "atr": atr, "bhav_close": bhav_close, "c1514": c1514,
        "side_pp": _sign(o - pp),
        "dist_pp_pct": (o - pp) / pp * 100,
        "dist_pp_atr": (o - pp) / atr if atr else None,
        "gap_pct": (o - pc) / pc * 100,
        "gap_atr": (o - pc) / atr if atr else None,
        "cpr_side": 1 if o > cp.tc else -1 if o < cp.bc else 0,
        "cpr_width_pct": cp.width_pct,
        # returns, %
        "r_oc_bhav": r(o, bhav_close),
        "r_o_1514": r(o, c1514),
        "e916": b916.o if b916 else None, "r_916": r(b916.o if b916 else None, c1514),
        "side15": _sign(c15.c - pp) if c15 else 0,
        "e930": b930.o if b930 else None, "r_930": r(b930.o if b930 else None, c1514),
        "side30": _sign(c30.c - pp) if c30 else 0,
        "e945": b945.o if b945 else None, "r_945": r(b945.o if b945 else None, c1514),
        "pp_touched_1015": _touched(bars, pp, M(10, 14)) or None,
        "pp_touched_day": _touched(bars, pp, M(15, 14)),
        "e1015": b1015.o if b1015 else None, "r_1015": r(b1015.o if b1015 else None, c1514),
        "gap_dir": gap_dir,
        "pc_filled_1015": bool(gap_dir) and _touched(bars, pc, M(10, 14)),
        "pc_filled_1514": bool(gap_dir) and _touched(bars, pc, M(15, 14)),
        "pc_filled_1015_after916": bool(gap_dir) and _touched(bars, pc, M(10, 14), M(9, 16)),
    }
    row["pp_touched_1015"] = bool(row["pp_touched_1015"])
    return row


# ── signal sources (harness) ────────────────────────────────────────────────

class BiasSources:
    """All tradeable variants in one generator (strategy name distinguishes them).
    Also collects ``day_features`` rows as a side effect (``self.days``)."""

    name = "bias_is"
    GAP_MIN_PCT, GAP_MAX_PCT = 0.25, 1.0

    def __init__(self, bhav_by_symbol: Callable[[str], dict[date, float]]):
        self.bhav_by_symbol = bhav_by_symbol
        self.days: list[dict] = []

    def generate(self, data: SymbolData) -> list[Intent]:
        bhav = self.bhav_by_symbol(data.symbol)
        out: list[Intent] = []
        for s in data.period_sessions:
            f = day_features(data.symbol, s, bhav.get(s.day))
            if f is None:
                continue
            self.days.append(f)
            out.extend(self._intents(data.symbol, s, f))
        return out

    def _intents(self, sym: str, s: DaySession, f: dict) -> list[Intent]:
        d, bars = s.day, s.bars
        pp, out = f["pp"], []
        t0 = bars[0].t

        def mk(strategy, side, signal_t, stop, target=None, time_exit_t=None, prio=0.0, **meta):
            out.append(Intent(strategy=strategy, symbol=sym, day=d, side=side,
                              signal_t=signal_t, stop=stop, target=target, trail_tf=5,
                              time_exit_t=time_exit_t, priority=prio, meta=meta))

        prio = abs(f["dist_pp_pct"])
        # A. open side of PP, 09:16 entry, stop at PP
        if f["side_pp"]:
            mk("pp_open_side", f["side_pp"], t0 + 60, pp, prio=prio)
            if f["atr"]:
                ref = bars[0].c
                mk("pp_open_side_atrstop", f["side_pp"], t0 + 60,
                   ref - f["side_pp"] * 0.5 * f["atr"], prio=prio)
        # B/C. first 15 / 30 min close vs PP
        if f["side15"]:
            mk("pp_first15_side", f["side15"], t0 + 15 * 60, pp, prio=prio)
        if f["side30"]:
            mk("pp_first30_side", f["side30"], t0 + 30 * 60, pp, prio=prio)
        # D. PP untouched 09:15–10:14 → trade the side at 10:15, stop at PP
        if f["side_pp"] and not f["pp_touched_1015"]:
            mk("pp_untouched_1015", f["side_pp"], t0 + 60 * 60, pp, prio=prio)
        # E. open outside CPR → trade that side, stop beyond the far CPR edge
        if f["cpr_side"]:
            stop = f["bc"] if f["cpr_side"] > 0 else f["tc"]
            mk("cpr_open_side", f["cpr_side"], t0 + 60, stop, prio=prio)
        # F. Camarilla H3/L3 first touch (open strictly inside L3–H3) → continuation
        h3, l3, h4, l4, c = f["h3"], f["l3"], f["h4"], f["l4"], f["prev_close"]
        if l3 < f["open"] < h3:
            touch = None
            for b in bars:
                up, dn = b.h >= h3, b.l <= l3
                if up or dn:
                    touch = (b, 1 if up and not dn else -1 if dn and not up else 0)
                    break
            if touch and touch[1]:
                b, side = touch
                tgt = h4 if side > 0 else l4
                mk("cam_touch_tgtH4", side, b.t + 60, c, target=tgt)
                mk("cam_touch_trail", side, b.t + 60, c)
                # book geometry: stop at the opposite band (L3/H3), target H4/L4
                mk("cam_touch_stopL3", side, b.t + 60, l3 if side > 0 else h3, target=tgt)
                # first 5m close beyond the touched band (at/after the touch bar)
                lvl = h3 if side > 0 else l3
                for b5 in s.resampled(5):
                    if b5.t + 300 <= b.t:
                        continue
                    if side * (b5.c - lvl) > 0:
                        mk("cam_5mclose_tgtH4", side, b5.t + 300, c, target=tgt)
                        break
        # G. small-gap fill: fade 0.25–1.0 % gaps toward the prior close
        g = f["gap_pct"]
        side, gap_px = -_sign(g), abs(f["open"] - c)
        stop = f["open"] - side * gap_px
        if self.GAP_MIN_PCT <= abs(g) <= self.GAP_MAX_PCT:
            mk("gapfill_60m", side, t0 + 60, stop, target=c, time_exit_t=t0 + 60 * 60)
            mk("gapfill_eod", side, t0 + 60, stop, target=c)
        elif 0.10 <= abs(g) < self.GAP_MIN_PCT:
            mk("gapfill_tiny_60m", side, t0 + 60, stop, target=c, time_exit_t=t0 + 60 * 60)
        return out


STRATEGY_NOTES = {
    "pp_open_side": "09:16 open, side = open vs PP, stop PP, trail, EOD",
    "pp_open_side_atrstop": "09:16 open, side = open vs PP, stop 0.5×ATR14(d) from the 09:15 close",
    "pp_first15_side": "09:30 open, side = 09:29 close vs PP, stop PP",
    "pp_first30_side": "09:45 open, side = 09:44 close vs PP, stop PP",
    "pp_untouched_1015": "10:15 open if PP untouched 09:15–10:14, side of PP, stop PP",
    "cpr_open_side": "09:16 open if open outside CPR, stop far CPR edge",
    "cam_touch_tgtH4": "next 1m open after first H3/L3 touch, stop = prior close (C), target H4/L4",
    "cam_touch_trail": "same entry/stop, no target (trail + EOD)",
    "cam_touch_stopL3": "next 1m open after first H3/L3 touch, stop = opposite band (L3/H3), target H4/L4",
    "cam_5mclose_tgtH4": "next open after first 5m close beyond the touched band, stop C, target H4/L4",
    "gapfill_60m": "gap 0.25–1.0 %: fade at 09:16, target prior close, stop open ± gap, exit 10:15",
    "gapfill_eod": "same, no time exit",
    "gapfill_tiny_60m": "gap 0.10–0.25 %: same as gapfill_60m (the 'small gap' the magnet stats favour)",
}


# ── statistics ──────────────────────────────────────────────────────────────

_COST = CostModel()


def net_pct(entry: float, signed_r_pct: float, side: int, notional: float) -> float:
    """Hypothetical open→close trade: signed return − 2×slippage − costs (% of notional)."""
    qty = max(1, math.floor(notional / entry))
    exit_ = entry * (1 + side * signed_r_pct / 100)
    cost = _COST.round_trip(entry, exit_, qty, side) / (qty * entry) * 100
    return signed_r_pct - 2 * SLIP_PCT - cost


def day_t(pairs: Sequence[tuple[date, float]]) -> tuple[float | None, int]:
    """t-stat of the mean with stock-days clustered by date (mean per day, then t over days)."""
    by = defaultdict(list)
    for d, v in pairs:
        by[d].append(v)
    means = [sum(v) / len(v) for v in by.values()]
    if len(means) < 3:
        return None, len(means)
    sd = statistics.stdev(means)
    return (statistics.fmean(means) / (sd / math.sqrt(len(means))) if sd else None), len(means)


def signed_stats(rows: Sequence[dict], side_key, ret_key: str, entry_key: str) -> dict:
    """side_key: field name or callable(row)->side. Returns n, continuation %,
    mean/median signed %, day-t, net % at each notional."""
    pts = []
    for r in rows:
        side = side_key(r) if callable(side_key) else r[side_key]
        v, e = r.get(ret_key), r.get(entry_key)
        if not side or v is None or not e:
            continue
        pts.append((r["day"], side, side * v, e))
    if not pts:
        return {"n": 0}
    sv = [p[2] for p in pts]
    t, nd = day_t([(p[0], p[2]) for p in pts])
    out = {"n": len(pts), "n_days": nd, "cont": sum(v > 0 for v in sv) / len(sv),
           "mean": statistics.fmean(sv), "median": statistics.median(sv), "t": t}
    for notional in NOTIONALS:
        nets = [net_pct(p[3], p[2], p[1], notional) for p in pts]
        out[f"net_{int(notional)}"] = statistics.fmean(nets)
        out[f"netwin_{int(notional)}"] = sum(x > 0 for x in nets) / len(nets)
    return out


def mech_null(rows: Sequence[dict]) -> dict[tuple[str, date], float]:
    """P(close on the open's side of PP) if the day's O→C return were drawn from the
    same stock's unconditional IS O→C returns (bhavcopy close)."""
    pool = defaultdict(list)
    for r in rows:
        if r["r_oc_bhav"] is not None:
            pool[r["symbol"]].append(r["r_oc_bhav"])
    out = {}
    for r in rows:
        p, side = pool[r["symbol"]], r["side_pp"]
        if not p or not side:
            continue
        thr = (r["pp"] / r["open"] - 1) * 100       # return needed to reach PP
        out[(r["symbol"], r["day"])] = sum(side * (x - thr) > 0 for x in p) / len(p)
    return out


# ── table rendering ─────────────────────────────────────────────────────────

def _p(x, nd=1):
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def _f(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}"


def _star(n):
    return " *" if n < 30 else ""


HDR = ("| condition | n | days | cont % | mean % | median % | t (day-clustered) | "
       "net % @₹50k | net % @₹5L | net win % @₹50k |")
SEP = "|---|---|---|---|---|---|---|---|---|---|"


def srow(label: str, st: dict) -> str:
    if not st.get("n"):
        return f"| {label} | 0 | | | | | | | | |"
    return (f"| {label} | {st['n']}{_star(st['n'])} | {st['n_days']} | {_p(st['cont'])} | "
            f"{_f(st['mean'])} | {_f(st['median'])} | {_f(st['t'], 2)} | "
            f"{_f(st['net_50000'])} | {_f(st['net_500000'])} | {_p(st['netwin_50000'])} |")


def _bucket(v: float | None, edges: Sequence[float]) -> str | None:
    if v is None:
        return None
    a = abs(v)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo <= a < hi:
            return f"{lo:g}–{hi:g}" if hi != math.inf else f"≥ {lo:g}"
    return None


def bias_tables(days: list[dict]) -> str:
    lines: list[str] = []
    rows = [r for r in days if r["side_pp"]]
    all_long = lambda r: 1  # noqa: E731
    up = [r for r in rows if r["side_pp"] > 0]
    dn = [r for r in rows if r["side_pp"] < 0]

    # T1 — persistence vs mechanical null
    null = mech_null(days)
    lines += ["### 1a. Open vs PP: persistence against a mechanical null", "",
          "`close` = official bhavcopy close (as in the Pivot Boss study). *Null* = the "
          "rate expected if each day's open→close return were drawn at random from the same "
          "stock's IS returns (i.e. pure \"the open is already on that side\").", "",
          "| condition | n | close same side of PP (actual) | mechanical null | excess | "
          "close beyond open, same direction | unconditional close > open |",
          "|---|---|---|---|---|---|---|"]
    uncond_up = sum(r["r_oc_bhav"] > 0 for r in days if r["r_oc_bhav"] is not None) / max(
        1, sum(r["r_oc_bhav"] is not None for r in days))
    for lab, sub, sd in (("open > PP", up, 1), ("open < PP", dn, -1)):
        sub = [r for r in sub if r["bhav_close"] is not None]
        act = sum(sd * (r["bhav_close"] - r["pp"]) > 0 for r in sub) / len(sub)
        nl = statistics.fmean(null[(r["symbol"], r["day"])] for r in sub)
        cont = sum(sd * r["r_oc_bhav"] > 0 for r in sub) / len(sub)
        lines.append(f"| {lab} | {len(sub)} | {_p(act)} | {_p(nl)} | {(act - nl) * 100:+.1f} pp | "
                 f"{_p(cont)} | {_p(uncond_up if sd > 0 else 1 - uncond_up)} "
                 f"({'close>open' if sd > 0 else 'close<open'}, all days) |")
    lines.append("")

    # T1b — tradeable open→close
    lines += ["### 1b. Open side of PP as a trade (side = +1 above PP / −1 below; signed returns)", "",
          "*cont %* = share of days the signed return is > 0. *mean/median* = signed return in %. "
          "Net = signed return − 0.04 % slippage − NSE costs at that notional (no stop, "
          "09:16 entry → 15:14 close). *t* clusters the 49 stocks by date (one mean per day).", "",
          HDR, SEP]
    for lab, sub in (("all days, LONG every stock (drift baseline)", days),):
        lines.append(srow(lab + ", 09:16→15:14", signed_stats(sub, all_long, "r_916", "e916")))
    for lab, sub in (("open > PP → long", up), ("open < PP → short", dn), ("both sides", rows)):
        lines.append(srow(lab + ", bhav O→C (descriptive)",
                      signed_stats(sub, "side_pp", "r_oc_bhav", "open")))
        lines.append(srow(lab + ", 09:15 open→15:14", signed_stats(sub, "side_pp", "r_o_1514", "open")))
        lines.append(srow(lab + ", **09:16 open→15:14 (tradeable)**",
                      signed_stats(sub, "side_pp", "r_916", "e916")))
    # drift-neutral version: subtract the same-day cross-sectional mean
    lines.append(srow("both sides, 09:16→15:14, **market-neutral** (minus same-day mean of all 49)",
                  signed_stats(_demean(rows, "r_916"), "side_pp", "r_916_dm", "e916")))
    lines.append("")

    # T2 — distance buckets
    for key, edges, title in (
            ("dist_pp_pct", (0, 0.25, 0.5, 1, 2, math.inf), "open distance from PP, % of PP"),
            ("dist_pp_atr", (0, 0.25, 0.5, 1, math.inf), "open distance from PP, daily ATR14 units"),
            ("gap_pct", (0, 0.25, 0.5, 1, 2, math.inf), "gap (09:15 open vs prior close), %")):
        lines += [f"### 1c. Open side of PP by {title} (09:16→15:14, signed by PP side)", "", HDR, SEP]
        for sd, slab in ((1, "long (open > PP)"), (-1, "short (open < PP)")):
            for b in _bucket_labels(edges):
                sub = [r for r in rows if r["side_pp"] == sd and _bucket(r[key], edges) == b]
                if key == "gap_pct":
                    for gd, glab in ((sd, "gap with PP side"), (-sd, "gap against PP side")):
                        sub2 = [r for r in sub if _sign(r["gap_pct"]) == gd]
                        lines.append(srow(f"{slab}, |gap| {b}, {glab}",
                                      signed_stats(sub2, "side_pp", "r_916", "e916")))
                else:
                    lines.append(srow(f"{slab}, {b}", signed_stats(sub, "side_pp", "r_916", "e916")))
        lines.append("")

    # T3 — intraday versions
    lines += ["### 1d. Intraday versions (entry at the next 1m open, exit 15:14, signed)", "", HDR, SEP]
    lines.append(srow("first-15m close (09:29) vs PP → enter 09:30",
                  signed_stats(days, "side15", "r_930", "e930")))
    lines.append(srow("first-30m close (09:44) vs PP → enter 09:45",
                  signed_stats(days, "side30", "r_945", "e945")))
    unt = [r for r in rows if not r["pp_touched_1015"]]
    tch = [r for r in rows if r["pp_touched_1015"]]
    for sd, lab in ((1, "above"), (-1, "below")):
        lines.append(srow(f"PP untouched 09:15–10:14, {lab} PP → enter 10:15",
                      signed_stats([r for r in unt if r["side_pp"] == sd], "side_pp", "r_1015", "e1015")))
    lines.append(srow("PP untouched by 10:15, both sides → enter 10:15",
                  signed_stats(unt, "side_pp", "r_1015", "e1015")))
    lines.append(srow("PP touched by 10:15 (reference; side = open vs PP) → 10:15",
                  signed_stats(tch, "side_pp", "r_1015", "e1015")))
    lines.append(srow("PP untouched by 10:15, **market-neutral**",
                  signed_stats(_demean(unt, "r_1015", universe=days), "side_pp", "r_1015_dm", "e1015")))
    for cs, lab in ((1, "open above TC → long"), (-1, "open below BC → short")):
        lines.append(srow(f"CPR: {lab}, 09:16",
                      signed_stats([r for r in days if r["cpr_side"] == cs], "cpr_side", "r_916", "e916")))
    lines.append(srow("CPR: open inside CPR (long, reference), 09:16",
                  signed_stats([r for r in days if r["cpr_side"] == 0], all_long, "r_916", "e916")))
    lines.append("")

    # T4 — stability within IS
    lines += ["### 1e. Stability inside IS (Apr 20 – Jun 30 vs Jul 1 – Aug 31), 09:16/10:15 → 15:14", "",
          HDR, SEP]
    for lab, flt in (("Apr–Jun", lambda r: r["day"] < IS_HALF), ("Jul–Aug", lambda r: r["day"] >= IS_HALF)):
        lines.append(srow(f"{lab}: open side of PP", signed_stats([r for r in rows if flt(r)],
                                                              "side_pp", "r_916", "e916")))
        lines.append(srow(f"{lab}: PP untouched by 10:15", signed_stats([r for r in unt if flt(r)],
                                                                   "side_pp", "r_1015", "e1015")))
        lines.append(srow(f"{lab}: long every stock (drift)", signed_stats([r for r in days if flt(r)],
                                                                      all_long, "r_916", "e916")))
    lines.append("")

    # T5 — gap fill rates (IS)
    lines += ["### 1f. Gap fill (prior close touched), IS", "",
          "| |gap| % | n | filled 09:15–10:14 | filled after 09:16 by 10:14 | filled by 15:14 | "
          "mean |gap| % |", "|---|---|---|---|---|---|"]
    for b in _bucket_labels((0, 0.25, 0.5, 1, 2, math.inf)):
        sub = [r for r in days if r["gap_dir"] and _bucket(r["gap_pct"], (0, 0.25, 0.5, 1, 2, math.inf)) == b]
        if not sub:
            continue
        lines.append(f"| {b} | {len(sub)}{_star(len(sub))} | {_p(_mean_b(sub, 'pc_filled_1015'))} | "
                 f"{_p(_mean_b(sub, 'pc_filled_1015_after916'))} | {_p(_mean_b(sub, 'pc_filled_1514'))} | "
                 f"{statistics.fmean(abs(r['gap_pct']) for r in sub):.2f} |")
    lines.append("")
    return "\n".join(lines)


def _mean_b(rows, k):
    return sum(bool(r[k]) for r in rows) / len(rows) if rows else None


def _bucket_labels(edges):
    return [f"{lo:g}–{hi:g}" if hi != math.inf else f"≥ {lo:g}"
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


def _demean(rows: list[dict], key: str, universe: list[dict] | None = None) -> list[dict]:
    """Copy rows with ``<key>_dm`` = return minus the same-day mean over ``universe``."""
    uni = universe if universe is not None else rows
    by = defaultdict(list)
    for r in uni:
        if r.get(key) is not None:
            by[r["day"]].append(r[key])
    mean = {d: sum(v) / len(v) for d, v in by.items()}
    out = []
    for r in rows:
        if r.get(key) is None:
            continue
        rr = dict(r)
        rr[key + "_dm"] = r[key] - mean[r["day"]]
        out.append(rr)
    return out


def bhav_loader(src) -> Callable[[str], dict[date, float]]:
    """Day → official close (bhavcopy), for outcome measurement only."""
    def f(sym: str) -> dict[date, float]:
        return {day_of(b.t): b.c for b in src.daily_bars(sym, src.max_date)}
    return f
