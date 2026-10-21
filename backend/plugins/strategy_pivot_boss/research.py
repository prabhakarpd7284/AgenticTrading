"""Per-stock-day statistics for re-measuring the book's edges on NSE, plus
pooled aggregation into markdown tables (used by ``manage.py pivot_research``).

Pure: rows are plain dicts, inputs are ``Bar`` lists and floats.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from .bars import Bar, ist_minute
from .levels import CPR, Camarilla, FloorPivots, MoneyZone
from .relationships import classify_open, two_day_relationship

IB_END_MIN = 10 * 60 + 15  # IB = 09:15–10:14 bars
LOW_N = 30

# Book (YM futures, Appendix A) reference numbers
BOOK_A = {
    "pp_touch": 63.0,
    "l1_touch": 73.3, "l2_touch": 38.2, "l3_touch": 17.6, "l4_touch": 5.5,
    "l1_beyond": 24.8, "l2_beyond": 12.1, "l3_beyond": 6.7, "l4_beyond": 3.0,
    "u_l1_touch": 67.2, "u_l2_touch": 63.9, "u_l3_touch": 34.4, "u_l4_touch": 11.5,
    "u_l1_beyond": 27.9, "u_l2_beyond": 23.0, "u_l3_beyond": 16.4, "u_l4_beyond": 6.6,
    "within_r1s1": 53.4, "open_above_close_above": 53.4, "open_below_close_below": 48.8,
    "pp_untouched_days": 37.0, "untouched_close_beyond_l1": 73.8, "touched_close_beyond_l1": 46.6,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-day primitives
# ─────────────────────────────────────────────────────────────────────────────


def first_hit(bars: Sequence[Bar], start: int, up: float | None = None,
              down: float | None = None) -> tuple[str, int]:
    """Scan from ``start``: 'up' (high ≥ up), 'down' (low ≤ down), 'both' when
    one bar hits both (order unknowable on 1m bars), or 'none'."""
    for i in range(max(start, 0), len(bars)):
        b = bars[i]
        u = up is not None and b.h >= up
        d = down is not None and b.l <= down
        if u and d:
            return "both", i
        if u:
            return "up", i
        if d:
            return "down", i
    return "none", -1


def touched(h: float, l: float, level: float) -> bool:  # noqa: E741
    return l <= level <= h


def trend_day(o: float, h: float, l: float, c: float, frac: float = 0.2) -> int:  # noqa: E741
    """+1/−1 when the day opens within ``frac`` of one extreme and closes within
    ``frac`` of the other (strict 'open-to-close' trend day); else 0."""
    rng = h - l
    if rng <= 0:
        return 0
    if o - l <= frac * rng and h - c <= frac * rng:
        return 1
    if h - o <= frac * rng and c - l <= frac * rng:
        return -1
    return 0


def close_at_extreme(h: float, l: float, c: float, frac: float = 0.2) -> int:  # noqa: E741
    rng = h - l
    if rng <= 0:
        return 0
    if h - c <= frac * rng:
        return 1
    if c - l <= frac * rng:
        return -1
    return 0


def ib_stats(bars: Sequence[Bar]) -> dict:
    ib = [b for b in bars if ist_minute(b.t) < IB_END_MIN]
    rest = [b for b in bars if ist_minute(b.t) >= IB_END_MIN]
    if not ib or not rest:
        return {}
    ib_h, ib_l = max(b.h for b in ib), min(b.l for b in ib)
    r_h, r_l = max(b.h for b in rest), min(b.l for b in rest)
    return {"ib_h": ib_h, "ib_l": ib_l, "ib_break_up": r_h > ib_h, "ib_break_dn": r_l < ib_l}


def camarilla_sequences(bars: Sequence[Bar], bars5: Sequence[Bar], cam: Camarilla,
                        day_open: float) -> dict:
    """Section C sequences on one day's 1m (and 5m) bars ≤ 15:14."""
    out: dict = {}
    inside = cam.l3 < day_open < cam.h3
    out["cam_open_inside"] = inside
    if inside:
        # H3: first touch from inside, then race H4 vs L3
        side, k = first_hit(bars, 0, up=cam.h3, down=cam.l3)
        for lvl, sgn in (("h3", "up"), ("l3", "down")):
            res = None
            if side == sgn:
                b = bars[k]
                if (sgn == "up" and b.h >= cam.h4) or (sgn == "down" and b.l <= cam.l4):
                    res = "break"
                else:
                    s2, _ = first_hit(bars, k + 1,
                                      up=cam.h4 if sgn == "up" else cam.h3,
                                      down=cam.l3 if sgn == "up" else cam.l4)
                    res = {"none": "none", "both": "both"}.get(s2) or (
                        ("break" if s2 == "up" else "opposite") if sgn == "up"
                        else ("break" if s2 == "down" else "opposite"))
            out[f"seq_{lvl}"] = res
    # H4/L4 close-beyond (cross) on 1m and on 5m, then race H5 vs H3 (L5 vs L3)
    for tf, series in (("1m", bars), ("5m", bars5)):
        prev_c = day_open
        res_up = res_dn = None
        for j, b in enumerate(series):
            if res_up is None and b.c > cam.h4 and prev_c <= cam.h4:
                res_up = _race_after(bars, b, series is bars, j, ext=cam.h5, back=cam.h3, side=1)
            if res_dn is None and b.c < cam.l4 and prev_c >= cam.l4:
                res_dn = _race_after(bars, b, series is bars, j, ext=cam.l5, back=cam.l3, side=-1)
            prev_c = b.c
        out[f"h4brk_{tf}"] = res_up
        out[f"l4brk_{tf}"] = res_dn
    return out


def _race_after(bars1m: Sequence[Bar], bar: Bar, is_1m: bool, j: int, ext: float,
                back: float, side: int) -> str:
    """After a close beyond H4 (side=1) / L4 (side=−1): 'ext' if the 5-level is
    reached first, 'back' if the 3-level is, 'both' (same 1m bar) or 'none'."""
    if (side > 0 and bar.h >= ext) or (side < 0 and bar.l <= ext):
        return "ext"
    if is_1m:
        start = j + 1
    else:
        end_min = ist_minute(bar.t) + 5
        start = next((i for i, b in enumerate(bars1m) if ist_minute(b.t) >= end_min), len(bars1m))
    up, down = (ext, back) if side > 0 else (back, ext)
    res = first_hit(bars1m, start, up=up, down=down)[0]
    if res in ("none", "both"):
        return res
    return "ext" if (res == "up") == (side > 0) else "back"


def gap_fill(bars: Sequence[Bar], prev_close: float, cpr: CPR) -> dict:
    """Did the session reach the prior close / the CPR by 10:15 and by 15:14?"""
    o = bars[0].o
    out = {"gap_pct": (o - prev_close) / prev_close * 100}
    early = [b for b in bars if ist_minute(b.t) < IB_END_MIN]
    lo_e, hi_e = min(b.l for b in early), max(b.h for b in early)
    lo, hi = min(b.l for b in bars), max(b.h for b in bars)
    if o > prev_close:
        out["pc_fill_1015"], out["pc_fill_eod"] = lo_e <= prev_close, lo <= prev_close
    elif o < prev_close:
        out["pc_fill_1015"], out["pc_fill_eod"] = hi_e >= prev_close, hi >= prev_close
    if o > cpr.tc:
        out["open_vs_cpr"] = "above"
        out["cpr_fill_1015"], out["cpr_fill_eod"] = lo_e <= cpr.tc, lo <= cpr.tc
    elif o < cpr.bc:
        out["open_vs_cpr"] = "below"
        out["cpr_fill_1015"], out["cpr_fill_eod"] = hi_e >= cpr.bc, hi >= cpr.bc
    else:
        out["open_vs_cpr"] = "inside"
    return out


def floor_row(o: float, h: float, l: float, c: float, f: FloorPivots, cp: CPR) -> dict:  # noqa: E741
    r = {"pp_touch": touched(h, l, f.pp), "cpr_touch": l <= cp.tc and h >= cp.bc}
    for n in range(1, 5):
        # "touched" = the level was reached (a gap straight through it counts)
        r[f"l{n}_touch"] = h >= f.resistance(n) or l <= f.support(n)
        r[f"l{n}_beyond"] = c > f.resistance(n) or c < f.support(n)
        r[f"r{n}_touch"], r[f"s{n}_touch"] = h >= f.resistance(n), l <= f.support(n)
        r[f"r{n}_beyond"], r[f"s{n}_beyond"] = c > f.resistance(n), c < f.support(n)
    r["within_r1s1"] = f.s1 <= c <= f.r1
    r["open_above_pp"] = o > f.pp
    r["close_above_pp"] = c > f.pp
    return r


def day_relationships(cp: CPR, cp_prev: CPR | None, cam: Camarilla, cam_prev: Camarilla | None,
                      mz_prev: MoneyZone | None, mz_prev2: MoneyZone | None,
                      tol_pct: float) -> dict:
    r: dict = {}
    if cp_prev:
        r["cpr_rel"] = two_day_relationship(cp.bc, cp.tc, cp_prev.bc, cp_prev.tc, tol_pct).value
    if cam_prev:
        r["cam_rel"] = two_day_relationship(cam.l3, cam.h3, cam_prev.l3, cam_prev.h3, tol_pct).value
    if mz_prev and mz_prev2:
        r["va_rel"] = two_day_relationship(mz_prev.val, mz_prev.vah, mz_prev2.val,
                                           mz_prev2.vah, tol_pct).value
    return r


def open_location(o: float, prev_h: float, prev_l: float, mz_prev: MoneyZone | None) -> str | None:
    if mz_prev is None:
        return None
    return classify_open(o, prev_h, prev_l, mz_prev.val, mz_prev.vah).category


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation → markdown
# ─────────────────────────────────────────────────────────────────────────────


def pct(rows: Iterable[dict], cond: Callable[[dict], bool]) -> tuple[float | None, int]:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return None, 0
    return 100.0 * sum(1 for r in rows if cond(r)) / n, n


def fmt_pct(v: float | None, n: int) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}%" + (" *" if n < LOW_N else "")


def percentile(vals: Sequence[float], q: float) -> float:
    s = sorted(vals)
    if not s:
        return math.nan
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def md_table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _side_avg(rows: list[dict], kind: str, k: int) -> str:
    """Per-side rate: mean of the R-side and S-side percentages (R% / S%)."""
    r, n = pct(rows, lambda x: x[f"r{k}_{kind}"])
    s, _ = pct(rows, lambda x: x[f"s{k}_{kind}"])
    if r is None:
        return "—"
    return f"{(r + s) / 2:.1f}% ({r:.1f} / {s:.1f})" + (" *" if n < LOW_N else "")


def section_a(rows: list[dict]) -> str:
    """Book numbers vs NSE. Each L-row shows the either-side rate and the
    per-side average (R% / S%) — the book's table is ambiguous between them."""
    unt = [r for r in rows if not r["pp_touch"]]
    tch = [r for r in rows if r["pp_touch"]]
    lines = []

    def add(label, book, sub, cond, side=None):
        v, n = pct(sub, cond)
        lines.append([label, book, fmt_pct(v, n), _side_avg(sub, *side) if side else "—", n])

    add("PP touched (all days)", f"{BOOK_A['pp_touch']}%", rows, lambda r: r["pp_touch"])
    add("CPR band (BC–TC) touched", "—", rows, lambda r: r["cpr_touch"])
    for k in range(1, 5):
        add(f"L{k} (R{k}/S{k}) reached", f"{BOOK_A[f'l{k}_touch']}%", rows,
            lambda r, k=k: r[f"l{k}_touch"], ("touch", k))
    for k in range(1, 5):
        add(f"Close beyond L{k}", f"{BOOK_A[f'l{k}_beyond']}%", rows,
            lambda r, k=k: r[f"l{k}_beyond"], ("beyond", k))
    add("PP untouched days", f"{BOOK_A['pp_untouched_days']}%", rows, lambda r: not r["pp_touch"])
    for k in range(1, 5):
        add(f"L{k} reached, PP untouched", f"{BOOK_A[f'u_l{k}_touch']}%", unt,
            lambda r, k=k: r[f"l{k}_touch"], ("touch", k))
    for k in range(1, 5):
        add(f"Close beyond L{k}, PP untouched", f"{BOOK_A[f'u_l{k}_beyond']}%", unt,
            lambda r, k=k: r[f"l{k}_beyond"], ("beyond", k))
    add("Close beyond L1, PP untouched (prose)", f"{BOOK_A['untouched_close_beyond_l1']}%", unt,
        lambda r: r["l1_beyond"])
    add("Close beyond L1, PP touched (prose)", f"{BOOK_A['touched_close_beyond_l1']}%", tch,
        lambda r: r["l1_beyond"])
    add("Close within S1–R1", f"{BOOK_A['within_r1s1']}%", rows, lambda r: r["within_r1s1"])
    add("Open > PP → close > PP", f"{BOOK_A['open_above_close_above']}%",
        [r for r in rows if r["open_above_pp"]], lambda r: r["close_above_pp"])
    add("Open < PP → close < PP", f"{BOOK_A['open_below_close_below']}%",
        [r for r in rows if not r["open_above_pp"]], lambda r: not r["close_above_pp"])
    return md_table(["Statistic", "Book (YM 08–09)", "NSE, either side",
                     "NSE, per-side avg (R / S)", "n"], lines)


def width_buckets(rows: list[dict], edges: Sequence[float], labels: Sequence[str]) -> str:
    out = []
    for i, lab in enumerate(labels):
        lo = edges[i]
        hi = edges[i + 1]
        b = [r for r in rows if lo <= r["cpr_width_pct"] < hi]
        n = len(b)
        if n == 0:
            out.append([lab, 0, "—", "—", "—", "—", "—", "—"])
            continue
        ra = [r["range_atr"] for r in b if r.get("range_atr") is not None]
        t1, _ = pct(b, lambda r: r["trend_day"] != 0)
        t2, _ = pct([r for r in b if r.get("range_atr") is not None], lambda r: r["trend_day2"] != 0)
        ibb = [r for r in b if "ib_break_up" in r]
        e, ne = pct(ibb, lambda r: r["ib_break_up"] or r["ib_break_dn"])
        bo, _ = pct(ibb, lambda r: r["ib_break_up"] and r["ib_break_dn"])
        out.append([lab, n, f"{sum(ra)/len(ra):.2f}" if ra else "—",
                    f"{percentile(ra, 50):.2f}" if ra else "—",
                    fmt_pct(t1, n), fmt_pct(t2, len(ra)), fmt_pct(e, ne), fmt_pct(bo, ne)])
    return md_table(["CPR width bucket", "n", "mean range/ATR14", "median range/ATR14",
                     "trend day (open/close at opposite 20% extremes)",
                     "trend day (close in 20% extreme & range ≥ ATR14)",
                     "IB broken (either side)", "IB broken both sides"], out)
