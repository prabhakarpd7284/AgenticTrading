"""The four Appendix-B entry triggers (spec §3) + a 1m → N-minute resampler.

Each ``*_series`` function returns one int per bar: +1 long trigger on that
bar's close, −1 short trigger, 0 none. Averages are simple moving averages of
the ``lookback`` bars *ending at* the evaluated bar (TradeStation
``Average(Range, 50)`` semantics); no trigger fires until a full lookback of
history exists, so feed a continuous multi-session series.

Interpretation notes (the spec's one-line summaries leave a few gaps):
- Wick reversal: body > 0 required for the wick ≥ 2.5×body clause; a body-less
  bar (O == C) sitting at its extreme (close in the top/bottom 25 %) only counts
  when its range ≥ the 50-bar average range — the "full-range bar at extreme"
  clause. Zero-range bars never trigger.
- Doji reversal: the doji must be entirely above (short) / below (long) the
  SMA(10) of closes at the doji bar; the confirming bar is the first of the next
  two bars that closes beyond the doji's low/high *and* is red/green.
"""
from __future__ import annotations

from collections.abc import Sequence

from .bars import SESSION_OPEN_MIN, Bar, ist_day, ist_minute

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rolling_mean(values: Sequence[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= n:
            s -= values[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Triggers
# ─────────────────────────────────────────────────────────────────────────────


def wick_reversal_series(bars: Sequence[Bar], wick_mult: float = 2.5,
                         close_pct: float = 0.25, lookback: int = 50) -> list[int]:
    avg_rng = _rolling_mean([b.h - b.l for b in bars], lookback)
    out = [0] * len(bars)
    for i, b in enumerate(bars):
        rng = b.h - b.l
        if rng <= 0 or avg_rng[i] is None:
            continue
        body = abs(b.c - b.o)
        top, bot = max(b.o, b.c), min(b.o, b.c)
        lower_wick, upper_wick = bot - b.l, b.h - top
        close_top = (b.h - b.c) <= close_pct * rng
        close_bot = (b.c - b.l) <= close_pct * rng
        if body > 0:
            if lower_wick >= wick_mult * body and close_top:
                out[i] = 1
            elif upper_wick >= wick_mult * body and close_bot:
                out[i] = -1
        elif rng >= avg_rng[i]:  # body-less bar at an extreme, full-size range
            if close_top:
                out[i] = 1
            elif close_bot:
                out[i] = -1
    return out


def extreme_reversal_series(bars: Sequence[Bar], body_pct: float = 0.525,
                            lookback: int = 50, range_mult: float = 2.0) -> list[int]:
    rngs = [b.h - b.l for b in bars]
    bodies = [abs(b.c - b.o) for b in bars]
    avg_rng = _rolling_mean(rngs, lookback)
    avg_body = _rolling_mean(bodies, lookback)
    out = [0] * len(bars)
    for i in range(1, len(bars)):
        if avg_rng[i] is None:
            continue
        p, b = bars[i - 1], bars[i]
        pr, pb = rngs[i - 1], bodies[i - 1]
        if pr <= 0 or pb < body_pct * pr or pr <= range_mult * avg_rng[i] or pb <= avg_body[i]:
            continue
        if p.c < p.o and b.c > b.o:  # big red bar then green → long
            out[i] = 1
        elif p.c > p.o and b.c < b.o:  # big green bar then red → short
            out[i] = -1
    return out


def outside_reversal_series(bars: Sequence[Bar], range_mult: float = 1.25,
                            lookback: int = 50) -> list[int]:
    avg_rng = _rolling_mean([b.h - b.l for b in bars], lookback)
    out = [0] * len(bars)
    for i in range(1, len(bars)):
        if avg_rng[i] is None:
            continue
        p, b = bars[i - 1], bars[i]
        if (b.h - b.l) < range_mult * avg_rng[i]:
            continue
        if b.l < p.l and b.c > p.h:
            out[i] = 1
        elif b.h > p.h and b.c < p.l:
            out[i] = -1
    return out


def doji_reversal_series(bars: Sequence[Bar], doji_pct: float = 0.10,
                         sma_len: int = 10, confirm_within: int = 2) -> list[int]:
    sma = _rolling_mean([b.c for b in bars], sma_len)
    out = [0] * len(bars)
    for i in range(len(bars)):
        for k in range(1, confirm_within + 1):
            j = i - k  # candidate doji
            if j < 0 or sma[j] is None:
                continue
            d = bars[j]
            rng = d.h - d.l
            if rng <= 0 or abs(d.c - d.o) > doji_pct * rng:
                continue
            # the confirm must be the FIRST qualifying bar after the doji
            b = bars[i]
            if d.l > sma[j] and b.c < d.l and b.c < b.o:
                if not any(bars[m].c < d.l and bars[m].c < bars[m].o for m in range(j + 1, i)):
                    out[i] = -1
                    break
            if d.h < sma[j] and b.c > d.h and b.c > b.o:
                if not any(bars[m].c > d.h and bars[m].c > bars[m].o for m in range(j + 1, i)):
                    out[i] = 1
                    break
    return out


def any_trigger_series(bars: Sequence[Bar]) -> list[int]:
    """First non-zero of wick, extreme, outside, doji (per bar)."""
    series = (wick_reversal_series(bars), extreme_reversal_series(bars),
              outside_reversal_series(bars), doji_reversal_series(bars))
    out = [0] * len(bars)
    for i in range(len(bars)):
        for s in series:
            if s[i]:
                out[i] = s[i]
                break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Resampler
# ─────────────────────────────────────────────────────────────────────────────


def resample(bars_1m: Sequence[Bar], minutes: int,
             session_open_min: int = SESSION_OPEN_MIN) -> list[Bar]:
    """Aggregate 1-minute bars into N-minute bars aligned to the 09:15 open.

    Buckets never span sessions. Missing minutes are simply absent (no
    synthetic bars); a partial final bucket is emitted as-is. ``t`` of an output
    bar = bucket start time.
    """
    out: list[Bar] = []
    cur_key = None
    for b in bars_1m:
        m = ist_minute(b.t)
        bucket = (m - session_open_min) // minutes
        key = (ist_day(b.t), bucket)
        if key != cur_key:
            start_t = b.t - (m - (session_open_min + bucket * minutes)) * 60
            out.append(Bar(start_t, b.o, b.h, b.l, b.c, b.v))
            cur_key = key
        else:
            x = out[-1]
            x.h = max(x.h, b.h)
            x.l = min(x.l, b.l)
            x.c = b.c
            x.v += b.v
    return out
