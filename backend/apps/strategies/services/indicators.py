"""Causal indicator series for the Strategy Studio rule evaluator.

Every function returns a list aligned with its input; element ``i`` uses only
bars ``0..i`` (no look-ahead) and is ``None`` until the indicator has warmed
up. The formulas match ``trading.utils.indicators`` (the library the live
screener's ``IndicatorEngine`` uses): SMA-seeded EMA, Wilder RSI, Wilder ATR,
population-std Bollinger Bands, session VWAP on typical price. The screener
engine itself recomputes fixed-period snapshots on every bar close, so it
can't serve configurable periods over a whole series — these are the same
maths as series (equivalence is unit-tested).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

Series = list  # list[float | None]


def sma(values: Sequence[float], period: int) -> Series:
    out: Series = [None] * len(values)
    if period <= 0:
        return out
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out


def ema(values: Sequence[float], period: int) -> Series:
    out: Series = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    k = 2 / (period + 1)
    cur = sum(values[:period]) / period
    out[period - 1] = cur
    for i in range(period, len(values)):
        cur = (values[i] - cur) * k + cur
        out[i] = cur
    return out


def rsi(closes: Sequence[float], period: int = 14) -> Series:
    n = len(closes)
    out: Series = [None] * n
    if n < period + 1:
        return out
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    ag = sum(max(0.0, d) for d in deltas[:period]) / period
    al = sum(max(0.0, -d) for d in deltas[:period]) / period

    def val(g, l_):
        return 100.0 if l_ == 0 else 100 - 100 / (1 + g / l_)

    out[period] = val(ag, al)
    for j, d in enumerate(deltas[period:], start=period + 1):
        ag = (ag * (period - 1) + max(0.0, d)) / period
        al = (al * (period - 1) + max(0.0, -d)) / period
        out[j] = val(ag, al)
    return out


def true_range(highs, lows, closes) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    for i in range(1, len(closes)):
        pc = closes[i - 1]
        out[i] = max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc))
    return out


def atr(highs, lows, closes, period: int = 14) -> Series:
    n = len(closes)
    out: Series = [None] * n
    if n < period + 1:
        return out
    tr = true_range(highs, lows, closes)
    cur = sum(tr[1:period + 1]) / period
    out[period] = cur
    for i in range(period + 1, n):
        cur = (cur * (period - 1) + tr[i]) / period
        out[i] = cur
    return out


def bollinger(closes: Sequence[float], period: int = 20, num_std: float = 2.0
              ) -> tuple[Series, Series, Series]:
    n = len(closes)
    up: Series = [None] * n
    mid: Series = [None] * n
    lo: Series = [None] * n
    for i in range(period - 1, n):
        w = closes[i - period + 1:i + 1]
        m = sum(w) / period
        sd = math.sqrt(sum((x - m) ** 2 for x in w) / period)
        mid[i], up[i], lo[i] = m, m + num_std * sd, m - num_std * sd
    return up, mid, lo


def highest_before(values: Sequence[float], period: int) -> Series:
    """Highest of the ``period`` bars BEFORE bar i (current bar excluded), so
    ``close > highest_before(high, 20)`` is a 20-bar breakout."""
    out: Series = [None] * len(values)
    for i in range(period, len(values)):
        out[i] = max(values[i - period:i])
    return out


def lowest_before(values: Sequence[float], period: int) -> Series:
    out: Series = [None] * len(values)
    for i in range(period, len(values)):
        out[i] = min(values[i - period:i])
    return out


def avg_before(values: Sequence[float], period: int) -> Series:
    """Mean of the ``period`` bars before bar i (e.g. average volume)."""
    out: Series = [None] * len(values)
    s = 0.0
    for i in range(len(values)):
        if i >= period:
            out[i] = s / period
        s += values[i]
        if i >= period:
            s -= values[i - period]
    return out


def session_vwap(highs, lows, closes, volumes, session_ids: Sequence) -> Series:
    """Cumulative typical-price VWAP reset at each session; None while the
    session has no volume (indices carry zero volume)."""
    out: Series = [None] * len(closes)
    cur_sess = object()
    pv = vol = 0.0
    for i in range(len(closes)):
        if session_ids[i] != cur_sess:
            cur_sess, pv, vol = session_ids[i], 0.0, 0.0
        tp = (highs[i] + lows[i] + closes[i]) / 3
        pv += tp * volumes[i]
        vol += volumes[i]
        out[i] = pv / vol if vol > 0 else None
    return out


def opening_range(highs, lows, minutes_of_day: Sequence[int], session_ids: Sequence,
                  bar_minutes: int, range_minutes: int, open_minute: int = 9 * 60 + 15
                  ) -> tuple[Series, Series]:
    """Session opening-range high/low over the first ``range_minutes``. Defined
    from the bar whose CLOSE completes the range onward (never earlier)."""
    n = len(highs)
    hi: Series = [None] * n
    lo: Series = [None] * n
    end = open_minute + range_minutes
    cur, rh, rl = object(), None, None
    for i in range(n):
        if session_ids[i] != cur:
            cur, rh, rl = session_ids[i], None, None
        m = minutes_of_day[i]
        if m < end:
            rh = highs[i] if rh is None else max(rh, highs[i])
            rl = lows[i] if rl is None else min(rl, lows[i])
        if m + bar_minutes >= end and rh is not None:
            hi[i], lo[i] = rh, rl
    return hi, lo
