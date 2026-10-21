"""Deterministic in-memory candles for Strategy Studio tests (no DB, no broker)."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

from plugins.strategy_pivot_boss.bars import Bar as PBar
from plugins.strategy_search.data import epoch_of_ist

from apps.strategies.services.candles import LoadedSymbol, frame_from_bars
from apps.strategies.services.spec import TIMEFRAMES


def sessions(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_bars(symbol: str, days: list[date], minutes: int, base: float = 1000.0, seed: int = 0):
    """Wavy intraday path: a daily drift plus a sine wave, so crosses happen."""
    bars = []
    px = base + seed * 37
    for k, d in enumerate(days):
        drift = 0.002 if (k + seed) % 3 else -0.003
        n = math.ceil(360 / minutes) if minutes else 1
        for j in range(n):
            if minutes:
                t = epoch_of_ist(datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=j * minutes))
            else:
                t = epoch_of_ist(datetime(d.year, d.month, d.day))
            o = px
            wave = math.sin((j + seed) / 4.0) * px * 0.002
            c = px * (1 + drift / n) + wave * 0.5
            h, l = max(o, c) * 1.0008, min(o, c) * 0.9992
            bars.append(PBar(t, round(o, 2), round(h, 2), round(l, 2), round(c, 2), 1000 + 50 * (j % 7)))
            px = c
    return bars


class FakeLoader:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def source_label(self, timeframe):
        return "fake candles (tests)"

    def load(self, symbol, timeframe, start, end, warmup_bars=60, keep_raw=False):
        self.calls.append((symbol, timeframe, start, end))
        if symbol in self.missing:
            return None
        days = sessions(start - timedelta(days=7), end)
        minutes = TIMEFRAMES[timeframe]
        bars = make_bars(symbol, days, minutes, seed=sum(map(ord, symbol)) % 5)
        prev = {}
        last = None
        for d in days:
            prev[d.isoformat()] = last
            day_bars = [b for b in bars if _day(b.t) == d.isoformat()]
            if day_bars:
                last = (max(b.h for b in day_bars), min(b.l for b in day_bars), day_bars[-1].c)
        frame, ebars = frame_from_bars(symbol, bars, timeframe, prev)
        in_range = [d.isoformat() for d in days if start <= d <= end]
        stats = {"bars": sum(1 for s in frame.session if s in set(in_range)), "sessions": len(in_range),
                 "first": in_range[0] if in_range else None, "last": in_range[-1] if in_range else None,
                 "gap_sessions": 0, "gap_days": []}
        return LoadedSymbol(symbol, frame, ebars, stats)


def _day(t: int) -> str:
    from plugins.strategy_search.data import day_of
    return day_of(t).isoformat()


def simple_spec(**over) -> dict:
    spec = {
        "meta": {"name": "test ema"},
        "universe": {"symbols": ["AAA", "BBB"]},
        "timeframe": "5m",
        "entry": {"long": {"left": {"ind": "ema", "source": "close", "period": 5}, "op": "crosses_above",
                           "right": {"ind": "ema", "source": "close", "period": 20}},
                  "short": None, "time_from": "09:30", "time_to": "14:30"},
        "exits": {"stop": {"kind": "atr", "value": 1.0, "atr_period": 14},
                  "target": {"kind": "r", "value": 2.0}, "trail": {"kind": "none"}},
        "limits": {"owner_cooldown": False},
    }
    spec.update(over)
    return spec
