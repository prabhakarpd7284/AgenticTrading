"""Compile a validated StrategySpec's condition tree into a per-bar evaluator.

``SeriesFrame`` holds one symbol's bars at the spec timeframe (plus session
ids, minute-of-day and the prior session's official H/L/C per bar).
``CompiledSpec.entry_masks(frame)`` returns two boolean lists (long, short):
``mask[i]`` is decided from bars ``0..i`` only. The same compiled object is
what a live/paper evaluator will call on completed bars (Studio plan §3:
one evaluator for backtest and live).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from apps.strategies.services import indicators as ind


@dataclass
class SeriesFrame:
    symbol: str
    timestamps: list[str]               # ISO IST, bar START
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    minute: list[int]                   # IST minute-of-day of bar start (0 for daily)
    session: list[str]                  # 'YYYY-MM-DD'
    prev_day: list[tuple | None]        # prior session (h, l, c) per bar, or None
    bar_minutes: int                    # 0 = daily
    _cache: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.close)

    def price(self, name: str) -> list[float]:
        return getattr(self, name)


def _key(d: dict) -> str:
    return json.dumps(d, sort_keys=True)


def indicator_series(frame: SeriesFrame, spec_ind: dict) -> list:
    """Series for one normalised indicator dict (kind + params)."""
    k = _key(spec_ind)
    if k in frame._cache:
        return frame._cache[k]
    kind = spec_ind["kind"]
    p = spec_ind.get("period")
    if kind == "ema":
        s = ind.ema(frame.price(spec_ind["source"]), p)
    elif kind == "sma":
        s = ind.sma(frame.price(spec_ind["source"]), p)
    elif kind == "rsi":
        s = ind.rsi(frame.price(spec_ind["source"]), p)
    elif kind == "atr":
        s = ind.atr(frame.high, frame.low, frame.close, p)
    elif kind in ("bb_upper", "bb_middle", "bb_lower"):
        up, mid, lo = ind.bollinger(frame.close, p, float(spec_ind["std"]))
        s = {"bb_upper": up, "bb_middle": mid, "bb_lower": lo}[kind]
    elif kind == "highest":
        s = ind.highest_before(frame.price(spec_ind["source"]), p)
    elif kind == "lowest":
        s = ind.lowest_before(frame.price(spec_ind["source"]), p)
    elif kind == "avg":
        s = ind.avg_before(frame.price(spec_ind["source"]), p)
    elif kind in ("prev_day_high", "prev_day_low", "prev_day_close"):
        j = {"prev_day_high": 0, "prev_day_low": 1, "prev_day_close": 2}[kind]
        s = [pd[j] if pd else None for pd in frame.prev_day]
    elif kind == "vwap":
        s = ind.session_vwap(frame.high, frame.low, frame.close, frame.volume, frame.session)
    elif kind == "day_open":
        s, cur, o = [], None, None
        for i in range(len(frame)):
            if frame.session[i] != cur:
                cur, o = frame.session[i], frame.open[i]
            s.append(o)
    elif kind == "gap_pct":
        opens = indicator_series(frame, {"kind": "day_open"})
        s = [((o - pd[2]) / pd[2] * 100) if (o is not None and pd and pd[2]) else None
             for o, pd in zip(opens, frame.prev_day)]
    elif kind in ("orb_high", "orb_low"):
        hi, lo = ind.opening_range(frame.high, frame.low, frame.minute, frame.session,
                                   frame.bar_minutes, int(spec_ind["minutes"]))
        s = hi if kind == "orb_high" else lo
    else:  # validated upstream; defensive
        raise ValueError(f"unknown indicator {kind}")
    frame._cache[k] = s
    return s


class CompiledSpec:
    def __init__(self, spec: dict):
        self.spec = spec
        self.named = {i["id"]: {k: v for k, v in i.items() if k != "id"}
                      for i in spec.get("indicators") or []}
        e = spec["entry"]
        self.long_node = e.get("long")
        self.short_node = e.get("short")
        self.daily = spec["timeframe"] == "1d"
        self.weekdays = set(e.get("weekdays") or [0, 1, 2, 3, 4])
        self.time_from = _hhmm(e.get("time_from")) if not self.daily else None
        self.time_to = _hhmm(e.get("time_to")) if not self.daily else None

    # ── operands / nodes ────────────────────────────────────────────────
    def operand(self, frame: SeriesFrame, op: dict) -> list:
        n = len(frame)
        if "value" in op:
            return [op["value"]] * n
        if "price" in op:
            base = frame.price(op["price"])
        elif "ref" in op:
            base = indicator_series(frame, self.named[op["ref"]])
        else:
            spec_ind = {k: v for k, v in op.items() if k not in ("ind", "bars_ago", "mult")}
            spec_ind["kind"] = op["ind"]
            base = indicator_series(frame, spec_ind)
        ba = op.get("bars_ago", 0)
        if ba:
            base = [None] * min(ba, n) + list(base[:n - ba])
        m = op.get("mult")
        if m:
            base = [None if v is None else v * m for v in base]
        return base

    def evaluate(self, frame: SeriesFrame, node: dict) -> list[bool]:
        n = len(frame)
        if "all" in node:
            out = [True] * n
            for k in node["all"]:
                out = [a and b for a, b in zip(out, self.evaluate(frame, k))]
            return out
        if "any" in node:
            out = [False] * n
            for k in node["any"]:
                out = [a or b for a, b in zip(out, self.evaluate(frame, k))]
            return out
        if "not" in node:
            # NOT of an undefined (warming-up) comparison stays False.
            inner = self.evaluate(frame, node["not"])
            defined = self._defined(frame, node["not"])
            return [d and not v for v, d in zip(inner, defined)]
        op = node["op"]
        a = self.operand(frame, node["left"])
        if op in ("rising", "falling"):
            k = node.get("bars", 1)
            return [i >= k and a[i] is not None and a[i - k] is not None
                    and (a[i] > a[i - k] if op == "rising" else a[i] < a[i - k]) for i in range(n)]
        b = self.operand(frame, node["right"])
        if op in (">", ">=", "<", "<="):
            f = {">": lambda x, y: x > y, ">=": lambda x, y: x >= y,
                 "<": lambda x, y: x < y, "<=": lambda x, y: x <= y}[op]
            return [x is not None and y is not None and f(x, y) for x, y in zip(a, b)]
        up = op == "crosses_above"
        out = [False] * n
        for i in range(1, n):
            if None in (a[i], b[i], a[i - 1], b[i - 1]):
                continue
            out[i] = (a[i] > b[i] and a[i - 1] <= b[i - 1]) if up else (a[i] < b[i] and a[i - 1] >= b[i - 1])
        return out

    def _defined(self, frame, node) -> list[bool]:
        n = len(frame)
        if "all" in node or "any" in node:
            kids = node.get("all") or node.get("any")
            out = [True] * n
            for k in kids:
                out = [x and y for x, y in zip(out, self._defined(frame, k))]
            return out
        if "not" in node:
            return self._defined(frame, node["not"])
        a = self.operand(frame, node["left"])
        b = self.operand(frame, node["right"]) if "right" in node else a
        return [x is not None and y is not None for x, y in zip(a, b)]

    # ── masks ───────────────────────────────────────────────────────────
    def entry_masks(self, frame: SeriesFrame) -> tuple[list[bool], list[bool]]:
        n = len(frame)
        lm = self.evaluate(frame, self.long_node) if self.long_node else [False] * n
        sm = self.evaluate(frame, self.short_node) if self.short_node else [False] * n
        allow = self._fill_allowed(frame)
        return [x and a for x, a in zip(lm, allow)], [x and a for x, a in zip(sm, allow)]

    def _fill_allowed(self, frame: SeriesFrame) -> list[bool]:
        """A signal on bar i fills on bar i+1: that bar must exist (same
        session intraday), fall inside the entry window and on an allowed weekday."""
        n = len(frame)
        out = [False] * n
        for i in range(n - 1):
            j = i + 1
            if not self.daily and frame.session[j] != frame.session[i]:
                continue
            if date.fromisoformat(frame.session[j]).weekday() not in self.weekdays:
                continue
            if not self.daily and not (self.time_from <= frame.minute[j] <= self.time_to):
                continue
            out[i] = True
        return out


def _hhmm(s) -> int:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)
