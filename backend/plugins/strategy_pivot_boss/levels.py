"""Pivot levels — floor/CPR/Camarilla, developing levels, higher time frames,
Money Zone (volume-at-price + TPO profile) and virgin-level tracking.

All functions take the *prior* period's H/L/C (spec §1) and are pure.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from .bars import Bar, ist_minute

# ─────────────────────────────────────────────────────────────────────────────
# Floor pivots (standard + expanded)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FloorPivots:
    pp: float
    r1: float
    r2: float
    r3: float
    r4: float
    s1: float
    s2: float
    s3: float
    s4: float

    def resistance(self, n: int) -> float:
        return (self.r1, self.r2, self.r3, self.r4)[n - 1]

    def support(self, n: int) -> float:
        return (self.s1, self.s2, self.s3, self.s4)[n - 1]


def floor_pivots(h: float, l: float, c: float) -> FloorPivots:  # noqa: E741
    rng = h - l
    pp = (h + l + c) / 3.0
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + rng
    s2 = pp - rng
    r3 = r1 + rng
    s3 = s1 - rng
    r4 = r3 + (r2 - r1)
    s4 = s3 - (s1 - s2)
    return FloorPivots(pp, r1, r2, r3, r4, s1, s2, s3, s4)


# ─────────────────────────────────────────────────────────────────────────────
# Central Pivot Range
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CPR:
    pp: float
    tc: float  # top of the range (max of the raw TC/BC)
    bc: float  # bottom of the range (min of the raw TC/BC)
    raw_tc: float  # 2·PP − BC before normalisation (may sit below BC)

    @property
    def width(self) -> float:
        return self.tc - self.bc

    @property
    def width_pct(self) -> float:
        return (self.tc - self.bc) / self.pp * 100.0 if self.pp else 0.0


def cpr(h: float, l: float, c: float) -> CPR:  # noqa: E741
    pp = (h + l + c) / 3.0
    bc = (h + l) / 2.0
    tc = 2 * pp - bc
    return CPR(pp=pp, tc=max(tc, bc), bc=min(tc, bc), raw_tc=tc)


def classify_cpr_width(width_pct: float, narrow: float = 0.25, wide: float = 0.75) -> str:
    """'narrow' | 'wide' | 'normal'. Defaults are the book's futures thresholds —
    pass NSE-calibrated percentile cut-offs instead (see pivot_research B)."""
    if width_pct < narrow:
        return "narrow"
    if width_pct > wide:
        return "wide"
    return "normal"


# ─────────────────────────────────────────────────────────────────────────────
# Camarilla (standard + expanded H5/L5)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Camarilla:
    h1: float
    h2: float
    h3: float
    h4: float
    h5: float
    l1: float
    l2: float
    l3: float
    l4: float
    l5: float

    @property
    def width(self) -> float:
        """Camarilla value band width (H3 − L3 = RANGE·0.55)."""
        return self.h3 - self.l3


def camarilla(h: float, l: float, c: float) -> Camarilla:  # noqa: E741
    rng = h - l
    k = rng * 1.1
    h5 = (h / l) * c if l else c
    l5 = c - (h5 - c)
    return Camarilla(
        h1=c + k / 12, h2=c + k / 6, h3=c + k / 4, h4=c + k / 2, h5=h5,
        l1=c - k / 12, l2=c - k / 6, l3=c - k / 4, l4=c - k / 2, l5=l5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Developing levels (running high/low + current close)
# ─────────────────────────────────────────────────────────────────────────────


def developing_cpr(bars: Sequence[Bar]) -> list[CPR]:
    """CPR computed after each bar from the running session H/L and that bar's
    close. The last element equals the next session's CPR."""
    out: list[CPR] = []
    hi, lo = -math.inf, math.inf
    for b in bars:
        hi, lo = max(hi, b.h), min(lo, b.l)
        out.append(cpr(hi, lo, b.c))
    return out


def developing_camarilla3(bars: Sequence[Bar]) -> list[tuple[float, float]]:
    """(L3, H3) after each bar from running H/L + current close (DC3)."""
    out: list[tuple[float, float]] = []
    hi, lo = -math.inf, math.inf
    for b in bars:
        hi, lo = max(hi, b.h), min(lo, b.l)
        k = (hi - lo) * 1.1 / 4
        out.append((b.c - k, b.c + k))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Higher time frames
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PeriodHLC:
    key: tuple
    first: date
    last: date
    o: float
    h: float
    l: float  # noqa: E741
    c: float


def _period_key(d: date, period: str) -> tuple:
    if period == "week":
        iso = d.isocalendar()
        return (iso[0], iso[1])
    if period == "month":
        return (d.year, d.month)
    if period == "year":
        return (d.year,)
    raise ValueError(f"unknown period {period!r}")


def aggregate_periods(daily: Iterable[tuple[date, float, float, float, float]],
                      period: str) -> list[PeriodHLC]:
    """Roll daily (date, o, h, l, c) rows (ascending) up to week/month/year."""
    out: list[PeriodHLC] = []
    cur = None
    for d, o, h, l, c in daily:  # noqa: E741
        k = _period_key(d, period)
        if cur is None or cur["key"] != k:
            if cur is not None:
                out.append(PeriodHLC(**cur))
            cur = {"key": k, "first": d, "last": d, "o": o, "h": h, "l": l, "c": c}
        else:
            cur["last"] = d
            cur["h"] = max(cur["h"], h)
            cur["l"] = min(cur["l"], l)
            cur["c"] = c
    if cur is not None:
        out.append(PeriodHLC(**cur))
    return out


def prior_period_hlc(daily: Sequence[tuple[date, float, float, float, float]],
                     on_date: date, period: str) -> PeriodHLC | None:
    """H/L/C of the last *completed* week/month/year before ``on_date``'s period."""
    key = _period_key(on_date, period)
    prior = [p for p in aggregate_periods((r for r in daily if r[0] < on_date), period)
             if p.key != key]
    return prior[-1] if prior else None


def higher_tf_pivots(daily, on_date: date, period: str):
    """(FloorPivots, CPR, Camarilla) from the prior week/month/year, or None."""
    p = prior_period_hlc(daily, on_date, period)
    if p is None:
        return None
    return floor_pivots(p.h, p.l, p.c), cpr(p.h, p.l, p.c), camarilla(p.h, p.l, p.c)


# ─────────────────────────────────────────────────────────────────────────────
# Money Zone — volume-at-price / TPO profile, POC/VPOC, 70 % value area
# ─────────────────────────────────────────────────────────────────────────────


def default_bin_size(price: float, tick: float = 0.05, frac: float = 0.0005) -> float:
    """~0.05 % of price rounded to a tick multiple (never below one tick)."""
    n = max(1, round(price * frac / tick))
    return round(n * tick, 6)


def volume_profile(bars: Sequence[Bar], bin_size: float) -> dict[int, float]:
    """Volume at price: each bar's volume spread equally over the bins its
    H–L spans. Keys are bin indices (price ≈ (idx + 0.5)·bin_size)."""
    prof: dict[int, float] = {}
    for b in bars:
        lo_i = math.floor(b.l / bin_size + 1e-9)
        # a high sitting exactly on a bin edge belongs to the bin below it
        hi_i = max(lo_i, math.ceil(b.h / bin_size - 1e-9) - 1)
        n = hi_i - lo_i + 1
        share = (b.v or 0.0) / n
        for i in range(lo_i, hi_i + 1):
            prof[i] = prof.get(i, 0.0) + share
    return prof


def tpo_profile(bars: Sequence[Bar], bin_size: float, period_minutes: int = 30,
                session_open_min: int = 9 * 60 + 15) -> dict[int, float]:
    """Market-profile TPO counts: one TPO per bin touched per 30-min period."""
    periods: dict[int, tuple[float, float]] = {}
    for b in bars:
        p = (ist_minute(b.t) - session_open_min) // period_minutes
        hi, lo = periods.get(p, (-math.inf, math.inf))
        periods[p] = (max(hi, b.h), min(lo, b.l))
    prof: dict[int, float] = {}
    for hi, lo in periods.values():
        lo_i = math.floor(lo / bin_size + 1e-9)
        for i in range(lo_i, max(lo_i, math.ceil(hi / bin_size - 1e-9) - 1) + 1):
            prof[i] = prof.get(i, 0.0) + 1.0
    return prof


def value_area(profile: dict[int, float], va_pct: float = 0.70) -> tuple[int, int, int]:
    """(poc_idx, val_idx, vah_idx). Start at the POC (ties → the one nearest the
    profile's weighted mean, then the lower), add the larger adjacent row each step
    (ties add both) until ``va_pct`` of the total is covered."""
    if not profile:
        raise ValueError("empty profile")
    lo_k, hi_k = min(profile), max(profile)
    total = sum(profile.values())
    mx = max(profile.values())
    mid = sum(k * v for k, v in profile.items()) / total if total else (lo_k + hi_k) / 2
    poc = min((k for k, v in profile.items() if v == mx), key=lambda k: (abs(k - mid), k))
    val = vah = poc
    acc = profile[poc]
    target = va_pct * total
    while acc < target and (val > lo_k or vah < hi_k):
        up = profile.get(vah + 1, 0.0) if vah < hi_k else -1.0
        dn = profile.get(val - 1, 0.0) if val > lo_k else -1.0
        if up > dn:
            vah += 1
            acc += up
        elif dn > up:
            val -= 1
            acc += dn
        else:  # tie: expand both sides that exist
            if vah < hi_k:
                vah += 1
                acc += max(up, 0.0)
            if val > lo_k:
                val -= 1
                acc += max(dn, 0.0)
    return poc, val, vah


@dataclass(frozen=True, slots=True)
class MoneyZone:
    bin_size: float
    vpoc: float  # max-volume price (volume profile)
    poc: float  # max-TPO price (30-min TPO profile)
    vah: float
    val: float
    basis: str  # "volume" | "tpo" — which profile the value area came from

    @property
    def width_pct(self) -> float:
        mid = (self.vah + self.val) / 2
        return (self.vah - self.val) / mid * 100 if mid else 0.0


def money_zone(bars: Sequence[Bar], bin_size: float | None = None,
               va_pct: float = 0.70, basis: str = "volume") -> MoneyZone:
    """Money Zone of one session's intraday bars. Prices are bin centres."""
    if not bars:
        raise ValueError("no bars")
    if bin_size is None:
        bin_size = default_bin_size(bars[0].o)
    vol = volume_profile(bars, bin_size)
    tpo = tpo_profile(bars, bin_size)
    if sum(vol.values()) <= 0:  # indices carry no volume → fall back to TPO
        vol, basis = tpo, "tpo"
    vpoc_i = value_area(vol, va_pct)[0]
    poc_i = value_area(tpo, va_pct)[0]
    _, val_i, vah_i = value_area(vol if basis == "volume" else tpo, va_pct)

    def px(i: int) -> float:
        return (i + 0.5) * bin_size

    return MoneyZone(bin_size=bin_size, vpoc=px(vpoc_i), poc=px(poc_i),
                     vah=px(vah_i), val=px(val_i), basis=basis)


# ─────────────────────────────────────────────────────────────────────────────
# Virgin levels
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class VirginLevel:
    price: float
    kind: str  # e.g. "vpoc", "vah", "val", "poc"
    born: int  # session index the level was computed from
    touched_at: int | None = None


@dataclass
class VirginLevelTracker:
    """A level from session N stays virgin until a later session's H–L spans
    it. Levels older than ``max_age`` sessions (~3 months) are dropped."""

    max_age: int = 63
    levels: list[VirginLevel] = field(default_factory=list)

    def add(self, price: float, kind: str, session: int) -> VirginLevel:
        lv = VirginLevel(price, kind, session)
        self.levels.append(lv)
        return lv

    def update(self, session: int, high: float, low: float) -> list[VirginLevel]:
        """Mark levels touched by ``session``'s range; returns newly touched."""
        hit = []
        for lv in self.levels:
            if lv.touched_at is None and session > lv.born and low <= lv.price <= high:
                lv.touched_at = session
                hit.append(lv)
        return hit

    def virgin(self, session: int) -> list[VirginLevel]:
        return [lv for lv in self.levels
                if lv.touched_at is None and session - lv.born <= self.max_age]
