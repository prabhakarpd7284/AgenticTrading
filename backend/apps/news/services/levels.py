"""Next-session pivot levels for the nightly scan — thin adapter.

Prefers the Pivot Boss toolkit (``plugins.strategy_pivot_boss.levels`` /
``.relationships``); if that module is missing or its API changes, falls
back to small local formulas (same book definitions). ``PIVOT_SOURCE``
says which path is active.

Inputs: daily H/L/C of the last two completed sessions (``Candle`` "1d"
from the NSE bhavcopy; indices aggregated from "1m" when no daily row).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from django.conf import settings
from django.utils import timezone

try:  # pragma: no cover - exercised implicitly
    from plugins.strategy_pivot_boss import levels as _pb_levels
    from plugins.strategy_pivot_boss import relationships as _pb_rel

    _pb_levels.floor_pivots, _pb_levels.cpr, _pb_levels.camarilla  # noqa: B018
    _pb_rel.two_day_relationship, _pb_rel.BIAS  # noqa: B018
    PIVOT_SOURCE = "pivot_boss"
except Exception:  # noqa: BLE001
    _pb_levels = _pb_rel = None
    PIVOT_SOURCE = "local"


@dataclass
class HLC:
    day: date
    h: float
    l: float  # noqa: E741
    c: float


# ── Local fallbacks (book formulas) ──────────────────────────────────────

def _floor_local(h, l, c):  # noqa: E741
    pp = (h + l + c) / 3
    rng = h - l
    r1, s1 = 2 * pp - l, 2 * pp - h
    r2, s2 = pp + rng, pp - rng
    r3, s3 = r1 + rng, s1 - rng
    return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "r4": r3 + (r2 - r1),
            "s1": s1, "s2": s2, "s3": s3, "s4": s3 - (s1 - s2)}


def _cpr_local(h, l, c):  # noqa: E741
    pp = (h + l + c) / 3
    bc = (h + l) / 2
    tc = 2 * pp - bc
    return {"pp": pp, "tc": max(tc, bc), "bc": min(tc, bc)}


def _camarilla_local(h, l, c):  # noqa: E741
    k = (h - l) * 1.1
    h5 = (h / l) * c if l else c
    return {"h1": c + k / 12, "h2": c + k / 6, "h3": c + k / 4, "h4": c + k / 2, "h5": h5,
            "l1": c - k / 12, "l2": c - k / 6, "l3": c - k / 4, "l4": c - k / 2,
            "l5": c - (h5 - c)}


_LOCAL_BIAS = {
    "higher": "bullish", "overlapping_higher": "moderately_bullish",
    "lower": "bearish", "overlapping_lower": "moderately_bearish",
    "unchanged": "sideways_breakout", "outside": "sideways", "inside": "breakout",
}


def _relationship_local(t_lo, t_hi, y_lo, y_hi, tol_pct=0.02) -> str:
    tol = tol_pct / 100 * (y_lo + y_hi) / 2
    if abs(t_hi - y_hi) <= tol and abs(t_lo - y_lo) <= tol:
        return "unchanged"
    if t_lo > y_hi:
        return "higher"
    if t_hi < y_lo:
        return "lower"
    if t_hi <= y_hi and t_lo >= y_lo:
        return "inside"
    if t_hi >= y_hi and t_lo <= y_lo:
        return "outside"
    return "overlapping_higher" if t_hi > y_hi else "overlapping_lower"


# ── Adapter ──────────────────────────────────────────────────────────────

def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    fields = getattr(obj, "__dataclass_fields__", None) or {}
    return {k: getattr(obj, k) for k in fields}


def floor_pivots(h, l, c) -> dict:  # noqa: E741
    if _pb_levels is not None:
        try:
            return _as_dict(_pb_levels.floor_pivots(h, l, c))
        except Exception:  # noqa: BLE001
            pass
    return _floor_local(h, l, c)


def cpr(h, l, c) -> dict:  # noqa: E741
    d = None
    if _pb_levels is not None:
        try:
            d = _as_dict(_pb_levels.cpr(h, l, c))
            d = {"pp": d["pp"], "tc": d["tc"], "bc": d["bc"]}
        except Exception:  # noqa: BLE001
            d = None
    d = d or _cpr_local(h, l, c)
    d["width_pct"] = (d["tc"] - d["bc"]) / d["pp"] * 100 if d["pp"] else 0.0
    return d


def camarilla(h, l, c) -> dict:  # noqa: E741
    if _pb_levels is not None:
        try:
            return _as_dict(_pb_levels.camarilla(h, l, c))
        except Exception:  # noqa: BLE001
            pass
    return _camarilla_local(h, l, c)


def relationship(t_lo, t_hi, y_lo, y_hi) -> tuple[str, str]:
    """(relationship, bias) of today's band vs yesterday's."""
    if _pb_rel is not None:
        try:
            rel = _pb_rel.two_day_relationship(t_lo, t_hi, y_lo, y_hi)
            return rel.value, _pb_rel.BIAS.get(rel, "")
        except Exception:  # noqa: BLE001
            pass
    rel = _relationship_local(t_lo, t_hi, y_lo, y_hi)
    return rel, _LOCAL_BIAS.get(rel, "")


def classify_width(width_pct: float) -> str:
    conf = getattr(settings, "NIGHTLY_SCAN", {})
    narrow = float(conf.get("CPR_NARROW_PCT", 0.25))
    wide = float(conf.get("CPR_WIDE_PCT", 0.75))
    if width_pct < narrow:
        return "narrow"
    if width_pct > wide:
        return "wide"
    return "normal"


def _r(d: dict) -> dict:
    return {k: round(float(v), 2) for k, v in d.items()}


def compute_levels(last: HLC, prev: HLC | None) -> dict:
    """Levels for the session after ``last`` + two-day relationships."""
    fp = floor_pivots(last.h, last.l, last.c)
    cp = cpr(last.h, last.l, last.c)
    cam = camarilla(last.h, last.l, last.c)
    out = {
        "source": PIVOT_SOURCE,
        "data_date": last.day.isoformat(),
        "prev_hlc": {"h": last.h, "l": last.l, "c": last.c},
        "floor": _r(fp),
        "cpr": {**_r({k: cp[k] for k in ("pp", "tc", "bc")}),
                "width_pct": round(cp["width_pct"], 4),
                "class": classify_width(cp["width_pct"])},
        "camarilla": _r(cam),
        "cpr_relationship": "", "camarilla_relationship": "", "bias": "",
    }
    if prev is not None:
        pcp = cpr(prev.h, prev.l, prev.c)
        pcam = camarilla(prev.h, prev.l, prev.c)
        rel, bias = relationship(cp["bc"], cp["tc"], pcp["bc"], pcp["tc"])
        crel, _ = relationship(cam["l3"], cam["h3"], pcam["l3"], pcam["h3"])
        out.update(cpr_relationship=rel, camarilla_relationship=crel, bias=bias)
    return out


# ── Data access ──────────────────────────────────────────────────────────

def last_two_sessions(symbol: str, on_or_before: date) -> tuple[HLC | None, HLC | None]:
    """Daily H/L/C for the last two sessions ≤ ``on_or_before``."""
    from apps.market_data.models import Candle

    end = timezone.make_aware(datetime.combine(on_or_before + timedelta(days=1), datetime.min.time()))
    rows = list(
        Candle.objects.filter(symbol__tradingsymbol=symbol, symbol__exchange="NSE",
                              interval="1d", t__lt=end)
        .order_by("-t").values_list("t", "h", "l", "c")[:2]
    )
    if len(rows) < 2:
        rows = _daily_from_minutes(symbol, end) or rows
    hlcs = [HLC(timezone.localtime(t).date(), float(h), float(l), float(c)) for t, h, l, c in rows]
    last = hlcs[0] if hlcs else None
    prev = hlcs[1] if len(hlcs) > 1 else None
    return last, prev


def _daily_from_minutes(symbol: str, end: datetime) -> list:
    """Aggregate the last two sessions of 1m candles into daily rows (indices)."""
    from apps.market_data.models import Candle

    start = end - timedelta(days=10)
    rows = (Candle.objects.filter(symbol__tradingsymbol=symbol, interval="1m",
                                  t__gte=start, t__lt=end)
            .order_by("t").values_list("t", "h", "l", "c"))
    days: dict[date, list] = {}
    for t, h, l, c in rows.iterator():  # noqa: E741
        d = timezone.localtime(t).date()
        agg = days.get(d)
        if agg is None:
            days[d] = [t, float(h), float(l), float(c)]
        else:
            agg[1] = max(agg[1], float(h))
            agg[2] = min(agg[2], float(l))
            agg[3] = float(c)
    ordered = sorted(days.items(), reverse=True)[:2]
    return [(v[0], v[1], v[2], v[3]) for _, v in ordered]
